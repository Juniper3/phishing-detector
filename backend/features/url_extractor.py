"""
url_extractor.py
ดึง features จาก URL สำหรับใช้ใน phishing detection model
อ้างอิงงานวิจัย: Phishing Website Detection using Machine Learning (2020-2024)
"""

import re
import math
import ipaddress
from urllib.parse import urlparse, parse_qs, unquote


# --- คำที่น่าสงสัยพบบ่อยใน phishing URL ---
SUSPICIOUS_WORDS = {
    "login", "signin", "verify", "secure", "account", "update",
    "banking", "confirm", "password", "credential", "wallet",
    "paypal", "apple", "google", "microsoft", "amazon", "facebook",
    "ebay", "netflix", "support", "helpdesk", "service", "alert",
    "suspended", "limited", "unusual", "access", "click", "free",
    "prize", "winner", "lucky", "gift", "bonus", "offer",
}

# --- TLD ที่พบบ่อยใน phishing sites ---
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",          # Freenom free TLDs
    "xyz", "top", "club", "online", "site",  # cheap TLDs
    "pw", "cc", "biz", "info",               # commonly abused
    "work", "click", "link", "buzz",
}

# --- URL shortening services ที่นิยม ---
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "short.link", "adf.ly", "bit.do",
    "shorte.st", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc",
}

# --- แบรนด์ดังที่มักถูก spoof ---
TARGET_BRANDS = {
    "paypal", "apple", "google", "microsoft", "amazon",
    "facebook", "instagram", "twitter", "netflix", "ebay",
    "wellsfargo", "bankofamerica", "citibank", "chase",
    "dhl", "fedex", "ups", "usps",
}


def _has_ip_address(hostname: str) -> int:
    """ตรวจสอบว่า hostname เป็น IP address แทน domain name หรือไม่"""
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        pass
    # IPv4 แบบ octal/hex ที่ซ่อนใน URL เช่น 0x7f000001
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", hostname):
        return 1
    return 0


def _get_entropy(text: str) -> float:
    """คำนวณ Shannon entropy — URL ที่สุ่มมากมักเป็น phishing/malware"""
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _count_subdomains(hostname: str) -> int:
    """นับจำนวน subdomain (ไม่รวม www และ domain หลัก)
    เช่น secure.login.paypal.com → 2 subdomains"""
    parts = hostname.split(".")
    # ลบ www prefix ออกก่อนนับ
    if parts and parts[0] == "www":
        parts = parts[1:]
    # domain หลัก + TLD = 2 ส่วนสุดท้าย → ที่เหลือคือ subdomains
    return max(0, len(parts) - 2)


def _get_tld(hostname: str) -> str:
    """ดึง TLD จาก hostname เช่น paypal.com → 'com'"""
    parts = hostname.rstrip(".").split(".")
    return parts[-1].lower() if parts else ""


def _get_registered_domain(hostname: str) -> str:
    """ดึง registered domain (ไม่รวม subdomains) เช่น a.b.paypal.com → 'paypal'"""
    parts = hostname.rstrip(".").split(".")
    # ส่วนที่ 2 จากท้าย (ก่อน TLD)
    return parts[-2].lower() if len(parts) >= 2 else hostname.lower()


def extract_url_features(url: str) -> dict:
    """
    ดึง features ทั้งหมดจาก URL string

    Args:
        url: URL ที่ต้องการวิเคราะห์ เช่น 'http://secure-login.paypal.com.evil.tk/verify'

    Returns:
        dict ของ features พร้อมใช้กับ XGBoost model
    """
    # --- Normalize URL ---
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full_url = url.lower()
    hostname_lower = hostname.lower()

    # --- 1. Lexical Features: ความยาวส่วนต่างๆ ของ URL ---
    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    # --- 2. Character Count Features: จำนวน special characters ---
    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_underscores = url.count("_")
    num_slashes = url.count("/")
    num_at = url.count("@")                # @ ใน URL บ่งชี้การหลอกลวง เช่น http://user@evil.com
    num_ampersand = url.count("&")
    num_question_mark = url.count("?")
    num_equal = url.count("=")
    num_percent = url.count("%")           # percent-encoded characters มากผิดปกติ
    num_digits = sum(c.isdigit() for c in url)
    num_double_slash = url.count("//") - 1  # หัก 1 ออก (เพราะ http:// นับด้วย)
    num_www = full_url.count("www")

    # สัดส่วนตัวเลข/ตัวอักษรพิเศษต่อความยาว URL ทั้งหมด
    digit_ratio = num_digits / url_length if url_length > 0 else 0.0
    special_chars = sum(1 for c in url if not c.isalnum() and c not in "-._~/")
    special_char_ratio = special_chars / url_length if url_length > 0 else 0.0

    # --- 3. Hostname Features ---
    has_ip = _has_ip_address(hostname)
    num_subdomains = _count_subdomains(hostname)
    hostname_num_dots = hostname.count(".")
    tld = _get_tld(hostname)
    registered_domain = _get_registered_domain(hostname)

    has_suspicious_tld = int(tld in SUSPICIOUS_TLDS)
    domain_has_digits = int(bool(re.search(r"\d", registered_domain)))
    domain_has_hyphen = int("-" in registered_domain)

    # --- 4. Path Features ---
    path_depth = len([p for p in path.split("/") if p])  # ความลึกของ directory
    has_exe = int(".exe" in path.lower())
    has_php = int(".php" in path.lower())
    has_html = int(path.lower().endswith((".html", ".htm")))

    # --- 5. Security / Protocol Features ---
    has_https = int(parsed.scheme.lower() == "https")
    has_port = int(parsed.port is not None)
    # port ที่ผิดปกติ (ไม่ใช่ 80/443) น่าสงสัย
    port_value = parsed.port if parsed.port is not None else 0
    has_non_standard_port = int(port_value not in (0, 80, 443))

    # --- 6. Query String Features ---
    params = parse_qs(query)
    num_params = len(params)

    # --- 7. Suspicious Pattern Features ---
    has_shortener = int(hostname_lower in URL_SHORTENERS or
                        any(s in hostname_lower for s in URL_SHORTENERS))

    # ตรวจ suspicious words ใน URL ทั้งหมด
    has_suspicious_words = int(
        any(word in full_url for word in SUSPICIOUS_WORDS)
    )

    # แบรนด์ดังอยู่ใน subdomain (ไม่ใช่ registered domain) → phishing สัญญาณ
    subdomains_part = hostname_lower.replace(registered_domain, "").replace(tld, "")
    brand_in_subdomain = int(any(brand in subdomains_part for brand in TARGET_BRANDS))

    # แบรนด์ดังอยู่ใน path → อาจเป็นการ spoof
    brand_in_path = int(any(brand in path.lower() for brand in TARGET_BRANDS))

    # registered domain มีชื่อแบรนด์แต่ไม่ใช่ official → typosquatting
    brand_in_domain = int(any(brand in registered_domain for brand in TARGET_BRANDS))

    # --- 8. Obfuscation Features ---
    # URL ที่มี hex encoding มาก เช่น %41%42%43 แทน "ABC"
    has_hex_encoding = int(bool(re.search(r"%[0-9a-fA-F]{2}", url)))
    num_hex_encoded = len(re.findall(r"%[0-9a-fA-F]{2}", url))

    # มี // ใน path (อาจเป็น redirect trick)
    has_redirect = int("//" in path)

    # --- 9. Statistical / Information-Theoretic Features ---
    url_entropy = _get_entropy(url)           # entropy สูง = URL สุ่มมาก = น่าสงสัย
    hostname_entropy = _get_entropy(hostname)

    # รวม features ทั้งหมดเป็น dict
    features = {
        # Lexical
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,

        # Character counts
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_underscores": num_underscores,
        "num_slashes": num_slashes,
        "num_at": num_at,
        "num_ampersand": num_ampersand,
        "num_question_mark": num_question_mark,
        "num_equal": num_equal,
        "num_percent": num_percent,
        "num_digits": num_digits,
        "num_double_slash": num_double_slash,
        "num_www": num_www,
        "digit_ratio": round(digit_ratio, 4),
        "special_char_ratio": round(special_char_ratio, 4),

        # Hostname
        "has_ip": has_ip,
        "num_subdomains": num_subdomains,
        "hostname_num_dots": hostname_num_dots,
        "has_suspicious_tld": has_suspicious_tld,
        "domain_has_digits": domain_has_digits,
        "domain_has_hyphen": domain_has_hyphen,

        # Path
        "path_depth": path_depth,
        "has_exe": has_exe,
        "has_php": has_php,
        "has_html": has_html,

        # Security
        "has_https": has_https,
        "has_port": has_port,
        "has_non_standard_port": has_non_standard_port,
        "port_value": port_value,

        # Query
        "num_params": num_params,

        # Suspicious patterns
        "has_shortener": has_shortener,
        "has_suspicious_words": has_suspicious_words,
        "brand_in_subdomain": brand_in_subdomain,
        "brand_in_path": brand_in_path,
        "brand_in_domain": brand_in_domain,

        # Obfuscation
        "has_hex_encoding": has_hex_encoding,
        "num_hex_encoded": num_hex_encoded,
        "has_redirect": has_redirect,

        # Statistical
        "url_entropy": round(url_entropy, 4),
        "hostname_entropy": round(hostname_entropy, 4),
    }

    return features


if __name__ == "__main__":
    test_urls = [
        "http://paypal.com.evil-login.tk/secure",
        "https://192.168.1.1/admin",
        "https://www.google.com",
        "https://bit.ly/test",
    ]

    highlight = {
        "has_suspicious_tld", "brand_in_subdomain", "has_shortener",
        "has_ip", "num_subdomains",
    }

    for u in test_urls:
        feats = extract_url_features(u)
        print(f"\nURL: {u}")
        print(f"  {'Feature':<25} {'Value':>8}  {'< focus' if True else ''}")
        print(f"  {'-'*40}")
        for k, v in feats.items():
            marker = " <--" if k in highlight else ""
            print(f"  {k:<25} {str(v):>8}{marker}")
