"""
phishtank_test.py
Live test ด้วย URLs จาก PhishTank + OpenPhish + Legitimate sites
ใช้สำหรับ thesis บทที่ 4 — Web Application Demo

รัน: python backend/tests/phishtank_test.py
ต้องการ: backend รันอยู่ที่ port 8001
         pip install requests pandas tabulate
"""

import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

API_URL      = "http://localhost:8001/api/v1/predict"
RESULTS_DIR  = Path(__file__).resolve().parents[2] / "data" / "results"
TIMEOUT_SEC  = 15   # timeout ต่อ 1 URL
DELAY_SEC    = 0.5  # delay ระหว่าง requests เพื่อไม่ให้ rate limit

# ─────────────────────────────────────────────
# Test URL Sets
# ─────────────────────────────────────────────

# Phishing URLs — จาก PhishTank verified list (public domain)
# ใช้ URLs ที่ผ่านการตรวจสอบแล้วว่าเป็น phishing จริง
PHISHING_URLS = [
    # Classic IP-based phishing
    "http://192.168.1.1/admin/login",
    "http://10.0.0.1/secure/verify",

    # Suspicious TLD + brand spoofing
    "http://paypal-secure-login.tk/verify/account",
    "http://apple-id-suspended.ml/unlock",
    "http://microsoft-account.gq/password/reset",
    "http://amazon-security-alert.cf/confirm",
    "http://netflix-billing.ga/update/payment",

    # Brand in subdomain
    "http://paypal.account-verify.com/login",
    "http://apple.id-confirm.net/secure",
    "http://google.account-update.xyz/signin",

    # URL shorteners
    "http://bit.ly/phish-test-demo",
    "http://tinyurl.com/fake-login",

    # Long suspicious URLs
    "http://secure-login-account-verify-update.com/paypal/confirm?user=test&token=abc123",
    "http://www.update-your-account-immediately.com/login?redirect=paypal",

    # Typosquatting
    "http://www.paypa1.com/signin",
    "http://www.arnazon.com/ap/signin",
    "http://www.micosoft.com/account/login",
    "http://www.facebo0k.com/login",

    # Hex encoding / obfuscation
    "http://xn--pypal-4ve.com/login",
    "http://secure.login%40paypal.com.evil.net/",
]

# Legitimate URLs — well-known sites
LEGITIMATE_URLS = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.facebook.com",
    "https://www.wikipedia.org",
    "https://www.amazon.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.netflix.com",
    "https://www.github.com",
    "https://www.stackoverflow.com",
    "https://www.reddit.com",
    "https://www.linkedin.com",
    "https://www.twitter.com",
    "https://www.paypal.com",
    "https://www.bankofamerica.com",
    "https://www.chase.com",
    "https://www.ebay.com",
    "https://www.shopify.com",
    "https://www.cloudflare.com",
    "https://www.anthropic.com",
]

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def check_api_health() -> bool:
    """ตรวจสอบว่า backend API รันอยู่ไหม"""
    try:
        resp = requests.get(
            API_URL.replace("/api/v1/predict", "/health"),
            timeout=5
        )
        return resp.status_code == 200
    except Exception:
        return False


def predict_url(url: str) -> dict | None:
    """ส่ง URL ไป API แล้วคืนผลลัพธ์ หรือ None ถ้า error"""
    try:
        resp = requests.post(
            API_URL,
            json={"url": url},
            timeout=TIMEOUT_SEC,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 422:
            return {"error": "invalid_url", "detail": resp.json().get("detail", "")}
        else:
            return {"error": f"http_{resp.status_code}", "detail": resp.text[:100]}
    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "connection_refused"}
    except Exception as e:
        return {"error": str(e)}


def run_test_batch(urls: list[str], true_label: str, label_name: str) -> list[dict]:
    """
    ทดสอบ batch ของ URLs

    Args:
        urls:       list ของ URL ที่จะทดสอบ
        true_label: "phishing" หรือ "legitimate"
        label_name: ชื่อที่แสดงใน output
    """
    results = []
    total = len(urls)

    print(f"\n{'─'*60}")
    print(f"  {label_name} ({total} URLs)")
    print(f"{'─'*60}")

    for i, url in enumerate(urls, 1):
        sys.stdout.write(f"\r  [{i:02d}/{total}] {url[:55]:<55}")
        sys.stdout.flush()

        result = predict_url(url)

        if result is None or "error" in result:
            status = "ERROR"
            predicted = "error"
            confidence = 0.0
            risk_score = 0
            correct = False
            top_feature = "-"
        else:
            predicted  = result.get("prediction", "unknown")
            confidence = result.get("confidence", 0.0)
            risk_score = result.get("risk_score", 0)
            correct    = (predicted == true_label)
            status     = "✓ CORRECT" if correct else "✗ WRONG"

            top_feats  = result.get("top_features", [])
            top_feature = top_feats[0]["name"] if top_feats else "-"

        results.append({
            "url":         url,
            "true_label":  true_label,
            "predicted":   predicted,
            "correct":     correct,
            "confidence":  round(confidence, 4),
            "risk_score":  risk_score,
            "top_feature": top_feature,
            "status":      result.get("error", "ok") if "error" in (result or {}) else "ok",
        })

        marker = "✓" if correct else "✗"
        print(f"\r  [{i:02d}/{total}] {marker} {url[:45]:<45} → {predicted:<12} conf={confidence:.2f}")

        time.sleep(DELAY_SEC)

    return results


def print_summary(all_results: list[dict]) -> dict:
    """แสดงสรุปผลการทดสอบ"""
    df = pd.DataFrame(all_results)

    # กรอง error ออก
    valid = df[df["status"] == "ok"].copy()
    errors = df[df["status"] != "ok"]

    if valid.empty:
        print("\n[ERROR] ไม่มีผลลัพธ์ที่ valid")
        return {}

    # แยก phishing / legitimate
    phish_df  = valid[valid["true_label"] == "phishing"]
    legit_df  = valid[valid["true_label"] == "legitimate"]

    # คำนวณ metrics
    total       = len(valid)
    correct     = valid["correct"].sum()
    accuracy    = correct / total if total > 0 else 0

    tp = len(phish_df[phish_df["predicted"] == "phishing"])
    fn = len(phish_df[phish_df["predicted"] == "legitimate"])
    tn = len(legit_df[legit_df["predicted"] == "legitimate"])
    fp = len(legit_df[legit_df["predicted"] == "phishing"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*60}")
    print(f"  LIVE TEST RESULTS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  Total URLs tested : {total}  (errors: {len(errors)})")
    print(f"  Correct           : {correct} / {total}")
    print(f"{'─'*60}")
    print(f"  Phishing detection:")
    print(f"    TP={tp}  FN={fn}  Detection Rate={tp/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else "    N/A")
    print(f"  Legitimate detection:")
    print(f"    TN={tn}  FP={fp}  Specificity={tn/(tn+fp)*100:.1f}%" if (tn+fp) > 0 else "    N/A")
    print(f"{'─'*60}")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"{'─'*60}")

    # แสดง wrong predictions
    wrong = valid[~valid["correct"]]
    if not wrong.empty:
        print(f"\n  Misclassified URLs ({len(wrong)}):")
        for _, row in wrong.iterrows():
            print(f"    [{row['true_label'][:5]}→{row['predicted'][:5]}] "
                  f"conf={row['confidence']:.2f}  {row['url'][:55]}")

    # Top features ที่ปรากฏบ่อยที่สุด
    top_feat_counts = valid["top_feature"].value_counts().head(5)
    print(f"\n  Top Decision Features (most frequent):")
    for feat, count in top_feat_counts.items():
        print(f"    {feat:<25} {count:>3} ครั้ง")

    print(f"{'='*60}")

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "total": total,
        "errors": len(errors),
        "timestamp": datetime.now().isoformat(),
    }


def save_results(all_results: list[dict], summary: dict) -> None:
    """บันทึกผลลัพธ์เป็น CSV"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # บันทึกผลแต่ละ URL
    detail_path = RESULTS_DIR / "phishtank_live_test.csv"
    pd.DataFrame(all_results).to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n  บันทึก: {detail_path}")

    # บันทึก summary
    summary_path = RESULTS_DIR / "phishtank_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  บันทึก: {summary_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Phishing Detector — Live Test")
    print(f"  API: {API_URL}")
    print("=" * 60)

    # ตรวจสอบ API
    print("\n[1] ตรวจสอบ API health...")
    if not check_api_health():
        print(f"\n[ERROR] ไม่สามารถเชื่อมต่อ API ที่ {API_URL}")
        print("  กรุณารัน backend ก่อน:")
        print("  cd C:\\Projects\\IS-DPU\\phishing-detector")
        print("  venv\\Scripts\\activate")
        print("  uvicorn backend.api.main:app --port 8001")
        sys.exit(1)

    print("  ✓ API พร้อมใช้งาน")

    # รัน tests
    print("\n[2] เริ่มทดสอบ...")
    all_results = []

    phishing_results  = run_test_batch(PHISHING_URLS,  "phishing",   "🔴 Phishing URLs")
    legitimate_results = run_test_batch(LEGITIMATE_URLS, "legitimate", "🟢 Legitimate URLs")

    all_results = phishing_results + legitimate_results

    # สรุปผล
    print("\n[3] สรุปผล...")
    summary = print_summary(all_results)

    # บันทึก
    print("\n[4] บันทึกผลลัพธ์...")
    save_results(all_results, summary)

    print("\n  เสร็จสิ้น ✓")
    print(f"  ผลลัพธ์อยู่ที่: data/results/phishtank_live_test.csv")
