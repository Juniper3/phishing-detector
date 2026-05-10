"""
test_blind_spot_analysis.py
วิเคราะห์ blind spots และผล before/after ของการพยายามเพิ่ม features

ผลการทดลอง (บันทึกไว้สำหรับ thesis):
  ก่อน (9 features):     paypal.com.evil-login.tk P=0.1216 → LEGIT ❌
  ลอง 12 features:       paypal.com.evil-login.tk P=0.0049 → LEGIT ❌ (แย่ลง!)
  หลัง (9 features revert): ผลเดิม

สาเหตุ: PhiUSIIL dataset distribution ≠ heuristic expectations
  has_suspicious_tld=1 → 93.5% LEGIT ใน PhiUSIIL
  brand_in_subdomain=1 → 97.3% LEGIT ใน PhiUSIIL
  ดังนั้น model เรียนว่า features เหล่านี้ = legitimate

รัน: python backend/tests/test_blind_spot_analysis.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.features.url_extractor import (
    extract_url_features, SUSPICIOUS_TLDS, TARGET_BRANDS, URL_SHORTENERS,
)
from backend.xai.shap_explainer import SHAPExplainer

SAVE_DIR     = ROOT / "backend" / "models" / "saved"
PHIUSIIL_CSV = ROOT / "data" / "raw" / "dataset.csv"


# ─────────────────────────────────────────────
# โหลด Model
# ─────────────────────────────────────────────

def load_artifacts():
    try:
        with open(SAVE_DIR / "best_model.pkl",    "rb") as f: model = pickle.load(f)
        with open(SAVE_DIR / "scaler.pkl",        "rb") as f: scaler = pickle.load(f)
        with open(SAVE_DIR / "feature_names.pkl", "rb") as f: feature_names = pickle.load(f)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}"); sys.exit(1)
    return model, scaler, feature_names


# ─────────────────────────────────────────────
# ตรวจ Feature-Label Correlation ใน PhiUSIIL
# ─────────────────────────────────────────────

def analyze_feature_label_correlation() -> None:
    """
    แสดงว่า features ที่เพิ่มมามี correlation กับ phishing label
    อย่างไรใน PhiUSIIL — ผลนี้สำคัญมากสำหรับ thesis
    """
    print(f"\n{'='*70}")
    print(f"  Feature-Label Correlation ใน PhiUSIIL Dataset (สำหรับ thesis)")
    print(f"{'='*70}")

    df = pd.read_csv(PHIUSIIL_CSV)
    if "URL" in df.columns: df = df.rename(columns={"URL": "url"})
    df["label"] = df["label"].astype(int)
    n_total = len(df)

    # ── has_suspicious_tld ──────────────────────────────────────────────
    tld = df["TLD"].fillna("").str.lower().str.strip(".")
    sus_mask = tld.isin(SUSPICIOUS_TLDS)
    n_sus = sus_mask.sum()
    n_sus_phi = int(df.loc[sus_mask, "label"].sum())
    n_sus_leg = n_sus - n_sus_phi
    top_tlds = tld[sus_mask].value_counts().head(5)

    print(f"\n  Feature: has_suspicious_tld")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  TLDs ใน SUSPICIOUS_TLDS set ที่พบ: {top_tlds.to_dict()}")
    print(f"  rows ที่ has_suspicious_tld=1 : {n_sus:,} / {n_total:,} ({n_sus/n_total*100:.1f}%)")
    print(f"  ในนั้น Phishing               : {n_sus_phi:,} ({n_sus_phi/n_sus*100:.1f}%)")
    print(f"  ในนั้น Legitimate             : {n_sus_leg:,} ({n_sus_leg/n_sus*100:.1f}%)")
    print(f"  ⚠ INSIGHT: {n_sus_leg/n_sus*100:.0f}% เป็น Legitimate → model เรียนว่า suspicious TLD = LEGIT")

    # ── brand_in_subdomain ──────────────────────────────────────────────
    domain = df["Domain"].fillna("").str.lower()
    def subdomain_part(d):
        parts = d.rstrip(".").split(".")
        return ".".join(parts[:-2]) if len(parts) > 2 else ""
    subdomain_series = domain.apply(subdomain_part)
    brand_mask = np.zeros(n_total, dtype=bool)
    for brand in TARGET_BRANDS:
        brand_mask |= subdomain_series.str.contains(brand, regex=False, na=False).values
    n_brand = brand_mask.sum()
    n_brand_phi = int(df.loc[brand_mask, "label"].sum())

    print(f"\n  Feature: brand_in_subdomain")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  rows ที่ brand_in_subdomain=1 : {n_brand:,} / {n_total:,} ({n_brand/n_total*100:.2f}%)")
    print(f"  ในนั้น Phishing               : {n_brand_phi:,} ({n_brand_phi/n_brand*100:.1f}%)")
    print(f"  ในนั้น Legitimate             : {n_brand-n_brand_phi:,} ({(n_brand-n_brand_phi)/n_brand*100:.1f}%)")
    print(f"  ⚠ INSIGHT: {(n_brand-n_brand_phi)/n_brand*100:.0f}% เป็น Legitimate → model เรียนว่า brand in subdomain = LEGIT")

    # ── has_shortener ──────────────────────────────────────────────────
    url_s = df["url"].fillna("").str.lower()
    short_mask = np.zeros(n_total, dtype=bool)
    for s in URL_SHORTENERS:
        short_mask |= url_s.str.contains(s, regex=False, na=False).values
    n_short = short_mask.sum()
    n_short_phi = int(df.loc[short_mask, "label"].sum())

    print(f"\n  Feature: has_shortener")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  rows ที่ has_shortener=1      : {n_short:,} / {n_total:,} ({n_short/n_total*100:.2f}%)")
    print(f"  ในนั้น Phishing               : {n_short_phi:,} ({n_short_phi/n_short*100:.1f}%)")
    print(f"  ในนั้น Legitimate             : {n_short-n_short_phi:,} ({(n_short-n_short_phi)/n_short*100:.1f}%)")
    print(f"  ~ INSIGHT: {n_short_phi/n_short*100:.0f}% Phishing — สัญญาณอ่อน แต่ยังเป็น positive correlation")

    print(f"""
  ┌─ Thesis Finding: Dataset Distribution ≠ Security Heuristics ──────────┐
  │                                                                         │
  │  Heuristic ที่นักวิจัย assume:                                           │
  │    TLD ฟรี (.tk/.ml/.ga) = phishing signal                              │
  │    แบรนด์ดังใน subdomain  = phishing signal                              │
  │                                                                         │
  │  ความเป็นจริงใน PhiUSIIL (real-world dataset):                          │
  │    has_suspicious_tld=1 → {n_sus_leg/n_sus*100:.0f}% LEGIT (startups ใช้ free TLD จริง)       │
  │    brand_in_subdomain=1 → {(n_brand-n_brand_phi)/n_brand*100:.0f}% LEGIT (api.paypal.com ถูกต้อง)            │
  │                                                                         │
  │  บทสรุป: URL-only lexical features ไม่เพียงพอสำหรับ domain-squatting     │
  │  ต้องการ: DNS-based features / WHOIS age / SSL certificate              │
  └─────────────────────────────────────────────────────────────────────────┘""")


# ─────────────────────────────────────────────
# Before/After Comparison Table
# ─────────────────────────────────────────────

def print_before_after(model, scaler, feature_names) -> None:
    """แสดงตารางเปรียบเทียบก่อน/หลัง ของ blind spot URLs"""
    print(f"\n{'='*70}")
    print(f"  Before / After Comparison — Blind Spot URLs")
    print(f"{'='*70}")
    print(f"  (model: {type(model).__name__}, features: {len(feature_names)})")

    # ผล BEFORE ที่บันทึกไว้จากการ experiment
    before_results = {
        "http://paypal.com.evil-login.tk/verify?user=john": {
            "9feat_before":  ("LEGIT", 0.1216),
            "12feat_attempt": ("LEGIT", 0.0049),  # แย่ลง!
        },
        "http://192.168.1.1/admin": {
            "9feat_before":  ("LEGIT", 0.2437),
            "12feat_attempt": ("LEGIT", 0.3800),
        },
        "https://bit.ly/3xAbc12": {
            "9feat_before":  ("LEGIT", 0.0018),
            "12feat_attempt": ("LEGIT", 0.0019),
        },
    }

    explainer = SHAPExplainer(model, feature_names)

    print(f"\n  {'URL':<45}  {'Before':>9}  {'12feat':>9}  {'After':>9}  Verdict")
    print(f"  {'─'*85}")

    for url, exp_dict in before_results.items():
        # คำนวณ after (model ปัจจุบัน = 9 features restored)
        f = extract_url_features(url)
        extracted = {n: float(f.get(n, 0.0)) for n in feature_names}
        X = np.array([extracted[n] for n in feature_names], dtype=np.float32).reshape(1, -1)
        Xs = scaler.transform(X)
        prob_after = float(model.predict_proba(Xs)[0][1])
        label_after = "PHISHING" if prob_after >= 0.5 else "LEGIT"

        before_lbl, before_p = exp_dict["9feat_before"]
        mid_lbl,    mid_p    = exp_dict["12feat_attempt"]

        verdict = "✓ Fixed" if label_after == "PHISHING" else "⚠ Still miss"

        url_short = url[:43] + ".." if len(url) > 45 else url
        print(
            f"  {url_short:<45}  "
            f"P={before_p:.4f}  P={mid_p:.4f}  P={prob_after:.4f}  {verdict}"
        )

    print(f"\n  Columns: Before(9feat original) | 12feat attempt | After(9feat restored)")
    print(f"\n  ┌─ Key Finding ──────────────────────────────────────────────────────────┐")
    print(f"  │  เพิ่ม 3 features → blind spot WORSE, ไม่ได้ช่วยเลย                    │")
    print(f"  │  สาเหตุ: PhiUSIIL dataset มี legitimate .tk/.brand-subdomain จำนวนมาก  │")
    print(f"  │  model เรียนกลับทิศ: suspicious TLD/brand = LEGIT                       │")
    print(f"  │  ข้อเสนอ thesis: URL-only features ไม่เพียงพอสำหรับ domain-squatting     │")
    print(f"  │  ต้องการ: DNS registration age, WHOIS, SSL cert, webpage content       │")
    print(f"  └─────────────────────────────────────────────────────────────────────────┘")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Blind Spot Analysis — URL Feature Engineering Experiment")
    print("=" * 70)

    model, scaler, feature_names = load_artifacts()
    print(f"\n  Model       : {type(model).__name__}")
    print(f"  Features    : {feature_names}")

    # 1. วิเคราะห์ correlation ใน training data
    analyze_feature_label_correlation()

    # 2. ตารางเปรียบเทียบ before/after
    print_before_after(model, scaler, feature_names)

    print(f"\n{'='*70}")
    print(f"  Analysis เสร็จสิ้น")
    print(f"{'='*70}")
