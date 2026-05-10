"""
test_predict_integration.py
ทดสอบ prediction pipeline ครบทุกขั้นตอนโดยไม่ต้องรัน API server

ทดสอบ 3 URL:
  1. URL phishing      — http://paypal.com.evil-login.tk
  2. URL legitimate    — https://www.google.com
  3. URL shortened     — https://bit.ly/something

รัน: python backend/tests/test_predict_integration.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.features.url_extractor import extract_url_features
from backend.xai.shap_explainer import SHAPExplainer

SAVE_DIR = ROOT / "backend" / "models" / "saved"


# ─────────────────────────────────────────────
# โหลด Model Artifacts
# ─────────────────────────────────────────────

def load_artifacts():
    try:
        with open(SAVE_DIR / "best_model.pkl",    "rb") as f: model = pickle.load(f)
        with open(SAVE_DIR / "scaler.pkl",        "rb") as f: scaler = pickle.load(f)
        with open(SAVE_DIR / "feature_names.pkl", "rb") as f: feature_names = pickle.load(f)
    except FileNotFoundError as e:
        print(f"[ERROR] {e} — รัน train.py ก่อน")
        sys.exit(1)
    return model, scaler, feature_names


# ─────────────────────────────────────────────
# แสดง Feature Comparison
# ─────────────────────────────────────────────

def print_feature_comparison(feature_names: list[str]) -> None:
    """
    เปรียบเทียบ features ที่ url_extractor มี vs features ที่ model ต้องการ
    แสดงว่า feature ใดถูกใช้จริง และตัวใดเป็นแค่ extra
    """
    # ดึง keys ทั้งหมดจาก extractor ด้วย URL ตัวอย่าง
    sample_feats = set(extract_url_features("https://example.com").keys())
    model_feats  = set(feature_names)

    covered = model_feats & sample_feats   # feature ที่ extractor cover ได้
    missing  = model_feats - sample_feats  # feature ที่ต้อง default=0
    extra    = sample_feats - model_feats  # feature ใน extractor ที่ไม่ได้ใช้

    print(f"\n{'='*70}")
    print(f"  Feature Comparison: url_extractor.py ↔ feature_names.pkl")
    print(f"{'='*70}")
    print(f"  url_extractor  : {len(sample_feats)} features รวม")
    print(f"  model ต้องการ  : {len(model_feats)} features")
    print(f"  Coverage       : {len(covered)}/{len(model_feats)} features ✅")
    if missing:
        print(f"  Missing        : {missing}  ← จะ default=0.0")
    else:
        print(f"  Missing        : ไม่มี — extractor cover ครบทุก feature ✅")
    print(f"\n  Features ที่ model ใช้จริง (ตามลำดับ):")
    for i, name in enumerate(feature_names, 1):
        status = "✅" if name in sample_feats else "⚠️ default=0"
        print(f"    {i:2d}. {name:<25s}  {status}")
    print(f"\n  Features พิเศษใน extractor (ไม่ได้ใช้ตอน train — {len(extra)} features):")
    for name in sorted(extra):
        print(f"       – {name}")


# ─────────────────────────────────────────────
# ทดสอบ URL เดียว
# ─────────────────────────────────────────────

def test_url(
    url: str,
    model,
    scaler,
    feature_names: list[str],
    explainer: SHAPExplainer,
) -> dict:
    """
    ทดสอบ prediction pipeline ครบทุกขั้นตอน:
    extract → map → scale → predict → SHAP
    """
    print(f"\n{'─'*70}")
    print(f"  URL: {url}")
    print(f"{'─'*70}")

    # 1. Extract features จาก URL จริง
    raw_feats = extract_url_features(url)

    # 2. Map ตาม feature_names (features ที่ไม่มีใน extractor → default 0.0)
    extracted = {f: round(float(raw_feats.get(f, 0.0)), 4) for f in feature_names}

    print(f"\n  Features ที่ extract ได้และใช้กับ model:")
    print(f"  {'Feature':<25s}  {'Extracted Value':>15}  {'In Extractor?'}")
    print(f"  {'─'*55}")
    for name in feature_names:
        val = extracted[name]
        src = "✅" if name in raw_feats else "⚠️ default=0"
        print(f"  {name:<25s}  {val:>15.4f}  {src}")

    # 3. Scale
    X_raw    = np.array([extracted[f] for f in feature_names], dtype=np.float32).reshape(1, -1)
    X_scaled = scaler.transform(X_raw)

    # 4. Predict
    pred_int = int(model.predict(X_scaled)[0])
    if hasattr(model, "predict_proba"):
        proba       = model.predict_proba(X_scaled)[0]
        confidence  = float(proba[pred_int])
        phishing_p  = float(proba[1])
    else:
        score       = model.decision_function(X_scaled)[0]
        confidence  = float(1 / (1 + np.exp(-score)))
        phishing_p  = confidence if pred_int == 1 else 1 - confidence

    label      = "PHISHING" if pred_int == 1 else "LEGITIMATE"
    risk_score = int(round(phishing_p * 100))

    print(f"\n  ┌─ Prediction ──────────────────────────────────────────┐")
    print(f"  │  Result     : {'⚠  ' + label if pred_int == 1 else '✓  ' + label:<20s}                    │")
    print(f"  │  Confidence : {confidence*100:>6.2f}%                                   │")
    print(f"  │  Risk Score : {risk_score:>3d} / 100                                 │")
    print(f"  └──────────────────────────────────────────────────────────┘")

    # 5. SHAP explanation
    scaled_dict = {f: float(v) for f, v in zip(feature_names, X_scaled[0])}
    shap_result = explainer.explain_local(scaled_dict)

    print(f"\n  Top SHAP Feature Contributions:")
    print(f"  {'Feature':<25s}  {'Raw Value':>9}  {'SHAP':>8}  Direction")
    print(f"  {'─'*65}")
    for feat in shap_result["top_features"]:
        name      = feat["feature_name"]
        raw_val   = extracted.get(name, 0.0)
        shap_val  = feat["shap_value"]
        direction = "↑ phishing" if feat["direction"] == "increases_risk" else "↓ safe"
        print(f"  {name:<25s}  {raw_val:>9.4f}  {shap_val:>+8.4f}  {direction}")

    return {
        "url": url, "prediction": label, "confidence": confidence,
        "risk_score": risk_score, "extracted_features": extracted,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Phishing Detector — Integration Test (url_extractor → predict)")
    print("=" * 70)

    model, scaler, feature_names = load_artifacts()
    print(f"\n  Model       : {type(model).__name__}")
    print(f"  Features    : {feature_names}")

    # แสดง feature comparison ก่อน
    print_feature_comparison(feature_names)

    # สร้าง SHAP explainer ครั้งเดียว — ใช้ซ้ำทุก URL
    explainer = SHAPExplainer(model, feature_names)

    # ─── 3 URL ทดสอบ ───────────────────────────────────────────────────────
    test_urls = [
        # 1. Phishing — ใช้ IP ใน domain + suspicious keywords + free TLD
        "http://paypal.com.evil-login.tk/verify?user=john&token=abc123&session=xyz",
        # 2. Legitimate — HTTPS, ไม่มี suspicious pattern
        "https://www.google.com/search?q=python+machine+learning+tutorial",
        # 3. URL shortener — hostname สั้น, ไม่ทราบ destination
        "https://bit.ly/3xAbc12",
    ]

    results = []
    for url in test_urls:
        r = test_url(url, model, scaler, feature_names, explainer)
        results.append(r)

    # ─── สรุปผล ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  {'URL':<55}  {'Result':<12}  {'Risk':>5}")
    print(f"  {'─'*68}")
    for r in results:
        url_short = r["url"][:53] + ".." if len(r["url"]) > 55 else r["url"]
        icon = "⚠" if r["prediction"] == "PHISHING" else "✓"
        print(f"  {icon} {url_short:<54}  {r['prediction']:<12}  {r['risk_score']:>3}/100")

    print(f"\n{'='*70}")
    print(f"  ✅ Integration test เสร็จสมบูรณ์")
    print(f"{'='*70}")
