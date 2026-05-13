"""
train.py (v3 — 9 URL-Only Features, Restored)
หมายเหตุจากการทดลอง v3 (12 features):
  - เพิ่ม has_suspicious_tld, brand_in_subdomain, has_shortener
  - พบว่า PhiUSIIL มี distribution ตรงข้ามกับ heuristic:
      has_suspicious_tld=1 → 93.5% LEGIT (ไม่ใช่ phishing)
      brand_in_subdomain=1 → 97.3% LEGIT
    เพราะ dataset มี legitimate .tk/.ml/.xyz startups จำนวนมาก
  - features เหล่านี้จึงทำให้ model เรียน "suspicious TLD = legit"
  - ผล: blind spot URLs ถูก predict เป็น legit ด้วย confidence สูงขึ้น (แย่ลง)

  ผลสรุปสำหรับ thesis: blind spot เกิดจาก dataset distribution ≠ heuristic
  การแก้ที่ถูกต้องคือ:
    1. ต้องการ dataset ที่มี domain-squatting examples โดยเฉพาะ
    2. หรือใช้ rule-based detection สำหรับ pattern เหล่านี้แยกต่างหาก

  ตัดสินใจ: revert กลับ 9 features เพื่อ maintain การ generalize บน ISCX

รัน: python backend/models/train.py
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.features.url_extractor import SUSPICIOUS_TLDS, TARGET_BRANDS, URL_SHORTENERS
from backend.models.feature_config import FEATURE_MAP, CANONICAL_FEATURES, PHI_LABEL_CANDIDATES

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

PHIUSIIL_CSV = ROOT / "data" / "raw" / "dataset.csv"
ISCX_CSV     = ROOT / "data" / "raw" / "iscx_url.csv"
SAVE_DIR     = ROOT / "backend" / "models" / "saved"
RESULTS_DIR  = ROOT / "data" / "results"

# ─────────────────────────────────────────────
# FEATURE_MAP และ CANONICAL_FEATURES import มาจาก feature_config.py
# ─────────────────────────────────────────────

_PHI_DROP = {"FILENAME", "URL", "url", "Domain", "TLD", "Title",
             "filename", "domain", "tld", "title"}


# ─────────────────────────────────────────────
# Helper: คำนวณ 3 features ใหม่
# ─────────────────────────────────────────────

def _compute_new_features_phi(df: pd.DataFrame) -> np.ndarray:
    """
    คำนวณ has_suspicious_tld, brand_in_subdomain, has_shortener
    จาก PhiUSIIL ที่มี TLD, Domain, URL columns
    ใช้ vectorized computation เพื่อ performance (235k rows)
    """
    n = len(df)

    # ── has_suspicious_tld ──────────────────────────────────────────────
    # TLD column มีค่าเช่น "com", "tk", "ml" → check ว่าอยู่ใน SUSPICIOUS_TLDS
    tld_series = df.get("TLD", pd.Series([""] * n, index=df.index))
    tld_series = tld_series.fillna("").astype(str).str.lower().str.strip(".")
    has_suspicious_tld = tld_series.isin(SUSPICIOUS_TLDS).astype(np.int8).values

    # ── brand_in_subdomain ──────────────────────────────────────────────
    # Domain column เช่น "paypal.com.evil-login.tk"
    # ตัดสอง parts สุดท้าย (SLD + TLD) ออก เหลือ subdomain portion
    domain_series = df.get("Domain", pd.Series([""] * n, index=df.index))
    domain_series = domain_series.fillna("").astype(str).str.lower()

    def _subdomain_part(d: str) -> str:
        """ดึง subdomain portion ทิ้ง SLD+TLD สองส่วนสุดท้าย"""
        parts = d.rstrip(".").split(".")
        return ".".join(parts[:-2]) if len(parts) > 2 else ""

    subdomain_series = domain_series.apply(_subdomain_part)
    brand_in_subdomain = np.zeros(n, dtype=np.int8)
    for brand in TARGET_BRANDS:
        # OR-accumulate: brand ใดก็ได้ใน subdomain → flag = 1
        brand_in_subdomain |= (
            subdomain_series.str.contains(brand, regex=False, na=False).astype(np.int8).values
        )

    # ── has_shortener ───────────────────────────────────────────────────
    # URL column — ตรวจว่า hostname เป็น URL shortener service
    url_series = df.get("URL", df.get("url", pd.Series([""] * n, index=df.index)))
    url_series = url_series.fillna("").astype(str).str.lower()
    has_shortener = np.zeros(n, dtype=np.int8)
    for shortener in URL_SHORTENERS:
        has_shortener |= (
            url_series.str.contains(shortener, regex=False, na=False).astype(np.int8).values
        )

    return np.column_stack([
        has_suspicious_tld, brand_in_subdomain, has_shortener,
    ]).astype(np.float32)


def _compute_new_features_iscx(df: pd.DataFrame) -> np.ndarray:
    """
    คำนวณ features ใหม่จาก ISCX (ไม่มี URL/Domain column)
    - has_suspicious_tld: คำนวณได้จาก 'tld' column ✅
    - brand_in_subdomain: ไม่มีข้อมูล → 0 (limitation ของ ISCX dataset)
    - has_shortener:      ไม่มีข้อมูล → 0 (limitation ของ ISCX dataset)
    """
    n = len(df)

    # ── has_suspicious_tld ──────────────────────────────────────────────
    # ISCX มี 'tld' column (ตัวเล็ก) เช่น "com", "tk", "net"
    tld_series = df.get("tld", pd.Series([""] * n, index=df.index))
    tld_series = tld_series.fillna("").astype(str).str.lower().str.strip(".")
    has_suspicious_tld = tld_series.isin(SUSPICIOUS_TLDS).astype(np.int8).values

    # ── brand_in_subdomain = 0, has_shortener = 0 ───────────────────────
    # ISCX ไม่มี URL/Domain string → ไม่สามารถคำนวณได้
    brand_in_subdomain = np.zeros(n, dtype=np.float32)
    has_shortener      = np.zeros(n, dtype=np.float32)

    return np.column_stack([
        has_suspicious_tld, brand_in_subdomain, has_shortener,
    ]).astype(np.float32)


# ─────────────────────────────────────────────
# 1. โหลด PhiUSIIL — map columns → canonical names
# ─────────────────────────────────────────────

def load_phiusiil() -> tuple[np.ndarray, np.ndarray]:
    """
    โหลด PhiUSIIL dataset และ map columns ไปยัง canonical feature names

    Returns:
        X (n_samples, 9), y (n_samples,)
    """
    if not PHIUSIIL_CSV.exists():
        raise FileNotFoundError(f"ไม่พบ PhiUSIIL dataset: {PHIUSIIL_CSV}")

    df = pd.read_csv(PHIUSIIL_CSV)

    # rename URL → url ถ้ายังไม่มี
    if "URL" in df.columns and "url" not in df.columns:
        df = df.rename(columns={"URL": "url"})

    # หา label column
    label_col = next((c for c in PHI_LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"ไม่พบ label column ใน PhiUSIIL — columns: {df.columns.tolist()}")
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # สร้าง feature matrix ตาม FEATURE_MAP (PhiUSIIL column)
    rows = []
    for canonical, (phi_col, _) in FEATURE_MAP.items():
        if phi_col in df.columns:
            rows.append(df[phi_col].fillna(0).values)
        else:
            print(f"  [WARN] PhiUSIIL: ไม่พบ column '{phi_col}' → ใส่ 0")
            rows.append(np.zeros(len(df)))

    X = np.column_stack(rows).astype(np.float32)
    y = df["label"].values

    print(f"  PhiUSIIL: {len(y):,} rows | Phishing={y.sum():,} | Legit={(y==0).sum():,}")
    return X, y


# ─────────────────────────────────────────────
# 2. โหลด ISCX — กรอง phishing/benign แล้ว map columns
# ─────────────────────────────────────────────

def load_iscx(include_all_malicious: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    โหลด ISCX dataset กรองเฉพาะ phishing vs benign
    (ไม่รวม defacement/malware/spam เพื่อให้เปรียบเทียบ apples-to-apples)

    Args:
        include_all_malicious: True = นับ malware/defacement/spam เป็น phishing ด้วย
    """
    if not ISCX_CSV.exists():
        raise FileNotFoundError(f"ไม่พบ ISCX dataset: {ISCX_CSV}")

    df = pd.read_csv(ISCX_CSV)
    label_col = "URL_Type_obf_Type"
    df[label_col] = df[label_col].str.lower().str.strip()

    if include_all_malicious:
        # ทุก non-benign → phishing (1)
        df["label"] = (df[label_col] != "benign").astype(int)
    else:
        # เฉพาะ phishing vs benign
        df = df[df[label_col].isin({"phishing", "benign"})].copy()
        df["label"] = (df[label_col] == "phishing").astype(int)

    # สร้าง feature matrix ตาม FEATURE_MAP (ISCX column)
    rows = []
    for canonical, (_, iscx_col) in FEATURE_MAP.items():
        if iscx_col in df.columns:
            rows.append(df[iscx_col].fillna(0).values)
        else:
            print(f"  [WARN] ISCX: ไม่พบ column '{iscx_col}' → ใส่ 0")
            rows.append(np.zeros(len(df)))

    X = np.column_stack(rows).astype(np.float32)
    y = df["label"].values

    label_desc = "phishing+benign" if not include_all_malicious else "phishing+benign+others"
    print(f"  ISCX ({label_desc}): {len(y):,} rows | Phishing={y.sum():,} | Legit={(y==0).sum():,}")
    return X, y


# ─────────────────────────────────────────────
# 3. Models
# ─────────────────────────────────────────────

def build_models() -> dict:
    """กำหนด 3 models สำหรับเปรียบเทียบ"""
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "SVM (LinearSVC)": LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=42,
        ),
    }


# ─────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────

def evaluate(model_name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """คำนวณ metrics ครบชุดสำหรับ model หนึ่งตัว"""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # LinearSVC: ใช้ decision_function แทน predict_proba
        y_score = model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred)

    m = {
        "model":      model_name,
        "accuracy":   accuracy_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred, zero_division=0),
        "recall":     recall_score(y_test, y_pred, zero_division=0),
        "f1":         f1_score(y_test, y_pred, zero_division=0),
        "auc_roc":    roc_auc_score(y_test, y_score),
        "n_test":     len(y_test),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }

    print(f"\n  ── {model_name}")
    print(f"     Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
          f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}")
    print(f"     CM: TN={m['tn']:,} FP={m['fp']:,} FN={m['fn']:,} TP={m['tp']:,}")

    return m


# ─────────────────────────────────────────────
# 5. Train + Evaluate หนึ่ง mode
# ─────────────────────────────────────────────

def run_eval_mode(
    mode_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> tuple[dict, dict]:
    """
    Train ทุก model บน X_train แล้ว evaluate บน X_test

    Returns:
        metrics_dict: {model_name: metrics}
        trained_dict: {model_name: model}
        scalers_dict: {model_name: scaler}
    """
    # Normalize — fit บน train เท่านั้น ป้องกัน data leakage
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    print(f"\n{'='*55}")
    print(f"  Mode: {mode_name}")
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")
    print(f"{'='*55}")

    models   = build_models()
    metrics  = {}
    trained  = {}
    scalers  = {}

    for name, model in models.items():
        model.fit(X_tr_sc, y_train)
        metrics[name]  = evaluate(name, model, X_te_sc, y_test)
        trained[name]  = model
        scalers[name]  = scaler

    return metrics, trained, scalers


# ─────────────────────────────────────────────
# 6. Feature Importance (tree-based models)
# ─────────────────────────────────────────────

def print_feature_importance(trained: dict) -> None:
    """แสดง feature importance สำหรับ tree-based models"""
    print(f"\n{'='*55}")
    print(f"  Feature Importance (URL-Only Features)")
    print(f"{'='*55}")

    tree_models = {k: v for k, v in trained.items() if k in ("XGBoost", "Random Forest")}

    for name, model in tree_models.items():
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        print(f"\n  {name}:")
        for rank, idx in enumerate(order, 1):
            feat = CANONICAL_FEATURES[idx]
            print(f"    {rank}. {feat:<25s} {importances[idx]:.4f}")


# ─────────────────────────────────────────────
# 7. Comparison Table
# ─────────────────────────────────────────────

def print_comparison(all_results: dict[str, dict]) -> None:
    """
    แสดงตารางเปรียบเทียบผลทั้ง 3 modes x 3 models
    all_results: {mode_name: {model_name: metrics}}
    """
    print(f"\n{'='*75}")
    print(f"  Comparison Table — URL-Only Features ({len(CANONICAL_FEATURES)} features)")
    print(f"{'='*75}")
    print(f"  {'Mode':<28} {'Model':<18} {'Acc':>6} {'F1':>6} {'AUC':>6} {'N':>8}")
    print(f"  {'─'*70}")

    for mode, models_metrics in all_results.items():
        for model_name, m in models_metrics.items():
            print(
                f"  {mode:<28} {model_name:<18} "
                f"{m['accuracy']:>6.4f} {m['f1']:>6.4f} {m['auc_roc']:>6.4f} "
                f"{m['n_test']:>8,}"
            )
        print(f"  {'─'*70}")


# ─────────────────────────────────────────────
# 8. บันทึก Best Model
#    ใช้ model ที่ดีที่สุดจาก combined training
#    เพื่อให้ API (/predict) ใช้งานได้
# ─────────────────────────────────────────────

def save_best_model(
    metrics:  dict,
    trained:  dict,
    scalers:  dict,
) -> str:
    """
    เลือก model ที่ F1 สูงสุดจาก combined training
    บันทึก best_model.pkl, scaler.pkl, feature_names.pkl
    feature_names ใช้ canonical names ที่ url_extractor.py คืนมา
    """
    best_name = max(metrics, key=lambda k: metrics[k]["f1"])
    best_f1   = metrics[best_name]["f1"]

    print(f"\n{'='*55}")
    print(f"  Best Model (Combined Training)")
    print(f"  → {best_name}  F1={best_f1:.4f}")
    print(f"{'='*55}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    with open(SAVE_DIR / "best_model.pkl",    "wb") as f:
        pickle.dump(trained[best_name], f)
    with open(SAVE_DIR / "scaler.pkl",        "wb") as f:
        pickle.dump(scalers[best_name], f)
    with open(SAVE_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(CANONICAL_FEATURES, f)

    print(f"  บันทึกที่: {SAVE_DIR}")
    print(f"    best_model.pkl  → {best_name}")
    print(f"    scaler.pkl")
    print(f"    feature_names.pkl → {CANONICAL_FEATURES}")

    return best_name


# ─────────────────────────────────────────────
# 9. Save CSV Results
# ─────────────────────────────────────────────

def save_csv(all_results: dict[str, dict]) -> None:
    """บันทึกผล metrics ทั้งหมดเป็น CSV"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode, models_metrics in all_results.items():
        for model_name, m in models_metrics.items():
            rows.append({
                "mode":      mode,
                "model":     model_name,
                "n_test":    m["n_test"],
                "accuracy":  round(m["accuracy"],  4),
                "precision": round(m["precision"], 4),
                "recall":    round(m["recall"],    4),
                "f1":        round(m["f1"],        4),
                "auc_roc":   round(m["auc_roc"],   4),
                "tn":        m["tn"], "fp": m["fp"],
                "fn":        m["fn"], "tp": m["tp"],
            })

    out = RESULTS_DIR / "train_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  บันทึก CSV: {out}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Phishing Detector — Multi-Dataset Training (v3)")
    print(f"  URL-Only Features: {len(CANONICAL_FEATURES)}")
    print(f"  {CANONICAL_FEATURES}")
    print("=" * 55)

    # ── โหลด datasets ──────────────────────────────────────
    print("\n[1] โหลด Datasets")
    print("─" * 55)
    try:
        X_phi, y_phi = load_phiusiil()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    try:
        X_isc, y_isc = load_iscx(include_all_malicious=False)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    # ── ข้อมูลรวม ──────────────────────────────────────────
    X_combined = np.vstack([X_phi, X_isc])
    y_combined = np.concatenate([y_phi, y_isc])
    print(f"\n  Combined: {len(y_combined):,} rows | "
          f"Phishing={y_combined.sum():,} | Legit={(y_combined==0).sum():,}")

    # ══════════════════════════════════════════════════════════
    # Mode A: Combined Dataset, 80/20 Stratified Split
    # ══════════════════════════════════════════════════════════
    print("\n[2] Evaluation Modes")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_combined, y_combined,
        test_size=0.2, random_state=42, stratify=y_combined,
    )
    m_a, trained_a, scalers_a = run_eval_mode(
        "A: Combined (80/20 split)", X_tr, y_tr, X_te, y_te
    )

    # ══════════════════════════════════════════════════════════
    # Mode B: Train PhiUSIIL → Test ISCX (cross-dataset)
    # ══════════════════════════════════════════════════════════
    m_b, trained_b, scalers_b = run_eval_mode(
        "B: Train PhiUSIIL → Test ISCX", X_phi, y_phi, X_isc, y_isc
    )

    # ══════════════════════════════════════════════════════════
    # Mode C: Train ISCX → Test PhiUSIIL (cross-dataset)
    # ══════════════════════════════════════════════════════════
    m_c, trained_c, scalers_c = run_eval_mode(
        "C: Train ISCX → Test PhiUSIIL", X_isc, y_isc, X_phi, y_phi
    )

    # ── Feature Importance จาก combined training ──────────────
    print_feature_importance(trained_a)

    # ── ตารางเปรียบเทียบ ──────────────────────────────────────
    all_results = {"A: Combined (80/20)": m_a,
                   "B: Phi→ISCX":        m_b,
                   "C: ISCX→Phi":        m_c}
    print_comparison(all_results)

    # ── วิเคราะห์ Generalization ──────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Generalization Analysis")
    print(f"{'='*55}")

    # เปรียบเทียบ F1 ใน mode A vs B สำหรับ XGBoost
    for model_name in ("XGBoost", "Random Forest", "SVM (LinearSVC)"):
        f1_a = m_a[model_name]["f1"]
        f1_b = m_b[model_name]["f1"]
        f1_c = m_c[model_name]["f1"]
        gap  = f1_a - ((f1_b + f1_c) / 2)
        print(f"\n  {model_name}")
        print(f"    Combined F1  : {f1_a:.4f}")
        print(f"    Phi→ISCX F1  : {f1_b:.4f}")
        print(f"    ISCX→Phi F1  : {f1_c:.4f}")
        print(f"    Avg cross-ds : {(f1_b+f1_c)/2:.4f}  (gap={gap:+.4f})")

    # ── บันทึก Best Model (จาก combined training) ─────────────
    best = save_best_model(m_a, trained_a, scalers_a)

    # ── บันทึก CSV ────────────────────────────────────────────
    save_csv(all_results)

    print("\n" + "=" * 55)
    print(f"  เสร็จสิ้น — Best model: {best}")
    print(f"  Feature set: {len(CANONICAL_FEATURES)} URL-only features")
    print(f"  API พร้อมใช้งาน — url_extractor.py คืน canonical names ตรงกัน")
    print("=" * 55)
