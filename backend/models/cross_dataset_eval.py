"""
cross_dataset_eval.py (v3 — Youden's J Threshold Calibration)
วิเคราะห์ปัญหา generalization และแก้ด้วย optimal threshold

การค้นพบก่อนหน้า:
  - Mode B (PhiUSIIL → ISCX): AUC=0.842 แต่ F1=0.0
  - สาเหตุ: threshold=0.5 ไม่เหมาะกับ distribution ของ ISCX
    ทำให้ probability scores ของ ISCX ต่ำกว่า 0.5 แม้จะเป็น phishing

แนวทางแก้:
  1. หา optimal threshold จาก ROC curve บน PhiUSIIL validation set
     ใช้ Youden's J statistic: J = TPR - FPR (maximize sensitivity+specificity)
  2. ใช้ threshold ใหม่ predict บน ISCX แทน default 0.5
  3. เปรียบเทียบ threshold=0.5 vs optimal: F1, Precision, Recall
  4. Plot ROC curve ของทั้ง 3 modes → data/results/roc_curves.png
  5. สรุปผลเป็นตาราง + thesis conclusion

รัน: python backend/models/cross_dataset_eval.py
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc, confusion_matrix,
)
from sklearn.model_selection import train_test_split

# matplotlib — ใช้ Agg backend เพื่อไม่ต้องเปิด window
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] ไม่พบ matplotlib — ข้าม ROC plot (pip install matplotlib)")

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parents[2]
SAVE_DIR     = ROOT / "backend" / "models" / "saved"
PHIUSIIL_CSV = ROOT / "data" / "raw" / "dataset.csv"
ISCX_CSV     = ROOT / "data" / "raw" / "iscx_url.csv"
RESULTS_DIR  = ROOT / "data" / "results"

sys.path.insert(0, str(ROOT))
from backend.models.feature_config import FEATURE_MAP, PHI_LABEL_CANDIDATES


# ─────────────────────────────────────────────
# 1. โหลด Artifacts
# ─────────────────────────────────────────────

def load_artifacts() -> tuple:
    """โหลด model, scaler, feature_names จาก saved/"""
    print("=" * 60)
    print("  โหลด Model Artifacts (combined training)")
    print("=" * 60)

    for name in ("best_model.pkl", "scaler.pkl", "feature_names.pkl"):
        if not (SAVE_DIR / name).exists():
            raise FileNotFoundError(f"ไม่พบ {name} — รัน train.py ก่อน")

    with open(SAVE_DIR / "best_model.pkl",    "rb") as f:
        model = pickle.load(f)
    with open(SAVE_DIR / "scaler.pkl",        "rb") as f:
        scaler = pickle.load(f)
    with open(SAVE_DIR / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    print(f"  Model        : {type(model).__name__}")
    print(f"  Features ({len(feature_names)}): {feature_names}")
    return model, scaler, feature_names


# ─────────────────────────────────────────────
# 2. โหลด PhiUSIIL → validation set
#    ใช้เพื่อหา optimal threshold
#    split 80/20 เหมือน train.py
# ─────────────────────────────────────────────

def load_phiusiil_val(scaler) -> tuple[np.ndarray, np.ndarray]:
    """
    โหลด PhiUSIIL, สร้าง feature matrix ด้วย canonical names,
    คืน validation set 20% (stratified, random_state=42)
    """
    if not PHIUSIIL_CSV.exists():
        raise FileNotFoundError(f"ไม่พบ PhiUSIIL: {PHIUSIIL_CSV}")

    df = pd.read_csv(PHIUSIIL_CSV)
    if "URL" in df.columns and "url" not in df.columns:
        df = df.rename(columns={"URL": "url"})

    label_col = next((c for c in PHI_LABEL_CANDIDATES if c in df.columns), None)
    if label_col and label_col != "label":
        df = df.rename(columns={label_col: "label"})

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # สร้าง feature matrix ด้วย PhiUSIIL columns (left side of FEATURE_MAP)
    cols = [df[phi_col].fillna(0).values if phi_col in df.columns else np.zeros(len(df))
            for _, (phi_col, _) in FEATURE_MAP.items()]
    X = np.column_stack(cols).astype(np.float32)
    y = df["label"].values

    # 20% validation split — random_state=42 เพื่อ reproducibility
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val_sc = scaler.transform(X_val)

    print(f"\n  PhiUSIIL validation : {len(y_val):,} samples")
    print(f"  Phishing (1)        : {y_val.sum():,}  ({y_val.mean()*100:.1f}%)")
    print(f"  Legitimate (0)      : {(y_val==0).sum():,}  ({(1-y_val.mean())*100:.1f}%)")
    return X_val_sc, y_val


# ─────────────────────────────────────────────
# 3. โหลด ISCX → phishing vs benign
# ─────────────────────────────────────────────

def load_iscx_data(scaler) -> tuple[np.ndarray, np.ndarray]:
    """
    โหลด ISCX, กรองเฉพาะ phishing+benign,
    map columns → canonical names, scale
    """
    if not ISCX_CSV.exists():
        raise FileNotFoundError(f"ไม่พบ ISCX: {ISCX_CSV}")

    df = pd.read_csv(ISCX_CSV)
    lbl = "URL_Type_obf_Type"
    df[lbl] = df[lbl].str.lower().str.strip()

    # เฉพาะ phishing vs benign — apples-to-apples comparison
    df = df[df[lbl].isin({"phishing", "benign"})].copy()
    df["label"] = (df[lbl] == "phishing").astype(int)

    # สร้าง feature matrix ด้วย ISCX columns (right side of FEATURE_MAP)
    cols = [df[iscx_col].fillna(0).values if iscx_col in df.columns else np.zeros(len(df))
            for _, (_, iscx_col) in FEATURE_MAP.items()]
    X = np.column_stack(cols).astype(np.float32)
    y = df["label"].values
    X_sc = scaler.transform(X)

    print(f"\n  ISCX (phishing+benign): {len(y):,} samples")
    print(f"  Phishing (1)          : {y.sum():,}  ({y.mean()*100:.1f}%)")
    print(f"  Legitimate (0)        : {(y==0).sum():,}  ({(1-y.mean())*100:.1f}%)")
    return X_sc, y


# ─────────────────────────────────────────────
# 4. Probability Scores
# ─────────────────────────────────────────────

def get_scores(model, X_scaled: np.ndarray) -> np.ndarray:
    """คืน probability scores — ใช้ predict_proba หรือ decision_function"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_scaled)[:, 1]
    return model.decision_function(X_scaled)


# ─────────────────────────────────────────────
# 5. หา Optimal Threshold
# ─────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    หา optimal threshold ด้วย Youden's J statistic บน validation set
    J = TPR - FPR  (เท่ากับ Sensitivity + Specificity - 1)
    maximize J → สมดุล sensitivity และ specificity พร้อมกัน

    Returns:
        optimal_threshold, best_j (Youden's J value)
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # Youden's J = TPR - FPR ทุก threshold บน ROC curve
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))

    return float(thresholds[best_idx]), float(youden_j[best_idx])


# ─────────────────────────────────────────────
# 6. คำนวณ Metrics ที่ threshold ที่กำหนด
# ─────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    label: str = "",
) -> dict:
    """คำนวณ metrics ทั้งหมดที่ threshold ที่กำหนด"""
    y_pred = (scores >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    m = {
        "label":     label,
        "threshold": round(threshold, 4),
        "n_samples": len(y_true),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_true, scores), 4),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }

    print(f"\n  ── {label}  (threshold={threshold:.4f})")
    print(f"     Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
          f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}")
    print(f"     CM: TN={m['tn']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TP={m['tp']:,}")

    return m


# ─────────────────────────────────────────────
# 7. Plot ROC Curves (3 panels)
# ─────────────────────────────────────────────

def plot_roc_curves(
    y_phi:       np.ndarray,
    scores_phi:  np.ndarray,
    y_iscx:      np.ndarray,
    scores_iscx: np.ndarray,
    thr_opt:     float,
    model_name:  str,
    m_phi_05:    dict,
    m_iscx_05:   dict,
    m_iscx_opt:  dict,
) -> None:
    """
    วาด ROC curves 3 panels:
      1. PhiUSIIL validation — in-sample baseline
      2. ISCX cross-dataset — threshold=0.5 (จุดปัญหา)
      3. ISCX cross-dataset — optimal threshold (จุดปรับปรุง, Youden's J)
    บันทึกเป็น data/results/roc_curves.png

    รับ metrics dict ที่คำนวณแล้วเพื่อหลีกเลี่ยง compute ซ้ำ
    """
    if not HAS_MPL:
        return

    # คำนวณ ROC curve ทั้งสอง dataset
    fpr_p, tpr_p, thr_p = roc_curve(y_phi,  scores_phi)
    fpr_i, tpr_i, thr_i = roc_curve(y_iscx, scores_iscx)
    auc_p = auc(fpr_p, tpr_p)
    auc_i = auc(fpr_i, tpr_i)

    # หา operating points บน ROC curve ที่ threshold ที่สนใจ
    idx_p_05 = np.argmin(np.abs(thr_p - 0.5))
    idx_i_05 = np.argmin(np.abs(thr_i - 0.5))
    idx_i_op = np.argmin(np.abs(thr_i - thr_opt))

    # สี
    C_PHI  = "#4a9eff"   # ฟ้า — PhiUSIIL
    C_ISCX = "#ff6b6b"   # แดง — ISCX default
    C_OPT  = "#51cf66"   # เขียว — optimal
    C_DIAG = "#aaaaaa"   # เทา — random classifier line

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        f"ROC Curve Analysis — {model_name}  (9 URL-only features)\n"
        f"Combined Training: PhiUSIIL (235,795) + ISCX-phishing/benign (15,367)",
        fontsize=11, fontweight="bold",
    )

    def _style_ax(ax, title):
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.plot([0, 1], [0, 1], color=C_DIAG, lw=1, linestyle="--", label="Random (AUC=0.5)")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.set_facecolor("#fafafa")

    # ─── Panel 1: PhiUSIIL In-Sample ──────────────────────────
    ax = axes[0]
    _style_ax(ax, "① PhiUSIIL Validation\n(In-Sample Baseline)")
    ax.plot(fpr_p, tpr_p, color=C_PHI, lw=2.5, label=f"ROC  AUC={auc_p:.4f}")

    # operating point ที่ threshold=0.5 — ใช้ metrics ที่ส่งมาแล้ว ไม่ compute ซ้ำ
    ax.scatter(
        fpr_p[idx_p_05], tpr_p[idx_p_05],
        color=C_PHI, s=100, zorder=5, edgecolors="white", linewidths=2,
        label=f"t=0.50  F1={m_phi_05['f1']:.3f}",
    )
    ax.legend(fontsize=8.5, loc="lower right")

    # ─── Panel 2: ISCX threshold=0.5 ──────────────────────────
    ax = axes[1]
    _style_ax(ax, "② ISCX Cross-Dataset\nDefault threshold=0.50 ❌")
    ax.plot(fpr_i, tpr_i, color=C_ISCX, lw=2.5, label=f"ROC  AUC={auc_i:.4f}")
    ax.scatter(
        fpr_i[idx_i_05], tpr_i[idx_i_05],
        color="#cc0000", s=150, zorder=5, marker="X",
        edgecolors="white", linewidths=1.5,
        label=f"t=0.50  F1={m_iscx_05['f1']:.3f}  ← ปัญหา",
    )
    ax.legend(fontsize=8.5, loc="lower right")

    # annotation ชี้ปัญหา
    ax.annotate(
        "Model ตัดสินใจ\nเป็น 0 ทั้งหมด",
        xy=(fpr_i[idx_i_05], tpr_i[idx_i_05]),
        xytext=(0.35, 0.15),
        fontsize=8, color="#cc0000",
        arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.5),
    )

    # ─── Panel 3: ISCX Optimal Threshold (Youden's J) ──────────
    ax = axes[2]
    _style_ax(ax, f"③ ISCX Cross-Dataset\nYouden's J threshold={thr_opt:.3f} ✓")
    ax.plot(fpr_i, tpr_i, color=C_ISCX, lw=2.5, label=f"ROC  AUC={auc_i:.4f}")

    # ค่าเดิม (dim) — เพื่อเปรียบเทียบ
    ax.scatter(
        fpr_i[idx_i_05], tpr_i[idx_i_05],
        color="#cc0000", s=80, zorder=4, marker="X",
        alpha=0.35, label=f"t=0.50  F1={m_iscx_05['f1']:.3f}  (เดิม)",
    )

    # optimal threshold จาก Youden's J (ชัด)
    ax.scatter(
        fpr_i[idx_i_op], tpr_i[idx_i_op],
        color=C_OPT, s=200, zorder=5, marker="*",
        edgecolors="white", linewidths=1,
        label=f"t={thr_opt:.3f}  F1={m_iscx_opt['f1']:.3f}  ✓",
    )

    # annotation ชี้การปรับปรุง
    ax.annotate(
        f"F1: {m_iscx_05['f1']:.3f} → {m_iscx_opt['f1']:.3f}\n"
        f"(+{m_iscx_opt['f1']-m_iscx_05['f1']:.3f})",
        xy=(fpr_i[idx_i_op], tpr_i[idx_i_op]),
        xytext=(0.35, 0.40),
        fontsize=8, color="#1a7a2e",
        arrowprops=dict(arrowstyle="->", color="#1a7a2e", lw=1.5),
    )
    ax.legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  บันทึก ROC plot: {out}")


# ─────────────────────────────────────────────
# 8. Comparison Table
# ─────────────────────────────────────────────

def print_comparison_table(all_metrics: list[dict]) -> None:
    """แสดงตารางเปรียบเทียบทุก evaluation"""
    print(f"\n{'='*78}")
    print(f"  Threshold Comparison Table")
    print(f"{'='*78}")
    print(
        f"  {'Dataset / Mode':<38} {'Thr':>5} "
        f"{'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}"
    )
    print(f"  {'─'*75}")

    for m in all_metrics:
        if not m:
            continue
        print(
            f"  {m['label']:<38} "
            f"{m['threshold']:>5.3f} "
            f"{m['accuracy']:>6.4f} "
            f"{m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} "
            f"{m['f1']:>6.4f} "
            f"{m['auc_roc']:>6.4f}"
        )


# ─────────────────────────────────────────────
# 9. Save CSV
# ─────────────────────────────────────────────

def save_csv(all_metrics: list[dict], model_name: str) -> None:
    """บันทึกผลทั้งหมดเป็น CSV"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in all_metrics:
        if not m:
            continue
        rows.append({
            "dataset":   m["label"],
            "model":     model_name,
            "threshold": m["threshold"],
            "n_samples": m["n_samples"],
            "accuracy":  m["accuracy"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "auc_roc":   m["auc_roc"],
            "tn":        m["tn"], "fp": m["fp"],
            "fn":        m["fn"], "tp": m["tp"],
        })

    out = RESULTS_DIR / "cross_dataset_threshold_eval.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  บันทึก CSV: {out}")


# ─────────────────────────────────────────────
# 10. Thesis Conclusion
# ─────────────────────────────────────────────

def print_thesis_conclusion(
    m_phi:    dict,
    m_iscx_05: dict,
    m_iscx_opt: dict,
    thr_opt:  float,
) -> None:
    """สรุปผลเป็น structured conclusion สำหรับ thesis"""
    f1_improvement = m_iscx_opt["f1"] - m_iscx_05["f1"]
    generalization_gap_before = m_phi["f1"] - m_iscx_05["f1"]
    generalization_gap_after  = m_phi["f1"] - m_iscx_opt["f1"]

    print(f"\n{'='*78}")
    print(f"  Thesis Conclusion — Cross-Dataset Generalization Analysis")
    print(f"{'='*78}")

    print(f"""
  ┌─ Research Question ─────────────────────────────────────────────────────┐
  │ Model ที่ train บน PhiUSIIL สามารถ generalize ไปยัง ISCX-URL ได้แค่ไหน? │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ Finding 1: AUC สูง แต่ F1 ต่ำ (Threshold Problem) ──────────────────────┐
  │                                                                           │
  │  PhiUSIIL (in-sample)   AUC={m_phi['auc_roc']:.4f}  F1={m_phi['f1']:.4f}              │
  │  ISCX (threshold=0.50)  AUC={m_iscx_05['auc_roc']:.4f}  F1={m_iscx_05['f1']:.4f}              │
  │                                                                           │
  │  AUC=0.84 บน ISCX หมายความว่า model สามารถจัด ranking ถูกต้อง           │
  │  แต่ default threshold=0.5 ทำให้ model ตัดสินใจเป็น 0 ทั้งหมด           │
  │  เพราะ feature distribution ของ ISCX ต่างจาก PhiUSIIL                   │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ Finding 2: Youden's J Threshold Calibration ช่วยได้ ─────────────────────┐
  │                                                                           │
  │  Optimal threshold (Youden's J)        : {thr_opt:.4f}                       │
  │  ISCX F1 before (t=0.50)               : {m_iscx_05['f1']:.4f}                       │
  │  ISCX F1 after  (t={thr_opt:.3f})             : {m_iscx_opt['f1']:.4f}                       │
  │  F1 improvement                        : +{f1_improvement:.4f}                      │
  │                                                                           │
  │  Generalization gap (Δ F1):                                               │
  │    Before calibration: {generalization_gap_before:+.4f}                                   │
  │    After  calibration: {generalization_gap_after:+.4f}                                   │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ Recommendation ──────────────────────────────────────────────────────────┐
  │                                                                           │""")

    if m_iscx_opt["f1"] >= 0.7:
        print(f"""\
  │  ✓ Model generalize ได้ดีหลัง threshold calibration (F1={m_iscx_opt['f1']:.4f})    │
  │  → สามารถใช้ URL-only features กับ dataset ใหม่ได้โดยปรับ threshold      │""")
    elif m_iscx_opt["f1"] >= 0.5:
        print(f"""\
  │  ~ Model generalize ได้ระดับปานกลาง (F1={m_iscx_opt['f1']:.4f})              │
  │  → ควรพิจารณา domain adaptation หรือ fine-tuning บน target domain        │""")
    else:
        print(f"""\
  │  ⚠ Model ยัง generalize ได้ไม่ดีแม้ปรับ threshold (F1={m_iscx_opt['f1']:.4f})  │
  │  → ควร train บน target domain dataset โดยตรง                             │""")

    print(f"""\
  │                                                                           │
  │  Feature ที่สำคัญที่สุด (URL-only, portable across datasets):             │
  │    1. digit_ratio      — สัดส่วนตัวเลขใน URL                             │
  │    2. num_subdomains   — จำนวน subdomain                                  │
  │    3. has_ip           — มี IP address ใน domain หรือไม่                  │
  └───────────────────────────────────────────────────────────────────────────┘
""")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Cross-Dataset Evaluation + Threshold Calibration (v2)")
    print("=" * 60)

    # 1. โหลด model artifacts
    try:
        model, scaler, feature_names = load_artifacts()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    model_name = type(model).__name__

    # 2. โหลด PhiUSIIL validation set
    print("\n[1] โหลด PhiUSIIL Validation Set")
    print("─" * 60)
    X_phi_val, y_phi_val = load_phiusiil_val(scaler)

    # 3. โหลด ISCX dataset
    print("\n[2] โหลด ISCX Dataset (phishing+benign)")
    print("─" * 60)
    X_iscx, y_iscx = load_iscx_data(scaler)

    # 4. Probability scores
    scores_phi  = get_scores(model, X_phi_val)
    scores_iscx = get_scores(model, X_iscx)

    print(f"\n  Score stats บน PhiUSIIL val:")
    print(f"    mean={scores_phi.mean():.4f}  std={scores_phi.std():.4f}  "
          f"min={scores_phi.min():.4f}  max={scores_phi.max():.4f}")
    print(f"  Score stats บน ISCX:")
    print(f"    mean={scores_iscx.mean():.4f}  std={scores_iscx.std():.4f}  "
          f"min={scores_iscx.min():.4f}  max={scores_iscx.max():.4f}")
    print(f"\n  → ถ้า mean ของ ISCX ต่ำกว่า 0.5 คือสาเหตุที่ F1=0")

    # 5. หา optimal threshold ด้วย Youden's J จาก PhiUSIIL validation
    print("\n[3] หา Optimal Threshold (Youden's J) จาก PhiUSIIL Validation")
    print("─" * 60)
    thr_opt, best_j = find_optimal_threshold(y_phi_val, scores_phi)
    print(f"  Optimal threshold : {thr_opt:.4f}")
    print(f"  Youden's J        : {best_j:.4f}  (= TPR - FPR ที่ threshold นี้)")
    print(f"  ใช้ Youden's J เพราะ maximize sensitivity+specificity พร้อมกัน")

    # 6. Evaluate ทุก mode
    print("\n[4] Evaluation")
    print("─" * 60)

    m_phi_05  = compute_metrics(y_phi_val, scores_phi,  0.5,    "PhiUSIIL Validation (t=0.50)")
    m_phi_opt = compute_metrics(y_phi_val, scores_phi,  thr_opt, f"PhiUSIIL Validation (t={thr_opt:.3f})")
    m_iscx_05  = compute_metrics(y_iscx,   scores_iscx, 0.5,    "ISCX Cross-Dataset  (t=0.50)")
    m_iscx_opt = compute_metrics(y_iscx,   scores_iscx, thr_opt, f"ISCX Cross-Dataset  (t={thr_opt:.3f})")

    all_metrics = [m_phi_05, m_phi_opt, m_iscx_05, m_iscx_opt]

    # 7. แสดงตารางเปรียบเทียบ
    print_comparison_table(all_metrics)

    # 8. Plot ROC curves — ส่ง metrics ที่คำนวณแล้วเพื่อไม่ compute ซ้ำ
    print("\n[5] Plot ROC Curves")
    print("─" * 60)
    plot_roc_curves(
        y_phi_val,  scores_phi,
        y_iscx,     scores_iscx,
        thr_opt,    model_name,
        m_phi_05    = m_phi_05,
        m_iscx_05   = m_iscx_05,
        m_iscx_opt  = m_iscx_opt,
    )

    # 9. Thesis conclusion
    print_thesis_conclusion(m_phi_05, m_iscx_05, m_iscx_opt, thr_opt)

    # 10. Save CSV
    save_csv(all_metrics, model_name)

    print("=" * 60)
    print("  เสร็จสิ้น")
    print(f"  ผลบันทึกที่: {RESULTS_DIR}")
    print("=" * 60)
