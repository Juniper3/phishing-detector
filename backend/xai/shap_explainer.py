"""
shap_explainer.py
XAI module สำหรับ phishing detection — อธิบายการตัดสินใจของ model ด้วย SHAP values
"""

import sys
import numpy as np
import pandas as pd
import shap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# model types ที่รองรับ TreeExplainer (เร็วกว่า KernelExplainer มาก)
_TREE_MODEL_TYPES = (
    "XGBClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
)


class SHAPExplainer:
    """
    Wrapper สำหรับ SHAP explainer ทุกประเภท
    เลือก explainer ให้อัตโนมัติตาม model type
    """

    def __init__(self, model, feature_names: list[str]):
        """
        Args:
            model:         trained sklearn/xgboost model
            feature_names: ชื่อ features ตามลำดับที่ model ใช้ train
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = self._init_explainer(model)

    def _init_explainer(self, model) -> shap.Explainer:
        """เลือก explainer ที่เหมาะสมตาม model type"""
        model_type = type(model).__name__

        if model_type in _TREE_MODEL_TYPES:
            # TreeExplainer คำนวณแบบ exact — เร็วและแม่นยำสำหรับ tree-based models
            return shap.TreeExplainer(model)

        elif model_type in ("LinearSVC", "LogisticRegression", "LinearRegression"):
            # LinearExplainer ใช้กับ linear models — เร็วกว่า KernelExplainer
            return shap.LinearExplainer(model, np.zeros((1, len(self.feature_names))))

        else:
            # KernelExplainer ใช้ได้กับทุก model แต่ช้ากว่า (ใช้ sampling)
            print(f"[SHAP] {model_type} ไม่ใช่ tree/linear model — ใช้ KernelExplainer (อาจช้า)")
            background = np.zeros((1, len(self.feature_names)))
            return shap.KernelExplainer(model.predict_proba, background)

    def _dict_to_array(self, features_dict: dict) -> np.ndarray:
        """แปลง features dict → numpy array โดยเรียงตาม feature_names"""
        return np.array(
            [features_dict.get(f, 0.0) for f in self.feature_names],
            dtype=np.float32,
        ).reshape(1, -1)

    def _extract_1d_shap(self, raw, n_features: int) -> np.ndarray:
        """
        แปลง SHAP output หลากหลาย format → 1D array shape (n_features,)

        SHAP คืน format ต่างกันตาม model/version:
          - ndarray (n_samples, n_features)           ← XGBoost binary
          - list[ (n_samples, n_features), ... ]      ← RandomForest binary (per class)
          - list[ (n_features,), ... ]                ← บาง SHAP version
        """
        if isinstance(raw, list):
            # binary classification → เลือก class 1 (phishing)
            arr = np.asarray(raw[1])
        else:
            arr = np.asarray(raw)

        arr = arr.squeeze()   # ลด dimensions ที่ขนาด 1 ออก → (n_features,) หรือ (n_samples, n_features)

        if arr.ndim == 2:
            arr = arr[0]      # เลือก sample แรก → (n_features,)

        return arr.astype(np.float64)

    # ──────────────────────────────────────────────
    # Local Explanation (URL เดียว)
    # ──────────────────────────────────────────────

    def explain_local(self, features_dict: dict) -> dict:
        """
        อธิบาย prediction ของ URL เดียว

        Args:
            features_dict: dict ของ features จาก url_extractor หรือ html_extractor

        Returns:
            dict ที่ประกอบด้วย:
              - prediction:   0 (legitimate) หรือ 1 (phishing)
              - confidence:   % ความมั่นใจของ model (0.0–100.0)
              - shap_values:  dict {feature_name: shap_value}
              - top_features: list 5 features ที่มีผลมากสุด
        """
        X = self._dict_to_array(features_dict)

        # คำนวณ SHAP values แล้วแปลงเป็น 1D array (n_features,)
        raw = self.explainer.shap_values(X)
        shap_vals = self._extract_1d_shap(raw, len(self.feature_names))

        # Prediction + confidence
        prediction = int(self.model.predict(X)[0])
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = round(float(proba[prediction]) * 100, 2)
        else:
            # LinearSVC ไม่มี predict_proba ใช้ decision_function แทน
            score = self.model.decision_function(X)[0]
            # sigmoid approximation
            confidence = round(float(1 / (1 + np.exp(-score))) * 100, 2)

        # จับคู่ feature_name กับ shap_value
        shap_dict = {
            name: round(float(val), 6)
            for name, val in zip(self.feature_names, shap_vals)
        }

        # Top-5 features เรียงตาม |shap_value| มากไปน้อย
        sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = [
            {
                "feature_name": name,
                "shap_value":   val,
                # shap_value > 0 → เพิ่มความเสี่ยง phishing, < 0 → ลดความเสี่ยง
                "direction": "increases_risk" if val > 0 else "decreases_risk",
            }
            for name, val in sorted_feats[:5]
        ]

        return {
            "prediction":   prediction,
            "confidence":   confidence,
            "shap_values":  shap_dict,
            "top_features": top_features,
        }

    # ──────────────────────────────────────────────
    # Global Explanation (ทั้ง dataset)
    # ──────────────────────────────────────────────

    def explain_global(self, X_df: pd.DataFrame) -> dict:
        """
        สรุป feature importance รวมจาก SHAP values ของทั้ง dataset

        Args:
            X_df: DataFrame ของ features (columns ต้องตรงกับ feature_names)

        Returns:
            dict {feature_name: mean_abs_shap_value} เรียงจากมากไปน้อย
        """
        # เรียงคอลัมน์ให้ตรงกับ feature_names เสมอ
        X = X_df[self.feature_names].values.astype(np.float32)

        raw = self.explainer.shap_values(X)
        # เลือก class 1 (phishing) และให้แน่ใจว่าเป็น 2D (n_samples, n_features)
        if isinstance(raw, list):
            shap_matrix = np.asarray(raw[1])
        else:
            shap_matrix = np.asarray(raw)
        if shap_matrix.ndim == 1:
            shap_matrix = shap_matrix.reshape(1, -1)

        # mean |SHAP value| ต่อ feature = global importance
        mean_abs = np.abs(shap_matrix).mean(axis=0)
        importance = {
            name: round(float(val), 6)
            for name, val in zip(self.feature_names, mean_abs)
        }
        # เรียงจากสำคัญมากไปน้อย
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # ──────────────────────────────────────────────
    # Force Plot Data (สำหรับ Frontend)
    # ──────────────────────────────────────────────

    def get_force_plot_data(self, features_dict: dict) -> dict:
        """
        เตรียมข้อมูลสำหรับ render SHAP force plot ใน React frontend

        Returns:
            JSON-serializable dict พร้อม base_value และ feature contributions
        """
        X = self._dict_to_array(features_dict)
        raw = self.explainer.shap_values(X)
        shap_vals = self._extract_1d_shap(raw, len(self.feature_names))

        # base_value คือค่าเฉลี่ยของ model output (expected value)
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_value = float(self.explainer.expected_value[1])
        else:
            base_value = float(self.explainer.expected_value)

        # features array สำหรับแสดง feature value จริงที่ใช้
        feature_values = [features_dict.get(f, 0.0) for f in self.feature_names]

        # รวม contributions — แยก positive (push toward phishing) และ negative
        contributions = [
            {
                "feature":       name,
                "value":         float(fval),
                "shap_value":    round(float(sval), 6),
                "direction":     "increases_risk" if sval > 0 else "decreases_risk",
            }
            for name, fval, sval in zip(self.feature_names, feature_values, shap_vals)
            if abs(sval) > 1e-6   # ตัด feature ที่ shap_value ≈ 0 ออก (ไม่มีผล)
        ]
        # เรียงตาม |shap_value|
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        output_value = base_value + float(np.sum(shap_vals))

        return {
            "base_value":    round(base_value, 6),
            "output_value":  round(output_value, 6),
            "contributions": contributions,
        }


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import pickle
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from backend.features.url_extractor import extract_url_features

    SAVE_DIR = ROOT / "backend" / "models" / "saved"

    # โหลด saved model (ต้อง run train.py ก่อน)
    try:
        with open(SAVE_DIR / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(SAVE_DIR / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
    except FileNotFoundError:
        print("[ERROR] ไม่พบ saved model — รัน train.py ก่อน")
        sys.exit(1)

    explainer = SHAPExplainer(model, feature_names)

    test_urls = [
        "https://www.google.com/search?q=python",
        "http://paypa1-secure.account-verify.evil.tk/login.php",
    ]

    for url in test_urls:
        feats = extract_url_features(url)
        result = explainer.explain_local(feats)

        label = "PHISHING" if result["prediction"] == 1 else "LEGITIMATE"
        print(f"\nURL: {url}")
        print(f"  → {label} (confidence: {result['confidence']}%)")
        print(f"  Top features:")
        for f in result["top_features"]:
            sign = "+" if f["direction"] == "increases_risk" else "-"
            print(f"    {sign} {f['feature_name']:<30s} {f['shap_value']:+.4f}")

        force_data = explainer.get_force_plot_data(feats)
        print(f"  Force plot: base={force_data['base_value']:.4f} → output={force_data['output_value']:.4f}")
        print(f"  Contributions: {len(force_data['contributions'])} features")
