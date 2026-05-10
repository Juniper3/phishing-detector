"""
routes/predict.py
POST /predict — รับ URL แล้ว classify + explain ด้วย SHAP
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from backend.features.url_extractor import extract_url_features
from backend.xai.shap_explainer import SHAPExplainer

log = logging.getLogger(__name__)

router = APIRouter()

# ─────────────────────────────────────────────
# Module-level cache — set โดย main.py ตอน lifespan startup
# ใช้เป็น source of truth หลัก เพื่อหลีกเลี่ยงปัญหา request.app scope
# ─────────────────────────────────────────────

_model = None
_scaler = None
_feature_names = None


def set_model_cache(model, scaler, feature_names) -> None:
    """เรียกจาก main.py lifespan เพื่อ inject model เข้า module scope"""
    global _model, _scaler, _feature_names
    _model = model
    _scaler = scaler
    _feature_names = feature_names
    log.info("set_model_cache: model=%s  scaler=%s  features=%s",
             type(model).__name__ if model is not None else None,
             type(scaler).__name__ if scaler is not None else None,
             len(feature_names) if feature_names is not None else None)


def get_model_cache() -> tuple:
    """คืน (model, scaler, feature_names) จาก module cache — ค่าปัจจุบันเสมอ"""
    return _model, _scaler, _feature_names


# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """ตรวจ URL format — ต้องขึ้นต้นด้วย http:// หรือ https://"""
        v = v.strip()
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("URL ไม่ถูกต้อง ต้องขึ้นต้นด้วย http:// หรือ https://")
        return v


class FeatureContribution(BaseModel):
    name: str
    value: float
    shap_value: float
    direction: str   # "increases_risk" | "decreases_risk"


class PredictResponse(BaseModel):
    url: str
    prediction: str                    # "phishing" | "legitimate"
    confidence: float                  # 0.0 – 1.0
    risk_score: int                    # 0 – 100
    top_features: list[FeatureContribution]
    extracted_features: dict[str, float]  # feature values ที่ extract ได้จาก URL จริง
    processing_time_ms: float


# ─────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    """
    รับ URL → extract features → predict → explain ด้วย SHAP

    - 422: URL format ผิด (Pydantic validator)
    - 503: model ยังไม่ได้โหลด (train.py ยังไม่ได้รัน)
    - 500: error ที่ไม่คาดคิด
    """
    model         = _model
    scaler        = _scaler
    feature_names = _feature_names

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model ยังไม่พร้อม กรุณารัน train.py ก่อน แล้ว restart server",
        )

    t_start = time.perf_counter()

    try:

        # 1. Extract URL features
        features_dict = extract_url_features(body.url)

        # 2. Map features จาก extractor → model feature order
        #    features ที่ model ต้องการแต่ extractor ไม่มี → default 0.0
        extracted_features = {
            f: round(float(features_dict.get(f, 0.0)), 4)
            for f in feature_names
        }

        X_raw = np.array(
            [extracted_features[f] for f in feature_names],
            dtype=np.float32,
        ).reshape(1, -1)
        X_scaled = scaler.transform(X_raw)

        # 3. Predict
        prediction_int = int(model.predict(X_scaled)[0])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            confidence_raw = float(proba[prediction_int])
        else:
            # LinearSVC: sigmoid approximation จาก decision_function
            score = model.decision_function(X_scaled)[0]
            confidence_raw = float(1 / (1 + np.exp(-score)))

        # 4. SHAP explanation — ใช้ scaled features ที่ model เห็น
        scaled_features_dict = {
            f: float(v) for f, v in zip(feature_names, X_scaled[0])
        }
        explainer = SHAPExplainer(model, feature_names)
        shap_result = explainer.explain_local(scaled_features_dict)

        # 5. แปลง top_features ให้ตรง schema
        #    ใช้ original value (ก่อน scale) เพื่อให้ผู้ใช้อ่านเข้าใจง่าย
        top_features = [
            FeatureContribution(
                name=f["feature_name"],
                value=extracted_features.get(f["feature_name"], 0.0),
                shap_value=round(f["shap_value"], 4),
                direction=f["direction"],
            )
            for f in shap_result["top_features"]
        ]

        # risk_score: แปลง phishing probability เป็น 0-100
        phishing_proba = confidence_raw if prediction_int == 1 else (1 - confidence_raw)
        risk_score = int(round(phishing_proba * 100))

        processing_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

        log.info(
            "predict  url=%-50s  result=%-10s  conf=%.2f  ms=%.1f",
            body.url[:50], "phishing" if prediction_int == 1 else "legitimate",
            confidence_raw, processing_time_ms,
        )

        return PredictResponse(
            url=body.url,
            prediction="phishing" if prediction_int == 1 else "legitimate",
            confidence=round(confidence_raw, 4),
            risk_score=risk_score,
            top_features=top_features,
            extracted_features=extracted_features,
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")
