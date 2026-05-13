"""
routes/predict.py
POST /predict — รับ URL แล้ว classify + explain ด้วย SHAP
รองรับทั้ง URL-only mode และ URL+HTML mode
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
from backend.features.html_extractor import extract_html_features
from backend.xai.shap_explainer import SHAPExplainer

log = logging.getLogger(__name__)

router = APIRouter()

# ─────────────────────────────────────────────
# Module-level cache — set โดย main.py ตอน lifespan startup
# ─────────────────────────────────────────────

_model = None
_scaler = None
_feature_names = None
_explainer = None


def set_model_cache(model, scaler, feature_names) -> None:
    global _model, _scaler, _feature_names, _explainer
    _model = model
    _scaler = scaler
    _feature_names = feature_names
    if model is not None and feature_names is not None:
        _explainer = SHAPExplainer(model, feature_names)
    log.info("set_model_cache: model=%s  scaler=%s  features=%s  explainer=%s",
             type(model).__name__ if model is not None else None,
             type(scaler).__name__ if scaler is not None else None,
             len(feature_names) if feature_names is not None else None,
             type(_explainer).__name__ if _explainer is not None else None)


def get_model_cache() -> tuple:
    return _model, _scaler, _feature_names


# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    url: str
    fetch_html: bool = False   # True = ดึง HTML จริงจาก URL (ช้ากว่า แต่ข้อมูลมากกว่า)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
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
    prediction: str                       # "phishing" | "legitimate"
    confidence: float                     # 0.0 – 1.0
    risk_score: int                       # 0 – 100
    top_features: list[FeatureContribution]
    extracted_features: dict[str, float]  # URL features ที่ใช้ใน model
    html_features: dict[str, float] | None  # HTML features (ถ้า fetch_html=True)
    html_status: str                      # "ok" | "timeout" | "error" | "skipped"
    processing_time_ms: float


# ─────────────────────────────────────────────
# HTML Feature Labels (สำหรับแสดงผลใน frontend)
# ─────────────────────────────────────────────

HTML_FEATURE_LABELS = {
    "num_external_links":  "External Links",
    "num_internal_links":  "Internal Links",
    "external_link_ratio": "External Link Ratio",
    "num_images":          "Images",
    "num_scripts":         "Scripts",
    "num_iframes":         "iFrames",
    "has_favicon":         "Has Favicon",
    "title_match_domain":  "Title Matches Domain",
    "has_login_form":      "Has Login Form",
    "form_action_external":"Form Action External",
    "null_links_ratio":    "Null Links Ratio",
    "has_copyright":       "Has Copyright",
    "meta_refresh":        "Meta Refresh",
    "num_hidden_elements": "Hidden Elements",
}

# ─────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    """
    รับ URL → extract features → predict → explain ด้วย SHAP

    Params:
        fetch_html: False (default) = URL features เท่านั้น (เร็ว ~50ms)
                    True            = URL + HTML features (ช้า ~2-5s ขึ้นอยู่กับเว็บ)

    Errors:
        422: URL format ผิด
        503: model ยังไม่โหลด
        500: error ที่ไม่คาดคิด
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
        # ── 1. Extract URL Features ───────────────────────────────────
        features_dict = extract_url_features(body.url)

        extracted_features = {
            f: round(float(features_dict.get(f, 0.0)), 4)
            for f in feature_names
        }

        # ── 2. Extract HTML Features (optional) ──────────────────────
        html_features_dict = None
        html_status        = "skipped"

        if body.fetch_html:
            try:
                raw_html = extract_html_features(url=body.url)
                html_features_dict = {
                    k: round(float(v), 4)
                    for k, v in raw_html.items()
                }
                html_status = "ok"
                log.info("html_features fetched for %s", body.url[:50])
            except TimeoutError:
                html_status = "timeout"
                log.warning("html_fetch timeout: %s", body.url[:50])
            except ConnectionError as e:
                html_status = "error"
                log.warning("html_fetch error: %s — %s", body.url[:50], str(e)[:80])
            except Exception as e:
                html_status = "error"
                log.warning("html_fetch unexpected: %s — %s", body.url[:50], str(e)[:80])

        # ── 3. Build Feature Vector (URL features เท่านั้น → model) ──
        X_raw = np.array(
            [extracted_features[f] for f in feature_names],
            dtype=np.float32,
        ).reshape(1, -1)
        X_scaled = scaler.transform(X_raw)

        # ── 4. Predict ────────────────────────────────────────────────
        prediction_int = int(model.predict(X_scaled)[0])
        if hasattr(model, "predict_proba"):
            proba          = model.predict_proba(X_scaled)[0]
            confidence_raw = float(proba[prediction_int])
        else:
            score          = model.decision_function(X_scaled)[0]
            confidence_raw = float(1 / (1 + np.exp(-score)))

        # ── 5. SHAP Explanation ───────────────────────────────────────
        scaled_features_dict = {
            f: float(v) for f, v in zip(feature_names, X_scaled[0])
        }
        shap_result  = _explainer.explain_local(scaled_features_dict)

        top_features = [
            FeatureContribution(
                name=f["feature_name"],
                value=extracted_features.get(f["feature_name"], 0.0),
                shap_value=round(f["shap_value"], 4),
                direction=f["direction"],
            )
            for f in shap_result["top_features"]
        ]

        # ── 6. Risk Score ─────────────────────────────────────────────
        phishing_proba = confidence_raw if prediction_int == 1 else (1 - confidence_raw)

        # ถ้ามี HTML features — ปรับ risk score ตาม HTML signals
        if html_features_dict and html_status == "ok":
            html_risk_boost = _compute_html_risk_boost(html_features_dict)
            # blend: 80% model score + 20% HTML signals
            phishing_proba = min(1.0, phishing_proba * 0.8 + html_risk_boost * 0.2)

        risk_score = int(round(phishing_proba * 100))

        processing_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

        log.info(
            "predict  url=%-50s  result=%-10s  conf=%.2f  html=%s  ms=%.1f",
            body.url[:50],
            "phishing" if prediction_int == 1 else "legitimate",
            confidence_raw,
            html_status,
            processing_time_ms,
        )

        return PredictResponse(
            url=body.url,
            prediction="phishing" if prediction_int == 1 else "legitimate",
            confidence=round(confidence_raw, 4),
            risk_score=risk_score,
            top_features=top_features,
            extracted_features=extracted_features,
            html_features=html_features_dict,
            html_status=html_status,
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")


# ─────────────────────────────────────────────
# HTML Risk Boost
# rule-based scoring จาก HTML features
# คืนค่า 0.0 – 1.0 (phishing probability จาก HTML)
# ─────────────────────────────────────────────

def _compute_html_risk_boost(html: dict) -> float:
    """
    คำนวณ phishing risk จาก HTML features
    ใช้ rule-based scoring (ไม่ใช่ ML) เพื่อ complement URL model

    Risk factors:
    - has_login_form=1  (+0.3) — มี password input
    - form_action_external=1 (+0.4) — form ส่งไป domain อื่น = อันตรายมาก
    - meta_refresh=1    (+0.2) — redirect อัตโนมัติ
    - num_iframes > 0   (+0.15) — hidden iframes
    - null_links_ratio > 0.5 (+0.2) — links ส่วนใหญ่เป็น #
    - title_match_domain=0 (+0.1) — title ไม่ตรงกับ domain

    Legitimacy factors:
    - has_copyright=1   (-0.1) — มี copyright notice
    - has_favicon=1     (-0.05) — มี favicon (เล็กน้อย)
    """
    score = 0.0

    # Risk indicators
    if html.get("has_login_form"):
        score += 0.30
    if html.get("form_action_external"):
        score += 0.40
    if html.get("meta_refresh"):
        score += 0.20
    if html.get("num_iframes", 0) > 0:
        score += 0.15
    if html.get("null_links_ratio", 0) > 0.5:
        score += 0.20
    if html.get("title_match_domain") == 0:
        score += 0.10

    # Legitimacy indicators
    if html.get("has_copyright"):
        score -= 0.10
    if html.get("has_favicon"):
        score -= 0.05

    return max(0.0, min(1.0, score))
