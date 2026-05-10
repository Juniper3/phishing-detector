"""
main.py
FastAPI application สำหรับ Phishing Detector
รัน: uvicorn backend.api.main:app --reload
"""

import sys
import pickle
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.api.routes.predict import router as predict_router, set_model_cache, get_model_cache

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SAVE_DIR = ROOT / "backend" / "models" / "saved"


def _load_artifact(path: Path):
    """โหลด pickle file — คืน None ถ้าไม่เจอ"""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_into_state(app_instance: FastAPI) -> None:
    """
    โหลด model, scaler, feature_names แล้วเก็บใน:
      1. app.state          — สำหรับ request.app.state pattern
      2. predict module cache — รับประกัน access ได้เสมอ
    """
    model         = _load_artifact(SAVE_DIR / "best_model.pkl")
    scaler        = _load_artifact(SAVE_DIR / "scaler.pkl")
    feature_names = _load_artifact(SAVE_DIR / "feature_names.pkl")

    if model is None or scaler is None or feature_names is None:
        log.warning(
            "ไม่พบ saved model ใน %s "
            "— รัน backend/models/train.py ก่อนใช้งาน /predict",
            SAVE_DIR,
        )
    else:
        log.info(
            "โหลด model สำเร็จ: %s (%d features)",
            type(model).__name__,
            len(feature_names),
        )
        # inject เข้า predict module — วิธีที่รับประกัน 100%
        set_model_cache(model, scaler, feature_names)

    # เก็บใน app.state ด้วยเผื่อ request.app.state ทำงานถูกต้อง
    app_instance.state.model         = model
    app_instance.state.scaler        = scaler
    app_instance.state.feature_names = feature_names

    log.info(
        "DEBUG lifespan — app id=%d  state id=%d  model in state=%s",
        id(app_instance),
        id(app_instance.state),
        type(app_instance.state.model).__name__ if app_instance.state.model is not None else None,
    )


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_into_state(app)
    yield


app = FastAPI(
    title="Phishing Detector API",
    description=(
        "REST API สำหรับตรวจจับ phishing URL "
        "ด้วย XGBoost + SHAP explainability\n\n"
        "**Stack:** FastAPI · XGBoost · SHAP · React"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — อนุญาต Vite dev server (5173) และ CRA (3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Core Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    return {"message": "Phishing Detector API", "version": "1.0"}


@app.get("/health", tags=["General"])
async def health():
    """ตรวจสอบสถานะ server และ model"""
    model, _, _ = get_model_cache()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model is not None else None,
    }


# ─────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────

app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])
