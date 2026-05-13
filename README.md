# Phishing Detector

ระบบตรวจจับ Phishing URL ด้วย Machine Learning พร้อม XAI (SHAP)  
**IS ปริญญาโท — Dhurakij Pundit University**

**Stack:** FastAPI · XGBoost · SHAP · React · Tailwind CSS

---

## โครงสร้างโปรเจค

```
phishing-detector/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app + lifespan startup
│   │   └── routes/predict.py   # POST /predict endpoint
│   ├── features/
│   │   ├── url_extractor.py    # 43 URL features
│   │   └── html_extractor.py   # 14 HTML features (optional)
│   ├── models/
│   │   ├── feature_config.py   # FEATURE_MAP shared config
│   │   ├── train.py            # training pipeline (3 models × 3 modes)
│   │   ├── cross_dataset_eval.py # Youden's J threshold calibration
│   │   └── saved/              # best_model.pkl, scaler.pkl, feature_names.pkl
│   ├── xai/
│   │   └── shap_explainer.py   # SHAP TreeExplainer wrapper
│   └── tests/
│       ├── test_predict_integration.py  # pipeline end-to-end test
│       ├── test_blind_spot_analysis.py  # feature engineering experiment
│       └── phishtank_test.py            # live URL test via API
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── URLInput.jsx        # URL input + validation
│           ├── ResultCard.jsx      # prediction badge + risk bar
│           ├── SHAPChart.jsx       # horizontal bar chart (recharts)
│           ├── FeatureTable.jsx    # 9 features + SHAP cross-reference
│           ├── HTMLFeaturePanel.jsx # HTML risk signals
│           └── URLHistory.jsx      # localStorage history
├── data/
│   ├── raw/                    # dataset.csv (PhiUSIIL), iscx_url.csv (ISCX)
│   └── results/                # train_comparison.csv, roc_curves.png
└── requirements.txt
```

---

## Model

| | ค่า |
|---|---|
| Algorithm | XGBoost (XGBClassifier) |
| Features | 9 URL-only lexical features |
| Training data | PhiUSIIL (235,795) + ISCX (15,367) combined |
| F1 Score | 0.9679 |
| AUC-ROC | 0.9757 |
| Threshold | 0.4368 (Youden's J) |

**9 Features:** `url_length`, `hostname_length`, `has_ip`, `num_digits`, `digit_ratio`, `special_char_ratio`, `url_entropy`, `num_subdomains`, `num_equal`

---

## วิธีติดตั้งและรัน

### ข้อกำหนด
- Python 3.13+
- Node.js 18+

### 1. Clone และเตรียม environment

```bash
git clone <repo-url>
cd phishing-detector

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. เตรียม Dataset

วาง dataset ไว้ที่:
```
data/raw/dataset.csv      # PhiUSIIL Phishing URL Dataset
data/raw/iscx_url.csv     # ISCX-URL-2016 Dataset
```

### 3. Train Model

```bash
python backend/models/train.py
```

ผลลัพธ์ถูกบันทึกที่ `backend/models/saved/`

### 4. รัน Backend

```bash
uvicorn backend.api.main:app --reload --port 8001
```

API พร้อมใช้งานที่ `http://localhost:8001`  
Docs: `http://localhost:8001/docs`

### 5. รัน Frontend

```bash
cd frontend
npm install
npm run dev
```

เปิด `http://localhost:5173`

---

## API

### `POST /api/v1/predict`

```json
// Request
{
  "url": "https://example.com/login",
  "fetch_html": false
}

// Response
{
  "url": "https://example.com/login",
  "prediction": "legitimate",
  "confidence": 0.9823,
  "risk_score": 2,
  "top_features": [...],
  "extracted_features": {...},
  "html_features": null,
  "html_status": "skipped",
  "processing_time_ms": 12.4
}
```

`fetch_html: true` — ดึง HTML จริงเพื่อวิเคราะห์เพิ่มเติม (ช้าขึ้น ~2-5s)

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "model_type": "XGBClassifier" }
```

---

## Experiments

### Cross-Dataset Evaluation

```bash
python backend/models/cross_dataset_eval.py
```

วิเคราะห์ generalization gap และ threshold calibration ด้วย Youden's J  
ผล: `data/results/cross_dataset_threshold_eval.csv`, `roc_curves.png`

### Blind Spot Analysis

```bash
python backend/tests/test_blind_spot_analysis.py
```

วิเคราะห์ว่าทำไม URL เช่น `paypal.com.evil-login.tk` ถูก miss  
พบว่า dataset distribution ≠ security heuristics

### Integration Test

```bash
python backend/tests/test_predict_integration.py
```

ทดสอบ pipeline ครบทุกขั้น: extract → scale → predict → SHAP

---

## Key Findings (Thesis)

1. **Youden's J threshold (0.4368)** ช่วยให้ F1 บน ISCX cross-dataset ดีขึ้นอย่างมีนัยสำคัญ จาก F1=0.0 (threshold=0.5) เป็น F1 ที่ยอมรับได้
2. **Blind spots:** domain-squatting (.tk) และ IP-based phishing — URL-only lexical features ไม่เพียงพอ ต้องการ DNS age / WHOIS / SSL cert
3. **Feature engineering experiment:** เพิ่ม `has_suspicious_tld`, `brand_in_subdomain`, `has_shortener` ทำให้ model แย่ลง เพราะ PhiUSIIL มี 94% legitimate .tk sites → model เรียนกลับทิศ
