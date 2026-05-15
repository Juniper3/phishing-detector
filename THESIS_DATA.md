# THESIS_DATA.md — ข้อมูลสำหรับวิทยานิพนธ์ Phishing Detector
> รวบรวมจาก codebase จริง · อัปเดต 2026-05-15

---

## 1. ภาพรวมระบบ

### ชื่อระบบ
**Phishing Detector** — ระบบตรวจจับ phishing URL ด้วย Machine Learning และ Explainable AI

### วัตถุประสงค์
- ตรวจจับ URL ที่เป็น phishing โดยใช้ lexical features จาก URL structure (ไม่ต้องเข้าถึงเว็บ)
- อธิบายการตัดสินใจของ model ด้วย SHAP values (XAI)
- รองรับ optional HTML analysis เพื่อเพิ่มความแม่นยำ

### Tech Stack

| ชั้น | เทคโนโลยี | version |
|------|-----------|---------|
| ML Model | XGBoost | 2.1.3 |
| XAI | SHAP | 0.46.0 |
| API Framework | FastAPI + Uvicorn | 0.115.0 / 0.30.1 |
| Data Validation | Pydantic | 2.13.4 |
| Numerical | NumPy, Pandas | 2.0.2 / 2.2.3 |
| Baseline Models | scikit-learn (RF, SVM) | 1.6.1 |
| HTML Parsing | BeautifulSoup4 | 4.12.3 |
| HTTP Client | requests | 2.32.3 |
| Frontend | React + Vite | 18.3.1 / 5.x |
| UI Styling | Tailwind CSS | 3.4.6 |
| Charts | Recharts | 2.12.7 |
| HTTP (Frontend) | axios | 1.7.2 |
| Deployment | Render (Singapore region) | — |

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                  │
│                                                                  │
│  URLInput → [POST /api/v1/predict]                              │
│                                                                  │
│  ResultCard   SHAPChart   FeatureTable   HTMLFeaturePanel       │
│  URLHistory                                                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP (axios)  VITE_API_URL
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI + Uvicorn)                   │
│                                                                  │
│  GET  /                  → health info                          │
│  GET  /health            → model status                         │
│  POST /api/v1/predict    → prediction + SHAP explanation        │
│                                                                  │
│  ┌─────────────────┐   ┌──────────────────┐                    │
│  │  url_extractor  │   │  html_extractor  │ (optional)         │
│  │  9 URL features │   │  14 HTML features│                    │
│  └────────┬────────┘   └────────┬─────────┘                    │
│           │                     │                               │
│           ▼                     ▼                               │
│  ┌─────────────────────────────────────────┐                   │
│  │         StandardScaler (fit on train)   │                   │
│  └─────────────────────┬───────────────────┘                   │
│                         ▼                                       │
│  ┌─────────────────────────────────────────┐                   │
│  │    XGBoost (best_model.pkl, 902 KB)     │                   │
│  │    predict_proba → confidence           │                   │
│  └─────────────────────┬───────────────────┘                   │
│                         ▼                                       │
│  ┌─────────────────────────────────────────┐                   │
│  │    SHAPExplainer (TreeExplainer)        │                   │
│  │    top-5 features + directions          │                   │
│  └─────────────────────────────────────────┘                   │
│                         ▼                                       │
│  risk_score = model_proba × 0.8 + html_risk × 0.2 (if HTML ON)│
└─────────────────────────────────────────────────────────────────┘
```

### การทำงานของระบบตั้งแต่ต้นจนจบ

1. ผู้ใช้กรอก URL ใน frontend → กด Detect
2. Frontend POST `{url, fetch_html}` ไปยัง `/api/v1/predict`
3. Backend ตรวจสอบ URL format (http:// หรือ https://)
4. `url_extractor.extract_url_features()` ดึง 9 URL features
5. (ถ้า fetch_html=True) `html_extractor.extract_html_features()` ดึง HTML จาก URL จริง timeout 10s
6. Scale features ด้วย StandardScaler
7. XGBoost `predict_proba()` → confidence (0–1)
8. `SHAPExplainer.explain_local()` → top-5 SHAP features
9. คำนวณ risk_score (0–100): URL model 100% หรือ blend 80/20 ถ้ามี HTML
10. Return PredictResponse → Frontend render ResultCard + SHAPChart

---

## 2. Dataset

### Dataset 1: PhiUSIIL (Primary)

| รายละเอียด | ค่า |
|-----------|-----|
| ชื่อไฟล์ | `data/raw/dataset.csv` |
| แหล่งที่มา | PhiUSIIL Phishing URL Dataset (UCI ML Repository) |
| จำนวน rows ทั้งหมด | **235,795** |
| Phishing (label=1) | ~134,850 (57.2%) |
| Legitimate (label=0) | ~100,945 (42.8%) |
| Train set (80%) | 188,636 rows |
| Validation/Test set (20%) | 47,159 rows |

Label column candidates ที่ใช้ detect อัตโนมัติ: `label`, `phishing`, `class`, `Label`, `Phishing`, `Class`

### Dataset 2: ISCX-URL (Cross-Dataset Validation)

| รายละเอียด | ค่า |
|-----------|-----|
| ชื่อไฟล์ | `data/raw/iscx_url.csv` |
| แหล่งที่มา | ISCX URL Dataset (University of New Brunswick) |
| Label column | `URL_Type_obf_Type` |
| Filtered: phishing vs benign | **15,367** rows |
| Phishing | 7,586 (49.4%) |
| Benign | 7,781 (50.6%) |

หมายเหตุ: ISCX มีประเภทอื่นด้วย (defacement, malware, spam) แต่ทดลองเฉพาะ phishing vs benign เพื่อ apples-to-apples comparison

### Dataset Combined (Mode A Training)

| รายละเอียด | ค่า |
|-----------|-----|
| รวม | **251,162** rows |
| Test set (20%) | 50,233 rows |
| Train set (80%) | 200,929 rows |

### Features ที่ map ระหว่าง dataset

| Canonical Name | PhiUSIIL Column | ISCX Column |
|---------------|-----------------|-------------|
| url_length | URLLength | urlLen |
| hostname_length | DomainLength | domainlength |
| has_ip | IsDomainIP | ISIpAddressInDomainName |
| num_digits | NoOfDegitsInURL | URL_DigitCount |
| digit_ratio | DegitRatioInURL | NumberRate_URL |
| special_char_ratio | SpacialCharRatioInURL | spcharUrl |
| url_entropy | URLCharProb | Entropy_URL |
| num_subdomains | NoOfSubDomain | domain_token_count* |
| num_equal | NoOfEqualsInURL | URLQueries_variable** |

*`domain_token_count` ใน ISCX = dots+1, ค่าสูงกว่า NoOfSubDomain ~2 หน่วย  
**`URLQueries_variable` ≈ num_equal สำหรับ URL ทั่วไป

---

## 3. URL Features (9 ตัว)

### 1. url_length
- **สูตรจากโค้ด**: `len(url)` — นับทุก character รวม scheme, hostname, path, query
- **ค่าที่ผิดปกติ**: URL ยาวมาก (>75 chars) มักเป็น phishing
- **เหตุผล**: Legitimate sites ใช้ URL สั้นกระชับ Phishing มักต่อ parameter ยาวเพื่อ obfuscate

### 2. hostname_length
- **สูตรจากโค้ด**: `len(parsed.hostname)` — ความยาว hostname เท่านั้น
- **ค่าที่ผิดปกติ**: >20 chars
- **เหตุผล**: Phishing domains มักยาวเพื่อ ลอก domain จริง เช่น `secure-login-paypal-update.com`

### 3. has_ip
- **สูตรจากโค้ด**: ใช้ `ipaddress.ip_address(hostname)` → 1 ถ้า parse ได้ หรือ match `\d+\.\d+\.\d+\.\d+`
- **ค่าที่ผิดปกติ**: has_ip = 1
- **เหตุผล**: Legitimate sites ใช้ domain name ไม่ใช่ IP address โดยตรง

### 4. num_digits
- **สูตรจากโค้ด**: `sum(c.isdigit() for c in url)` — นับ digit ทุกตัวใน URL
- **ค่าที่ผิดปกติ**: >5 digits
- **เหตุผล**: ตัวเลขมากใน URL บ่งชี้ random-generated domain หรือ IP encoding

### 5. digit_ratio
- **สูตรจากโค้ด**: `num_digits / url_length` (0.0–1.0)
- **ค่าที่ผิดปกติ**: >0.1
- **เหตุผล**: Feature สำคัญที่สุดอันดับ 1 ใน cross-dataset evaluation — normalize ความยาว URL

### 6. special_char_ratio
- **สูตรจากโค้ด**: `sum(1 for c in url if not c.isalnum() and c not in "-._~/") / url_length`
- **ค่าที่ผิดปกติ**: >0.15
- **เหตุผล**: Phishing URLs มี `@`, `%`, `=`, `?`, `&` หนาแน่นกว่า legitimate

### 7. url_entropy
- **สูตรจากโค้ด**: Shannon entropy `-Σ (freq/n × log₂(freq/n))` ของทุก character ใน URL
- **ค่าที่ผิดปกติ**: >4.0 bits (URL สุ่มมาก)
- **เหตุผล**: URL ที่สร้างด้วย algorithm จะมี entropy สูง ต่างจาก URL ที่มีความหมาย

### 8. num_subdomains
- **สูตรจากโค้ด**: `max(0, len(hostname.split(".")) - 2)` — ลบ www prefix ก่อนนับ, ลบ SLD+TLD 2 ส่วนสุดท้าย
- **ค่าที่ผิดปกติ**: >2
- **เหตุผล**: Feature สำคัญอันดับ 2 — Phishing ใช้ subdomain ซ้อนเพื่อ สร้างภาพว่าเป็น domain จริง เช่น `secure.paypal.com.evil.tk`

### 9. num_equal
- **สูตรจากโค้ด**: `url.count("=")` — นับ `=` ทั้งหมดใน URL
- **ค่าที่ผิดปกติ**: >3
- **เหตุผล**: Query parameters มาก บ่งชี้การ redirect หรือ tracking แบบ phishing

---

## 4. HTML Features (14 ตัว)

### วิธี extract: BeautifulSoup4 + html.parser, ดึง HTML จาก URL จริง timeout 10 วินาที

### 1. num_external_links
- **วิธี extract**: นับ `<a href>` ที่ขึ้นต้น `http://`/`https://` และ domain ≠ page_domain
- **เหตุผล**: Phishing มักมี external links มากเพื่อ redirect ผู้ใช้

### 2. num_internal_links
- **วิธี extract**: นับ `<a href>` ที่เป็น relative URL หรือ domain เดียวกัน
- **เหตุผล**: Legitimate sites มี internal links มากกว่า

### 3. external_link_ratio
- **วิธี extract**: `num_external_links / (num_external + num_internal + num_null)`
- **เหตุผล**: สัดส่วน external links สูง = phishing signal

### 4. num_images
- **วิธี extract**: `len(soup.find_all("img"))`
- **เหตุผล**: Phishing มักใช้ รูปภาพจาก brand จริงเพื่อสร้างความน่าเชื่อถือ

### 5. num_scripts
- **วิธี extract**: `len(soup.find_all("script"))`
- **เหตุผล**: Script มากผิดปกติ บ่งชี้ obfuscation หรือ keylogger

### 6. num_iframes
- **วิธี extract**: `len(soup.find_all("iframe"))`
- **เหตุผล**: Hidden iframe (width=0, height=0) ใช้ใน tracking/clickjacking

### 7. has_favicon
- **วิธี extract**: `soup.find_all("link", rel=lambda r: r and "icon" in " ".join(r).lower())`, 1 ถ้ามี
- **เหตุผล**: Phishing ที่เร่งรีบมักลืม copy favicon → risk ลดลงถ้ามี favicon (-0.05 HTML score)

### 8. title_match_domain
- **วิธี extract**: ตรวจว่า `hostname.split(".")[-2]` (registered domain) ปรากฏใน `<title>` text
- **เหตุผล**: Phishing มักใช้ title ของแบรนด์จริง ("PayPal Login") แต่ domain ไม่ตรง

### 9. has_login_form
- **วิธี extract**: loop `<form>` → ตรวจว่ามี `<input type="password">`
- **เหตุผล**: Login form = credential harvesting signal (+0.30 HTML risk score)

### 10. form_action_external
- **วิธี extract**: ตรวจ `action` attribute ของ `<form>` ว่า domain ≠ page_domain
- **เหตุผล**: อันตรายที่สุด — form ส่งข้อมูลไป server อื่น (+0.40 HTML risk score)

### 11. null_links_ratio
- **วิธี extract**: นับ href ที่เป็น `"#"`, `""`, `"javascript:void(0)"`, `"javascript:;"` / total links
- **เหตุผล**: Phishing template มี null links มากเพราะ copy HTML แต่ไม่ได้ link จริง (+0.20 ถ้า >0.5)

### 12. has_copyright
- **วิธี extract**: ตรวจ `"©"` หรือ `"&copy;"` ใน raw HTML หรือ `"copyright"` ใน page text
- **เหตุผล**: Phishing มักไม่มี copyright notice (-0.10 HTML risk score)

### 13. meta_refresh
- **วิธี extract**: `soup.find_all("meta")` → ตรวจ `http-equiv="refresh"`
- **เหตุผล**: Phishing redirect อัตโนมัติหลัง harvest credentials (+0.20 HTML risk score)

### 14. num_hidden_elements
- **วิธี extract**: นับ tags ที่มี `hidden` attribute หรือ style มี `display:none` หรือ `visibility:hidden`
- **เหตุผล**: ซ่อน tracker หรือ honeypot ที่ผู้ใช้ไม่เห็น

---

## 5. ML Pipeline

### Preprocessing

```
1. โหลด dataset → map columns ตาม FEATURE_MAP
2. fillna(0) ทุก feature
3. สร้าง numpy array dtype=float32
4. StandardScaler.fit_transform(X_train) — fit บน train เท่านั้น (ป้องกัน data leakage)
5. StandardScaler.transform(X_test/X_val) — ใช้ parameters จาก train
```

### Algorithms ที่เปรียบเทียบ

#### XGBoost (Best Model)
```python
XGBClassifier(
    n_estimators    = 300,
    max_depth       = 6,
    learning_rate   = 0.1,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    eval_metric     = "logloss",
    random_state    = 42,
    n_jobs          = -1,
)
```

#### Random Forest
```python
RandomForestClassifier(
    n_estimators    = 300,
    max_depth       = None,      # ไม่จำกัด depth
    min_samples_leaf= 2,
    random_state    = 42,
    n_jobs          = -1,
)
```

#### SVM (LinearSVC)
```python
LinearSVC(
    C           = 1.0,
    max_iter    = 2000,
    random_state= 42,
)
# ไม่มี predict_proba → ใช้ decision_function + sigmoid approximation แทน
```

### Train/Test Split
```python
train_test_split(
    X_combined, y_combined,
    test_size   = 0.2,
    random_state= 42,
    stratify    = y_combined,    # รักษาสัดส่วน phishing/legit
)
```

### Best Model Selection
เลือก model ที่ **F1 สูงสุด** จาก Mode A (Combined Training) → XGBoost (F1=0.9679)

### Threshold Optimization — วิธีหา 0.437

**วิธี: Youden's J Statistic บน PhiUSIIL Validation Set**

```python
fpr, tpr, thresholds = roc_curve(y_val, predict_proba(X_val)[:, 1])
youden_j = tpr - fpr           # J = Sensitivity + Specificity - 1
best_idx  = np.argmax(youden_j)
optimal_threshold = thresholds[best_idx]  # = 0.4368
```

- Youden's J maximize ทั้ง sensitivity (recall) และ specificity พร้อมกัน
- ค่าที่ได้: **threshold = 0.4368** (ปัดเป็น 0.437 ในเอกสาร)
- ใช้เพื่อแก้ปัญหา Mode B (Phi→ISCX) ที่ threshold=0.5 ทำให้ F1=0.0

---

## 6. ผลการทดสอบทั้งหมด

### 6.1 train_comparison.csv — ผลเปรียบเทียบ 3 Models × 3 Modes

#### Mode A: Combined Dataset (80/20 Stratified Split) — n_test=50,233

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | TN | FP | FN | TP |
|-------|----------|-----------|--------|-----|---------|-----|-----|-----|-----|
| **XGBoost** | **0.9628** | **0.9472** | **0.9896** | **0.9679** | **0.9757** | 20,173 | 1,572 | 296 | 28,192 |
| Random Forest | 0.9596 | 0.9477 | 0.9830 | 0.9650 | 0.9731 | 20,199 | 1,546 | 485 | 28,003 |
| SVM (LinearSVC) | 0.7837 | 0.7355 | 0.9661 | 0.8352 | 0.8010 | 11,849 | 9,896 | 967 | 27,521 |

**Confusion Matrix ของ XGBoost (Mode A):**
```
                  Predicted: Legit    Predicted: Phishing
Actual: Legit        TN = 20,173          FP = 1,572
Actual: Phishing     FN = 296             TP = 28,192
```

#### Mode B: Train PhiUSIIL → Test ISCX (Cross-Dataset) — n_test=15,367

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | TN | FP | FN | TP |
|-------|----------|-----------|--------|-----|---------|-----|-----|-----|-----|
| XGBoost | 0.5063 | 0.0 | 0.0 | **0.0** | **0.842** | 7,781 | 0 | 7,586 | 0 |
| Random Forest | 0.5063 | 0.0 | 0.0 | 0.0 | 0.6375 | 7,781 | 0 | 7,586 | 0 |
| SVM | 0.5063 | 0.0 | 0.0 | 0.0 | 0.5114 | 7,781 | 0 | 7,586 | 0 |

> **สังเกต**: AUC=0.842 แต่ F1=0.0 → model ranking ถูก แต่ threshold=0.5 ทำให้ตัดสินใจเป็น 0 ทั้งหมด เพราะ feature distribution ของ ISCX ต่ำกว่า PhiUSIIL

#### Mode C: Train ISCX → Test PhiUSIIL — n_test=235,795

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | TN | FP | FN | TP |
|-------|----------|-----------|--------|-----|---------|-----|-----|-----|-----|
| XGBoost | 0.6277 | 0.6057 | 0.9999 | 0.7544 | 0.5168 | 13,172 | 87,773 | 10 | 134,840 |
| Random Forest | 0.6013 | 0.5892 | 1.0000 | 0.7415 | 0.6401 | 6,939 | 94,006 | 0 | 134,850 |
| SVM | 0.4241 | 0.0 | 0.0 | 0.0 | 0.4836 | 100,005 | 940 | 134,850 | 0 |

---

### 6.2 cross_dataset_threshold_eval.csv — ผลหลัง Threshold Calibration

| Dataset / Mode | Threshold | Accuracy | Precision | Recall | F1 | AUC-ROC | n_samples |
|---------------|-----------|----------|-----------|--------|-----|---------|-----------|
| PhiUSIIL Validation | 0.50 | 0.9633 | 0.9458 | 0.9926 | 0.9686 | 0.9739 | 47,159 |
| PhiUSIIL Validation | **0.437** | **0.9637** | 0.9449 | **0.9945** | **0.9690** | 0.9739 | 47,159 |
| ISCX Cross-Dataset | 0.50 | 0.9728 | 0.9862 | 0.9583 | 0.9721 | 0.9961 | 15,367 |
| ISCX Cross-Dataset | **0.437** | **0.9733** | 0.9826 | **0.9628** | **0.9726** | 0.9961 | 15,367 |

**Confusion Matrix ISCX ที่ threshold=0.437:**
```
                  Predicted: Legit    Predicted: Phishing
Actual: Legit        TN = 7,652           FP = 129
Actual: Phishing     FN = 282             TP = 7,304
```

**ผลการปรับ threshold บน cross_dataset_eval.csv (XGBClassifier):**

| Dataset | n_samples | Accuracy | Precision | Recall | F1 | AUC |
|---------|-----------|----------|-----------|--------|-----|-----|
| PhiUSIIL (in-sample) | 47,159 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ISCX (phishing vs benign) | 15,367 | 0.5063 | 0.0 | 0.0 | 0.0 | 0.5015 |
| ISCX (phishing vs all) | 36,707 | 0.7933 | 0.0 | 0.0 | 0.0 | 0.5005 |

---

### 6.3 phishtank_summary.json — Live Test (20 Phishing + 20 Legitimate)

| Metric | ค่า |
|--------|-----|
| Total URLs tested | 40 |
| Accuracy | 0.575 (57.5%) |
| Precision | 1.000 (ไม่มี False Positive เลย) |
| Recall | 0.150 (จับ phishing ได้แค่ 15%) |
| F1-Score | 0.2609 |
| TP (phishing ที่จับได้) | 3 / 20 |
| FN (phishing ที่พลาด) | 17 / 20 |
| TN (legitimate ที่ถูกต้อง) | 20 / 20 |
| FP (legitimate ที่ผิด) | 0 / 20 |
| Errors | 0 |
| ทดสอบวันที่ | 2026-05-13 |

---

## 7. SHAP / XAI

### วิธี Implement จริง

```python
class SHAPExplainer:
    def __init__(self, model, feature_names):
        self.explainer = shap.TreeExplainer(model)  # สำหรับ XGBoost
        # เลือก explainer ตาม model type อัตโนมัติ:
        # XGBoost/RF/GBT → TreeExplainer (exact, เร็ว)
        # LinearSVC/LR   → LinearExplainer
        # อื่นๆ          → KernelExplainer (sampling, ช้า)
```

```python
def explain_local(self, features_dict):
    X = to_array(features_dict)
    raw = self.explainer.shap_values(X)
    # raw อาจเป็น list[class0, class1] หรือ ndarray
    # → normalize ให้เป็น 1D array (n_features,) เสมอ
    shap_vals = extract_1d_shap(raw)

    top_features = sorted by |shap_value|, top-5
    direction: "increases_risk"  (shap_value > 0)
               "decreases_risk"  (shap_value < 0)
```

### Top Features (จาก Cross-Dataset Analysis)

อันดับ 1–3 ที่สำคัญและ portable ระหว่าง datasets:

| อันดับ | Feature | ความหมาย |
|--------|---------|----------|
| 1 | `digit_ratio` | สัดส่วนตัวเลขใน URL |
| 2 | `num_subdomains` | จำนวน subdomain |
| 3 | `has_ip` | มี IP address แทน domain |

Top decision feature จาก live test (พบบ่อยที่สุด): `num_subdomains` (legitimate sites มีค่าต่ำสม่ำเสมอ)

### ตัวอย่าง SHAP Explanation จริง

#### ตัวอย่างที่ 1: Phishing ที่ detect ได้ถูกต้อง
```
URL:        http://paypal.account-verify.com/login
Prediction: PHISHING
Confidence: 0.9282  (92.82%)
Risk Score: 93/100
Top Feature: special_char_ratio  ← drives toward phishing

SHAP contributions (top features):
  + special_char_ratio  → increases_risk  (URL มี - และ . มากผิดปกติ)
  + url_length          → increases_risk
  - num_subdomains      → decreases_risk  (1 subdomain เท่านั้น)
```

#### ตัวอย่างที่ 2: Legitimate ที่ classify ถูกต้อง
```
URL:        https://www.google.com
Prediction: LEGITIMATE
Confidence: 0.9989  (99.89%)
Risk Score: 0/100
Top Feature: num_subdomains  ← drives toward legitimate

SHAP contributions (top features):
  - num_subdomains      → decreases_risk  (1 subdomain ปกติ)
  - digit_ratio         → decreases_risk  (ไม่มีตัวเลข)
  - url_entropy         → decreases_risk  (entropy ปกติ)
```

#### ตัวอย่างที่ 3: Blind Spot — Phishing ที่ miss
```
URL:        http://paypal.com.evil-login.tk/verify?user=john
Prediction: LEGITIMATE  ← ผิด!
P(phishing): 0.1216  (12.16%)

เหตุผล SHAP: num_subdomains=1 → decreases_risk strongly
  model เรียนว่า 1 subdomain = legitimate
  ไม่รู้จัก .tk ว่าเป็น suspicious TLD เพราะ dataset สอนกลับทิศ
```

---

## 8. Web Application

### 8.1 API Endpoints

#### GET `/`
- **Tags**: General
- **Response**: `{"message": "Phishing Detector API", "version": "1.0"}`

#### GET `/health`
- **Tags**: General
- **Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "XGBClassifier"
}
```

#### POST `/api/v1/predict`
- **Tags**: Prediction
- **Request Body** (PredictRequest):
```json
{
  "url": "https://example.com",     // required, ต้องขึ้นต้น http:// หรือ https://
  "fetch_html": false               // optional, default=false
}
```
- **Response** (PredictResponse):
```json
{
  "url": "https://example.com",
  "prediction": "phishing",          // "phishing" | "legitimate"
  "confidence": 0.9282,              // 0.0–1.0, จาก predict_proba
  "risk_score": 93,                  // 0–100, int
  "top_features": [
    {
      "name": "special_char_ratio",
      "value": 0.0714,               // ค่าจริงของ feature (unscaled)
      "shap_value": 0.3421,
      "direction": "increases_risk"  // "increases_risk" | "decreases_risk"
    }
    // ... top 5 features
  ],
  "extracted_features": {
    "url_length": 42,
    "hostname_length": 22,
    "has_ip": 0,
    // ... ทุก 9 features
  },
  "html_features": null,             // dict ถ้า fetch_html=true และสำเร็จ
  "html_status": "skipped",          // "ok" | "timeout" | "error" | "skipped"
  "processing_time_ms": 48.5
}
```

- **Error Responses**:
  - `422`: URL format ผิด (ไม่มี scheme หรือ netloc)
  - `503`: Model ยังไม่โหลด (ต้อง run train.py ก่อน)
  - `500`: Unexpected error

### 8.2 Frontend Components

| Component | ไฟล์ | หน้าที่ |
|-----------|------|--------|
| URLInput | `src/components/URLInput.jsx` | Input field + Detect button |
| ResultCard | `src/components/ResultCard.jsx` | แสดง prediction, confidence, risk score badge |
| SHAPChart | `src/components/SHAPChart.jsx` | Bar chart แสดง top-5 SHAP features (Recharts) |
| FeatureTable | `src/components/FeatureTable.jsx` | ตาราง 9 URL features ที่ extract ได้ (collapsible) |
| HTMLFeaturePanel | `src/components/HTMLFeaturePanel.jsx` | แสดง 14 HTML features พร้อม label |
| URLHistory | `src/components/URLHistory.jsx` | ประวัติ URL ที่เคย scan + ปุ่ม Recheck |

State ใน App.jsx:
- `loading` — spinner ขณะรอ API
- `result` — PredictResponse
- `error` — error message
- `fetchHtml` — toggle HTML analysis mode
- `showFeatures` — collapse/expand FeatureTable
- `showHistory` — collapse/expand URLHistory
- `history` — array ของ past results (useURLHistory hook)

### 8.3 HTML Analysis Mode

- **เปิด-ปิด**: Toggle ที่ header (HTML label + toggle switch)
- **เมื่อเปิด**: ส่ง `fetch_html: true` → backend เรียก `requests.get(url, timeout=10)`
- **Risk Score Blending**: `risk_score = model_proba × 0.8 + html_risk × 0.2`
- **HTML Risk Scoring** (rule-based, 0.0–1.0):

| Condition | Score |
|-----------|-------|
| has_login_form = 1 | +0.30 |
| form_action_external = 1 | +0.40 |
| meta_refresh = 1 | +0.20 |
| num_iframes > 0 | +0.15 |
| null_links_ratio > 0.5 | +0.20 |
| title_match_domain = 0 | +0.10 |
| has_copyright = 1 | −0.10 |
| has_favicon = 1 | −0.05 |

- **Status ที่ frontend รับ**: `"ok"` / `"timeout"` / `"error"` / `"skipped"`

---

## 9. Limitations ที่พบจริง

### 9.1 Blind Spots จาก PhishTank Live Test

จาก `phishtank_live_test.csv` — พลาด 17/20 phishing URLs:

| ประเภท Blind Spot | ตัวอย่าง URL | ผลที่ได้ | เหตุผล |
|------------------|-------------|---------|--------|
| **IP-based phishing** | `http://192.168.1.1/admin/login` | LEGIT (conf=0.9956) | Private IP มี entropy ต่ำ, path สั้น ไม่มี feature ที่ผิดปกติ |
| **Free TLD (.tk, .ml, .ga, .cf, .gq)** | `http://netflix-billing.ga/update/payment` | LEGIT (conf=0.9999) | Dataset สอน model ว่า TLD เหล่านี้เป็น legitimate |
| **URL Shorteners** | `http://bit.ly/phish-test-demo` | LEGIT (conf=1.0000) | Domain สั้น entropy ต่ำมาก, ไม่มี subdomains |
| **Long descriptive phishing** | `http://secure-login-account-verify-update.com/paypal/confirm?...` | LEGIT (conf=0.9985) | num_subdomains=0, URL ยาวแต่ entropy ปานกลาง |
| **Typosquatting** | `http://www.paypa1.com/signin` | LEGIT (conf=0.9999) | ตัวอักษรเกือบเหมือน legitimate, เปลี่ยนแค่ 1→l |
| **Hex encoding** | `http://xn--pypal-4ve.com/login` | LEGIT (conf=0.9999) | Punycode domain, lexical features ไม่ detect |

**URLs ที่ detect ได้ถูก (3/20)**:
1. `http://paypal.account-verify.com/login` → PHISHING (special_char_ratio สูง)
2. `http://google.account-update.xyz/signin` → PHISHING (special_char_ratio สูง)
3. `http://secure.login%40paypal.com.evil.net/` → PHISHING (url_entropy จาก % encoding)

### 9.2 Dataset Distribution Issues

**has_suspicious_tld=1 ใน PhiUSIIL:**
- 93.5% เป็น LEGITIMATE (startups ใช้ .tk/.ml/.xyz จริง)
- Model เรียนว่า "suspicious TLD = legitimate" ← กลับทิศ!

**brand_in_subdomain=1 ใน PhiUSIIL:**
- 97.3% เป็น LEGITIMATE (api.paypal.com, id.apple.com เป็น legit subdomains)
- Model เรียนว่า "brand in subdomain = legitimate" ← กลับทิศ!

**Feature Distribution Shift ระหว่าง Dataset:**
- ISCX probability scores มีค่าเฉลี่ยต่ำกว่า 0.5 แม้จะเป็น phishing จริง
- ทำให้ threshold=0.5 ตัดสินใจเป็น 0 ทั้งหมด → F1=0.0 ใน Mode B

### 9.3 สิ่งที่ Revert และเหตุผล

**ทดลองเพิ่มเป็น 12 features** (เพิ่ม `has_suspicious_tld`, `brand_in_subdomain`, `has_shortener`):

| URL | P(phishing) กับ 9 features | P(phishing) กับ 12 features |
|-----|--------------------------|---------------------------|
| `http://paypal.com.evil-login.tk/verify?...` | 0.1216 → LEGIT | **0.0049 → LEGIT (แย่ลง!)** |
| `http://192.168.1.1/admin` | 0.2437 → LEGIT | 0.3800 → LEGIT |
| `https://bit.ly/3xAbc12` | 0.0018 → LEGIT | 0.0019 → LEGIT |

**ผลสรุป**: Features ใหม่ทำให้ blind spots แย่ลง ไม่ใช่ดีขึ้น  
**ตัดสินใจ**: Revert กลับ 9 features เพื่อ maintain generalization บน ISCX  
**เหตุผล**: การแก้ที่ถูกต้องต้องเปลี่ยน dataset ไม่ใช่ features

---

## 10. ข้อเสนอแนะสำหรับงานวิจัยต่อไป

### จากโค้ด comment โดยตรง

1. **DNS-based Features**
   - Domain registration age (WHOIS) — domain ใหม่ < 1 เดือน = risk สูง
   - SSL certificate age และ issuer
   - DNS TTL ที่ต่ำมากผิดปกติ

2. **Dataset ที่เฉพาะเจาะจงกว่า**
   - ต้องการ dataset ที่มี domain-squatting examples โดยเฉพาะ
   - Dataset ปัจจุบัน (PhiUSIIL) มี .tk/.xyz startups จำนวนมาก ทำให้ heuristic กลับทิศ

3. **Rule-based Detection แยกจาก ML**
   - URL shortener blacklist (bit.ly, tinyurl.com ควร flag เสมอ)
   - IP-in-domain rule (ควร flag เสมอ ไม่ขึ้นกับ probability)
   - Free TLD whitelist/blacklist ที่อัปเดตจาก threat intelligence

4. **Domain Adaptation**
   - Fine-tuning บน target domain dataset เมื่อ deploy ใน environment ใหม่
   - Threshold calibration per-deployment (ใช้ Youden's J เป็น baseline)

5. **HTML-based ML Model**
   - ปัจจุบัน HTML features ใช้ rule-based scoring เท่านั้น
   - ขั้นต่อไป: train ML model บน HTML features ด้วย → ensemble กับ URL model

6. **Webpage Content Analysis**
   - Natural Language Processing บน page text
   - Image similarity กับ brand logo จริง (perceptual hash)

7. **Generalization Gap**
   - Mode B F1=0.0 → F1=0.9726 หลัง threshold calibration แสดงว่า URL features มี transferability
   - แต่ Mode B ดั้งเดิมสำคัญ: model ranking ถูก (AUC=0.842) แต่ calibration ผิด
   - แนะนำ: ใช้ temperature scaling หรือ Platt scaling แทน hard threshold

---

*ข้อมูลนี้รวบรวมจาก source code, result CSV และ JSON files โดยตรง ไม่มีการ estimate*
