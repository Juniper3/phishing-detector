"""
feature_config.py
Shared feature configuration — ใช้ร่วมกันระหว่าง train.py และ cross_dataset_eval.py

นิยาม FEATURE_MAP ที่ map canonical feature names (url_extractor.py)
ไปยัง column names ของแต่ละ dataset เพื่อป้องกัน divergence เมื่อเพิ่ม/ลด features
"""

from collections import OrderedDict

# ─────────────────────────────────────────────
# URL-Only Feature Mapping
#
# เลือกเฉพาะ features ที่มีใน dataset ทั้งสอง
# canonical name (url_extractor.py) → (PhiUSIIL column, ISCX column)
#
# หมายเหตุ:
#   num_subdomains: PhiUSIIL=NoOfSubDomain (actual subdomain count)
#                   ISCX=domain_token_count (dots+1, ค่าสูงกว่า ~2)
#                   model จะ learn relationship นี้เองจากทั้งสอง dataset
#   num_equal:      PhiUSIIL=NoOfEqualsInURL (count of '=')
#                   ISCX=URLQueries_variable (count of query vars)
#                   สำหรับ URL ทั่วไป: num_equal ≈ URLQueries_variable
# ─────────────────────────────────────────────

FEATURE_MAP: OrderedDict = OrderedDict([
    ("url_length",         ("URLLength",            "urlLen")),
    ("hostname_length",    ("DomainLength",          "domainlength")),
    ("has_ip",             ("IsDomainIP",            "ISIpAddressInDomainName")),
    ("num_digits",         ("NoOfDegitsInURL",       "URL_DigitCount")),
    ("digit_ratio",        ("DegitRatioInURL",       "NumberRate_URL")),
    ("special_char_ratio", ("SpacialCharRatioInURL", "spcharUrl")),
    ("url_entropy",        ("URLCharProb",           "Entropy_URL")),
    ("num_subdomains",     ("NoOfSubDomain",         "domain_token_count")),
    ("num_equal",          ("NoOfEqualsInURL",       "URLQueries_variable")),
])

# Canonical feature names ตามลำดับที่ model ใช้ train
CANONICAL_FEATURES: list[str] = list(FEATURE_MAP.keys())

# Label column candidates ใน PhiUSIIL dataset (ชื่อ column ไม่สม่ำเสมอ)
PHI_LABEL_CANDIDATES: tuple[str, ...] = (
    "label", "phishing", "class", "Label", "Phishing", "Class"
)
