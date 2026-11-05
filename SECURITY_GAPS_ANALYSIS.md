# SECURITY_GAPS_ANALYSIS — การวิเคราะห์ช่องว่างด้านความปลอดภัย
**ประเภท:** เอกสารภายใน — เตรียมก่อน Partner รายแรก
**วันที่:** พฤศจิกายน 2026
**สถานะ:** Draft สำหรับทีมวิศวกรรม

---

## 1. สรุปสถานะปัจจุบัน

> **อัปเดต: พฤศจิกายน 2026**

| ด้าน | สถานะ | หมายเหตุ |
|---|---|---|
| Hash customer_id | ✅ implement แล้ว — src/data_security.py | `hash_customer_id`, `hash_dataframe_ids` พร้อม HMAC-SHA256 + per-store salt |
| ตัดคอลัมน์ที่ไม่ได้ขอ | ✅ implement แล้ว — src/data_security.py | `filter_columns()` + `ALLOWED_COLUMNS` allowlist; hook เข้า ingest.py แล้ว |
| DPA / Terms | ยังไม่มี — ต้องผ่านทนายความก่อน | ต้องทำก่อน partner รายแรก |
| Tenant isolation | partially — path structure documented, enforce at API layer | ตอนนี้ข้อมูลทุกร้านอยู่รวมกัน |
| Pooled training consent | config flag ใน DPA — ต้องเพิ่มใน business_config | Champion v7 pool dessert + Dunnhumby |
| API key management | ✅ implement แล้ว — APIKeyManager ใน src/data_security.py | per-partner scoped keys, file-based registry, revocation support |
| TLS (in transit) | ยังไม่ทราบ — ขึ้นอยู่กับ deploy | ขึ้นอยู่กับ deploy environment |
| Encryption at rest | ยังไม่ทราบ — ขึ้นอยู่กับ storage | ขึ้นอยู่กับ storage ที่ใช้ |
| Audit log | ✅ implement แล้ว — AuditLogger ใน src/data_security.py | append-only NDJSON, retention 2 ปี, thread-safe |
| Data retention / auto-delete | ✅ implement แล้ว — src/data_retention.py | `RetentionCleaner` พร้อม dry_run mode และ tier-based retention |
| Right to erasure | ✅ implement แล้ว — src/data_retention.py | `ErasureProcessor` ลบ CSV records + enqueue สำหรับ retrain exclusion |
| Breach notification process | ✅ implement แล้ว — docs/BREACH_RUNBOOK.md | timeline 72h, contact checklist, template email Thai + PDPC |
| LLM data boundary | policy documented — enforce ตอนเปลี่ยน LLM | ถ้าเปลี่ยน local llama → cloud ต้องไม่ส่ง row-level data |

---

## 2. สิ่งที่ต้องทำก่อน Partner รายแรก (Priority 1)

### 2.1 Hash customer_id ที่ฝั่ง pipeline

**ทำอะไร:** Pipeline ต้องรับ hashed id เท่านั้น หรือ hash เองทันทีตอนรับข้อมูลก่อนเข้า processing ใดๆ

**ทำไม:** ถ้า partner ส่ง raw customer_id (เช่น phone number หรือ email) และระบบเก็บค้างในไฟล์ระหว่าง processing — กลายเป็นเราเก็บ PII โดยไม่จำเป็น ซึ่งขัดหลัก PDPA (data minimization)

**วิธีทำ:**
```python
import hashlib

def hash_customer_id(raw_id: str, salt: str) -> str:
    """
    salt ควรเป็น per-store secret ที่เก็บใน env var
    ไม่ใช่ hardcode — เพื่อให้ร้านต่าง salt กัน
    """
    return hashlib.sha256(f"{salt}:{raw_id}".encode()).hexdigest()
```

- เพิ่ม hashing step เป็นขั้นตอนแรกสุดใน `csv_pipeline_engine.py` ก่อน feature extraction
- บันทึก mapping (raw → hashed) ไว้ที่ฝั่ง partner เท่านั้น เราไม่เก็บ
- ถ้า partner ส่งมา hashed แล้ว — ระบบตรวจว่า format ถูกต้อง (SHA-256 hex 64 chars) และใช้ค่านั้นโดยตรง

---

### 2.2 Column filtering — ตัดคอลัมน์ที่ไม่อยู่ใน allowlist

**ทำอะไร:** Reject หรือ drop ทุกคอลัมน์ที่ไม่อยู่ใน allowlist ทันทีตอน ingestion — ก่อนบันทึกหรือ process ใดๆ

**ทำไม:** ถ้า partner ส่ง CSV ที่มีชื่อ, เบอร์โทร, หรือ email ติดมา — ถ้าไม่ตัดทิ้ง ระบบจะเก็บและ log ข้อมูล PII โดยไม่ได้ตั้งใจ

**วิธีทำ:**
```python
ALLOWED_COLUMNS = {
    "customer_id", "timestamp", "total_price",
    "product_name", "category",
    "is_treated", "post_promo_visit"  # optional
}

def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    extra = set(df.columns) - ALLOWED_COLUMNS
    if extra:
        logger.warning(f"Dropping non-allowlisted columns: {extra}")
    return df[[col for col in df.columns if col in ALLOWED_COLUMNS]]
```

- เพิ่ม `filter_columns()` ก่อน step แรกของ pipeline
- Log ว่าตัดคอลัมน์อะไรออกบ้าง (ไม่ log ค่า — log แค่ชื่อคอลัมน์)

---

### 2.3 DPA Template

**ทำอะไร:** ร่าง Data Processing Agreement ระหว่าง Partner (Data Processor) กับเรา (Sub-processor) ให้พร้อมก่อน onboard partner รายแรก

**ทำไม:** PDPA กำหนดให้มีสัญญาในลักษณะนี้ระหว่าง processor และ sub-processor เป็นข้อบังคับ ถ้าไม่มี — ทั้งสองฝ่ายรับ liability ร่วมกัน

**วิธีทำ:**
- ระบุ scope ของข้อมูลที่เราประมวลผล (6 คอลัมน์เท่านั้น)
- ระบุ purpose (campaign targeting เท่านั้น)
- ระบุ data retention period
- ระบุ sub-processor chain (เรา → cloud provider ถ้าใช้)
- ระบุ right to audit
- ระบุ breach notification obligation (72 ชั่วโมง)
- แนะนำ: ให้ทนายความที่เชี่ยวชาญ PDPA ทบทวนก่อน sign

---

### 2.4 API Key System — Scoped Per Partner

**ทำอะไร:** เปลี่ยนจาก single `CRM_API_KEY` ใน env var เป็น key management system ที่รองรับหลาย partner

**ทำไม:** ตอนนี้ถ้า key หลุด — partner ทุกรายได้รับผลกระทบ ไม่มีทางทราบว่าใครเรียก API บ้าง และไม่มีทาง revoke เฉพาะรายได้

**วิธีทำ:**
```python
# schema: api_keys table (หรือ YAML config ถ้ายังไม่มี DB)
{
  "key_id": "pk_live_xxxx",         # prefix บอก scope
  "key_hash": "sha256_of_key",      # เก็บแค่ hash — ไม่เก็บ plaintext
  "partner_id": "partner_A",
  "store_scope": ["store_01"],      # จำกัดว่าเข้าถึงข้อมูลร้านไหนได้
  "created_at": "2026-11-01",
  "revoked_at": null
}
```

- Generate key ครั้งเดียว — แสดงให้ partner copy แล้วเก็บแค่ hash
- Rotate: สร้าง key ใหม่ → revoke key เก่า (overlap 24h)
- ตรวจ `store_scope` ทุกครั้งที่ call endpoint

---

### 2.5 Rate Limiting — ป้องกัน Abuse

**ทำอะไร:** จำกัด request rate ต่อ API key

**ทำไม:** ถ้าไม่มี rate limit — partner (หรือ attacker ที่ได้ key) สามารถ enumerate ข้อมูลได้ไม่จำกัด หรือ DoS ระบบได้

**วิธีทำ:**
```python
# ตัวอย่างด้วย slowapi (FastAPI middleware)
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_api_key)  # key per API key ไม่ใช่ per IP

@app.get("/v1/plans/{month}/events/{event_id}/customers")
@limiter.limit("100/minute")
async def get_customers(...):
    ...
```

- Endpoint ที่ return รายชื่อลูกค้า (`/customers`) — เข้มกว่า endpoint อื่น
- Return `429 Too Many Requests` พร้อม `Retry-After` header

---

## 3. สิ่งที่ต้องทำก่อนใช้ข้อมูลข้ามร้าน (Priority 2)

### 3.1 Tenant Isolation — แยกข้อมูลต่อร้าน

**ทำอะไร:** แยก data storage ต่อ store_id อย่างชัดเจน ทั้ง raw data, processed data, และ model features

**ทำไม:** ตอนนี้ข้อมูลทุกร้านอยู่รวมกันใน `data/processed/` — ถ้า partner A เข้าถึง API ด้วย key ของตัวเอง อาจได้เห็นข้อมูล partner B โดยไม่ตั้งใจ

**วิธีทำ:**
- เปลี่ยน path structure เป็น `data/processed/{store_id}/`
- Middleware ตรวจ `store_scope` ของ API key ทุก request
- Output payload (`broadcast_list.csv`) แยกต่อร้านเสมอ

---

### 3.2 Pooled Training Consent — Explicit Opt-In

**ทำอะไร:** เพิ่มมาตราใน DPA ที่ให้ partner opt-in อย่างชัดเจนว่ายินยอมให้ใช้ข้อมูลในการ train โมเดลรวม (pooled training)

**ทำไม:** Champion v7 ถูก train บนข้อมูลจาก dessert store และ Dunnhumby รวมกัน หากไม่มี consent ที่ชัดเจน — การ train ลักษณะนี้อาจขัด PDPA ในส่วนของ purpose limitation

**วิธีทำ:**
- เพิ่มใน DPA: `consent_pooled_training: true/false` ต่อ partner
- ถ้า `false` → train โมเดลแยกต่อร้าน (store-specific model) เท่านั้น
- เพิ่ม opt-out mechanism: partner แจ้งได้ว่าต้องการถอน consent → ต้อง retrain โมเดลโดยไม่รวมข้อมูลนั้น

---

### 3.3 Output Firewall — ผลลัพธ์ต้องใช้ข้อมูลของร้านตัวเองเท่านั้น

**ทำอะไร:** ตรวจสอบว่า output payload ของร้าน A ถูก generate จาก inference บนข้อมูล store A เท่านั้น — แม้ว่าโมเดลจะ train จากหลายร้าน

**ทำไม:** ถ้า inference path ผิด — ร้าน A อาจได้รับ recommendation ที่ถูก influence โดยข้อมูลของร้าน B โดยที่ไม่มี consent

**วิธีทำ:**
- เพิ่ม `store_id` check ใน `csv_pipeline_engine.py` ตอน produce output
- Log `store_id` ที่ใช้ในทุก inference run
- Unit test: ให้ร้าน A และ B ส่งข้อมูลที่ disjoint กัน → verify ว่า output ไม่ cross

---

## 4. สิ่งที่ต้องทำก่อน Scale (Priority 3)

### 4.1 Auto-Delete — Retention Policy

**ทำอะไร:** กำหนด data retention period และสร้าง scheduled job ที่ลบข้อมูลหลังครบกำหนด

**ทำไม:** PDPA กำหนดว่าต้องไม่เก็บข้อมูลนานกว่าที่จำเป็น — ไม่มี policy = เก็บไปเรื่อยๆ

**วิธีทำ:**
- กำหนด retention: raw POS data → 90 วัน, processed output → 12 เดือน
- สร้าง cron job หรือ scheduled task ที่ลบไฟล์เก่า
- Log การลบพร้อม timestamp (ไม่ log เนื้อหา)

---

### 4.2 Audit Logging — บันทึกการเข้าถึง

**ทำอะไร:** Log ทุก significant event: upload, API call, model training

**ทำไม:** ถ้าเกิด incident — ไม่มีทางสอบสวนได้ว่าใครทำอะไร เมื่อไหร่ กับข้อมูลอะไร

**วิธีทำ:**
```python
# Log schema (ไม่ log ค่าข้อมูล — log แค่ metadata)
{
  "event": "api_call",
  "timestamp": "2026-11-04T10:30:00Z",
  "partner_id": "partner_A",
  "store_id": "store_01",
  "endpoint": "/v1/plans/2026-11/events",
  "ip": "x.x.x.x",
  "status_code": 200,
  "records_returned": 42
}
```

- เก็บ audit log แยกจาก application log
- Immutable: append-only (ไม่แก้ไขย้อนหลัง)
- Retention ของ audit log ≥ 2 ปี

---

### 4.3 Right to Erasure — กระบวนการลบข้อมูลลูกค้า

**ทำอะไร:** สร้างกระบวนการที่ partner สามารถ request ลบข้อมูลลูกค้าคนใดคนหนึ่งออกจากระบบได้

**ทำไม:** PDPA ให้สิทธิ์ data subject ในการขอลบข้อมูล (right to erasure) — ถ้าไม่มีกระบวนการ ไม่สามารถ comply ได้

**วิธีทำ:**
- API endpoint: `DELETE /v1/customers/{hashed_id}` (ต้องการ elevated permission)
- ลบจาก: raw data, processed features, output payload, training dataset
- ถ้าลบจาก training data → flag ว่าโมเดลถัดไปต้อง retrain โดยไม่รวม record นั้น
- ส่ง confirmation กลับพร้อม timestamp

---

### 4.4 Breach Notification Runbook

**ทำอะไร:** สร้าง runbook ที่ระบุขั้นตอนชัดเจนเมื่อเกิด data breach

**ทำไม:** PDPA กำหนดว่าต้องแจ้ง PDPC ภายใน **72 ชั่วโมง** หลังรับทราบ breach และแจ้ง data subject โดยไม่ชักช้า — ถ้าไม่มี runbook มีโอกาสพลาด deadline สูง

**ขั้นตอนใน Runbook (ร่าง):**

| ชั่วโมง | ขั้นตอน |
|---|---|
| 0 | ระบุขอบเขต breach — ข้อมูลอะไร กี่ record กี่ partner |
| 1 | Contain — revoke key ที่ถูก compromise, isolate affected store |
| 4 | ประเมิน severity — มี PII จริงหรือไม่ (ถ้า hash ทั้งหมด ความเสี่ยงลดลงมาก) |
| 24 | แจ้ง partner ที่ได้รับผลกระทบ |
| 72 | แจ้ง PDPC (ถ้า breach มีความเสี่ยงสูงต่อ data subject) |
| หลัง 72h | แจ้ง data subject (ถ้า PDPC กำหนด) |

---

### 4.5 LLM Data Boundary Enforcement

**ทำอะไร:** ถ้ามีการใช้ Cloud LLM (เช่นเปลี่ยนจาก local llama) — กำหนด policy ชัดเจนว่าส่งได้แค่ aggregate สถิติ ไม่ใช่ row-level data

**ทำไม:** Row-level transaction data ที่ส่งไป Cloud LLM อาจถูก log หรือใช้ train โมเดลของ provider — ซึ่งเป็นการ transfer ข้อมูลออกนอก processing boundary โดยไม่ได้รับ consent

**วิธีทำ:**
- Policy: LLM calls รับได้แค่ aggregated features เช่น `avg_spent`, `segment_name` — ไม่ใช่ raw transactions
- ถ้าใช้ LLM generate message → input = `{segment, action_type, reason}` เท่านั้น
- ถ้าเปลี่ยน LLM provider → ทบทวน Data Processing Agreement กับ provider ใหม่

---

## 5. ความเสี่ยงที่ยังเหลืออยู่หลัง Implement ทั้งหมด (Residual Risks)

แม้จะ implement ทุกรายการข้างต้นแล้ว ยังมีความเสี่ยงที่ลดได้แต่กำจัดไม่ได้ 100%:

### 5.1 Inference Attack (Model Inversion)
โมเดลที่เทรนจากหลายร้าน อาจ leak ข้อมูลบางส่วนผ่าน model weights ได้ในบางกรณี — เช่น attacker ที่ query โมเดลหลายครั้งอาจ infer ข้อมูลของร้านที่ตนไม่มีสิทธิ์เข้าถึง

**ระยะยาว:** Federated learning แก้ปัญหานี้ได้ (train โมเดลแบบกระจาย ไม่ส่ง raw data มารวมกัน) แต่มี engineering cost สูง

### 5.2 Re-identification
Hashed `customer_id` + purchase pattern อาจถูก re-identify ได้ถ้า attacker มีข้อมูลเสริม (เช่น รู้ว่า customer X ซื้ออะไรในช่วงเวลานั้น) — hash อย่างเดียวไม่ได้ป้องกันได้ทุกกรณี

**แนวทาง:** เพิ่ม per-store salt ใน hash (ตาม 2.1) ลดความเสี่ยงได้มาก

### 5.3 Proxy Label Quality Risk
ถ้าไม่มี `is_treated` / `post_promo_visit` — โมเดลใช้ proxy label ซึ่งอาจแม่นน้อยกว่า ส่งผลให้ส่งโปรผิดคน ซึ่งเป็น **ความเสี่ยงทางธุรกิจ ไม่ใช่ security** — แต่ต้องแจ้ง partner ให้ชัดเจน

---

## 6. สรุป Priority Matrix

| Priority | Action | Timeline | Owner |
|---|---|---|---|
| 1 | Hash customer_id ที่ pipeline | ก่อน partner รายแรก | |
| 1 | Column filtering (allowlist) | ก่อน partner รายแรก | |
| 1 | DPA Template | ก่อน partner รายแรก | |
| 1 | API key system (per-partner, scoped) | ก่อน partner รายแรก | |
| 1 | Rate limiting | ก่อน partner รายแรก | |
| 2 | Tenant isolation (data separation) | ก่อนใช้ข้อมูลข้ามร้าน | |
| 2 | Pooled training consent (DPA clause) | ก่อนใช้ข้อมูลข้ามร้าน | |
| 2 | Output firewall (inference store check) | ก่อนใช้ข้อมูลข้ามร้าน | |
| 3 | Auto-delete (retention policy) | ก่อน scale | |
| 3 | Audit logging | ก่อน scale | |
| 3 | Right to erasure (delete API) | ก่อน scale | |
| 3 | Breach notification runbook | ก่อน scale | |
| 3 | LLM boundary enforcement | ถ้าเปลี่ยนเป็น cloud LLM | |

---

*เอกสารภายใน — ไม่สำหรับเผยแพร่*
*ปรับปรุงล่าสุด: พฤศจิกายน 2026*
