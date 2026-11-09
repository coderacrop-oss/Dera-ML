# Codera Platform — Screen Spec
> สำหรับโชแปง (Repo 2 — Codera Platform)
> อัปเดต: พฤศจิกายน 2026

---

## ข้อตกลงเบื้องต้น

- Codera Platform คือ **web app standalone** — ไม่ได้ embed ใน DotDash
- **Multi-tenant**: แต่ละร้านมี `store_id` ของตัวเอง — ร้านเห็นเฉพาะข้อมูลของตัวเอง
- DotDash เป็นแค่ 1 client ที่ใช้งาน — ไม่ใช่ platform ทั้งหมด
- ทุก API call ต้องส่ง header `X-API-Key` — โชแปงจัดการ key management อยู่แล้ว
- `store_id` ดึงจาก session หลัง login — ไม่มี screen ไหนให้ user กรอก `store_id` เอง
- **Model Tier**: ร้านใหม่ใช้ Category Model (Global → F&B/Retail/Beauty/Services) → upgrade เป็น Store Model อัตโนมัติเมื่อมี data ≥ 12 เดือน
- POS CSV columns ขั้นต่ำ: `customer_id`, `transaction_id`, `timestamp`, `total_price`, `product_name`, `category`

---

## Overview — Navigation Structure

```
Codera Platform
    ├── GROUP 1: Onboarding (เฉพาะร้านใหม่ / setup ครั้งแรก)
    │       ├── 1.1 Sign Up / Login
    │       ├── 1.2 Store Profile + Brand Kit
    │       ├── 1.3 Data Connection
    │       └── 1.4 Ready Screen
    │
    ├── GROUP 2: Dashboard (entry point หลัง onboard แล้ว)
    │       ├── 2.1 Segment Health Dashboard       ← หน้าหลัก
    │       └── 2.2 Notification Center
    │
    ├── GROUP 3: Campaign Planning
    │       ├── 3.1 New Campaign
    │       ├── 3.2 Goal Comparison
    │       ├── 3.3 Plan Results + Manual Advisor
    │       └── 3.4 Campaign Calendar
    │
    ├── GROUP 4: Content Studio
    │       ├── 4.1 Content Brief
    │       ├── 4.2 Creative Preview
    │       └── 4.3 Approve / Request Regenerate
    │
    ├── GROUP 5: Execution Center
    │       ├── 5.1 Choose Channel
    │       ├── 5.2 Meta Connect
    │       ├── 5.3 LINE OA Connect
    │       └── 5.4 Launch & Track
    │
    ├── GROUP 6: Analytics
    │       ├── 6.1 Campaign History
    │       ├── 6.2 Ad Performance
    │       ├── 6.3 Outcome Tracking
    │       └── 6.4 Model Status + Retrain
    │
    └── GROUP 7: Settings
            ├── 7.1 Brand Kit edit
            ├── 7.2 Data Completeness Tracker
            └── 7.3 Connections
```

---

## GROUP 1 — Onboarding

---

### Screen 1.1: Sign Up / Login

#### วัตถุประสงค์
จุดเข้าของ Platform — ให้ร้านสมัครใช้งานหรือ login เข้าสู่ระบบ รองรับ multi-tenant (ร้านแต่ละเจ้ามี account แยกกัน)

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| ไม่มี API Codera ML — ใช้ auth ของ Platform เอง | — | — |

> Auth flow จัดการภายใน Repo 2 — ไม่ผ่าน ML engine endpoints ทั้ง 9 ตัว

#### Input / User Actions

**Tab: สมัครใหม่**

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| ชื่อร้าน | text | ✅ | ใช้เป็น `store_name` เบื้องต้น |
| อีเมล | email | ✅ | ใช้ login |
| รหัสผ่าน | password | ✅ | min 8 ตัว |
| ยืนยันรหัสผ่าน | password | ✅ | ต้องตรงกัน |
| ประเภทธุรกิจ | dropdown | ✅ | F&B / Retail / Beauty & Health / Services (ใช้กำหนด Category Model tier) |

**Tab: Login**

| Field | ประเภท | Required |
|-------|--------|----------|
| อีเมล | email | ✅ |
| รหัสผ่าน | password | ✅ |
| "จำฉันไว้" | checkbox | ❌ |

#### แสดงผล
- หลัง login สำเร็จ: redirect ไป Screen 2.1 (Segment Health) ถ้าเคย onboard แล้ว
- ถ้า account ใหม่ที่ยังไม่มี data: redirect ไป Screen 1.2

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| อีเมลซ้ำในระบบ | "อีเมลนี้มีบัญชีอยู่แล้ว — เข้าสู่ระบบแทน?" |
| รหัสผ่านไม่ตรงกัน | inline error ใต้ช่อง "รหัสผ่านไม่ตรงกัน" |
| login ผิด | "อีเมลหรือรหัสผ่านไม่ถูกต้อง" ไม่บอกว่าอันไหนผิด (security) |
| session หมดอายุ | redirect กลับ Login พร้อม toast: "Session หมดอายุ — กรุณา login ใหม่" |

---

### Screen 1.2: Store Profile + Brand Kit Setup

#### วัตถุประสงค์
เก็บข้อมูลร้านและ Brand Kit ที่ใช้ gen content ทุก campaign — ทำครั้งแรกตอน onboard แล้วแก้ได้ภายหลังใน Settings (Screen 7.1)

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| ไม่มี ML endpoint — บันทึกลง Platform database เอง | — | — |

#### Input / User Actions

**Section A — ข้อมูลร้าน**

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| ชื่อร้าน (`brand_name`) | text | ✅ | แสดงใน content ที่ gen |
| ประเภทธุรกิจ | dropdown | ✅ | F&B / Retail / Beauty & Health / Services — กำหนด Category Model tier |
| จังหวัด / พื้นที่ | text | ❌ | บริบทสำหรับ content gen |
| URL โลโก้ | image upload | ❌ | แสดงใน Platform, อ้างอิงใน content |

**Section B — Brand Voice**

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| โทนเสียง (`tone`) | dropdown | ✅ | friendly / professional / playful / luxurious — ถ้าไม่กรอก ใช้ category default |
| คำที่ห้ามใช้ (`forbidden_words`) | tag input | ❌ | เช่น "ถูก", "ลดราคา" — กด Enter เพิ่มแต่ละคำ |
| คำที่ต้องการใช้ (`preferred_words`) | tag input | ❌ | เช่น "สมาชิก", "พิเศษสำหรับคุณ" |

**Section C — Visual Direction**

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| สไตล์ภาพ (`photo_style`) | dropdown | ❌ | minimal / lifestyle / product_focused / warm_tones / dark_moody |
| layout preference (`layout_preference`) | dropdown | ❌ | text_heavy / image_heavy / balanced |
| ตัวอย่าง creative ที่ชอบ (`reference_examples`) | image upload (max 3) | ❌ | ใช้เป็น reference ใน image direction — ไม่ใช่ input ให้ image API โดยตรง |

#### แสดงผล
- Progress stepper: **1. Store Profile** → 2. Data Connection → 3. พร้อมใช้งาน
- Preview panel ด้านขวา: แสดงตัวอย่าง content tone แบบ real-time เมื่อ user กรอก `brand_name` + `tone`

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| ไม่กรอก Brand Kit เลย (กด Skip) | ใช้ category default tone — แสดง notice: "เราจะใช้ tone มาตรฐานของ [ประเภทธุรกิจ] ก่อน แก้ได้ภายหลังใน Settings" |
| `reference_examples` ไฟล์ใหญ่เกิน 5MB | "ไฟล์ใหญ่เกิน 5MB — กรุณาบีบอัดก่อน" |
| `forbidden_words` ซ้ำกับ `preferred_words` | inline warning: "คำนี้มีอยู่ทั้งฝั่ง forbidden และ preferred — กรุณาลบออกฝั่งหนึ่ง" |

---

### Screen 1.3: Data Connection

#### วัตถุประสงค์
ให้ร้านเชื่อม data source เพื่อให้ ML engine วิเคราะห์ได้ — MVP รองรับ MeriPOS connector และ CSV upload เท่านั้น

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| ไม่มี ML endpoint — เป็น Platform data ingestion เอง | — | — |
| `GET` | `/segment/health` | หลัง ingest สำเร็จ เพื่อตรวจว่า ML รับ data แล้ว |

#### Input / User Actions

**Option A — MeriPOS Connector**

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| MeriPOS API Key | text | ✅ | ร้านดึงจาก MeriPOS admin |
| MeriPOS Store ID | text | ✅ | unique per branch |
| ช่วงข้อมูลที่ดึง | date range | ❌ | default: ย้อนหลัง 12 เดือน |

ปุ่ม **"เชื่อมต่อ MeriPOS"** → Platform pull data → process → รอ ML engine พร้อม

**Option B — CSV Upload**

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| ไฟล์ POS (.csv) | file upload | ✅ | columns: `customer_id`, `transaction_id`, `timestamp`, `total_price`, `product_name`, `category` |
| ไฟล์ CRM (.csv) | file upload | ❌ | columns: `customer_id`, `hashed_email`, `hashed_phone`, `line_uid` — unlock LINE OA + Meta |

ปุ่ม **"อัปโหลด"** → validate columns → encode base64 → ส่ง ML engine

**Data Completeness — Unlock Prompt** (แสดงหลัง upload สำเร็จ)

แสดง progress bar ต่อ feature:

| Feature | สิ่งที่ unlock เมื่อมี | วิธีเพิ่ม |
|---------|----------------------|---------|
| `discount_amount` column | PSM Mode A (แม่นกว่า) | เพิ่มใน POS CSV |
| `line_uid` ใน CRM | LINE OA broadcast | อัปโหลด CRM CSV |
| `hashed_email` / `hashed_phone` ใน CRM | Meta audience upload | อัปโหลด CRM CSV |
| ข้อมูล ≥ 12 เดือน | Store Model (personal) | รอ / เพิ่มข้อมูลย้อนหลัง |

#### แสดงผล
- Progress stepper: 1. Store Profile → **2. Data Connection** → 3. พร้อมใช้งาน
- **สถานะการ ingest**: spinner → "กำลังประมวลผล data..." → สำเร็จ: ✅ "ประมวลผลเสร็จแล้ว" + จำนวน transaction ที่ได้
- Data Completeness bar (แสดงทันทีหลัง ingest): แต่ละ feature มี progress indicator + tooltip อธิบายว่า unlock อะไร

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| POS CSV columns ไม่ครบ | error list: "ขาด column ต่อไปนี้: [list]" + download template CSV |
| transaction < 90 วัน | warning: "ข้อมูลน้อยกว่า 90 วัน — ML จะใช้ Category Model แทน Store Model" ยังดำเนินต่อได้ |
| MeriPOS API Key ผิด | "เชื่อมต่อ MeriPOS ไม่สำเร็จ — ตรวจสอบ API Key อีกครั้ง" |
| ไฟล์ใหญ่เกิน 50MB | "ไฟล์ใหญ่เกิน 50MB — กรุณาแบ่ง upload เป็น batch" |
| `GET /segment/health` คืน `MODEL_NOT_READY` (503) | "ML กำลังประมวลผลข้อมูลครั้งแรก — อาจใช้เวลา 5-10 นาที" + auto-refresh ทุก 30 วินาที |

---

### Screen 1.4: Ready Screen

#### วัตถุประสงค์
สรุปภาพรวม segment ครั้งแรก — ให้ร้านเห็นว่า ML วิเคราะห์ลูกค้าออกมาเป็นยังไง ก่อนเริ่มใช้งานจริง

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `GET` | `/segment/health` | โหลด screen |
| `GET` | `/model/status` | โหลด screen (parallel) |

#### Input / User Actions
- ไม่มี input — ดึง `store_id` จาก session อัตโนมัติ
- ปุ่ม **"เริ่มใช้งาน"** → ไป Screen 2.1

#### แสดงผล

**Section 1 — Model Tier Badge**

| Field | Label ที่แสดง |
|-------|--------------|
| `champion_version` | "โมเดลที่ใช้: v[X]" |
| ประเภท tier | badge: "Category Model — [ประเภทธุรกิจ]" หรือ "Store Model" |
| `calibration.trend` | badge: stable สีเขียว / degrading สีส้ม |

**Section 2 — Segment Summary (จาก `GET /segment/health`)**

แสดง segment cards ทั้ง 10 IBM segments:

| Field | Label ที่แสดง |
|-------|--------------|
| segment name | ชื่อ segment |
| `segments[name].count` | จำนวนลูกค้า |
| `segments[name].trend` | arrow: `growing` ↑ / `stable` → / `shrinking` ↓ |
| `total_customers` | "ลูกค้าทั้งหมด X คน" (ด้านบน) |

**Section 3 — Data Completeness Bar**

Progress bar แสดง % ข้อมูลที่มี — คำนวณจาก feature ที่ ingest ได้:
- POS มีครบ 4 columns หลัก → 40%
- มี `discount_amount` → +20%
- มี CRM (email/phone) → +20%
- มี `line_uid` → +10%
- ข้อมูล ≥ 12 เดือน → +10%

แสดง unlock prompt สำหรับ feature ที่ยังขาด: "เพิ่ม [X] เพื่อ unlock [Y]"

**Section 4 — ML Recommendation**

| Field | Label ที่แสดง |
|-------|--------------|
| `recommendation.goal` | "ML แนะนำ: [goal label]" |
| `recommendation.reason` | เหตุผล |
| `recommendation.urgency` | badge: `high` → สีแดง / `medium` → สีส้ม / `low` → สีเขียว |

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| `MODEL_NOT_READY` (503) | spinner + "ML กำลังประมวลผล data ครั้งแรก..." + auto-refresh ทุก 30 วิ |
| segment ทุกตัวเป็น 0 | "ยังไม่มี data เพียงพอ — ต้องการ transaction อย่างน้อย 90 วัน" + ปุ่ม "เพิ่ม Data" |
| ร้านใช้ Category Model | notice banner สีฟ้า: "คุณกำลังใช้ Category Model (F&B) — เมื่อมีข้อมูลครบ 12 เดือน ระบบจะ upgrade เป็น Store Model อัตโนมัติ" |

---

## GROUP 2 — Dashboard

---

### Screen 2.1: Segment Health Dashboard

#### วัตถุประสงค์
Entry point หลักของ Platform หลัง onboard เสร็จ — แสดงสถานะ segment ปัจจุบัน, ML recommendation, และ alerts ที่ต้องสนใจ

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `GET` | `/segment/health` | โหลด screen |
| `GET` | `/model/status` | โหลด screen (parallel) |

#### Input / User Actions
- ไม่มี input — ดึง `store_id` จาก session อัตโนมัติ
- ปุ่ม Refresh icon → reload ทั้ง 2 endpoints

#### แสดงผล

**ส่วนบน — ML Engine Status** (จาก `GET /model/status`)

| Field | Label ที่แสดง |
|-------|--------------|
| `champion_version` | "โมเดล v[X]" |
| `model_type` | "ประเภท: [model_type]" |
| `auuc` | "AUUC Score: [X]" (tooltip อธิบาย AUUC) |
| `calibration.trend` | badge: `stable` → สีเขียว "แม่นดี" / `degrading` → สีส้ม "เริ่มคลาดเคลื่อน" |
| `calibration.recommendation` | ข้อความใต้ badge |
| `promoted_at` | "อัปเดตล่าสุด: DD MMM YYYY" |

**Model Tier Badge** (ด้านบนขวา)

| Tier | Badge ที่แสดง |
|------|--------------|
| Category Model | "Category: [F&B/Retail/Beauty/Services]" สีฟ้า |
| Store Model | "Store Model — Personalized" สีม่วง |

**ส่วนกลาง — Segment Cards** (จาก `GET /segment/health`)

แสดง 10 cards (IBM RFM segments) แบบ 2×5 grid หรือ scroll:

| Field | Label ที่แสดง |
|-------|--------------|
| segment name (key ของ `segments`) | ชื่อ segment |
| `segments[name].count` | จำนวนลูกค้า (ตัวเลขใหญ่) |
| `segments[name].prev_count` | "(เมื่อเดือนที่แล้ว: X คน)" สีเทา |
| `segments[name].trend` | arrow: `growing` ↑ สีแดง (ถ้า At-Risk/Churn) / ↑ สีเขียว (ถ้า Champions) / `stable` → สีเทา / `shrinking` ↓ |
| `total_customers` | "ลูกค้าทั้งหมด X คน" (header) |
| `snapshot_at` | "ข้อมูล ณ DD MMM YYYY HH:mm" |

> สี trend arrow ขึ้นกับ segment type: At-Risk / Churn Risk / About to Sleep / Hibernating → growing = แดง (แย่), Champions / Loyal / Potential Loyalist → growing = เขียว (ดี)

**Alerts Banner** (จาก `segments.alerts[]` — แสดงเฉพาะเมื่อมี alert)

| Field | Label ที่แสดง |
|-------|--------------|
| `alerts[*].level` | icon: `warning` → ⚠️ สีแดง / `info` → ℹ️ สีส้ม |
| `alerts[*].segment` | ชื่อ segment ที่มีปัญหา |
| `alerts[*].message` | ข้อความ alert |

**ส่วนล่าง — ML Recommendation**

| Field | Label ที่แสดง |
|-------|--------------|
| `recommendation.goal` | "แนะนำ: [goal label]" |
| `recommendation.reason` | เหตุผล |
| `recommendation.urgency` | badge: `high` → "ด่วน" สีแดง / `medium` → "ควรทำเร็วๆ นี้" สีส้ม / `low` → "รอได้" สีเขียว |

#### Actions

| ปุ่ม | พาไปที่ |
|------|---------|
| **"สร้างแคมเปญ"** (primary CTA) | Screen 3.1: New Campaign |
| **"ดูประวัติแคมเปญ"** | Screen 6.1: Campaign History |
| **"ดูการแจ้งเตือน"** (badge ถ้ามี alert) | Screen 2.2: Notification Center |
| Refresh icon | reload `/segment/health` + `/model/status` |

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| กำลังโหลด | skeleton cards 10 ใบ |
| `MODEL_NOT_READY` (503) | banner สีเหลือง: "ML กำลังประมวลผล — กรุณารอสักครู่" ซ่อนปุ่ม "สร้างแคมเปญ" |
| `calibration.trend = "degrading"` | banner สีส้ม: "โมเดลเริ่มคลาดเคลื่อน — ดูรายละเอียดใน Model Status" + ลิงก์ไป Screen 6.4 |
| ไม่มี data เลย (ร้านใหม่ที่ข้ามขั้น) | "ยังไม่มีข้อมูล — เชื่อมต่อ data ก่อน" + ปุ่ม "เชื่อมต่อ Data" → Screen 1.3 |
| network error | toast error + ปุ่ม retry |

---

### Screen 2.2: Notification Center

#### วัตถุประสงค์
รวม alerts ทุกประเภทในที่เดียว — Churn Risk โต, calibration degrading, campaign ที่ต้องวัดผล (ครบ 30 วัน)

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `GET` | `/segment/health` | ดึง `alerts[]` |
| `GET` | `/model/status` | ตรวจ `calibration.trend` |
| `GET` | `/campaign/history` | ตรวจ campaign ที่ `status = "awaiting_outcome"` และ `outcome_due_at` ผ่านมาแล้ว |

#### Input / User Actions
- Filter dropdown: "ทั้งหมด" / "Segment Alerts" / "Model Alerts" / "Campaign Reminders"
- คลิก notification item → redirect ไป screen ที่เกี่ยวข้อง

#### แสดงผล

**Notification List** — เรียงจากใหม่ไปเก่า + level สำคัญก่อน

แต่ละ item แสดง:

| Field | Label ที่แสดง |
|-------|--------------|
| icon | ⚠️ warning / ℹ️ info / 🔔 reminder |
| title | สรุปสั้นๆ เช่น "Churn Risk โต 34%" |
| body | ข้อความเต็ม (จาก `alerts[*].message` หรือ generate จาก calibration / campaign) |
| timestamp | "X ชั่วโมงที่แล้ว" / "เมื่อวาน" |
| CTA ลิงก์ | เช่น "สร้าง Win-Back Campaign →" / "ดู Model Status →" / "วัดผลแคมเปญ →" |

**ประเภท Notification:**

| ประเภท | trigger | CTA |
|--------|---------|-----|
| Churn Risk โต | `alerts[level=warning, segment=Churn Risk]` | "สร้าง Win-Back Campaign" → Screen 3.1 |
| Calibration Degrading | `calibration.trend = "degrading"` | "ดู Model Status" → Screen 6.4 |
| Campaign ต้องวัดผล | campaign ที่ `outcome_due_at` ≤ วันนี้ และ `status = "awaiting_outcome"` | "วัดผลแคมเปญ" → Screen 6.3 |
| At-Risk เพิ่ม | `alerts[level=info, segment=At-Risk]` | "วางแผน Campaign" → Screen 3.1 |

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| ไม่มี notification เลย | illustration + "ทุกอย่างเรียบร้อยดี — ไม่มีการแจ้งเตือนตอนนี้" |
| กำลังโหลด | skeleton list 3 items |
| network error | toast error + retry |

---

## GROUP 3 — Campaign Planning

---

### Screen 3.1: New Campaign

#### วัตถุประสงค์
จุดเริ่มต้นสร้างแคมเปญ — ให้ร้านเลือกว่าจะให้ ML แนะนำทุกอย่าง หรือจะกรอกแผนเองแล้วให้ ML วิเคราะห์

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `GET` | `/segment/health` | ดึง recommendation แสดง context |

#### Input / User Actions

**Mode selector** (2 ตัวเลือกใหญ่):

| Mode | คำอธิบาย | ไปที่ |
|------|---------|-------|
| **ML แนะนำให้** | "ให้ AI วางแผน goal, segment, และ budget ที่เหมาะสมที่สุด" | Screen 3.2: Goal Comparison |
| **กรอกเองแล้วให้ ML วิเคราะห์** | "มีแผนอยู่แล้ว ต้องการให้ ML ตรวจสอบก่อน" | Screen 3.3: Plan Results (manual advisor mode) |

**ถ้าเลือก ML Mode** — user กรอก budget ก่อน:

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| Budget (฿) | number input | ✅ | min ฿500 |
| Mode | toggle | ✅ | "กำหนดงบ" (budget) / "กำหนดกำไร" (target_profit) |
| Target profit (฿) | number input | ถ้า mode = target_profit | — |

**ถ้าเลือก Manual Mode** — user กรอกแผนเบื้องต้น:

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| Goal | dropdown | ✅ | maximize_profit / win_back / retain_loyal / maximize_reach |
| Budget (฿) | number | ✅ | — |
| Segment ที่ต้องการ target | multi-select | ❌ | IBM 10 segments — ถ้าไม่เลือก ML เลือกให้ |
| Promo type | dropdown | ❌ | discount_15 / discount_20 / points_x2 / free_item / notification_only |
| วันที่ตั้งใจยิง | date picker | ❌ | ใช้ตรวจ overlap |

#### แสดงผล
- แสดง context card: "ML แนะนำ [goal] เพราะ [reason]" (จาก `recommendation` ของ `/segment/health`)
- แสดง Churn Risk + At-Risk count ให้เห็นว่า urgent แค่ไหน

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| Budget < ฿500 | inline error: "งบขั้นต่ำ ฿500" |
| `MODEL_NOT_READY` | ซ่อน ML Mode / แสดงแค่ Manual Mode |
| ไม่มี segment data | disable ปุ่มทั้งหมด + "กรุณาเชื่อมต่อ data ก่อน" |

---

### Screen 3.2: Goal Comparison

#### วัตถุประสงค์
ก่อนเลือก goal ให้ ML วางแผน — แสดงผลประมาณการทั้ง 4 goals พร้อมกันเพื่อให้ร้านเปรียบเทียบและตัดสินใจ

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `POST` | `/campaign/plan` | เรียก 4 ครั้งพร้อมกัน (1 goal ต่อ 1 request) — parallel |

Request body ที่ส่งสำหรับแต่ละ goal:
```json
{
  "store_id": "<จาก session>",
  "pos_data": "<base64 POS CSV>",
  "mode": "<budget หรือ target_profit>",
  "budget": <user input>,
  "goal": "<maximize_profit | win_back | retain_loyal | maximize_reach>"
}
```

> `pos_data` ที่ส่ง: Platform ดึง POS ล่าสุดจาก store data ที่ ingest ไว้ — user ไม่ต้อง upload ซ้ำ

#### Input / User Actions
- Budget มาจาก Screen 3.1 แล้ว — ไม่ต้องกรอกซ้ำ
- ปุ่ม **"ปรับงบ"** → แสดง input ให้แก้ budget → เรียก 4 API ใหม่
- ปุ่ม **"เลือก goal นี้"** ในแต่ละ card → ไป Screen 3.3

#### แสดงผล

แสดง 4 cards เรียง 2×2 grid (desktop) หรือ horizontal scroll (mobile):

**แต่ละ card แสดง:**

| Field | Label ที่แสดง |
|-------|--------------|
| `goal_label` | ชื่อ goal ภาษาไทย |
| `summary.customers_selected` | "เลือกลูกค้า X คน" |
| `summary.expected_incremental_profit` | "กำไรคาดการณ์ ฿X" (ตัวเลขใหญ่ สีเขียว) |
| `summary.roi` | "ROI X.Xx" |
| `summary.actual_spend` | "ใช้งบ ฿X" |
| `summary.budget_utilisation_pct` | progress bar แสดง % งบที่ใช้ |

**Goal labels:**

| `goal` value | Label ภาษาไทย |
|-------------|--------------|
| `maximize_profit` | กำไรสูงสุด |
| `win_back` | ดึงลูกค้ากลับ |
| `retain_loyal` | รักษาลูกค้าประจำ |
| `maximize_reach` | เข้าถึงคนมากที่สุด |

**Badge "แนะนำ"**: highlight card ที่ให้ `expected_incremental_profit` สูงสุด

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| กำลังโหลด (4 API calls parallel) | skeleton cards 4 ใบ พร้อม loading spinner |
| goal ใดโหลดสำเร็จก่อน | แสดง card นั้นได้เลย อีก card ค้างเป็น skeleton |
| goal ใด error | card นั้น: "ประมวลผล goal นี้ไม่ได้ — ลองใหม่" + retry ปุ่มเฉพาะ card |
| `viable_after_guardrail` = 0 ทุก goal | banner: "ลูกค้าที่ตรงเงื่อนไขไม่เพียงพอ — ลองเพิ่ม budget หรือเปลี่ยน goal" |
| budget น้อยเกินทุก goal | แสดง minimum budget ที่แนะนำต่อ goal |

---

### Screen 3.3: Plan Results + Manual Advisor

#### วัตถุประสงค์
แสดงรายละเอียดเต็มของแผนแคมเปญหลัง user เลือก goal หรือ analyze แผน manual — ให้ร้านดู segment breakdown, action breakdown และ feasibility ก่อนไปขั้นถัดไป

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `POST` | `/campaign/plan` | โหลด screen (ใช้ผลที่มีอยู่แล้วจาก Screen 3.2 ถ้ามาจาก ML mode — ไม่เรียกซ้ำ) |
| `POST` | `/campaign/analyze` | เมื่อ user กด "วิเคราะห์ก่อน" (ML mode) หรือโหลด screen ทันที (manual mode) |

**Manual mode request body:**
```json
{
  "store_id": "<จาก session>",
  "pos_data": "<base64 POS CSV>",
  "proposed_plan": {
    "goal": "<goal ที่กรอกใน 3.1>",
    "mode": "budget",
    "budget": <budget>,
    "segments": ["<segment ที่เลือก>"],
    "promo_type": "<promo_type>",
    "scheduled_date": "<วันที่ยิง>"
  },
  "campaign_history": "<promo_log จาก /campaign/history>"
}
```

#### Input / User Actions
- ไม่มี input เพิ่มเติม — ข้อมูลมาจาก Screen 3.1 / 3.2 ทั้งหมด
- ปุ่ม **"วิเคราะห์ก่อน"** (ML mode) → เรียก `POST /campaign/analyze` → แสดง Analysis Section ใน screen เดิม
- ปุ่ม **"ไปสร้าง Content"** (primary) → ไป Screen 4.1
- ปุ่ม **"Export Audience เท่านั้น"** (secondary) → ไป Screen 5.1 ข้ามขั้น Content Studio
- ปุ่ม **"เลือก goal อื่น"** → กลับ Screen 3.2

#### แสดงผล

**Section 1 — Plan Summary**

| Field | Label ที่แสดง |
|-------|--------------|
| `plan_id` | Plan ID (เล็กๆ เป็น reference) |
| `goal_label` | ชื่อ goal ที่เลือก |
| `summary.customers_selected` | "ลูกค้าที่เลือก X คน" |
| `summary.viable_after_guardrail` | "ผ่าน guardrail X คน" (tooltip: "ML กรองลูกค้าที่โปรไม่คุ้มทุนออกแล้ว") |
| `summary.actual_spend` | "งบที่ใช้จริง ฿X" |
| `summary.expected_incremental_profit` | "กำไรคาดการณ์ ฿X" (ตัวเลขใหญ่ สีเขียว) |
| `summary.roi` | "ROI X.Xx" |
| `summary.budget_utilisation_pct` | "ใช้งบ X% จากทั้งหมด" |

**Section 2 — Segment Breakdown**

แสดงเป็น horizontal bar chart หรือ table จาก `segment_breakdown`:

| Column | ข้อมูล |
|--------|--------|
| Segment name | key เช่น "At-Risk", "Champions" |
| จำนวนลูกค้า | value |
| % ของ customers_selected | คำนวณ client-side |

**Section 3 — Action Breakdown**

แสดงเป็น table จาก `action_breakdown`:

| Column | ข้อมูล |
|--------|--------|
| Action | key เช่น "Win-Back: ส่วนลด 15%" |
| จำนวนลูกค้า | value |

**Section 4 — Feasibility** (แสดงเฉพาะ `mode = "target_profit"`)

| Field | Label |
|-------|-------|
| `feasibility.target_achievable` | badge: สำเร็จ / ไม่สำเร็จ |
| `feasibility.profit_gap` | "ขาด ฿X จากเป้า" (ถ้า not achievable) |
| `feasibility.max_achievable_profit` | "กำไรสูงสุดที่ทำได้ ฿X" |
| `feasibility.max_budget_needed` | "งบที่ต้องใช้ ฿X" |

**Section 5 — Analysis Results** (แสดงเมื่อมีผล `POST /campaign/analyze`)

`overall_verdict` banner:
- `recommended` → banner สีเขียว "แผนนี้น่าทำ"
- `proceed_with_caution` → banner สีเหลือง "ทำได้ แต่มีข้อควรระวัง"
- `not_recommended` → banner สีแดง "ML ไม่แนะนำแผนนี้"

4 check cards:

| Check | Fields ที่แสดง |
|-------|--------------|
| Profitability | `checks.profitability.status` badge + message + `net` (กำไร/ขาดทุน) |
| Overlap | `checks.overlap.status` badge + `overlap_count` คน + message |
| Segment Readiness | `checks.segment_readiness.status` badge + `trend` + message |
| Budget Efficiency | `checks.budget_efficiency.status` badge + `gain` + message |

ถ้ามี `suggestion`:

| Field | Label |
|-------|-------|
| `suggestion.message` | กล่องข้อความแนะนำ สีเหลือง |
| `suggestion.adjusted_budget` | "งบที่แนะนำ ฿X" |
| `suggestion.adjusted_estimated_profit` | "กำไรที่คาดหวัง ฿X" |
| `suggestion.adjusted_roi` | "ROI ที่คาดหวัง X.Xx" |

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| `action_breakdown` เป็น empty object | "ไม่มี action สำหรับแคมเปญนี้" |
| `overall_verdict = "not_recommended"` แล้วกด "ไปสร้าง Content" | confirmation dialog: "ML ไม่แนะนำแผนนี้ — ต้องการดำเนินต่อหรือไม่?" |
| `/campaign/analyze` loading | inline spinner ที่ Section 5 ไม่ block ทั้ง screen |
| `/campaign/analyze` error | "วิเคราะห์ไม่ได้ตอนนี้ — ดำเนินต่อหรือ retry?" |

---

### Screen 3.4: Campaign Calendar

#### วัตถุประสงค์
ภาพรวม campaign ที่วางแผนไว้และที่ผ่านมาในรูปแบบ calendar — ช่วยให้ร้านเห็น cadence และป้องกันยิงแคมเปญซ้ำในช่วงเวลาเดียวกัน

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `GET` | `/campaign/history` | โหลด screen |

#### Input / User Actions
- Toggle view: **"Monthly"** / **"Weekly"**
- คลิก campaign event ใน calendar → แสดง popup สรุปข้อมูล campaign นั้น
- ปุ่ม **"+ สร้างแคมเปญ"** → Screen 3.1

#### แสดงผล

**Calendar View** — event ต่อ campaign จาก `campaigns[]`:

| Field | Label ที่แสดง |
|-------|--------------|
| `executed_at` | วันที่ event ใน calendar |
| `goal_label` | ชื่อ event |
| `status` | สีของ event: `active` → เขียว / `awaiting_outcome` → ส้ม / `completed` → เทา / `plan_only` → ฟ้า |

**Popup เมื่อคลิก event:**

| Field | Label ที่แสดง |
|-------|--------------|
| `goal_label` | Goal |
| `customers_targeted` | "ส่งถึง X คน" |
| `actual_spend` | "งบ ฿X" |
| `status` badge | สถานะ |
| `outcome_due_at` | "วัดผลภายใน: DD MMM YYYY" (ถ้า status = awaiting_outcome) |

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| ไม่มีประวัติ | calendar ว่าง + "ยังไม่มีแคมเปญ — เริ่มสร้างแคมเปญแรกของคุณ" + ปุ่ม "สร้างแคมเปญ" |
| campaign หลายตัวใน 1 วัน | แสดง stacked dots + "X แคมเปญ" คลิกเพื่อขยาย |

---

## GROUP 4 — Content Studio

---

### Screen 4.1: Content Brief

#### วัตถุประสงค์
แสดง ML insight → strategy → art direction ที่ generate ขึ้นมาสำหรับแคมเปญนี้ ก่อนที่จะ generate ภาพและ copy จริง

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| ไม่มี ML endpoint ตรงๆ — เรียก content generator ภายใน Platform | — | — |

> Content generator flow: `plan_id` + `brand_kit` → Claude API (strategy layer) → art direction layer → แสดงผลใน screen นี้

#### Input / User Actions

**Mode selector:**

| Mode | คำอธิบาย |
|------|---------|
| **ML-driven (Promo Content)** | auto-fill จาก plan — segment, action, promo_type, top_product |
| **Brand-driven (Awareness)** | ร้านเลือก content_type เอง ไม่ต้องมี promo |

**ถ้าเลือก Brand-driven:**

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| Content Type | dropdown | ✅ | new_menu / service_highlight / brand_story / seasonal / bestseller |
| สินค้า / เมนูที่ต้องการ highlight | text | ❌ | ถ้าไม่กรอก ใช้ top_product จาก POS data |
| ช่องทาง | multi-select | ✅ | Meta / LINE OA / Email / SMS |

#### แสดงผล

**Section 1 — ML Insight (auto-fill จาก plan)**

| Field | Label ที่แสดง |
|-------|--------------|
| segment name | "กลุ่มเป้าหมาย: [segment]" |
| action type | "Action: [action]" |
| promo_type | "โปรโมชัน: [promo_type]" |
| top_product | "สินค้าหลัก: [top_product]" |
| send_window | "เวลาที่แนะนำส่ง: [HH:mm-HH:mm]" |

**Section 2 — Strategy Layer** (จาก Claude API)

| Field | Label ที่แสดง |
|-------|--------------|
| key_message | "Key Message" |
| tone | "โทน: [tone]" |
| emotion | "อารมณ์ที่ต้องการ: [emotion]" |
| call_to_action | "CTA หลัก: [cta]" |

**Section 3 — Art Direction Preview**

| Field | Label ที่แสดง |
|-------|--------------|
| visual_brief | "Visual Brief" (paragraph) |
| composition | "Composition: [composition]" |
| mood | "Mood: [mood]" |
| color_palette | "Color Palette" (แสดงเป็น color swatches) |
| `photo_style` จาก Brand Kit | "Style: [photo_style]" |

**ด้านล่าง — Brand Kit ที่ใช้**

แสดง summary: brand_name, tone, forbidden_words, preferred_words ที่จะ apply

ปุ่ม **"แก้ Brand Kit"** → เปิด modal edit Brand Kit ชั่วคราว ไม่ต้องออกจาก flow

#### Actions

| ปุ่ม | พฤติกรรม |
|------|---------|
| **"Generate Creative"** (primary) | ส่งไป Screen 4.2 — เรียก image API + Claude copy gen |
| **"แก้ strategy ก่อน"** | แสดง editable field สำหรับ strategy layer — user แก้แล้วค่อย Generate |
| **"ย้อนกลับ"** | กลับ Screen 3.3 |

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| ไม่มี Brand Kit | warning: "ยังไม่ได้ตั้งค่า Brand Kit — เราจะใช้ค่า default ของ [ประเภทธุรกิจ]" + ปุ่ม "ตั้งค่า Brand Kit" |
| top_product ไม่มีใน POS | "ไม่พบข้อมูลสินค้าใน data — กรอกชื่อสินค้าที่ต้องการ highlight" |
| strategy generation ล้มเหลว | "สร้าง strategy ไม่ได้ตอนนี้ — ลองใหม่" + retry |

---

### Screen 4.2: Creative Preview

#### วัตถุประสงค์
แสดงภาพและ copy ที่ generate มาแล้ว ให้ร้านดูก่อน approve — แสดง per-channel ถ้าเลือกหลายช่องทาง

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| ไม่มี ML endpoint — image API + Claude API ภายใน Platform | — | — |

> Image API: pluggable (เจ้ายังไม่ตัดสินใจ — Firefly / Stability AI / Ideogram / fal.ai) — interface เดิมไม่เปลี่ยน

#### Input / User Actions
- Tab selector: แต่ละ channel ที่เลือก (Meta / LINE OA / Email / SMS)
- ดู preview full-screen โดยคลิกที่ภาพ

#### แสดงผล

**Per Channel Preview** — แต่ละ tab แสดง:

| Field | Label ที่แสดง |
|-------|--------------|
| ภาพที่ generate | preview image (ขนาดที่ถูก spec ตาม channel เช่น 1200×628 Meta) |
| `headline` | "Headline" |
| `body` | "Body Copy" |
| `cta` | "CTA" |
| `meta_primary_text` | "Primary Text" (เฉพาะ Meta tab) |
| `image_direction` | "Image Direction" (แสดงเป็น text reference — ร้านดูว่าภาพสอดคล้องกับ direction ไหม) |

**Character count:**
- Meta headline: max 40 chars — แสดง counter
- Meta primary text: max 125 chars — แสดง counter
- LINE OA: max 500 chars

**Image API badge** (มุมล่างขวาของภาพ): "Generated by [ชื่อ provider — pluggable]"

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| กำลัง generate | loading skeleton image + "กำลัง generate ภาพ..." (อาจใช้ 10-30 วิ) |
| image API error | placeholder สีเทา + "สร้างภาพไม่ได้ตอนนี้ — ดำเนินต่อโดยไม่มีภาพ หรือ retry" |
| copy เกิน character limit | inline error สีแดงใต้ field + "เกิน X ตัวอักษร — Claude จะปรับให้อัตโนมัติถ้ากด Regenerate" |

---

### Screen 4.3: Approve / Request Regenerate

#### วัตถุประสงค์
ให้ร้าน approve creative ก่อน launch หรือขอ regenerate ถ้าไม่พอใจ — มี request feedback เพื่อ improve prompt

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| ไม่มี ML endpoint ตรงๆ | — | — |

#### Input / User Actions

**ถ้า Approve:**
- ปุ่ม **"Approve ทั้งหมด"** หรือ approve per-channel → ไป Screen 5.1

**ถ้า Regenerate:**

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| ต้องการแก้อะไร | dropdown multi-select | ✅ | ภาพ / headline / body copy / tone / ทั้งหมด |
| feedback เพิ่มเติม | textarea | ❌ | เช่น "ต้องการภาพที่อบอุ่นกว่านี้" |

ปุ่ม **"Regenerate"** → กลับ Screen 4.2 พร้อม feedback ใหม่

**Manual edit:**
- ร้านสามารถ edit `headline`, `body`, `cta` ได้โดยตรง ก่อน approve (rich text inline edit)

#### แสดงผล
- Summary ของ creative ที่จะ approve ต่อ channel
- Status per channel: ✅ Approved / ⏳ Pending / 🔄 Regenerating

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| Regenerate ซ้ำ > 3 ครั้ง | notice: "ถ้าต้องการ control มากขึ้น ลอง edit manual หรือ update Brand Kit" |
| approve บาง channel ยังไม่ครบ | warning: "ยังมี X channel ที่ยังไม่ approve — approve ทั้งหมดก่อน launch หรือ launch เฉพาะ channel ที่พร้อม?" |

---

## GROUP 5 — Execution Center

---

### Screen 5.1: Choose Channel

#### วัตถุประสงค์
ให้ร้านเลือกว่าจะยิงแคมเปญผ่านช่องทางไหน — แต่ละช่องทางแสดงสถานะ connection และ prerequisite ที่ต้องมี

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `POST` | `/audience/export` | user กด confirm channel selection |

Request body ที่ส่งอัตโนมัติ:
```json
{
  "plan_id": "<จาก plan>",
  "store_id": "<จาก session>",
  "crm_data": "<base64 CRM CSV ถ้ามี>"
}
```

#### Input / User Actions

**Channel Cards** (เลือกได้หลายอัน):

| Channel | Prerequisite | สถานะ ถ้ายังไม่เชื่อม |
|---------|-------------|---------------------|
| **Meta Ads** | connect Meta ad account | "เชื่อมต่อ Meta" → Screen 5.2 |
| **LINE OA** | connect Channel Access Token + มี `line_uid` ใน CRM | "เชื่อมต่อ LINE OA" → Screen 5.3 |
| **Email** | มี `hashed_email` ใน CRM | warning: "ต้องการ email ในข้อมูลลูกค้า" |
| **SMS** | มี `hashed_phone` ใน CRM | warning: "ต้องการเบอร์โทรในข้อมูลลูกค้า" |
| **PDF / CSV Export** | ไม่มี prerequisite | พร้อมใช้เสมอ |

แต่ละ card แสดง:
- จำนวน audience ที่จะเข้าถึงได้ (จาก audience export summary)
- สถานะ connection: ✅ Connected / ⚠️ Not Connected / 🔒 ต้องการข้อมูลเพิ่ม

#### แสดงผล

**Ad Budget Recommendation** (สำหรับ Meta):

| ก่อน connect Meta | หลัง connect Meta |
|-------------------|------------------|
| "CPM ประมาณ ฿30-80 (F&B benchmark)" | "CPM ของคุณเฉลี่ย ฿X (จาก Meta performance)" |
| "งบแนะนำสำหรับ warm audience: ฿X" | "งบแนะนำตาม CPM จริง: ฿X" |

**Cost Summary:**

| Field | Label |
|-------|-------|
| promo cost (จาก plan) | "ต้นทุนโปรโมชัน ฿X" |
| ad spend estimate | "ค่าโฆษณาโดยประมาณ ฿X" |
| total | "รวมทั้งหมด ฿X" |
| net ROI estimate | "Net ROI ประมาณ X.Xx" |

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| ไม่มี audience เลยใน channel ที่เลือก | disable channel นั้น + "ไม่มี [LINE UID / email / phone] ในข้อมูลลูกค้า" |
| `POST /audience/export` loading | "กำลัง prepare audience..." spinner ห้ามกดซ้ำ |
| `PLAN_NOT_FOUND` (404) | "แผนนี้หมดอายุ — กลับไปสร้างแผนใหม่" |
| เลือกแค่ PDF Export | ไม่เรียก `/audience/export` — download plan เป็น PDF โดยตรง |

---

### Screen 5.2: Meta Connect

#### วัตถุประสงค์
ให้ร้าน connect Meta ad account เพื่อให้ Platform สร้าง campaign + pull performance กลับมาได้

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| ไม่มี ML endpoint — Meta API ภายใน Platform | — | — |

#### Input / User Actions

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| Meta Ad Account ID | text | ✅ | format: `act_XXXXXXXXX` |
| Meta Access Token | password | ✅ | token ที่มีสิทธิ์ `ads_management` + `ads_read` |

ปุ่ม **"ทดสอบการเชื่อมต่อ"** → ตรวจว่า token ถูกต้องและมีสิทธิ์พอก่อน save

ปุ่ม **"Save & Continue"** → บันทึก credentials + กลับ Screen 5.1

> **Security note:** Access Token เก็บ encrypted ใน Platform — ไม่แสดงซ้ำหลัง save

#### แสดงผล
- ขั้นตอน guide: "วิธีหา Meta Access Token" (expandable accordion)
- Permissions checklist ที่ต้องการ: `ads_management`, `ads_read`, `business_management`
- หลัง test สำเร็จ: ✅ "เชื่อมต่อสำเร็จ — Ad Account: [ชื่อ account]"

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| Token ไม่มีสิทธิ์ `ads_management` | "Token นี้ไม่มีสิทธิ์สร้าง campaign — ตรวจสอบ permissions ใน Meta Business Suite" |
| Ad Account ถูก disable | "Ad Account นี้ถูกระงับ — กรุณาติดต่อ Meta" |
| Token หมดอายุ (เมื่อใช้งานจริง) | notification: "Meta Token หมดอายุ — กรุณา reconnect" + ลิงก์ไป Screen 7.3 |
| `CHANNEL_ERROR` (502) | "ติดต่อ Meta ไม่ได้ตอนนี้ — ลองใหม่ภายหลัง" |

---

### Screen 5.3: LINE OA Connect

#### วัตถุประสงค์
ให้ร้าน connect LINE OA ด้วย Channel Access Token เพื่อให้ Platform broadcast ข้อความถึง LINE UID ที่มีในฐานข้อมูล

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| ไม่มี ML endpoint — LINE Messaging API ภายใน Platform | — | — |

#### Input / User Actions

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| LINE Channel Access Token | password | ✅ | ดึงจาก LINE Developers Console |
| LINE Channel ID | text | ✅ | ใช้ verify identity |

ปุ่ม **"ทดสอบการเชื่อมต่อ"** → ส่ง test message ไปที่ LINE OA ของร้านก่อน save

ปุ่ม **"Save & Continue"** → บันทึก credentials + กลับ Screen 5.1

#### แสดงผล
- ขั้นตอน guide: "วิธีหา Channel Access Token จาก LINE Developers Console" (expandable)
- Warning: "LINE ไม่เปิด API ให้ lookup LINE UID จาก email/phone — ต้องมี `line_uid` ใน CRM data เท่านั้น"
- แสดง `audiences.line_oa.count` ที่จะส่งถึงได้จาก audience export

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| Token ผิด | "Channel Access Token ไม่ถูกต้อง — ตรวจสอบใน LINE Developers Console" |
| `line_uid` ใน CRM = 0 | warning: "ไม่มี LINE UID ใน CRM — เชื่อมต่อได้แต่จะส่งถึง 0 คน ต้องการดำเนินต่อหรือไม่?" |
| `CHANNEL_ERROR` (502) | "ติดต่อ LINE OA ไม่ได้ตอนนี้ — ลองใหม่ภายหลัง" |

---

### Screen 5.4: Launch & Track

#### วัตถุประสงค์
หน้า confirm ก่อน launch จริง — แสดงสรุปทุกอย่าง แล้วกด Execute เพื่อส่ง campaign พร้อมกันทุก channel ที่เลือก

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `POST` | `/campaign/execute` | user กด "Launch Campaign" |

> ⚠️ Endpoint นี้ยังรอ research — UI นี้ spec ไว้แต่ยังไม่ implement จนกว่า research เสร็จ

Request body ที่ Platform ส่ง:
```json
{
  "export_id": "<จาก audience export>",
  "store_id": "<จาก session>",
  "channels": ["meta", "line_oa"],
  "line_oa_config": {
    "channel_access_token": "<จาก 5.3>",
    "message": {
      "type": "text",
      "text": "<copy ที่ approve จาก 4.3>"
    }
  },
  "meta_config": {
    "ad_account_id": "<จาก 5.2>",
    "access_token": "<จาก 5.2>"
  }
}
```

#### Input / User Actions
- ไม่มี input เพิ่ม — ทุกอย่างเตรียมมาแล้ว
- ปุ่ม **"Launch Campaign"** (primary — สีแดง/ส้ม เพราะ action ไม่ reversible) → execute
- ปุ่ม **"แก้ไข"** → กลับ Screen ที่เกี่ยวข้อง

#### แสดงผล

**Pre-launch Summary:**

| Section | ข้อมูล |
|---------|--------|
| Campaign Goal | goal_label |
| Audience | customers_selected คน ต่อ channel |
| Promo Actions | action_breakdown list |
| Creative | thumbnail per channel |
| Channels | badge ต่อ channel ที่ enable |
| Budget | promo cost + ad spend + total |

**หลัง Launch สำเร็จ:**

แสดงผลจาก `POST /campaign/execute`:

| Field | Label ที่แสดง |
|-------|--------------|
| `execute_id` | "Execute ID: [X]" (reference สำหรับ track) |
| `executed_at` | "ส่งเมื่อ: DD MMM YYYY HH:mm" |
| `results.line_oa.sent_count` | "LINE OA: ส่งสำเร็จ X คน" |
| `results.line_oa.failed_count` | "ส่งไม่ได้ X คน" (ถ้า > 0 → แสดงสีส้ม) |
| `results.meta.status` | "Meta: Uploaded" / "Meta: Error" |
| `results.meta.custom_audience_id` | "Custom Audience ID: [X]" |
| `outcome_due_at` (คำนวณ = executed_at + 30 วัน) | "วัดผลได้หลัง: DD MMM YYYY" |

#### Actions หลัง launch

| ปุ่ม | พาไปที่ |
|------|---------|
| **"ดู Ad Performance"** (ถ้า Meta) | Screen 6.2 |
| **"ตั้ง Reminder วัดผล"** | บันทึก notification ใน Platform สำหรับ Screen 6.3 |
| **"กลับ Dashboard"** | Screen 2.1 |

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| `CHANNEL_ERROR` Line OA (502) | "LINE OA ส่งไม่ได้ตอนนี้ — Meta ยังดำเนินต่อ ต้องการ retry LINE OA ภายหลังไหม?" |
| `CHANNEL_ERROR` Meta (502) | "Meta upload ล้มเหลว — ต้องการ retry หรือ export CSV แทน?" |
| endpoint ยังไม่พร้อม (⏳) | แสดง placeholder: "Launch ยังไม่พร้อม — Export Audience CSV แทนก่อน" |

---

## GROUP 6 — Analytics

---

### Screen 6.1: Campaign History

#### วัตถุประสงค์
แสดงประวัติ campaign ทั้งหมดของร้าน — ทุก status ตั้งแต่ draft ถึงที่วัดผลแล้ว

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `GET` | `/campaign/history` | โหลด screen + pagination |

Query params: `store_id`, `limit=20`, `offset=0`

#### Input / User Actions
- Filter: status dropdown ("ทั้งหมด" / "กำลังวิ่ง" / "รอวัดผล" / "เสร็จสิ้น" / "แบบร่าง")
- Date range picker (client-side filter)
- คลิก row → accordion ขยาย

#### แสดงผล

**Campaign History Table:**

| Field | Label ที่แสดง |
|-------|--------------|
| `plan_id` | Plan ID (เล็ก) |
| `goal_label` | Goal |
| `executed_at` | วันที่ execute (DD MMM YYYY HH:mm) |
| `customers_targeted` | จำนวนลูกค้า |
| `actual_spend` | งบ (฿) |
| `status` | badge (ดูด้านล่าง) |
| `outcome_due_at` | วันที่ต้องวัดผล (ถ้า status = awaiting_outcome) |

**Status badge:**

| `status` value | Badge | สี |
|----------------|-------|----|
| `plan_only` | "แบบร่าง" | เทา |
| `active` | "กำลังวิ่ง" | เขียว |
| `awaiting_outcome` | "รอวัดผล" | ส้ม |
| `completed` | "เสร็จสิ้น" | ม่วง |

**Accordion เมื่อคลิก row:**

| Field | Label ที่แสดง |
|-------|--------------|
| `segment_breakdown` (ถ้ามี) | "Segment Breakdown" table |
| `action_breakdown` (ถ้ามี) | "Action Breakdown" table |
| `outcome` (ถ้ามี) | สรุป: predicted_profit, actual_revenue, lift_pct |

#### Actions

| ปุ่ม / Interaction | พฤติกรรม |
|------------------|---------|
| **"ดูผลลัพธ์"** (status = `completed`) | ไป Screen 6.3 |
| **"วัดผลตอนนี้"** (status = `awaiting_outcome` และ ≥ 30 วัน) | ไป Screen 6.3 mode กรอก |
| **"ดู Ad Performance"** (ถ้า Meta) | ไป Screen 6.2 |
| **"สร้างแคมเปญใหม่"** | ไป Screen 3.1 |
| Pagination | โหลด page ถัดไป (offset += 20) |

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| ไม่มีประวัติเลย | illustration + "ยังไม่มีแคมเปญ — เริ่มสร้างแคมเปญแรก" + ปุ่ม "สร้างแคมเปญ" |
| กำลังโหลด | skeleton table 5 rows |
| row ที่ awaiting_outcome แต่ < 30 วัน | ไม่แสดงปุ่ม "วัดผล" — แสดง tooltip "รอครบ 30 วัน (เหลือ X วัน)" |

---

### Screen 6.2: Ad Performance

#### วัตถุประสงค์
แสดง Meta ad performance จริงที่ pull กลับมาผ่าน Meta API — CTR, conversion, CPM ต่อ campaign เพื่อปรับ content และ budget ครั้งต่อไป

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| ไม่มี ML endpoint — pull จาก Meta API ภายใน Platform | — | — |

#### Input / User Actions
- เลือก campaign จาก dropdown (default: campaign ล่าสุด)
- Date range picker
- ปุ่ม **"Refresh"** → pull ข้อมูลจาก Meta API ใหม่

#### แสดงผล

**Overview Cards:**

| Metric | Label |
|--------|-------|
| CTR | "Click-Through Rate X%" |
| CPM | "Cost per 1,000 Impressions ฿X" |
| Conversions | "Conversion X ครั้ง" |
| Total Spend | "Meta Spend ฿X" |
| ROAS | "Return on Ad Spend X.Xx" |

**Performance Chart:** line chart แสดง CTR และ CPM ตามวัน

**Audience Performance:** breakdown ตาม custom_audience vs lookalike

**Content Performance:**
- Headline ที่ CTR สูงสุด → highlight เป็น "winning creative"
- แสดง feedback loop: "ML จะ learn จาก headline นี้เพื่อ campaign ถัดไป"

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| campaign ยังไม่ได้ผ่าน Meta | "ไม่มีข้อมูล Meta สำหรับ campaign นี้ — ใช้ channel Meta เท่านั้นถึงจะมีข้อมูล" |
| Meta Token หมดอายุ | banner: "Meta Token หมดอายุ — Reconnect เพื่อ pull ข้อมูล" + ลิงก์ไป Screen 7.3 |
| campaign เพิ่ง launch (< 24 ชม.) | "ข้อมูลยังไม่พร้อม — Meta ใช้เวลาอย่างน้อย 24 ชม. ในการ report" |

---

### Screen 6.3: Outcome Tracking

#### วัตถุประสงค์
30 วันหลัง execute — ให้ร้านส่ง POS ล่าสุดมาวัด actual lift เทียบกับที่ ML predict ไว้ เพื่อปรับ calibration

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `POST` | `/campaign/record-outcome` | user กด "บันทึกผล" |

Request body:
```json
{
  "execute_id": "<จาก campaign history>",
  "store_id": "<จาก session>",
  "post_pos_data": "<base64 POS CSV 30 วันหลัง execute>",
  "observation_days": 30
}
```

> Platform ดึง POS 30 วันล่าสุดจาก store data อัตโนมัติ — ถ้า connect MeriPOS ไม่ต้อง upload ซ้ำ ถ้าใช้ CSV ต้อง upload ใหม่

#### Input / User Actions

**ถ้า connect MeriPOS:** ไม่มี input — Platform pull อัตโนมัติ
**ถ้าใช้ CSV upload:** ต้อง upload POS CSV ใหม่ (30 วันหลัง execute)

**Confirm form ก่อนส่ง:**
- แสดง campaign ที่กำลังวัดผล: execute_id, goal, วันที่ execute, จำนวนลูกค้า
- แสดง date range ของ POS ที่ใช้วัด: "POS วันที่ X ถึง Y (X วัน)"
- ปุ่ม **"บันทึกผล"**

#### แสดงผล

**Section 1 — Campaign Reference**

| Field | Label ที่แสดง |
|-------|--------------|
| `execute_id` | Execute ID |
| `observation_days` | "วัดผล X วัน" |

**Section 2 — ผลลัพธ์จริง vs ที่คาดการณ์**

| Field | Label ที่แสดง |
|-------|--------------|
| `result.predicted_profit` | "กำไรที่ ML คาดการณ์ ฿X" |
| `result.actual_revenue` | "รายได้จริง ฿X" (ตัวเลขใหญ่) |
| `result.calibration_ratio` | "Calibration Ratio X.XXx" (tooltip อธิบาย) |
| `result.trend` | badge: `improving` → สีเขียว / `stable` → สีน้ำเงิน / `degrading` → สีส้ม |
| `result.recommendation` | ข้อความแนะนำ |

**Calibration Ratio interpretation:**
- 0.90–1.10 → "โมเดลแม่นมาก" สีเขียว
- 0.80–0.90 หรือ 1.10–1.20 → "แม่นพอใช้" สีเหลือง
- < 0.80 หรือ > 1.20 → "โมเดลคลาดเคลื่อน — ควร retrain" สีส้ม

**Section 3 — Lift Analysis**

| Field | Label ที่แสดง |
|-------|--------------|
| `comparison.baseline_conversion_rate` | "Conversion Rate กลุ่มปกติ X%" |
| `comparison.treated_conversion_rate` | "Conversion Rate กลุ่มที่ได้รับ promo X%" |
| `comparison.lift_pct` | "Lift +X%" (ตัวเลขใหญ่ สีเขียว) |

แสดงเป็น before/after bar chart

**Section 4 — Model Health** (แสดงเฉพาะ `result.trend = "degrading"`)

Banner สีส้ม: "โมเดลเริ่มคลาดเคลื่อน — ควร retrain ด้วย POS ล่าสุด" + ปุ่ม "ไป Retrain" → Screen 6.4

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| กำลังประมวลผล | "กำลังวิเคราะห์ผลแคมเปญ..." full-screen loading (อาจใช้ 10–30 วิ) |
| `PLAN_NOT_FOUND` (404) | "ไม่พบข้อมูลแคมเปญนี้ — ติดต่อทีม Codera" |
| POS มีข้อมูลน้อยกว่า 30 วัน | warning: "POS มีข้อมูลเพียง X วัน — ผลที่ได้อาจไม่ครบถ้วน" ยังส่งได้ |
| `result.lift_pct` < 0 | แสดง negative lift สีแดง + "โปรครั้งนี้ไม่ได้ผลตามที่คาด — ML จะ learn จากข้อมูลนี้เพื่อปรับ campaign ถัดไป" |
| ส่งผลซ้ำ (execute_id เดิม) | "คุณบันทึกผลแคมเปญนี้ไปแล้ว" แสดงผลเดิม |

---

### Screen 6.4: Model Status + Retrain

#### วัตถุประสงค์
แสดงสถานะโมเดลปัจจุบันและ model tier ของร้าน — ให้ทีม Codera หรือร้านที่ unlock Store Model trigger retrain ได้

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `GET` | `/model/status` | โหลด screen |
| `POST` | `/model/retrain` | user กด "Retrain โมเดล" |

#### Input / User Actions

**Retrain form** (แสดงเฉพาะเมื่อร้านมี Store Model หรือ calibration degrading):

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| PSM Mode | dropdown | ❌ (default: discount) | discount / campaign_list / proxy |
| Observation months | number | ❌ (default: 3) | 1-6 เดือน |
| Auto promote | toggle | ❌ (default: on) | promote ถ้า AUUC ชนะ champion ปัจจุบัน |

ปุ่ม **"Retrain โมเดล"** → เรียก `POST /model/retrain`

#### แสดงผล

**Section 1 — Model Tier**

| Field | Label ที่แสดง |
|-------|--------------|
| Tier ปัจจุบัน | badge: "Global" / "Category: [ประเภท]" / "Store Model" |
| เงื่อนไข upgrade | progress bar: "ข้อมูล X/12 เดือน — อีก Y เดือนจะ unlock Store Model" |

**Section 2 — Champion Model** (จาก `GET /model/status`)

| Field | Label ที่แสดง |
|-------|--------------|
| `champion_version` | "Champion: v[X]" |
| `model_type` | "ประเภท: [model_type]" |
| `auuc` | "AUUC Score: [X]" |
| `calibration.trend` | badge |
| `calibration.avg_calibration_ratio` | "Calibration Ratio เฉลี่ย: [X]" |
| `calibration.recommendation` | ข้อความแนะนำ |
| `promoted_at` | "อัปเดตล่าสุด: DD MMM YYYY" |

**Section 3 — Retrain Result** (แสดงหลัง retrain สำเร็จ จาก `POST /model/retrain`)

| Field | Label ที่แสดง |
|-------|--------------|
| `psm_result.matched_pairs` | "PSM Matched Pairs: X" |
| `psm_result.features_balanced` | "Features Balanced: X/5" |
| `evaluation.new_model_auuc` | "New Model AUUC: [X]" |
| `evaluation.champion_auuc` | "Champion AUUC: [X]" |
| `evaluation.winner` | badge: "New Model ชนะ" สีเขียว / "Champion ยังดีกว่า" สีเทา |
| `promotion.promoted` | badge: "Promoted → v[X]" / "ไม่ promote" |

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| ร้านยังใช้ Category Model | ซ่อน Retrain button — แสดง "Retrain จะ unlock เมื่อมีข้อมูลครบ 12 เดือน" |
| `RETRAIN_FAILED` (500) | "Retrain ล้มเหลว — ติดต่อทีม Codera" |
| retrain กำลังทำงาน | "กำลัง retrain... อาจใช้เวลา 5-10 นาที" + disable ปุ่ม Retrain |
| `calibration.trend = "degrading"` | banner สีส้มที่ด้านบน: "โมเดลเริ่มคลาดเคลื่อน — แนะนำ retrain" |

---

## GROUP 7 — Settings

---

### Screen 7.1: Brand Kit Edit

#### วัตถุประสงค์
ให้ร้านแก้ Brand Kit ที่ตั้งไว้ตอน onboard — การแก้จะ apply ทุก content gen ใหม่ที่สร้างหลังจากนี้

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| ไม่มี ML endpoint — บันทึกลง Platform database | — | — |

#### Input / User Actions

เหมือน Screen 1.2 ทุกประการ แต่ pre-fill ด้วยค่าที่บันทึกไว้แล้ว

**Fields:**

| Field | ประเภท |
|-------|--------|
| `brand_name` | text |
| ประเภทธุรกิจ | dropdown |
| `tone` | dropdown |
| `forbidden_words` | tag input |
| `preferred_words` | tag input |
| `photo_style` | dropdown |
| `layout_preference` | dropdown |
| `reference_examples` | image upload (max 3) — แสดง current images |

ปุ่ม **"บันทึก"** → save + แสดง success toast "Brand Kit อัปเดตแล้ว"
ปุ่ม **"ยกเลิก"** → discard changes

#### แสดงผล
- Preview panel: แสดงตัวอย่าง content tone แบบ real-time เมื่อ user แก้ `tone` หรือ `brand_name`
- Notice: "การเปลี่ยนแปลงจะ apply กับ content ที่สร้างใหม่เท่านั้น — campaign ที่ approve ไปแล้วไม่เปลี่ยน"

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| `forbidden_words` ซ้ำกับ `preferred_words` | inline warning ก่อน save |
| ลบ `reference_examples` ทั้งหมด | confirm dialog: "ลบ reference images ทั้งหมด ใช้ photo_style เป็น visual guide เท่านั้น — ยืนยัน?" |

---

### Screen 7.2: Data Completeness Tracker

#### วัตถุประสงค์
แสดง data ที่มีอยู่ของร้านและสิ่งที่ unlock ได้เมื่อเพิ่ม data — ให้ร้านเห็นว่า "เพิ่ม X แล้วได้ Y"

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `GET` | `/segment/health` | ดูว่า ML ใช้ feature อะไรได้บ้าง |
| `GET` | `/model/status` | ดู tier ปัจจุบัน |

#### Input / User Actions
- ปุ่ม **"+ เพิ่ม Data"** → ไป Screen 1.3 (Data Connection) เพื่อ update

#### แสดงผล

**Progress Bar หลัก:** "Data Completeness: X%"

**Feature breakdown:**

| Feature | สถานะ | Unlock |
|---------|-------|--------|
| POS transaction | ✅ X records | ML segmentation |
| `discount_amount` | ✅ / ❌ | PSM Mode A (แม่นกว่า) |
| `product_name` + `category` | ✅ / ❌ | action recommendation ระบุสินค้าได้ |
| CRM: email/phone | ✅ X records / ❌ | Meta audience upload, Email/SMS |
| CRM: `line_uid` | ✅ X records / ❌ | LINE OA broadcast |
| ข้อมูล ≥ 12 เดือน | ✅ X เดือน / ❌ X/12 | Store Model |

**Last sync:** "sync ล่าสุด: DD MMM YYYY HH:mm" (MeriPOS) หรือ "upload ล่าสุด: DD MMM YYYY" (CSV)

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| ยังไม่มี data เลย | "ยังไม่มีข้อมูล — เชื่อมต่อ data เพื่อเริ่มใช้งาน" + ปุ่ม "เชื่อมต่อ Data" |
| MeriPOS sync ล้มเหลวล่าสุด | warning: "MeriPOS sync ล้มเหลวเมื่อ [วัน] — ตรวจสอบ connection ใน Connections" |

---

### Screen 7.3: Connections

#### วัตถุประสงค์
รวม connection ทั้งหมดในที่เดียว — MeriPOS, Meta, LINE OA — ให้ร้านดูสถานะและ reconnect ได้

#### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| ไม่มี ML endpoint — เป็น Platform connection management | — | — |

#### Input / User Actions
- แต่ละ connection card มีปุ่ม "แก้ไข" / "ทดสอบ" / "ตัดการเชื่อมต่อ"

#### แสดงผล

**Connection Cards:**

**1. MeriPOS**

| Field | Label |
|-------|-------|
| สถานะ | ✅ Connected / ❌ Not Connected |
| Store ID | "Store: [MeriPOS Store ID]" |
| Last sync | "sync ล่าสุด: DD MMM YYYY HH:mm" |
| Auto sync frequency | "sync ทุก: [ความถี่]" |

ปุ่ม: "ทดสอบ Connection" / "แก้ไข Credentials" / "ตัดการเชื่อมต่อ"

**2. Meta Ads**

| Field | Label |
|-------|-------|
| สถานะ | ✅ Connected / ❌ Not Connected / ⚠️ Token หมดอายุ |
| Ad Account | "Act: [ad_account_id]" |
| Permissions | checklist: ads_management ✅ / ads_read ✅ |
| Token status | "Token: Valid" / "Token: Expired" |

ปุ่ม: "Reconnect Meta" (→ Screen 5.2) / "ทดสอบ" / "ตัดการเชื่อมต่อ"

**3. LINE OA**

| Field | Label |
|-------|-------|
| สถานะ | ✅ Connected / ❌ Not Connected |
| Channel ID | "Channel: [channel_id]" |
| Token status | "Token: Valid" |

ปุ่ม: "Reconnect LINE OA" (→ Screen 5.3) / "ทดสอบ" / "ตัดการเชื่อมต่อ"

#### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| ไม่มี connection เลย | "ยังไม่มีการเชื่อมต่อ — เพิ่ม connection เพื่อ unlock ฟีเจอร์ต่างๆ" |
| ตัดการเชื่อมต่อ Meta | confirm dialog: "การตัดการเชื่อมต่อจะหยุด ad performance pull — ยืนยัน?" |
| ตัดการเชื่อมต่อ MeriPOS | confirm dialog: "POS data จะไม่ sync อัตโนมัติ — ต้อง upload CSV เอง ยืนยัน?" |

---

## Error Handling ภาพรวม

| Error Code | HTTP | สิ่งที่แสดงใน UI |
|-----------|------|----------------|
| `INVALID_INPUT` | 400 | toast: "ข้อมูลไม่ถูกต้อง — กรุณาตรวจสอบและลองใหม่" + highlight field ที่ผิด |
| `PLAN_NOT_FOUND` | 404 | inline error พร้อมปุ่ม "สร้างแผนใหม่" |
| `STORE_MISMATCH` | 403 | "store_id ไม่ตรงกัน — กรุณา logout แล้ว login ใหม่" |
| `MODEL_NOT_READY` | 503 | banner สีเหลือง + ซ่อนปุ่ม generate plan + auto-retry |
| `CHANNEL_ERROR` | 502 | toast: "เชื่อมต่อ [LINE OA / Meta] ไม่ได้ตอนนี้ — ลองใหม่ภายหลัง" |
| `RETRAIN_FAILED` | 500 | "ระบบขัดข้องระหว่าง retrain — กรุณาติดต่อทีม Codera" |
| network timeout | — | toast: "หมดเวลาเชื่อมต่อ — ลองใหม่" + retry ปุ่ม |
| session expired | — | redirect Login + toast "Session หมดอายุ" |

---

## Notes สำหรับโชแปง

**Data ที่ Platform ต้องเตรียมก่อนเรียก ML API:**
- `pos_data` — ดึงจาก store data ที่ ingest ไว้ → encode base64 → ส่ง API (user ไม่เห็น process นี้)
- `crm_data` (optional) — เช่นเดียวกัน
- `store_id` — จาก session หลัง login

**IBM 10 Segments ที่ใช้ใน `segment_breakdown`:**
Champions / Loyal Customers / Potential Loyalist / Promising / New Customers / Need Attention / About to Sleep / At-Risk / Hibernating / Churn Risk

**Content generation stack:**
- Strategy + copy: Claude API (claude-sonnet-4-6)
- Image: image API — **pluggable** (Firefly / Stability AI / Ideogram / fal.ai — ยังไม่ตัดสินใจ) → ออกแบบ interface ให้ swap ได้ไม่ต้อง refactor

**Endpoints ที่ทำได้เลย:**
- `POST /campaign/plan` ✅
- `POST /campaign/analyze` ✅
- `POST /audience/export` ✅
- `GET /segment/health` ✅
- `GET /campaign/history` ✅
- `GET /model/status` ✅
- `POST /model/retrain` ✅
- `POST /campaign/record-outcome` ✅

**Endpoints ที่รอ research:**
- `POST /campaign/execute` ⏳ — รอ research LINE OA multicast format + Meta Ads API create campaign format

**Multi-tenant สิ่งที่ต้องระวัง:**
- ทุก query ต้อง filter ด้วย `store_id` จาก session — ห้าม cross-tenant data leak
- API Key ต่อ store — ไม่ share key ระหว่างร้าน
- Brand Kit, connections credentials ทุกอย่าง scope ต่อ `store_id`
