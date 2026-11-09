# PARTNER_SLIDES_DRAFT — Technical Briefing: ML Campaign Engine
**ผู้ฟัง:** ทีม Product & Engineering ของ Partner CRM
**วัตถุประสงค์:** ทำความเข้าใจระบบก่อนเริ่ม integrate เป็น API สำหรับ MVP
**วันที่:** พฤศจิกายน 2026

---

## Slide 1 — ระบบนี้ทำอะไร

> **"วิเคราะห์ข้อมูล POS → บอกว่าควรส่งโปรให้ใคร ด้วยอะไร เพราะอะไร"**

### Flow (6 กล่อง)

```
[ POS Data ]
     ↓
[ 6 Signals ]
  (ข้อมูลพฤติกรรมลูกค้า)
     ↓
[ 3 Questions ]
  (ส่งไหม? ส่งอะไร? ส่งเมื่อไหร่?)
     ↓
[ 1 Decision / คน ]
  (should_send + action + timing)
     ↓
[ Payload to CRM ]
  (JSON รายคน)
     ↓
[ Send ]
  (CRM ส่ง LINE/SMS ตาม send_datetime)
```

---

## Slide 2 — ต้องการข้อมูลอะไรบ้าง (Input)

### 6 คอลัมน์เท่านั้น

| คอลัมน์ | คำอธิบาย |
|---|---|
| `customer_id` | **hash ก่อนส่ง** — เราไม่เห็น id จริง |
| `timestamp` | วันเวลาที่ซื้อ |
| `total_price` | ยอดต่อบิล |
| `product_name` | ชื่อสินค้า |
| `category` | หมวดสินค้า |
| `is_treated` + `post_promo_visit` | (optional) ถ้ามี — โมเดลแม่นขึ้น |

### ไม่ต้องการ
- ไม่ต้องการ ชื่อ / เบอร์โทร / อีเมล / ที่อยู่
- ถ้ามีคอลัมน์เกิน → ระบบตัดทิ้งทันที ไม่เก็บ

> **นี่คือ feature ไม่ใช่ข้อจำกัด** — ลด PDPA exposure ตั้งแต่ต้น

---

## Slide 3 — ระบบวิเคราะห์อะไรบ้าง และทำไม (Core Logic)

### 6 Signals

| # | Signal | ตอบคำถาม |
|---|---|---|
| 1 | `avg_spent` + `top_category` | โปรนี้ร้านได้กำไรไหม? |
| 2 | `uplift_score` (ML) | โปรมีผลกับคนนี้ไหม? |
| 3 | `prob_control` (ML) | เขาจะซื้อเองอยู่แล้วไหม? |
| 4 | `cluster_segment` + `freq_trend` | คนนี้คือใคร กำลังห่างหายไหม? |
| 5 | `time_context` (peak/off-peak) | มักมาช่วงไหน? |
| 6 | `in_stock` | ของที่เขาชอบยังมีไหม? |

### 3 คำถามหลัก

1. **คนนี้คือใคร?** — Champions / At-Risk / Hibernating / ... (10 กลุ่ม IBM)
2. **โปรมีผลไหม?** — uplift > 0 และ margin ผ่านเกณฑ์
3. **ส่งได้ไหมตอนนี้?** — stock มี + ไม่ใช่ช่วงร้านยุ่ง

> **Speaker note:** uplift_score คือ "incremental lift" — ถ้าส่ง โอกาสซื้อเพิ่มขึ้นเท่าไหร่จากพฤติกรรมปกติของเขา ไม่ใช่แค่ "โอกาสซื้อ"

---

## Slide 4 — ระบบกรองยังไงก่อนส่ง (CFO Guardrail)

### Funnel

```
ลูกค้าทั้งหมด (100%)
         ↓
  [ กรองออก 3 กลุ่ม ]
  ┌─────────────────────────────────────────────────────┐
  │ 1. ซื้ออยู่แล้วโดยไม่ต้องจูงใจ (sure_thing → no discount)│
  │ 2. ส่วนลดไม่คุ้มต้นทุน (margin < 15%)               │
  │ 3. ของหมด / ร้านยุ่ง / peak อื่น                   │
  └─────────────────────────────────────────────────────┘
         ↓
  [ คนที่ "คุ้มส่ง" จริงๆ ]
  ประมาณ 50% ของ customer base*
```

> *ตัวเลขจากการทดสอบจริง Champion v7: send rate 50.1%

### ผลคือ
- ร้านไม่เสียเงินกับคนที่ซื้อเองอยู่แล้ว
- โปรทุกอันผ่าน profit check ก่อนส่ง
- กรณีของหมด → ไม่ส่งโปรที่เสนอไม่ได้จริง

---

## Slide 5 — ผลลัพธ์ที่ได้ (Output Payload)

### ตัวอย่าง JSON ต่อลูกค้า 1 คน

```json
{
  "customer_id": "hashed-id",
  "segment": "at_risk",
  "recommended_action": "Win-Back Offer",
  "send_datetime": "2026-11-08T10:30:00",
  "expected_profit": 84.50,
  "advisor_reason": "ห่างจากร้าน 47 วัน — ส่งโปรดึงกลับ",
  "final_source": "ml",
  "should_send": 1
}
```

### รูปแบบ
- 1 record ต่อ 1 ลูกค้า ต่อ 1 เดือน
- CRM ใช้ `send_datetime` เพื่อ trigger การส่ง
- `advisor_reason` ใช้แสดงใน dashboard หรือส่งเป็น message ได้เลย
- `should_send: 0` = ระบบแนะนำว่าไม่ส่งในรอบนี้

---

## Slide 6 — เอาไปต่อยอดทำอะไรได้บ้าง (Integration Use Cases)

- **CRM trigger** — ส่ง LINE / SMS อัตโนมัติตาม `send_datetime` ที่ระบบคำนวณให้
- **Dashboard** — แสดง segment breakdown + guardrail funnel ว่ากรองออกเพราะอะไร
- **Promo calendar** — operator ป้อนโปรที่วางแผน → ระบบตรวจว่าคุ้มไหม เปรียบ 3 scenario
- **Budget control** — ใส่ ceiling งบ → ระบบเลือกกลุ่มที่ ROI ดีที่สุดก่อน
- **Feedback loop** — เชื่อม POS หลังส่งโปร → โมเดลแม่นขึ้นรอบถัดไป (causal label จริง)

---

## Slide 7 — Integration Flow (Partner ต้องทำอะไร)

### 3 ขั้นตอน

**Step 1 — Upload**
- ส่ง POS CSV (6 คอลัมน์, hashed `customer_id`) ผ่าน API
- ระบบตรวจ schema ให้อัตโนมัติ

**Step 2 — รัน Analysis**
- เรียก endpoint `POST /v1/analyze`
- รับ payload JSON กลับ (1 record / ลูกค้า)

**Step 3 — ส่งโปร**
- CRM map `hashed_id` กลับเป็น customer จริงในระบบของ partner
- ส่งโปรตาม `send_datetime` และ `recommended_action`

> **Speaker note:** mapping กลับเป็น customer จริงเกิดในระบบ partner เท่านั้น เราไม่เคยเห็น identity จริง

---

## Slide 8 — ความปลอดภัยและ PDPA

### Role ของแต่ละฝ่าย

| ฝ่าย | บทบาท |
|---|---|
| ร้านค้า | Data Controller |
| Partner CRM | Data Processor |
| Codera (เรา) | Sub-processor |

### หลักการ
- เราไม่เคยเห็น customer ตัวจริง — รับแค่ hashed id
- ตัดคอลัมน์ที่ไม่ได้ขอออกทันทีตอน ingestion
- **DPA ต้องมีก่อนเริ่มส่งข้อมูลจริง**
- แยกข้อมูลต่อร้านค้า (tenant isolation) — ข้อมูลร้าน A ไม่ปะปนกับร้าน B

---

## Slide 9 — สิ่งที่ต้องรู้ก่อนใช้งาน (Limitations)

- **ตัวเลข expected_profit เป็นการประมาณ** — ไม่ใช่การรับประกัน ขึ้นกับ margin และ conversion จริง
- **ถ้าไม่มี `is_treated` / `post_promo_visit`** — โมเดลใช้ proxy label (แม่นน้อยกว่า แต่ยังใช้งานได้)
- **ผลดีขึ้นทุกรอบที่เชื่อม feedback** — รอบแรกอาจยังไม่ optimize เต็มที่
- **Multi-store pooled training ต้องได้รับ consent ก่อน** — ถ้าต้องการใช้ข้อมูลข้ามร้านเพื่อเพิ่มความแม่นยำ

---

## Slide 10 — Next Step

### ทำได้เลยตอนนี้

1. **ส่ง test dataset** → เราตรวจ schema ให้ฟรี และส่ง payload ตัวอย่างกลับ
2. **ดู payload ตัวอย่าง** → ทีม CRM ประเมิน integration effort
3. **เซ็น DPA** → เริ่มส่งข้อมูลจริง → ขอ API key

---

*เอกสารนี้เป็น draft สำหรับ partner briefing — ไม่สำหรับเผยแพร่สาธารณะ*
*ปรับปรุงล่าสุด: พฤศจิกายน 2026*
