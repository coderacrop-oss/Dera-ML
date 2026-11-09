# Codera — Screen Spec สำหรับ DotDash Integration
> สำหรับดิว (Repo 3 — DotDash/MeriPOS integration)
> อัปเดต: พฤศจิกายน 2026

---

## ข้อตกลงเบื้องต้น

- Codera เป็น **module ที่ build เข้าไปใน DotDash platform ที่มีอยู่แล้ว** — ไม่ใช่ app ใหม่
- DotDash จัดการ auth, store session, POS/CRM sync เองทั้งหมด — ดิวไม่ต้องทำ
- `store_id` ดึงจาก session context ของ DotDash ที่มีอยู่แล้ว ไม่ต้องให้ user กรอก
- ทุก API call ต้องส่ง header `X-API-Key` — DotDash manages this ตามระบบ auth เดิม
- POS data และ CRM data ที่ DotDash sync ไว้แล้ว ให้ encode เป็น base64 ก่อนส่ง API

---

## Overview — Module Structure

```
DotDash Sidebar / Nav
    └── Codera 🎯
            ├── Screen 1: Segment Health          ← entry point
            ├── Screen 5: Campaign History        ← accessible anytime
            └── [Flow: สร้างแคมเปญใหม่]
                    ├── Screen 2: Goal Comparison
                    ├── Screen 3: Plan Results
                    ├── Screen 4: Audience Export
                    └── Screen 6: Outcome Tracking (30 วันหลัง)
```

---

## Screen 1: Segment Health Dashboard

### วัตถุประสงค์
Entry point ของ Codera module — แสดงสถานะ segment ปัจจุบันของร้านและ ML recommendation ก่อนที่ user จะตัดสินใจสร้างแคมเปญ

### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `GET` | `/segment/health` | โหลด screen |
| `GET` | `/model/status` | โหลด screen (parallel) |

### Input
- ไม่มี user input — ดึงข้อมูลอัตโนมัติจาก `store_id` ใน session

### แสดงผล

**ส่วนบน — ML Engine Status** (จาก `GET /model/status`)

| Field | Label ที่แสดง |
|-------|--------------|
| `champion_version` | "โมเดลเวอร์ชัน" |
| `model_type` | "ประเภท" |
| `auuc` | "AUUC Score" |
| `calibration.trend` | badge: `stable` → สีเขียว / `degrading` → สีส้ม |
| `calibration.recommendation` | ข้อความ recommendation ใต้ badge |
| `promoted_at` | "อัปเดตล่าสุด" |

**ส่วนกลาง — Segment Overview** (จาก `GET /segment/health`)

แสดง segment cards แต่ละ card มี:

| Field | Label ที่แสดง |
|-------|--------------|
| `segment_name` | ชื่อ segment (เช่น "Champions", "At-Risk") |
| `customer_count` | จำนวนลูกค้า |
| `pct_of_total` | % ของลูกค้าทั้งหมด |
| `trend` | arrow icon: `growing` ↑ / `stable` → / `shrinking` ↓ |
| `avg_recency_days` | "ซื้อล่าสุดเมื่อ X วัน" |
| `avg_frequency` | "ซื้อเฉลี่ย X ครั้ง" |
| `avg_spend` | "ยอดเฉลี่ย ฿X" |

**ส่วนล่าง — ML Recommendation**

| Field | Label ที่แสดง |
|-------|--------------|
| `recommendation.priority_segment` | "แนะนำโฟกัส segment นี้ก่อน" |
| `recommendation.reason` | เหตุผล (เช่น "At-Risk กำลังโต 23% — ควรทำ win-back ก่อนลูกค้าหาย") |
| `recommendation.suggested_goal` | goal ที่ ML แนะนำ |

### Actions

| ปุ่ม | พาไปที่ |
|------|---------|
| **"สร้างแคมเปญ"** (primary CTA) | Screen 2: Goal Comparison |
| **"ดูประวัติแคมเปญ"** | Screen 5: Campaign History |
| **"Refresh"** icon | reload `GET /segment/health` + `/model/status` |

### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| `GET /segment/health` loading | skeleton cards ทุก segment |
| `MODEL_NOT_READY` (503) | banner สีเหลือง: "โมเดลยังไม่พร้อม — กำลังประมวลผล POS ครั้งแรก" ซ่อนปุ่ม "สร้างแคมเปญ" |
| `calibration.trend = "degrading"` | banner สีส้ม: "โมเดลเริ่มคลาดเคลื่อน — ควร retrain เมื่อมีข้อมูล POS เพิ่ม" |
| segment count = 0 | "ยังไม่มีข้อมูลลูกค้าเพียงพอ — POS ต้องมีอย่างน้อย 90 วัน" |
| network error | toast error + ปุ่ม retry |

---

## Screen 2: Goal Comparison

### วัตถุประสงค์
ก่อนที่ user จะเลือก goal ให้ ML วางแผนให้ — แสดงผลประมาณการทั้ง 4 goals พร้อมกัน เพื่อให้ร้านเปรียบเทียบและตัดสินใจได้ว่าวันนี้อยากได้อะไร

### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `POST` | `/campaign/plan` | เรียก 4 ครั้งพร้อมกัน (1 goal ต่อ 1 request) |

> เรียก `/campaign/plan` สำหรับ goal ทั้ง 4 แบบ parallel โดยใช้ budget เดิมที่ user กรอก

### Input

User กรอกก่อนที่จะเห็น comparison:

| Field | ประเภท | Required | หมายเหตุ |
|-------|--------|----------|---------|
| Budget (฿) | number input | ✅ | ส่งเป็น `budget` ใน request body |

Request body ที่ส่งสำหรับแต่ละ goal:
```json
{
  "store_id": "<จาก session>",
  "pos_data": "<base64 POS CSV จาก DotDash>",
  "mode": "budget",
  "budget": <user input>,
  "goal": "<maximize_profit | win_back | retain_loyal | maximize_reach>"
}
```

### แสดงผล

แสดง comparison cards 4 ใบ เรียงแบบ 2×2 grid หรือ horizontal scroll บนมือถือ

แต่ละ card แสดง:

| Field | Label ที่แสดง |
|-------|--------------|
| `goal_label` | ชื่อ goal ภาษาไทย (เช่น "กำไรสูงสุด") |
| `summary.customers_selected` | "เลือกลูกค้า X คน" |
| `summary.expected_incremental_profit` | "กำไรคาดการณ์ ฿X" (highlight สีเขียว) |
| `summary.roi` | "ROI X.Xx" |
| `summary.actual_spend` | "ใช้งบ ฿X" |
| `summary.budget_utilisation_pct` | progress bar แสดง % งบที่ใช้ |

**Goal labels:**

| `goal` value | Label ภาษาไทย |
|-------------|--------------|
| `maximize_profit` | กำไรสูงสุด ⭐ (default) |
| `win_back` | ดึงลูกค้ากลับ |
| `retain_loyal` | รักษาลูกค้าประจำ |
| `maximize_reach` | เข้าถึงคนมากที่สุด |

ML recommendation badge: highlight card ที่ให้ `expected_incremental_profit` สูงสุดด้วย badge "แนะนำ"

### Actions

| ปุ่ม | พฤติกรรม |
|------|---------|
| **"เลือก goal นี้"** ในแต่ละ card | โหลด Screen 3 ด้วย goal ที่เลือก |
| **"ปรับงบ"** | user แก้ budget → เรียก 4 API ใหม่ |
| **"ย้อนกลับ"** | กลับ Screen 1 |

### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| กำลังโหลด (4 API calls) | skeleton cards 4 ใบ พร้อม loading spinner |
| goal ใดโหลดสำเร็จก่อน | แสดง card นั้นได้เลย อีก card ค้างเป็น skeleton |
| goal ใด error | card นั้นแสดง "ไม่สามารถประมวลผล goal นี้ได้ — ลองใหม่" + retry ปุ่มเฉพาะ card |
| `viable_after_guardrail` = 0 | card แสดง "ลูกค้าที่ตรงเงื่อนไขไม่เพียงพอสำหรับ goal นี้" |
| budget น้อยเกินไปสำหรับทุก goal | แสดง banner แนะนำ minimum budget |

---

## Screen 3: Plan Results

### วัตถุประสงค์
แสดงรายละเอียดเต็มของแผนแคมเปญหลังจาก user เลือก goal แล้ว — ให้ร้านดู segment breakdown, action breakdown และ feasibility ก่อนตัดสินใจ export audience

### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `POST` | `/campaign/plan` | โหลด screen (ใช้ผลที่มีอยู่แล้วจาก Screen 2 ไม่เรียกซ้ำ) |
| `POST` | `/campaign/analyze` | ถ้า user กด "วิเคราะห์ก่อน" (optional flow) |

> ถ้ามาจาก Screen 2 ให้ reuse response เดิม ไม่ต้องเรียก `/campaign/plan` ซ้ำ

### Input
- Goal ที่เลือกจาก Screen 2 (ส่งต่อมาพร้อม plan response)
- ถ้า user กด "วิเคราะห์ก่อน": ไม่มี input เพิ่ม — ส่ง `plan_id` ให้ `/campaign/analyze` ประมวลผล

### แสดงผล

**Section 1 — Plan Summary**

| Field | Label ที่แสดง |
|-------|--------------|
| `plan_id` | Plan ID (แสดงเล็กๆ เป็น reference) |
| `goal_label` | ชื่อ goal ที่เลือก |
| `summary.customers_selected` | "ลูกค้าที่เลือก" |
| `summary.viable_after_guardrail` | "ผ่าน guardrail" (tooltip อธิบาย: "ML กรองลูกค้าที่โปรไม่คุ้มทุนออกแล้ว") |
| `summary.actual_spend` | "งบที่ใช้จริง ฿X" |
| `summary.expected_incremental_profit` | "กำไรคาดการณ์ ฿X" (ตัวเลขใหญ่ สีเขียว) |
| `summary.roi` | "ROI X.Xx" |
| `summary.budget_utilisation_pct` | "ใช้งบ X% จากทั้งหมด" |

**Section 2 — Segment Breakdown**

แสดงเป็น horizontal bar chart หรือ table จาก `segment_breakdown` object:

| Key (segment name) | Value |
|-------------------|-------|
| เช่น `"At-Risk"` | จำนวนลูกค้า |
| เช่น `"Champions"` | จำนวนลูกค้า |

**Section 3 — Action Breakdown**

แสดงเป็น table หรือ list จาก `action_breakdown` object:

| Key (action label) | Value |
|-------------------|-------|
| เช่น `"Win-Back: ส่วนลด 15%"` | จำนวนลูกค้า |
| เช่น `"VIP Reward: แจกแต้ม X2"` | จำนวนลูกค้า |

**Section 4 — Feasibility**

แสดงเฉพาะเมื่อ `mode = "target_profit"`:

| Field | Label |
|-------|-------|
| `feasibility.target_achievable` | badge: สำเร็จ / ไม่สำเร็จ |
| `feasibility.profit_gap` | "ขาด ฿X จากเป้า" (ถ้า not achievable) |
| `feasibility.max_achievable_profit` | "กำไรสูงสุดที่ทำได้ ฿X" |
| `feasibility.max_budget_needed` | "งบที่ต้องใช้ ฿X" |

**Section 5 — Analysis Results** (แสดงเฉพาะเมื่อ user กด "วิเคราะห์ก่อน")

แสดงผลจาก `POST /campaign/analyze` — 4 check cards:

| Check | Fields ที่แสดง |
|-------|--------------|
| Profitability | `checks.profitability.status` badge + `checks.profitability.message` + net profit |
| Overlap | `checks.overlap.status` badge + `checks.overlap.overlap_count` คน + message |
| Segment Readiness | `checks.segment_readiness.status` badge + trend + message |
| Budget Efficiency | `checks.budget_efficiency.status` badge + gain ที่เพิ่มได้ + message |

`overall_verdict` แสดงที่ด้านบนของ section:
- `recommended` → banner สีเขียว "แผนนี้น่าทำ"
- `proceed_with_caution` → banner สีเหลือง "ทำได้ แต่มีข้อควรระวัง"
- `not_recommended` → banner สีแดง "ML ไม่แนะนำแผนนี้"

ถ้ามี `suggestion` แสดงกล่องแยก:
- `suggestion.message`
- `suggestion.adjusted_budget` / `adjusted_estimated_profit` / `adjusted_roi`

### Actions

| ปุ่ม | พฤติกรรม |
|------|---------|
| **"วิเคราะห์ก่อน"** (secondary) | เรียก `POST /campaign/analyze` → แสดง Section 5 ใน screen เดิม |
| **"Export Audience"** (primary) | ไป Screen 4 — ส่ง `plan_id` ต่อ |
| **"เลือก goal อื่น"** | กลับ Screen 2 |
| **"ย้อนกลับ"** | กลับ Screen 1 |

> `can_proceed = true` เสมอ — ร้านยังกด Export ได้แม้ overall_verdict = not_recommended แต่ให้แสดง confirmation dialog ก่อน

### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| `action_breakdown` เป็น empty object | "ไม่มี action สำหรับแคมเปญนี้" |
| `segment_breakdown` เป็น empty object | "ไม่มี segment ที่ตรงเงื่อนไข" |
| `overall_verdict = "not_recommended"` แล้วกด Export | confirmation dialog: "ML ไม่แนะนำแผนนี้ — ต้องการดำเนินต่อหรือไม่?" |
| `/campaign/analyze` loading | inline spinner ที่ Section 5 ไม่ block ทั้ง screen |
| `/campaign/analyze` error | "วิเคราะห์ไม่ได้ตอนนี้ — ดำเนินต่อหรือ retry?" |

---

## Screen 4: Audience Export

### วัตถุประสงค์
หลังอนุมัติแผนแล้ว — export targeting audiences สำหรับ LINE OA และ Meta Ads พร้อม download ไฟล์

### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `POST` | `/audience/export` | user กด "Export" |

### Input

Request body ที่ส่งอัตโนมัติ — user ไม่ต้องกรอก:
```json
{
  "plan_id": "<จาก Screen 3>",
  "store_id": "<จาก session>",
  "crm_data": "<base64 CRM CSV จาก DotDash ถ้ามี>"
}
```

> `crm_data` ใส่ถ้า DotDash มี CRM sync อยู่แล้ว — ถ้าไม่มีให้ส่งโดยไม่มี field นี้ (API จะ fallback เป็น hashed customer_id)

### แสดงผล

**Section 1 — Export Summary**

| Field | Label ที่แสดง |
|-------|--------------|
| `export_id` | Export ID (reference) |
| `summary.total_customers` | "ลูกค้าทั้งหมด X คน" |
| `summary.match_rate_pct` | "Match rate X%" (tooltip: "% ลูกค้าที่จับคู่กับ LINE UID หรือ hashed contact ได้") |
| `summary.custom_audience` | "Custom Audience X คน" |
| `summary.lookalike_seed` | "Lookalike Seed X คน" |
| `summary.exclusion` | "Exclusion X คน" |

**Section 2 — LINE OA**

| Field | Label ที่แสดง |
|-------|--------------|
| `audiences.line_oa.count` | "ส่งถึงลูกค้า X คนใน LINE OA" |
| `audiences.line_oa.send_list` | ไม่แสดง raw list — แสดงเฉพาะจำนวน |

> LINE OA send_list นี้ใช้ส่งต่อให้ DotDash/LINE OA integration ที่มีอยู่แล้ว ดิวเชื่อมกับระบบ LINE OA ที่ DotDash มีตามที่ตกลงไว้

**Section 3 — Meta Ads Files**

| Field | Label ที่แสดง | Action |
|-------|--------------|--------|
| `audiences.meta.custom_audience_file` | "Custom Audience" | ปุ่ม Download CSV |
| `audiences.meta.lookalike_seed_file` | "Lookalike Seed" | ปุ่ม Download CSV |
| `audiences.meta.exclusion_file` | "Exclusion" | ปุ่ม Download CSV |

**Section 4 — Per-Segment Breakdown**

Table จาก `per_segment` object:

| Column | ข้อมูล |
|--------|--------|
| Segment | ชื่อ segment (key ของ `per_segment`) |
| Custom | `per_segment[segment].custom` คน |
| Lookalike | `per_segment[segment].lookalike` คน |
| Exclusion | `per_segment[segment].exclusion` คน |

**Section 5 — Warnings**

แสดงเฉพาะเมื่อ `warnings` array ไม่ว่าง — แสดงเป็น list สีส้ม

### Actions

| ปุ่ม | พฤติกรรม |
|------|---------|
| **"Download Custom Audience"** | download `meta.custom_audience_file` |
| **"Download Lookalike Seed"** | download `meta.lookalike_seed_file` |
| **"Download Exclusion List"** | download `meta.exclusion_file` |
| **"Download ทั้งหมด"** | download ทั้ง 3 ไฟล์ |
| **"ดูประวัติแคมเปญ"** | ไป Screen 5 |
| **"เสร็จสิ้น"** | กลับ Screen 1 |

### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| กำลัง export (loading) | progress indicator + "กำลัง generate ไฟล์..." ห้ามกด Export ซ้ำ |
| `PLAN_NOT_FOUND` (404) | "แผนนี้หมดอายุหรือไม่มีในระบบ — กลับไปสร้างแผนใหม่" |
| `STORE_MISMATCH` (403) | "store_id ไม่ตรงกัน — กรุณาติดต่อทีม Codera" |
| `summary.match_rate_pct` < 50% | warning banner: "Match rate ต่ำ — CRM อาจไม่มี LINE UID ครบ" |
| `audiences.line_oa.count` = 0 | warning: "ไม่มี LINE UID ใน CRM — ส่ง LINE OA ไม่ได้ แต่ Meta export ยังได้" |
| network error ระหว่าง export | "Export ล้มเหลว — ลองใหม่" + retry ปุ่ม |

---

## Screen 5: Campaign History

### วัตถุประสงค์
แสดงประวัติแคมเปญทั้งหมดของร้าน — ทั้งที่กำลังดำเนินการและที่ผ่านมา เพื่อให้ติดตามผลและวางแผนครั้งถัดไป

### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `GET` | `/campaign/history` | โหลด screen |

### Input
- ไม่มี user input — ดึงข้อมูลอัตโนมัติจาก `store_id` ใน session
- Optional filter: date range picker (client-side filter บน data ที่ได้มาแล้ว)

### แสดงผล

**ตาราง Campaign History**

| Field | Label ที่แสดง |
|-------|--------------|
| `plan_id` | Plan ID |
| `goal_label` | Goal |
| `generated_at` | วันที่สร้าง (format: DD MMM YYYY HH:mm) |
| `summary.customers_selected` | จำนวนลูกค้า |
| `summary.actual_spend` | งบที่ใช้ (฿) |
| `summary.expected_incremental_profit` | กำไรคาดการณ์ (฿) |
| `status` | badge: `draft` / `exported` / `executed` / `outcome_recorded` |

**Status badge colors:**
- `draft` → สีเทา "แบบร่าง"
- `exported` → สีฟ้า "Export แล้ว"
- `executed` → สีเขียว "ส่งแล้ว"
- `outcome_recorded` → สีม่วง "วัดผลแล้ว"

### Actions

| ปุ่ม / Interaction | พฤติกรรม |
|------------------|---------|
| คลิก row | ขยาย accordion แสดง segment_breakdown และ action_breakdown ของแคมเปญนั้น |
| **"ดูผลลัพธ์"** (เฉพาะ row ที่ status = `outcome_recorded`) | ไป Screen 6 — ส่ง `execute_id` |
| **"บันทึกผล"** (เฉพาะ row ที่ status = `executed` และผ่านมา ≥ 30 วัน) | ไป Screen 6 — mode: กรอกข้อมูล |
| **"สร้างแคมเปญใหม่"** | ไป Screen 2 |
| **"Refresh"** icon | reload `GET /campaign/history` |

### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| ไม่มีประวัติเลย | illustration + "ยังไม่มีแคมเปญ — เริ่มสร้างแคมเปญแรกของคุณ" + ปุ่ม "สร้างแคมเปญ" |
| กำลังโหลด | skeleton table 5 rows |
| network error | toast error + retry |
| row ที่ status = `executed` แต่ < 30 วัน | ไม่แสดงปุ่ม "บันทึกผล" — แสดง tooltip "รอครบ 30 วันก่อนวัดผล (เหลือ X วัน)" |

---

## Screen 6: Outcome Tracking

### วัตถุประสงค์
30 วันหลังจาก execute แคมเปญ — ให้ร้านส่ง POS ล่าสุดมาวัด actual lift เทียบกับที่ ML predict ไว้ และดูผลความแม่นยำของโมเดล

### API ที่เรียก

| Method | Path | เวลาที่เรียก |
|--------|------|-------------|
| `POST` | `/campaign/record-outcome` | user กด "บันทึกผลแคมเปญ" |

### Input

> DotDash sync POS อัตโนมัติอยู่แล้ว — ดิวดึง POS ช่วง 30 วันหลัง execute แล้ว encode base64 ส่ง API

Request body ที่ส่ง:
```json
{
  "execute_id": "<จาก campaign history>",
  "store_id": "<จาก session>",
  "post_pos_data": "<base64 POS CSV 30 วันหลัง execute>",
  "observation_days": 30
}
```

**Form ที่ user เห็น (อธิบาย context ไม่ใช่ input จริง):**
- แสดงข้อมูล campaign ที่กำลังวัดผล: วันที่ execute, goal, จำนวนลูกค้า
- แสดง date range ของ POS ที่จะใช้วัด: "POS วันที่ X ถึง Y"
- ปุ่ม confirm ก่อนส่ง

### แสดงผล

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
| `result.calibration_ratio` | "Calibration Ratio X.XXx" |
| `result.trend` | badge: `improving` → สีเขียว / `stable` → สีน้ำเงิน / `degrading` → สีส้ม |
| `result.recommendation` | ข้อความแนะนำ |

**Calibration ratio interpretation:**
- 0.90–1.10 → "โมเดลแม่นมาก" สีเขียว
- 0.80–0.90 หรือ 1.10–1.20 → "แม่นพอใช้" สีเหลือง
- < 0.80 หรือ > 1.20 → "โมเดลคลาดเคลื่อน" สีส้ม

**Section 3 — Lift Analysis**

| Field | Label ที่แสดง |
|-------|--------------|
| `comparison.baseline_conversion_rate` | "Conversion Rate กลุ่มปกติ X%" |
| `comparison.treated_conversion_rate` | "Conversion Rate กลุ่มที่ได้รับ promo X%" |
| `comparison.lift_pct` | "Lift +X%" (ตัวเลขใหญ่ สีเขียว) |

แสดงเป็น before/after bar comparison chart

**Section 4 — Model Health (ถ้า `result.trend = "degrading"`)**

Banner สีส้ม: "โมเดลเริ่มคลาดเคลื่อน — ควรรัน retrain ด้วย POS ล่าสุด" + ปุ่ม "แจ้งทีม Codera"

### Actions

| ปุ่ม | พฤติกรรม |
|------|---------|
| **"บันทึกผลแคมเปญ"** (ก่อนส่ง) | เรียก `POST /campaign/record-outcome` |
| **"ดูประวัติแคมเปญ"** (หลังบันทึก) | กลับ Screen 5 |
| **"สร้างแคมเปญใหม่"** (หลังบันทึก) | ไป Screen 2 |
| **"แจ้งทีม Codera"** (ถ้า degrading) | trigger notification ไปหาทีม Codera (implement ตาม channel ที่ DotDash มี) |

### Edge Cases / Empty States

| กรณี | สิ่งที่แสดง |
|------|-----------|
| กำลังประมวลผล | "กำลังวิเคราะห์ผลแคมเปญ..." full-screen loading (อาจใช้เวลา 10–30 วินาที) |
| `PLAN_NOT_FOUND` (404) | "ไม่พบข้อมูลแคมเปญนี้ — กรุณาติดต่อทีม Codera" |
| POS data ไม่ครบ 30 วัน | warning: "POS มีข้อมูลเพียง X วัน — ผลที่ได้อาจไม่ครบถ้วน" ยังส่งได้ |
| `result.lift_pct` < 0 | แสดง negative lift สีแดง + "โปรครั้งนี้ไม่ได้ผลตามที่คาด — ML จะ learn จากข้อมูลนี้" |
| ส่งผลซ้ำ (execute_id เดิม) | "คุณบันทึกผลแคมเปญนี้ไปแล้ว" แสดงผลเดิมแทน |

---

## Error Handling ภาพรวม

| Error Code | HTTP | สิ่งที่แสดงใน UI |
|-----------|------|----------------|
| `INVALID_INPUT` | 400 | toast: "ข้อมูลไม่ถูกต้อง — กรุณาตรวจสอบและลองใหม่" |
| `PLAN_NOT_FOUND` | 404 | inline error ใน screen พร้อมปุ่ม "สร้างแผนใหม่" |
| `STORE_MISMATCH` | 403 | "store_id ไม่ตรงกัน — กรุณาติดต่อทีม Codera" (ไม่มี retry) |
| `MODEL_NOT_READY` | 503 | banner สีเหลือง + ซ่อนปุ่ม generate plan |
| `CHANNEL_ERROR` | 502 | toast: "เชื่อมต่อ LINE OA หรือ Meta ไม่ได้ตอนนี้ — ลองใหม่ภายหลัง" |
| `RETRAIN_FAILED` | 500 | "ระบบขัดข้อง — กรุณาแจ้งทีม Codera" |
| network timeout | — | toast: "หมดเวลาเชื่อมต่อ — ลองใหม่" + retry ปุ่ม |

---

## Notes สำหรับดิว

**Data ที่ DotDash ต้องเตรียมก่อนเรียก API:**
- `pos_data` — POS CSV encode base64 (columns: `customer_id`, `transaction_id`, `timestamp`, `total_price`, `product_name`, `category`)
- `crm_data` (optional) — CRM CSV encode base64 (columns: `customer_id`, `hashed_email`, `hashed_phone`, `line_uid`)
- `store_id` — ดึงจาก session context ที่มีอยู่แล้ว

**IBM 10 Segments ที่จะเจอใน `segment_breakdown`:**

Champions / Loyal Customers / Potential Loyalist / Promising / Need Attention / About to Sleep / At-Risk / Can't Lose Them / Hibernating / Lost

**Endpoints ที่พร้อมใช้แล้ว:**
- `POST /campaign/plan` ✅
- `POST /campaign/analyze` ✅
- `POST /audience/export` ✅
- `GET /model/status` ✅
- `GET /segment/health` ✅
- `GET /campaign/history` ✅
- `POST /campaign/record-outcome` ✅

**Endpoints ที่ยังไม่พร้อม:**
- `POST /campaign/execute` ⏳ — รอโชแปง research LINE OA multicast + Meta upload format ก่อน

**Endpoints ที่ไม่ต้องใช้ใน DotDash UI:**
- `POST /model/retrain` — เป็น internal operation ของทีม Codera ไม่ expose ให้ร้านกด
