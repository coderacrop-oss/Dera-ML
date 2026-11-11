# Codera ML Engine — API Spec v1
> สำหรับโชแปง (Repo 2 — Codera Platform)
> อัปเดต: พฤศจิกายน 2026

---

## Overview

Repo 2 expose REST API ให้ DotDash (Repo 3) และ platform ของเราเรียกใช้
Repo 1 (ML engine) รันอยู่ภายใน Repo 2 — ไม่ expose ตรง

```
DotDash UI / Codera Platform
        ↓  REST API (Repo 2)
    Codera ML Engine (Repo 1)
```

## Authentication

ทุก endpoint ต้องส่ง API key ใน header:
```
X-API-Key: <key>
```
โชแปงจัดการ key management อยู่แล้ว — ใช้ระบบเดิม

---

## Endpoints

---

### 1. POST /campaign/plan

**วัตถุประสงค์**: รัน ML pipeline → ได้แผนแคมเปญ พร้อม segment breakdown

**ใช้เมื่อ**: ร้านกรอก goal + งบ แล้วกด "Generate Plan"

#### Request Body

```json
{
  "store_id": "store_abc",
  "pos_data": "<base64-encoded CSV string>",
  "mode": "budget",
  "budget": 5000,
  "goal": "maximize_profit"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `store_id` | string | ✅ | identifier ของร้าน |
| `pos_data` | string | ✅ | POS CSV encoded base64 — columns: customer_id, transaction_id, timestamp, total_price, product_name, category (+ quantity optional) |
| `mode` | enum | ✅ | `"budget"` หรือ `"target_profit"` |
| `budget` | float | ถ้า mode=budget | งบสูงสุดในหน่วยบาท |
| `target_profit` | float | ถ้า mode=target_profit | กำไรที่ต้องการในหน่วยบาท |
| `goal` | enum | ❌ (default: maximize_profit) | `maximize_profit` / `win_back` / `retain_loyal` / `maximize_reach` |
| `point_value_thb` | float > 0 | ❌ | ต้นทุนของ 1 แต้มสำหรับร้านนี้ (บาท) ถ้าส่งมา ข้อเสนอแบบแต้มจะคิดต้นทุน = แต้ม × point_value_thb × redemption_rate และกลุ่มนั้นมี `cost_source: "shop_provided"` ถ้าไม่ส่งใช้ค่าประมาณ 2% ของยอดซื้อเฉลี่ย (`cost_source: "estimated"`) |
| `redemption_rate` | float (0, 1] | ❌ (ต้องส่งคู่กับ `point_value_thb`; default 1.0) | สัดส่วนแต้มที่ลูกค้าแลกจริง |

#### Response 200

```json
{
  "plan_id": "plan_20261106_143022",
  "store_id": "store_abc",
  "generated_at": "2026-11-06T14:30:22Z",
  "mode": "budget",
  "goal": "maximize_profit",
  "goal_label": "กำไรสูงสุด (default)",
  "summary": {
    "total_customers": 1200,
    "viable_after_guardrail": 380,
    "customers_selected": 210,
    "actual_spend": 4850.00,
    "expected_incremental_profit": 28400.00,
    "roi": 5.86,
    "budget_utilisation_pct": 97.0
  },
  "action_breakdown": {
    "Win-Back: ส่วนลด 15%": 82,
    "VIP Reward: แจกแต้มพิเศษ 100 แต้ม": 64,
    "Early Win-Back: ส่วนลดพิเศษ": 41,
    "Welcome: ยินดีต้อนรับ": 23
  },
  "segment_breakdown": {
    "At-Risk": 95,
    "Champions": 64,
    "Churn Risk": 51
  },
  "feasibility": {
    "target_achievable": null,
    "profit_gap": null,
    "max_achievable_profit": 28400.00,
    "max_budget_needed": 4850.00
  },
  "dotdash_segments": [
    {
      "segment_name": "Potential Loyalist",
      "action_key": "potloyal_habit_builder",
      "recommended_action": "Habit Builder: สะสมแต้ม เพิ่มความถี่",
      "customer_count": 76,
      "customer_ids": [
        "c095",
        "c060",
        "..."
      ],
      "expected_profit": 20.67,
      "actual_spend": 16.5,
      "dotdash_offer": {
        "offer_type": "points",
        "points_amount": 60,
        "min_spend": 0,
        "expiry_days": 30,
        "action_key": "potloyal_habit_builder"
      }
    },
    {
      "segment_name": "Champions",
      "action_key": "champ_vip_bundle",
      "recommended_action": "VIP Bundle: เซตคู่คุ้มค่า (Bundle)",
      "customer_count": 31,
      "customer_ids": [
        "c041",
        "c098",
        "..."
      ],
      "expected_profit": 9.8,
      "actual_spend": 7.81,
      "dotdash_offer": {
        "offer_type": "points",
        "points_amount": 150,
        "min_spend": 0,
        "expiry_days": 30,
        "action_key": "champ_vip_bundle"
      }
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `dotdash_segments` | array — **1 กลุ่ม = 1 ของรางวัล**: แบ่งตาม (segment_name, action_key) ลูกค้าในกลุ่มได้ `dotdash_offer` เดียวกันทั้งหมด. ฟิลด์: `segment_name`, `action_key` (id เสถียรของกฎ ใช้จับคู่แทนชื่อ action), `recommended_action` (ข้อความที่พบบ่อยในกลุ่ม แสดงผลเท่านั้น), `customer_count`, `customer_ids`, `expected_profit`, `actual_spend`, `dotdash_offer` (`offer_type`, `points_amount` หรือ `coupon_type`+`coupon_value`, `min_spend`, `expiry_days`, `action_key`, **`bundle_items`** (คู่สินค้าที่พบบ่อยที่สุดในกลุ่ม หรือ null), **`bundle_coverage`** (สัดส่วนลูกค้าในกลุ่มนี้ (segment × action) ที่คู่นี้เป็นคู่ของตัวเอง 0–1; ลูกค้าที่ไม่มีคู่นับในตัวหาร), **`bundle_low_coverage`** (true เมื่อ coverage ต่ำกว่า `bundle_low_coverage_threshold` = 0.5)) และ **`cost_source`** ระดับกลุ่ม (`estimated` | `shop_provided`). ตัวอย่างข้างบนมาจากการรันจริงบนข้อมูลทดสอบ (fixture) ตัดรายชื่อลูกค้าให้สั้น |

#### Response 400

```json
{
  "error": "INVALID_INPUT",
  "message": "ต้องระบุ budget หรือ target_profit อย่างใดอย่างหนึ่ง"
}
```

---

### 2. POST /campaign/analyze

**วัตถุประสงค์**: ร้านกรอกแผนเอง → ML วิเคราะห์ว่าคุ้มทุนไหม ซ้ำซ้อนไหม timing ดีไหม พร้อม suggest ปรับ

**ใช้เมื่อ**: ร้านกด "วิเคราะห์ก่อน" หลังกรอกแผนเอง ก่อนกด Generate จริง

#### Request Body

```json
{
  "store_id": "store_abc",
  "pos_data": "<base64-encoded CSV string>",
  "proposed_plan": {
    "goal": "win_back",
    "mode": "budget",
    "budget": 5000,
    "segments": ["At-Risk", "Churn Risk"],
    "promo_type": "discount_20",
    "scheduled_date": "2026-11-15"
  },
  "campaign_history": [
    {
      "customer_id": "C001",
      "last_promo_date": "2026-10-20",
      "promo_type": "discount_15"
    }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `store_id` | string | ✅ | identifier ของร้าน |
| `pos_data` | string | ✅ | POS CSV encoded base64 |
| `proposed_plan` | object | ✅ | แผนที่ร้านกรอกเอง |
| `proposed_plan.goal` | enum | ✅ | `maximize_profit` / `win_back` / `retain_loyal` / `maximize_reach` |
| `proposed_plan.mode` | enum | ✅ | `budget` / `target_profit` |
| `proposed_plan.budget` | float | ถ้า mode=budget | งบสูงสุด |
| `proposed_plan.segments` | array | ❌ | segment ที่ต้องการ target — ถ้าไม่ส่ง ML เลือกให้ |
| `proposed_plan.promo_type` | string | ❌ | เช่น `discount_20`, `points_x2` |
| `proposed_plan.scheduled_date` | string (ISO 8601) | ❌ | วันที่ตั้งใจยิง — ใช้ตรวจ overlap |
| `campaign_history` | array | ❌ | ประวัติ promo ที่ลูกค้าเคยได้ — ถ้าไม่ส่งจะข้ามการตรวจ overlap |

#### Response 200

```json
{
  "analysis_id": "analysis_20261106_150000",
  "store_id": "store_abc",
  "analyzed_at": "2026-11-06T15:00:00Z",
  "proposed_plan": {
    "goal": "win_back",
    "mode": "budget",
    "budget": 5000,
    "segments": ["At-Risk", "Churn Risk"]
  },
  "checks": {
    "profitability": {
      "status": "warning",
      "message": "แผนนี้จะขาดทุน ฿2,300 ถ้าให้ส่วนลด 20% กับทุกคนใน segment",
      "estimated_cost": 8500.00,
      "estimated_profit": 6200.00,
      "net": -2300.00
    },
    "overlap": {
      "status": "warning",
      "message": "ลูกค้า 34 คนเพิ่งได้รับ promo เมื่อน้อยกว่า 21 วัน",
      "overlap_count": 34,
      "min_days_since_last_promo": 18
    },
    "segment_readiness": {
      "status": "ok",
      "message": "At-Risk และ Churn Risk กำลังโต — timing เหมาะสม",
      "at_risk_size": 95,
      "churn_risk_size": 51,
      "trend": "growing"
    },
    "budget_efficiency": {
      "status": "suggestion",
      "message": "ถ้า allocate งบตาม uplift จริงแทนแบ่งเท่ากัน profit เพิ่มขึ้น ฿4,200",
      "current_estimated_profit": 6200.00,
      "optimized_estimated_profit": 10400.00,
      "gain": 4200.00
    }
  },
  "overall_verdict": "not_recommended",
  "suggestion": {
    "message": "แนะนำให้ลด promo เป็น discount_15 และกรอง 34 คนที่เพิ่งได้ promo ออก",
    "adjusted_budget": 4100.00,
    "adjusted_estimated_profit": 9800.00,
    "adjusted_roi": 2.39
  },
  "can_proceed": true
}
```

| Field | Description |
|-------|-------------|
| `checks.profitability.status` | `ok` / `warning` / `danger` |
| `checks.overlap.status` | `ok` / `warning` |
| `checks.segment_readiness.status` | `ok` / `warning` |
| `checks.budget_efficiency.status` | `ok` / `suggestion` |
| `overall_verdict` | `recommended` / `not_recommended` / `proceed_with_caution` |
| `can_proceed` | always `true` — ร้านยังมีสิทธิ์ยืนยันแผนเดิมได้แม้ ML warn |

#### Response 400

```json
{
  "error": "INVALID_INPUT",
  "message": "proposed_plan.goal ต้องเป็น maximize_profit / win_back / retain_loyal / maximize_reach"
}
```

---

### 3. POST /audience/export

**วัตถุประสงค์**: เอา plan_id ที่ได้จาก /campaign/plan → export targeting audiences สำหรับ LINE OA และ Meta

**ใช้เมื่อ**: ร้านกด "Export Audiences" หลังดูแผนแล้ว ก่อนกด Execute

#### Request Body

```json
{
  "plan_id": "plan_20261106_143022",
  "store_id": "store_abc",
  "crm_data": "<base64-encoded CSV string>"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `plan_id` | string | ✅ | จาก /campaign/plan response |
| `store_id` | string | ✅ | ต้องตรงกับที่ใช้ตอน plan |
| `crm_data` | string | ❌ | CRM CSV encoded base64 — columns: customer_id, hashed_email, hashed_phone, line_uid — ถ้าไม่ส่งจะ fallback เป็น hashed customer_id |

#### Response 200

```json
{
  "export_id": "export_20261106_143055",
  "plan_id": "plan_20261106_143022",
  "store_id": "store_abc",
  "generated_at": "2026-11-06T14:30:55Z",
  "summary": {
    "total_customers": 210,
    "custom_audience": 134,
    "lookalike_seed": 51,
    "exclusion": 390,
    "match_rate_pct": 88.5
  },
  "audiences": {
    "line_oa": {
      "send_list": ["Uline00000001", "Uline00000002"],
      "count": 185
    },
    "meta": {
      "custom_audience_file": "exports/export_20261106_143055/all_custom.csv",
      "lookalike_seed_file": "exports/export_20261106_143055/all_lookalike.csv",
      "exclusion_file": "exports/export_20261106_143055/all_exclusion.csv"
    }
  },
  "per_segment": {
    "At-Risk": {
      "custom": 95,
      "lookalike": 0,
      "exclusion": 42
    },
    "Champions": {
      "custom": 0,
      "lookalike": 51,
      "exclusion": 20
    }
  },
  "warnings": []
}
```

---

### 4. POST /campaign/execute

**วัตถุประสงค์**: ร้านกด Execute → ส่ง LINE OA broadcast + upload Meta audience

**ใช้เมื่อ**: approve แผนแล้ว พร้อมส่งจริง

> ⚠️ Endpoint นี้ต้องการ LINE OA + Meta API ที่โชแปง research — implement หลัง research เสร็จ

#### Request Body

```json
{
  "export_id": "export_20261106_143055",
  "store_id": "store_abc",
  "channels": ["line_oa", "meta"],
  "line_oa_config": {
    "channel_access_token": "<token>",
    "message": {
      "type": "text",
      "text": "สวัสดี! มีโปรพิเศษสำหรับคุณวันนี้ 🎉"
    }
  },
  "meta_config": {
    "ad_account_id": "act_123456789",
    "access_token": "<token>"
  }
}
```

#### Response 200

```json
{
  "execute_id": "exec_20261106_143120",
  "store_id": "store_abc",
  "executed_at": "2026-11-06T14:31:20Z",
  "results": {
    "line_oa": {
      "status": "sent",
      "sent_count": 185,
      "failed_count": 2,
      "message_id": "mgs_xxxx"
    },
    "meta": {
      "status": "uploaded",
      "custom_audience_id": "23849...",
      "lookalike_audience_id": "23850...",
      "exclusion_audience_id": "23851..."
    }
  }
}
```

---

### 5. GET /segment/health

**วัตถุประสงค์**: ดู segment snapshot ปัจจุบันพร้อม trend และ ML recommendation — เร็วกว่า `/campaign/plan` เพราะไม่รัน optimizer

**ใช้เมื่อ**: โหลด Segment Health Dashboard ใน DotDash

#### Request

```
GET /segment/health?store_id=store_abc
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `store_id` | string | ✅ | identifier ของร้าน |

#### Response 200

```json
{
  "store_id": "store_abc",
  "snapshot_at": "2026-11-06T15:00:00Z",
  "total_customers": 1200,
  "segments": {
    "Champions":          { "count": 64,  "prev_count": 59,  "trend": "growing",  "trend_pct": 8.5,  "urgency_score": 42, "priority": "monitor",  "priority_color": "green",  "action_label": "รักษา" },
    "Loyal Customers":    { "count": 112, "prev_count": 115, "trend": "stable",   "trend_pct": -2.6, "urgency_score": 36, "priority": "low",      "priority_color": "green",  "action_label": "รักษา" },
    "Potential Loyalist": { "count": 98,  "prev_count": 90,  "trend": "growing",  "trend_pct": 8.9,  "urgency_score": 55, "priority": "medium",   "priority_color": "yellow", "action_label": "ทำได้" },
    "Promising":          { "count": 73,  "prev_count": 71,  "trend": "stable",   "trend_pct": 2.8,  "urgency_score": 47, "priority": "medium",   "priority_color": "yellow", "action_label": "ทำได้" },
    "New Customers":      { "count": 23,  "prev_count": 18,  "trend": "growing",  "trend_pct": 27.8, "urgency_score": 44, "priority": "medium",   "priority_color": "yellow", "action_label": "ทำได้" },
    "Need Attention":     { "count": 87,  "prev_count": 80,  "trend": "growing",  "trend_pct": 8.8,  "urgency_score": 68, "priority": "high",     "priority_color": "orange", "action_label": "ควรทำ" },
    "About to Sleep":     { "count": 61,  "prev_count": 63,  "trend": "stable",   "trend_pct": -3.2, "urgency_score": 72, "priority": "high",     "priority_color": "orange", "action_label": "ควรทำ" },
    "At-Risk":            { "count": 95,  "prev_count": 83,  "trend": "growing",  "trend_pct": 14.5, "urgency_score": 88, "priority": "critical", "priority_color": "red",    "action_label": "ทำก่อน" },
    "Hibernating":        { "count": 136, "prev_count": 140, "trend": "stable",   "trend_pct": -2.9, "urgency_score": 52, "priority": "medium",   "priority_color": "yellow", "action_label": "ทำได้" },
    "Can't Lose Them":    { "count": 51,  "prev_count": 38,  "trend": "growing",  "trend_pct": 34.2, "urgency_score": 95, "priority": "critical", "priority_color": "red",    "action_label": "ทำก่อน" },
    "Lost":               { "count": 0,   "prev_count": 0,   "trend": "none",     "trend_pct": null, "urgency_score": 0,  "priority": "monitor",  "priority_color": "gray",   "action_label": "—" }
  },
  "health_score": {
    "healthy_pct": 39.0,
    "at_risk_pct": 12.4,
    "churned_pct": 4.2,
    "label": "ดี"
  },
  "alerts": [
    {
      "level": "warning",
      "segment": "Can't Lose Them",
      "message": "Can't Lose Them โตขึ้น 34% จากเดือนที่แล้ว — แนะนำทำ Win-Back โดยเร็ว"
    },
    {
      "level": "info",
      "segment": "At-Risk",
      "message": "At-Risk เพิ่มขึ้น 14% — ควรวางแผน campaign ในเดือนนี้"
    }
  ],
  "recommendation": {
    "goal": "win_back",
    "reason": "Can't Lose Them และ At-Risk กำลังโต — Win-Back จะให้ ROI สูงสุดในสถานการณ์นี้",
    "urgency": "high"
  },
  "snapshot_history_available": true,
  "first_snapshot_at": "2026-10-01T00:00:00Z"
}
```

| Field | Description |
|-------|-------------|
| `segments[*].trend` | `growing` / `stable` / `shrinking` / `none` (ไม่มี history) |
| `segments[*].trend_pct` | % เปลี่ยนแปลงจาก prev_count — `null` ถ้าไม่มี history |
| `segments[*].urgency_score` | 0–100 — คำนวณจาก segment type × trend × size × avg_monetary |
| `segments[*].priority` | `critical` (≥80) / `high` (≥60) / `medium` (≥40) / `low` (≥20) / `monitor` |
| `segments[*].priority_color` | `red` / `orange` / `yellow` / `green` / `gray` |
| `segments[*].action_label` | `ทำก่อน` / `ควรทำ` / `ทำได้` / `รักษา` / `รอก่อน` / `—` (count=0) |
| `health_score` | aggregate สุขภาพลูกค้า — Repo 2 คำนวณจาก segment distribution |
| `health_score.healthy_pct` | % ลูกค้าใน Champions + Loyal + Potential Loyalist + Promising |
| `health_score.at_risk_pct` | % ใน At-Risk + Can't Lose Them + About to Sleep |
| `health_score.churned_pct` | % ใน Hibernating + Lost |
| `alerts[*].level` | `warning` (ต้องทำด่วน) / `info` (ควรสังเกต) |
| `recommendation.urgency` | `high` / `medium` / `low` |
| `snapshot_history_available` | `false` = เดือนแรก ยังไม่มี prev_count จริง |

#### Snapshot Storage Logic (Repo 2 ต้องทำ)

```
ทุกครั้งที่เรียก GET /segment/health → Repo 2 ต้อง:

1. เรียก ML engine → ได้ segment counts ปัจจุบัน
2. Save snapshot ลง DB:
   { store_id, snapshot_date, segment_name, count }
3. ดึง snapshot ล่าสุดก่อนหน้า (เดือนก่อน ± 15 วัน) → prev_count
4. คำนวณ trend_pct = (count - prev_count) / prev_count × 100
5. trend = "growing" ถ้า trend_pct > 5%, "shrinking" ถ้า < -5%, "stable" อื่นๆ
6. ถ้าไม่มี prev snapshot → prev_count=null, trend="none", snapshot_history_available=false

Pad IBM 10 segments เสมอ:
IBM_SEGMENTS = [Champions, Loyal Customers, Potential Loyalist, Promising,
                Need Attention, About to Sleep, At-Risk,
                Can't Lose Them, Hibernating, Lost]
→ segment ที่ ML ไม่ return → count=0, prev_count=0, trend="none",
  urgency_score=0, priority_color="gray", action_label="—"

Health Score คำนวณจาก:
  healthy_pct = (Champions + Loyal Customers + Potential Loyalist + Promising) / total × 100
  at_risk_pct = (At-Risk + Can't Lose Them + About to Sleep) / total × 100
  churned_pct = (Hibernating + Lost) / total × 100
  label = "ดี" ถ้า healthy_pct ≥ 40%, "ปานกลาง" ถ้า ≥ 25%, "ต้องดูแล" ถ้าต่ำกว่า

Alert Trigger Rules:
  warning: trend_pct > 20% บน At-Risk / Can't Lose Them / Hibernating
  info:    trend_pct > 10% บน Need Attention / About to Sleep
  (ไม่ alert segment ที่ count < 10 — noise เกิน)
```

#### Response 404

```json
{
  "error": "PLAN_NOT_FOUND",
  "message": "ไม่พบข้อมูลของ store_id นี้ในระบบ"
}
```

---

### 6. GET /campaign/history

**วัตถุประสงค์**: ดูประวัติ campaign ทั้งหมดของร้าน — ใช้แสดง history list และใช้ตรวจ overlap ใน `/campaign/analyze`

**ใช้เมื่อ**: โหลด Campaign History screen หรือเตรียม `campaign_history` ก่อนเรียก `/campaign/analyze`

#### Request

```
GET /campaign/history?store_id=store_abc&limit=20&offset=0
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `store_id` | string | ✅ | identifier ของร้าน |
| `limit` | int | ❌ (default: 20) | จำนวน record ต่อหน้า |
| `offset` | int | ❌ (default: 0) | สำหรับ pagination |

#### Response 200

```json
{
  "store_id": "store_abc",
  "total": 5,
  "campaigns": [
    {
      "execute_id": "exec_20261106_143120",
      "plan_id": "plan_20261106_143022",
      "export_id": "export_20261106_143055",
      "goal": "maximize_profit",
      "goal_label": "กำไรสูงสุด",
      "status": "awaiting_outcome",
      "customers_targeted": 210,
      "actual_spend": 4850.00,
      "executed_at": "2026-11-06T14:31:20Z",
      "outcome_due_at": "2026-12-06T14:31:20Z",
      "outcome": null
    },
    {
      "execute_id": "exec_20261010_090000",
      "plan_id": "plan_20261010_085500",
      "export_id": "export_20261010_085900",
      "goal": "win_back",
      "goal_label": "Win-Back",
      "status": "completed",
      "customers_targeted": 165,
      "actual_spend": 4200.00,
      "executed_at": "2026-10-10T09:00:00Z",
      "outcome_due_at": "2026-11-09T09:00:00Z",
      "outcome": {
        "predicted_profit": 18200.00,
        "actual_revenue": 20100.00,
        "calibration_ratio": 1.105,
        "lift_pct": 52.3
      }
    }
  ],
  "promo_log": [
    { "customer_id": "C001", "promo_type": "discount_15", "sent_at": "2026-10-10T09:00:00Z" },
    { "customer_id": "C002", "promo_type": "discount_15", "sent_at": "2026-10-10T09:00:00Z" }
  ]
}
```

| Field | Description |
|-------|-------------|
| `status` | `active` (กำลังวิ่ง) / `awaiting_outcome` (รอ 30 วัน) / `completed` (วัดผลแล้ว) / `plan_only` (ยังไม่ execute) |
| `outcome_due_at` | วันที่ควรส่ง post-campaign data = executed_at + 30 วัน |
| `promo_log` | flat list ใช้ส่งเป็น `campaign_history` ให้ `/campaign/analyze` |

#### Response 200 (ไม่มีประวัติ)

```json
{
  "store_id": "store_abc",
  "total": 0,
  "campaigns": [],
  "promo_log": []
}
```

---

### 7. GET /model/status

**วัตถุประสงค์**: ดูสถานะโมเดลปัจจุบัน

**ใช้เมื่อ**: dashboard แสดงว่า engine ใช้โมเดลไหนอยู่ แม่นแค่ไหน

#### Request

```
GET /model/status
```

#### Response 200

```json
{
  "champion_version": "v11",
  "model_type": "R-Learner (CausalForestDML)",
  "dataset": "psm",
  "auuc": 0.0033,
  "features": ["total_price", "quantity", "is_campaign_spike", "rfm_decile_segment", "frequency"],
  "promoted_at": "2026-11-06T10:00:00Z",
  "calibration": {
    "trend": "stable",
    "avg_calibration_ratio": 0.98,
    "recommendation": "โมเดลยังแม่นดี ไม่ต้อง retrain"
  }
}
```

---

### 8. POST /model/retrain

**วัตถุประสงค์**: retrain โมเดลบน POS data ใหม่ด้วย PSM

**ใช้เมื่อ**: Phase 0 — ได้ POS 12 เดือนจากร้านแล้ว หรือ calibration trend = degrading

#### Request Body

```json
{
  "store_id": "store_abc",
  "pos_data": "<base64-encoded CSV string>",
  "psm_mode": "discount",
  "observation_months": 3,
  "auto_promote": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `psm_mode` | enum | ❌ (default: discount) | `discount` / `campaign_list` / `proxy` |
| `observation_months` | int | ❌ (default: 3) | เดือนสุดท้ายที่ใช้เป็น outcome window |
| `auto_promote` | bool | ❌ (default: true) | promote ถ้า AUUC ชนะ champion ปัจจุบัน |

#### Response 200

```json
{
  "retrain_id": "retrain_20261106_150000",
  "store_id": "store_abc",
  "completed_at": "2026-11-06T15:05:00Z",
  "psm_result": {
    "mode": "discount",
    "matched_pairs": 1842,
    "features_balanced": "5/5",
    "all_balanced": true
  },
  "evaluation": {
    "new_model_auuc": 0.142,
    "champion_auuc": 0.0033,
    "winner": "new_model"
  },
  "promotion": {
    "promoted": true,
    "new_version": "v12",
    "previous_version": "v11"
  }
}
```

---

### 9. POST /campaign/record-outcome

**วัตถุประสงค์**: ส่ง POS หลังแคมเปญ 30 วัน → วัด actual lift vs predicted

**ใช้เมื่อ**: Phase 3 — ครบ 30 วันหลังส่งโปร

#### Request Body

```json
{
  "execute_id": "exec_20261106_143120",
  "store_id": "store_abc",
  "post_pos_data": "<base64-encoded CSV string>",
  "observation_days": 30
}
```

#### Response 200

```json
{
  "store_id": "store_abc",
  "execute_id": "exec_20261106_143120",
  "observation_days": 30,
  "result": {
    "predicted_profit": 28400.00,
    "actual_revenue": 31200.00,
    "calibration_ratio": 1.099,
    "trend": "improving",
    "recommendation": "โมเดลแม่นยำขึ้น — ไม่ต้อง retrain"
  },
  "comparison": {
    "baseline_conversion_rate": 0.32,
    "treated_conversion_rate": 0.51,
    "lift_pct": 59.4
  }
}
```

---

### 10. GET /send-date/recommend

**วัตถุประสงค์**: คำนวณวันที่เหมาะสมที่สุดสำหรับส่งแคมเปญในเดือนที่กำหนด โดยรวม buy-day affinity, payday window และ campaign overlap penalty

**ใช้เมื่อ**: ร้านต้องการ suggest วันส่ง ก่อนกด Execute หรือใน Campaign Planning screen

#### Request

```
GET /send-date/recommend?store_id=store_abc&segment=At-Risk&month=2026-11&goal=win_back&cooldown_days=14&top_n=3
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `store_id` | string | ✅ | identifier ของร้าน |
| `segment` | string | ✅ | IBM segment name เช่น `At-Risk`, `Champions` |
| `month` | string (YYYY-MM) | ✅ | เดือนเป้าหมาย เช่น `2026-11` |
| `goal` | enum | ❌ (default: maximize_profit) | `maximize_profit` / `win_back` / `retain_loyal` / `maximize_reach` |
| `cooldown_days` | int | ❌ (default: 14) | ระยะ cooldown ขั้นต่ำ (วัน) ระหว่าง campaign — 0–365 |
| `top_n` | int | ❌ (default: 3) | จำนวน alternative dates — 1–10 |

#### Response 200

```json
{
  "store_id": "store_abc",
  "segment": "At-Risk",
  "month": "2026-11",
  "goal": "win_back",
  "recommended_date": "2026-11-15",
  "score": 0.74,
  "reason": "วันศุกร์ที่ลูกค้า At-Risk ซื้อบ่อยที่สุด ตรงกับช่วง payday ต้นเดือน ไม่ซ้อน campaign ก่อนหน้า",
  "payday_window": true,
  "overlap_warning": false,
  "alternatives": [
    { "date": "2026-11-14", "score": 0.68, "reason": "วันพฤหัสก่อน payday" },
    { "date": "2026-11-22", "score": 0.61, "reason": "สิ้นเดือน — buy pattern รอง" },
    { "date": "2026-11-08", "score": 0.55, "reason": "วันศุกร์ถัดไป ไม่มี overlap" }
  ],
  "payday_dates": ["2026-11-15", "2026-11-30"]
}
```

| Field | Description |
|-------|-------------|
| `recommended_date` | วันที่แนะนำ (YYYY-MM-DD) |
| `score` | 0.0–1.0 — คะแนนรวม 5 signals |
| `reason` | อธิบายเหตุผลเป็นภาษาไทย |
| `payday_window` | true ถ้า recommended_date ตรงหรือใกล้ payday (±2 วัน) |
| `overlap_warning` | true ถ้ามี campaign ภายใน cooldown window |
| `alternatives` | top_n วันรอง เรียงตาม score |
| `payday_dates` | วัน payday ในเดือนนั้น — ใช้ highlight บน calendar |

#### Response 404

```json
{
  "error": { "code": "store_not_found", "message": "ไม่พบข้อมูลของ store_id นี้ กรุณาอัปโหลดข้อมูลผ่าน /campaign/plan ก่อน" }
}
```

#### Response 400

```json
{
  "error": { "code": "invalid_input", "message": "month ต้องเป็น YYYY-MM เช่น 2026-11" }
}
```

---

### 11. POST /model/retrain-request

**วัตถุประสงค์**: แจ้งขอ retrain โมเดล — ไม่รัน ML ทันที แต่เข้าคิวให้ scheduler รันคืนถัดไป

**ใช้เมื่อ**: ร้านหรือ platform กด "ขอ Retrain" เช่น หลังอัปโหลด POS ใหม่จำนวนมาก หรือเห็น accuracy_concern จาก `/campaign/record-outcome`

> ต่างจาก `POST /model/retrain` ตรงที่: retrain-request = queue เท่านั้น; retrain = background job รันทันที

#### Request Body

```json
{
  "store_id": "store_abc",
  "reason": "accuracy_concern",
  "psm_mode": "auto",
  "requested_by": "manager@store.com"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `store_id` | string | ✅ | identifier ของร้าน |
| `reason` | enum | ❌ (default: manual_request) | `manual_request` / `accuracy_concern` / `new_data` |
| `psm_mode` | enum | ❌ (default: auto) | `auto` / `discount` / `campaign_list` / `proxy` — `auto` = ให้ scheduler เลือก |
| `requested_by` | string | ❌ | email หรือ user identifier |

#### Response 200

```json
{
  "request_id": "retrain_req_a3f7c91b2e4d",
  "store_id": "store_abc",
  "queued_at": "2026-11-08T10:30:00+00:00",
  "status": "queued",
  "reason": "accuracy_concern",
  "psm_mode": "auto",
  "estimated_completion": "ภายใน 24 ชั่วโมง — scheduler รันกลางคืน"
}
```

| Field | Description |
|-------|-------------|
| `request_id` | unique ID รูปแบบ `retrain_req_<12 hex chars>` |
| `status` | ตอนนี้ return `queued` เสมอ — scheduler update เป็น `processing` / `completed` / `failed` เอง |
| `estimated_completion` | ข้อความภาษาไทย — ใช้แสดง toast บน UI ได้เลย |

#### Response 400

```json
{
  "error": { "code": "invalid_input", "message": "reason ต้องเป็นหนึ่งใน: accuracy_concern, manual_request, new_data" }
}
```

---

## Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `INVALID_INPUT` | 400 | ข้อมูลที่ส่งมาไม่ถูกต้อง |
| `PLAN_NOT_FOUND` | 404 | plan_id หรือ export_id ไม่มีในระบบ |
| `STORE_MISMATCH` | 403 | store_id ไม่ตรงกับ plan ที่ระบุ |
| `MODEL_NOT_READY` | 503 | ยังไม่มี champion model ในระบบ |
| `CHANNEL_ERROR` | 502 | LINE OA หรือ Meta API ตอบกลับ error |
| `RETRAIN_FAILED` | 500 | PSM หรือ training ล้มเหลว |

---

## Notes สำหรับโชแปง

**Data flow ภายใน Repo 2:**
```
รับ request → decode base64 CSV → save temp file
→ เรียก Repo 1 functions โดยตรง (import ไม่ใช่ HTTP call)
→ return result
→ ลบ temp file
```

**Endpoints ที่ทำได้เลย** (ไม่รอ external API):
- `POST /campaign/plan`
- `POST /campaign/analyze`
- `POST /audience/export`
- `GET /segment/health`
- `GET /campaign/history`
- `GET /model/status`
- `POST /model/retrain`
- `POST /campaign/record-outcome`
- `GET /send-date/recommend`
- `POST /model/retrain-request`

**Endpoints ที่รอ research ก่อน:**
- `POST /campaign/execute` — ต้องรู้ LINE OA multicast format + Meta upload format ก่อน

**POS CSV format ขั้นต่ำ:**
```
customer_id, transaction_id, timestamp, total_price, product_name, category
```
Optional: `quantity`, `discount_amount`, `promo_code`
