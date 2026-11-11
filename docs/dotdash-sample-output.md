# ตัวอย่าง output จริง (action_key / cost_source / bundle_*)

**ที่มา:** รันจริงด้วยโค้ด Dera-ML commit `2b52e4b` (branch `feat/dotdash-offer-contract`) บนข้อมูลทดสอบที่สร้างจาก `tests/fixtures_pos.py` (`make_pos_csv(n_customers=120, seed=0)`) **ไม่ใช่ข้อมูลร้านจริง** และใช้โมเดลที่เทรนในสคริปต์ตัวอย่าง **ไม่ใช่ champion v11** ตัวเลขจึงใช้ดูรูปแบบ ไม่ใช่ดูคุณภาพโมเดล ตัดรายชื่อลูกค้าให้สั้นด้วย `...`

## 1) `POST /campaign/plan` → `dotdash_segments` (ไม่ส่ง point_value_thb)

```json
[
  {
    "segment_name": "Churn Risk",
    "action_key": "st_loyalty_nurture",
    "recommended_action": "Loyalty Nurture: แต้มสะสม",
    "customer_count": 1,
    "customer_ids": [
      "c024",
      "..."
    ],
    "expected_profit": 0.95,
    "actual_spend": 0.4,
    "dotdash_offer": {
      "offer_type": "points",
      "points_amount": 50,
      "min_spend": 0,
      "expiry_days": 30,
      "action_key": "st_loyalty_nurture",
      "bundle_items": null,
      "bundle_coverage": 0.0,
      "bundle_low_coverage": true
    },
    "cost_source": "estimated"
  }
]
```

## 2) เมื่อส่ง `point_value_thb=0.002, redemption_rate=0.4`

```json
[
  {
    "segment_name": "Churn Risk",
    "action_key": "st_loyalty_nurture",
    "recommended_action": "Loyalty Nurture: แต้มสะสม",
    "customer_count": 1,
    "customer_ids": [
      "c024",
      "..."
    ],
    "expected_profit": 1.31,
    "actual_spend": 0.04,
    "dotdash_offer": {
      "offer_type": "points",
      "points_amount": 50,
      "min_spend": 0,
      "expiry_days": 30,
      "action_key": "st_loyalty_nurture",
      "bundle_items": null,
      "bundle_coverage": 0.0,
      "bundle_low_coverage": true
    },
    "cost_source": "shop_provided"
  }
]
```

## 3) `segments_latest.json` (ฟิลด์เดิมไม่เปลี่ยน + `action_key_breakdown` เพิ่มใหม่)

```json
[
  {
    "id": 1,
    "name": "Can't Lose Them",
    "customers": 48,
    "broadcast": 3,
    "suppressed": 45,
    "action_breakdown": {
      "Do Nothing (Do Not Disturb)": 44,
      "VIP Win-Back: กลับมาเป็น VIP อีกครั้ง": 3,
      "Do Nothing (Low Basket Size)": 1
    },
    "action_key_breakdown": {
      "none": 45,
      "st_cantlose_winback": 3
    }
  }
]
```

## 4) `final_campaign_engine_output.csv` — แถวที่ส่ง (เฉพาะบางคอลัมน์)

```
customer_id,recommended_action,expected_profit,promo_cost,should_send,action_key,cost_source
c024,Loyalty Nurture: แต้มสะสม,0.95,0.4,1,st_loyalty_nurture,estimated
c068,Loyalty Nurture: แต้มสะสม,0.37,0.19,1,st_loyalty_nurture,estimated
c051,Loyalty Nurture: แต้มสะสม,0.28,0.29,1,st_loyalty_nurture,estimated
```

คอลัมน์ทั้งหมด 43 คอลัมน์: คอลัมน์เดิมคงชื่อและลำดับ ส่วน `action_key`, `cost_source` อยู่ท้ายสุด: `['should_send', 'treatment_type', 'action_key', 'cost_source']`

## 5) แถวที่โปรของร้าน (operator) ทับ action ของ ML (รันด้วย PromoSpec ตามที่ ml-service ส่ง)

```
customer_id,recommended_action,expected_profit,promo_cost,should_send,action_key,cost_source,final_source
c044,Shop promo,0.05,0.12,1,operator_promo,operator,operator
c024,Shop promo,0.95,0.4,1,operator_promo,operator,operator
```

## 6) `broadcast_list.csv` (คอลัมน์ใหม่อยู่ท้าย)

```
customer_id,segment_name,recommended_action,offer,expected_profit,promo_cost,send_date,send_window,send_time_hint,channel,should_send,action_key,cost_source
c003,Can't Lose Them,VIP Win-Back: กลับมาเป็น VIP อีกครั้ง,ซื้อ [cake] คู่กับ [salad] ลด 15%,-0.86,0.0,2026-11-13,Off-Peak Window,Before Off-Peak Window,Email/SMS,1,st_cantlose_winback,estimated
c042,Can't Lose Them,VIP Win-Back: กลับมาเป็น VIP อีกครั้ง,ซื้อ [cake] คู่กับ [salad] ลด 15%,-1.08,0.0,2026-11-13,Off-Peak Window,Before Off-Peak Window,Email/SMS,1,st_cantlose_winback,estimated
```

## ความหมายของค่า

- `action_key`: id เสถียรของกฎ (ดู `docs/dotdash-action-keys.md`); `operator_promo` = โปรที่ร้านกำหนด (Promo Advisor เลือกแทน ML); ว่าง = ไม่มีโปร
- `cost_source`: `estimated` = ค่าประมาณ (อัตรา × ยอดซื้อเฉลี่ย) | `shop_provided` = คิดจากมูลค่าแต้มที่ร้านให้ | `operator` = ต้นทุนของโปรร้าน | `none` = ไม่มีโปร (ต้นทุน 0)
- `promo_cost`: ชื่อและหน่วยเดิม (บาทต่อลูกค้า) ไม่เปลี่ยน
- `bundle_coverage`: สัดส่วนลูกค้าในกลุ่ม (segment × action) ที่คู่สินค้านั้นเป็นคู่ของตัวเอง; `bundle_low_coverage` = true เมื่อต่ำกว่า 0.5
- หมายเหตุ: ตอนนี้ `point_value_thb`/`redemption_rate` มีผลที่ `POST /campaign/plan` เท่านั้น ยังไม่มีผลกับ `run_layer4` (เส้นที่ ml-service เรียก)
- ข้อจำกัดที่มีอยู่เดิม (ไม่ได้แก้ในรอบนี้): `promo_cost` ใน `broadcast_list.csv` เป็น 0.0 ทุกแถว (ตัวอย่างนี้: 17 แถว ไม่มีแถวไหนไม่ใช่ศูนย์) ส่วน `final_campaign_engine_output.csv` มีต้นทุนจริง (ตัวอย่างนี้: ทุกแถวที่ส่งไม่เป็นศูนย์) — ใช้ไฟล์หลังคิดต้นทุนรายคน
