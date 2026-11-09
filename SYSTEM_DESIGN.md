# ระบบแนะนำโปรโมชั่น — หลักการและสถาปัตยกรรม

## ภาพรวม

ระบบตัดสินใจว่าจะส่ง promo ให้ลูกค้าคนไหน และส่งอะไร โดยอิงจากพฤติกรรมการซื้อจริง
ทุก action ผ่านการ validate ว่า net margin ≥ 15% ก่อนส่ง

---

## Pipeline (ตามที่ควรเป็น)

```
CSV transactions
    │
    ▼
Layer 1: RFM + K-Means
    → แบ่งลูกค้าเป็น 3-5 กลุ่ม (Champions / At-Risk / Mid / Hibernating)
    → output: cluster_id + segment_name ต่อลูกค้า
    → ใช้เป็น unit ของ campaign (ไม่ใช่แค่ label)
    │
    ▼
Layer 2: Market Basket Analysis ต่อกลุ่ม
    → หา bundle offer ที่เหมาะกับแต่ละกลุ่ม (MBA per cluster)
    → Champions ซื้อสินค้าคู่ต่างจาก At-Risk
    → output: cluster_bundle_map = { cluster_id: "Bundle: item_A + item_B" }
    │
    ▼
Layer 3: Context per customer
    → Peak/Off-Peak (จากชั่วโมงที่ซื้อจริง)
    → Stock check (สินค้าในสต็อกหรือไม่)
    │
    ▼
Layer 4: Score รายคน — 2 steps
    │
    ├─ Step 1: ส่งไหม?
    │       มี champion model → X-learner uplift score
    │       ไม่มี champion   → rule-based (RFM segment + recency)
    │       ผล: 4 กลุ่ม
    │           persuadable    → ส่ง (ถ้าไม่ส่งเขาจะไม่ซื้อ)
    │           sure_thing     → ส่ง loyalty only (ซื้ออยู่แล้ว อย่า discount)
    │           do_not_disturb → ไม่ส่ง (uplift ติดลบ — ยิ่งส่งยิ่งเสียโอกาส)
    │           lost_cause     → ไม่ส่ง (uplift ≈ 0 และ baseline ต่ำ — ROI ติดลบ)
    │
    └─ Step 2: ส่งอะไร? (เฉพาะ persuadable + sure_thing)
            offer   = bundle ของกลุ่มลูกค้าจาก Layer 2
            timing  = Peak / Off-Peak จาก Layer 3
            validate: expected_profit > 0 AND net_margin ≥ 15%
            ถ้าไม่ผ่าน → Do Nothing (No Profitable Action)
```

---

## Decision Grid (cluster × uplift group)

```
                 sure_thing          persuadable         do_not_disturb  lost_cause
                 (ซื้ออยู่แล้ว)      (โน้มน้าวได้)
─────────────────────────────────────────────────────────────────────────────────
Champions        VIP bundle A+B      VIP bundle A+B      ไม่ส่ง          ไม่ส่ง
                 (ไม่มี discount)     (ไม่มี discount)
─────────────────────────────────────────────────────────────────────────────────
At-Risk          (หายาก)             Discount bundle C+D ไม่ส่ง          ไม่ส่ง
─────────────────────────────────────────────────────────────────────────────────
Mid-tier         Loyalty points      Upsell bundle       ไม่ส่ง          ไม่ส่ง
─────────────────────────────────────────────────────────────────────────────────
Hibernating      (หายาก)             Reactivation offer  ไม่ส่ง          ไม่ส่ง
─────────────────────────────────────────────────────────────────────────────────
```

แต่ละช่องในตาราง = 1 campaign (3-5 clusters × 2 contexts ≈ 3-5 campaigns จริง)

---

## หลักการสำคัญ

### 1. Cluster กำหนด "สินค้าที่แนะนำ" (What)
- Layer 2 MBA ต่อ cluster → bundle offer เฉพาะกลุ่ม
- Champions ได้ bundle จาก co-occurrence ของ Champions เท่านั้น
- At-Risk ได้ bundle ของตัวเอง

### 2. Uplift Model กำหนด "ส่งไหม" (Whether)
- X-learner ตอบ: ถ้าส่ง promo คนนี้จะ convert เพิ่มแค่ไหน
- uplift > 0 ไม่พอ — ต้องชนะ cost ด้วย (margin check)
- Cluster ไม่ได้อยู่ในสมการนี้โดยตรง แต่ feature อย่าง recency/frequency/avg_spent สะท้อน cluster อยู่แล้ว

### 3. Context กำหนด "ส่งเมื่อไหร่" (When)
- Peak Window: ช่วงที่ซื้อเยอะ → push โปรที่มี urgency
- Off-Peak Window: ช่วงเงียบ → ดึง traffic ด้วย deal พิเศษ

### 4. Margin Gate เป็น hard stop
```
expected_profit  = gross_margin × p_treated - promo_cost  > 0
net_margin       = (revenue - cogs - promo_cost) / revenue ≥ 15%
```
ถ้าไม่ผ่านทั้ง 2 เงื่อนไข → Do Nothing (ไม่ส่ง ดีกว่าขาดทุน)

---

## สถานะปัจจุบัน vs Target

| ส่วน | ตอนนี้ | Target |
|------|--------|--------|
| Pipeline | 1 pipeline (merged แล้ว) ✅ | 1 pipeline |
| Step 1/2 split | มีแล้ว ✅ | มี |
| cluster_bundle_map | ❌ ยังใช้ global bundle | 1 bundle ต่อ cluster |
| Champions no-discount | ❌ ยังได้ discount ถ้า persuadable | block discount ด้วย cluster check |
| Rule-based fallback | ❌ abort ถ้าไม่มี champion | RFM segment fallback |
| จำนวน campaigns | 6+ actions ปนกัน | 3-5 campaigns ชัดเจน |

---

## แผนระยะถัดไป: Incentive Type Personalization

### ปัญหาที่ระบบยังไม่ตอบ

ระบบปัจจุบันบอกได้ว่า "ส่งไหม" (uplift) และ "สินค้าอะไร" (MBA per cluster)
แต่ยังไม่บอกว่า "ให้ incentive แบบไหน" — discount / points / VIP access / reactivation hook

### แนวทาง Option A: Multi-Treatment Uplift Model

แทนที่จะ model binary (ส่ง/ไม่ส่ง) ให้ model CATE แยกต่อ treatment type:

```
treatment_1 = discount 15%
treatment_2 = loyalty points 2x
treatment_3 = VIP early access
treatment_4 = reactivation bundle

→ CATE_discount(customer_i)   = +0.12
→ CATE_points(customer_i)     = +0.08
→ CATE_vip(customer_i)        = +0.21   ← เลือกอันนี้
→ CATE_reactivation(customer_i)= +0.03

เลือก treatment ที่ CATE สูงสุด AND margin ผ่าน
```

### แนวทาง Option B: Cluster Heuristic (pragmatic, ทำได้ทันที)

ใช้ cluster segment name เป็น deterministic rule สำหรับ incentive type:

```python
Champions   → points / VIP access เท่านั้น (ไม่มี % discount เด็ดขาด)
At-Risk     → discount บน bundle (price incentive ดึงกลับ)
Mid-tier    → upsell nudge ("อีก X บาทได้ tier ถัดไป")
Hibernating → reactivation hook (emotional first, price second)
```

---

## Trade-off Analysis

### Option A: Multi-Treatment Uplift Model

**ได้อะไร**
- CATE รายคนต่อ treatment — Champions บางคน respond ต่อ discount จริงๆ ก็ได้
- Maximize incremental revenue รายคน ไม่ใช่แค่ segment average
- สามารถค้นพบว่า "Mid-tier ตอบ points ดีกว่า discount 30%" จากข้อมูลจริง

**เสียอะไร / ข้อจำกัด**

| ปัญหา | เหตุผล |
|-------|--------|
| ต้องการ randomized A/B/C/D data | ถ้า historical data ไม่ได้ random assign treatment → CATE estimate biased (confounding) |
| Sample size | 1,000 customers × 4 treatments × treat/control = 8 arms → แต่ละ arm ได้แค่ ~125 คน อาจไม่พอ |
| Cold start | treatment ที่ไม่เคยลอง (เช่น VIP access) ไม่มี CATE เลย |
| Exploration tax | ถ้า exploit อย่างเดียว → ไม่มีข้อมูลใหม่มา update model → ต้อง reserve 10-15% สำหรับ random treatment (ต้นทุน) |
| Maintenance | k treatment types = k models ที่ต้อง retrain, monitor drift แยกกัน |

**เมื่อไหร่ควรทำ**
- มี A/B test data ที่ treatment ถูก random assign จริงๆ (ไม่ใช่ observational)
- Customer base ≥ 5,000 คน ต่อ treatment arm
- มี infrastructure สำหรับ online experiment (record treatment ที่ให้จริง)

### Option B: Cluster Heuristic

**ได้อะไร**
- ทำได้ทันทีจาก data ที่มีอยู่
- Interpretable 100% — ทีม marketing เข้าใจและ trust ได้
- ง่ายต่อการ A/B test ทีหลัง (เปลี่ยน heuristic rule แล้ว compare)
- ไม่มี cold start problem

**เสียอะไร**

| ปัญหา | เหตุผล |
|-------|--------|
| Intra-cluster variance ถูกมองข้าม | Champions 20% อาจ respond ต่อ discount แต่ไม่ได้รับ |
| Rigid — ไม่เรียนรู้จาก outcome | ถ้า rule ผิด จะผิดซ้ำจนกว่าจะมีคนแก้ |
| Segment boundary arbitrary | ลูกค้าที่ RFM score อยู่ขอบ Champions/Mid ได้ treatment ต่างกันมาก |

**เมื่อไหร่ควรทำ**
- ข้อมูลไม่มี randomized treatment
- ต้องการ quick win ที่ explainable
- เป็น baseline ก่อน collect data สำหรับ Option A

---

## สรุปคำแนะนำ

```
ตอนนี้      → Option B (cluster heuristic) ก่อน
             เหตุผล: ข้อมูลที่มีไม่มี randomized treatment assignment
             ผล: ระบบ predictable, margin-safe, explainable

เมื่อมีข้อมูล → เพิ่ม treatment logging → collect 3-6 เดือน
             แล้ว train multi-treatment uplift (Option A) บน data นั้น

สิ่งสำคัญ:  เริ่ม log treatment ที่ส่งจริงตั้งแต่ตอนนี้
             ถ้าไม่มี log → ไม่มีทางทำ Option A ได้เลย
```

### ตารางสรุป

| มิติ | วิธีที่ถูกต้อง | ทำได้ตอนนี้ |
|------|--------------|------------|
| ส่งไหม (Who) | X-learner CATE binary | ✅ |
| สินค้าอะไร (What) | MBA per cluster | ⚠️ ต้อง wire cluster_bundle_map |
| incentive แบบไหน (How) | Multi-treatment CATE | ❌ ต้องรอ data → ใช้ heuristic ก่อน |
| สื่อสารยังไง (Tone) | cluster segment → LLM prompt | ❌ ต้องเพิ่ม |
| % เท่าไหร่ (Amount) | price sensitivity per segment | ❌ ใช้ cluster tier heuristic ก่อน |

---

## Multi-Treatment Uplift — Data Collection Plan

### Treatment Log Schema

ข้อมูลที่ต้อง collect ต่อ campaign run เพื่อ train multi-treatment model ในอนาคต:

```
customer_id     : string   — รหัสลูกค้า
campaign_date   : date     — วันที่ส่ง campaign
treatment_type  : string   — discount / points_vip / bundle / reactivation / none
converted       : int(0/1) — ซื้อภายใน X วันหลังรับ campaign หรือไม่
revenue_post    : float    — ยอดซื้อหลัง campaign (optional แต่ useful)
```

### ไฟล์ที่ pipeline สร้างให้แล้ว

- `data/processed/final_campaign_engine_output.csv` มีคอลัมน์ `treatment_type` แล้ว
- ต้องเพิ่ม: join กับ actual conversion data หลัง campaign ปิด

### เมื่อไหร่พร้อม train

- ต้องการ: ≥ 50 customers ต่อ treatment type ที่ได้ random assignment จริง
- เวลาที่ใช้เก็บ: ประมาณ 3-6 เดือน (ขึ้นกับ customer base)
- Script พร้อมแล้ว: `python src/train_multi_treatment.py --store-data STORE:transactions.csv:treatment_labels.csv`

### ข้อควรระวัง: Confounding

ถ้า treatment assignment ไม่ได้ random (เช่น ส่ง discount ให้แค่ At-Risk เสมอ):
- CATE estimate จะ biased — Champions จะดูเหมือนไม่ respond ต่อ discount เพราะไม่เคยได้รับ
- แนะนำ: reserve 5-10% ของลูกค้าสำหรับ random treatment assignment ใน campaign แต่ละรอบ
