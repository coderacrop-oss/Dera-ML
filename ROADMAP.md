# Codera ML Campaign Engine — Roadmap

> อัปเดตล่าสุด: พฤศจิกายน 2026 (เพิ่ม PSM + Model Registry integration)
> สรุปภาพรวมทั้งหมด: สิ่งที่ทำร่วมกันแล้ว · MVP เฟสแรก · และสิ่งที่ Codera พัฒนาต่อเพื่อหารายได้

---

## สิ่งที่ทำร่วมกันแล้ว (Engine พร้อมใช้)

### Core ML Pipeline
- [x] รับไฟล์ POS 7 คอลัมน์ (customer_id, transaction_id, timestamp, total_price, product_name, category + 2 optional)
- [x] IBM 10-segment clustering (K-Means + rank-percentile + 4 rule-based flags)
- [x] Uplift model — LightGBM R-Learner, Champion v9, Qini 1.21
- [x] 2-step decision engine: Hard Stop → Uplift Group → Action Rules → Profit Filter
- [x] CFO Guardrail: margin floor 15% + incremental profit check
- [x] Budget optimizer: greedy knapsack 1/2-approximation
- [x] Campaign scheduler: 3 waves/เดือน พร้อม control group split
- [x] Send rate 50.1% (Champion v9)

### Observational Retraining Pipeline (PSM)
- [x] **propensity_score_matching.py** — 3 modes รองรับทุกกรณีของ DotDash
  - Mode A `discount`: treated = มี discount_amount / promo_code ใน POS
  - Mode B `campaign_list`: treated = customer_id อยู่ใน campaign record
  - Mode C `proxy`: ไม่มีข้อมูลโปร → infer จาก purchase frequency spike (last resort)
- [x] **Balance diagnostics** — SMD per feature, caliper matching (Austin 2011)
- [x] **Time-split outcome** — feature window / observation window แยกอัตโนมัติ (default 3 เดือน)
- [x] **Feature alignment** — PSM ใช้ `SERVING_FEATURES` เดียวกับ `uplift_engine.py` ทุกประการ: `total_price, quantity, is_campaign_spike, rfm_decile_segment, frequency` — ไม่มี feature mismatch ตอน serve
- [x] **train_rlearner.py `--dataset psm`** — PSM → CausalForestDML → DR-Learner → AUUC comparison → auto-promote ถ้าชนะ champion
- [x] **rlearner_wrapper.py** — adapter `predict_uplift()` + `predict_control_proba()` ให้ serving pipeline ใช้ R-Learner / DR-Learner ได้โดยไม่ต้องแก้โค้ด

### Model Registry
- [x] **Versioned champion** — manifest.json + archive + atomic writes + flock
- [x] **Auto-promote** — `train_rlearner.py` promote ถ้า AUUC ชนะ champion ปัจจุบัน
- [x] **`--no-promote` flag** — ทดสอบโดยไม่ replace champion
- [x] **Rollback** — `registry.rollback(version)` คืน champion เวอร์ชันก่อนได้ทันที
- [x] **Serving transparent** — `campaign_planner.py` pick up champion ใหม่อัตโนมัติ ไม่ต้องแก้โค้ด
- [x] **Champion v11** — PSM R-Learner, 5 features aligned, baseline_control_prob จาก matched data

### Action Types (38 rules ครบ)
- [x] Win-Back, VIP Win-Back, Rescue, Early Win-Back
- [x] Milestone Reward, Welcome, Zero-cost notification
- [x] Loyalty Nurture, Bundle Upsell, Off-peak Traffic Driver
- [x] ครบ 10 IBM segments

### Operator Tools
- [x] Promo Advisor — ตรวจโปรร้าน Scenario A/B/C
- [x] Multi-promo optimizer — 1 โปร/คน/เดือน เลือก ROI สูงสุด
- [x] min_spend enforcement — ป้องกัน basket substitution
- [x] Cannibalization flag — discount profit สำหรับ sure-thing customers

### Campaign Planner
- [x] Budget-first mode: "มีงบ ฿X ได้กำไรเท่าไหร่"
- [x] Target-first mode: "อยากได้กำไร ฿X ต้องใช้งบเท่าไหร่"
- [x] 4 goals: maximize_profit / win_back / retain_loyal / maximize_reach

### Feedback Loop
- [x] record_campaign_outcome() — เทียบ predicted vs actual
- [x] get_calibration_stats() — trend: improving / stable / degrading
- [x] Auto-recommend: retrain เมื่อ trend degrading

### Agency Reports
- [x] audience_export() — customer_id list ต่อ segment สำหรับ LINE OA / Meta
- [x] post_campaign_report() — treated vs control lift per segment
- [x] monthly_trend_report() — segment migration + next campaign proposal

### Security & PDPA
- [x] HMAC-SHA256 per-store hashing
- [x] Column allowlist (filter ทันทีตอน ingest)
- [x] API key manager + Audit logger
- [x] Retention cleaner + Erasure processor
- [x] Breach runbook (ภาษาไทย H+0 ถึง H+72)

---

## Engagement Model — DotDash (4 Phases)

> เป้าหมาย: พิสูจน์ว่า "วิธีของเรา" ให้ผลดีกว่า "วิธีเดิม" วัดได้จริง
> DotDash มีระบบ: **POS + CRM + LINE OA Broadcast** — Codera เชื่อมเข้าทั้ง 3

### Phase 0 — Data & Model Calibration
- [ ] รับ POS 12 เดือนจาก DotDash
- [ ] รัน PSM Mode A (discount) หรือ Mode B (campaign_list) ตามข้อมูลที่มี
- [ ] Retrain R-Learner บน DotDash data → promote ถ้า AUUC ดีขึ้น
- [ ] สรุป segment profile ของ DotDash (10 segments, % แต่ละกลุ่ม)

### Phase 1 — Design
- [ ] ออกแบบ connector: POS API / CSV scheduled → Codera ingest
- [ ] ออกแบบ Campaign Planner UX สำหรับ DotDash (goal + งบ → กด Execute)
- [ ] ยืนยัน CRM มี `customer_id ↔ LINE UID` mapping ไหม (critical for delivery)

### Phase 2 — Implement
- [ ] **ad_targeting_export.py** — JSON custom/lookalike/exclusion audience สำหรับ Meta
- [ ] Connector: POS → Codera ingest อัตโนมัติ
- [ ] Connector: Codera → LINE OA Broadcast API (ส่งด้วย LINE UID list)
- [ ] Campaign Planner UI: ร้านกด approve → Codera push LINE OA ทันที
- [ ] **Content generator** — Claude API → LINE/Meta copy (ตัดสินใจ 5 ข้อก่อน — ดู project_content_layer.md)

### Phase 3 — Pilot & Evaluation
- [ ] รัน pilot campaign จริง — treated vs control
- [ ] **"วิธีเดิม vs ของเรา" report** — เทียบ baseline DotDash กับ post_campaign_report
- [ ] เก็บ real causal labels (promo response จริง) → retrain Mode B → AUUC ดีขึ้น
- [ ] ตัดสินใจ scale

### ทำมือได้ระหว่าง Phase 0-1
- audience export CSV → DotDash upload ขึ้น LINE OA เอง (ก่อน connector พร้อม)
- gen content → เอเจนซี่ approve → ส่ง (ไม่ auto)
- โค้ดโปร → generate list แนบไฟล์

### Flow เต็ม (Phase 2 เป็นต้นไป)

```
POS (auto) → Codera ML → วิเคราะห์ segment
                ↓
         Campaign Planner
         (goal + งบ → แผน)
                ↓
      DotDash กด Execute (approve)
                ↓
    CRM: customer_id → LINE UID
                ↓
    LINE OA Broadcast API → ลูกค้า
                ↓
    30 วัน → pull POS → วัดผล
    post_campaign_report → "วิธีเดิม vs ของเรา"
```

---

## สิ่งที่ Codera พัฒนาต่อ (หลัง MVP)

### Revenue Layer
- [ ] **Multi-brand isolation** — user/permission layer, แยก data per brand อย่างสมบูรณ์
- [ ] **Pricing module** — คำนวณ setup fee + monthly fee + cost per send อัตโนมัติ
- [ ] **Self-serve dashboard** — เอเจนซี่ upload POS เองได้ ไม่ต้องผ่าน Codera ทุกครั้ง

### Channel Automation (ตามลำดับ)
- [ ] LINE OA API integration — push ตาม send_datetime อัตโนมัติ
- [ ] Meta Ads API — create custom audience / lookalike / exclusion อัตโนมัติ
- [ ] Promo code generator + redemption tracking
- [ ] SMS, Email, TikTok (ภายหลัง)

### Model Quality
- [x] PSM retraining pipeline พร้อมใช้กับ DotDash 12 เดือน (Mode A/B/C) — features aligned กับ serving
- [ ] Real causal labels จาก DotDash (track promo response จริง) → retrain Mode B → ATE calibration ดีขึ้น
- [ ] Free Sample Add-On action type (รอ business approval เรื่อง cost of sample)
- [ ] Thai LLM upgrade — เมื่อ local model ภาษาไทยดีพอ เปลี่ยนจาก Claude API

#### วิธีวัดความแม่นยำโมเดล — Backtest Methodology (ยืนยันกับคุณอ้น Phase 0)

**หลักการ:** แบ่งข้อมูลตามเวลา — เทรนเดือน 1–9, ทดสอบเดือน 10–12 (โมเดลไม่เห็นข้อมูล test ก่อน)

**Track A — Segment Return Rate** _(ใช้เสมอ ไม่ต้องมีแคมเปญเก่า)_
- ทาย % ลูกค้าแต่ละ IBM segment ที่จะซื้อใน 30 วัน → เทียบกับจริง
- เกณฑ์ผ่าน: median MAPE ≤ 25% ทั้ง 10 segment
- ⚠️ ต้องยืนยันตัวเลข 25% กับคุณอ้น — ขึ้นอยู่กับ data ที่มี

**Track B — Campaign Revenue Accuracy** _(ใช้เมื่อมีแคมเปญเก่า ≥ 5 ตัวในช่วง test)_
- ทายรายได้รวมของ treated customers ใน 30 วัน → เทียบกับ actual receipt
- เกณฑ์ผ่าน: weighted MAPE ≤ 25% (weighted by campaign size เพื่อกัน small campaign bias)
- ⚠️ วัดได้แค่ accuracy ของการทาย volume ไม่ใช่ Lift — Lift วัดได้จาก pilot จริงเท่านั้น

**ถ้าไม่ผ่านเกณฑ์ → แสดงช่วงแทนตัวเลขเดียว**
- ใช้ CausalForestDML 70% confidence interval โดยตรง (`predict_interval(alpha=0.3)`)
- แสดง: `"คาดกำไร ฿900–1,600"` + badge "กำลังเรียนรู้"
- ถ้า CI กว้าง > 2× ของ point estimate → แสดง badge แม้ผ่านเกณฑ์ MAPE
- ⚠️ threshold 2× ต้องยืนยันกับคุณอ้น

**สิ่งที่ต้อง build: `src/model_backtester.py`** _(งานที่ 3 ในแผน 17–23 พ.ย.)_
- input: pos_df (12 เดือน), campaign_history (optional)
- auto-select Track A หรือ B ตาม campaign count
- output: `{track, mape_per_segment, weighted_mape, passed, ci_width_ratio}`

### Content Layer (ตัดสินใจก่อน build)
- [ ] content_generator.py — Claude API → LINE/Meta copy พร้อมใช้
- [ ] Brand voice config per brand
- [ ] Human review queue → auto-approve threshold
- [ ] A/B variant per segment

### Academic
- [ ] Research paper v2 — รอ dataset ใหม่ + real causal labels จาก pilot

---

## สรุป Priority

| Priority | งาน | Phase |
|----------|-----|-------|
| 🔴 ทำตอนนี้ | รับ POS 12 เดือน + รัน PSM retrain | Phase 0 |
| 🔴 ทำตอนนี้ | ยืนยัน CRM มี LINE UID mapping | Phase 1 |
| 🟡 Phase 1 | ออกแบบ connector + planner UX | Phase 1 |
| 🟡 Phase 2 | ad_targeting_export + content generator | Phase 2 |
| 🟡 Phase 2 | LINE OA connector + Campaign Planner UI | Phase 2 |
| 🟢 Phase 3 | Pilot campaign + comparison report | Phase 3 |
| 🟢 หลัง pilot | Multi-brand, self-serve, Meta API | Scale |
| 🟢 ภายหลัง | SMS, TikTok, Thai LLM | Scale |

---

## PSM Retraining — Command Reference

```bash
# Phase 0: retrain บน DotDash 12 เดือน
python src/train_rlearner.py \
  --dataset psm \
  --pos-file data/dotdash_12months.csv \
  --psm-mode discount          # Mode A: มี discount column
  --observation-months 3

# Mode B: มี campaign record
python src/train_rlearner.py \
  --dataset psm \
  --pos-file data/dotdash_12months.csv \
  --psm-mode campaign_list \
  --campaign-file data/dotdash_campaign_ids.txt

# ทดสอบโดยไม่ promote champion
python src/train_rlearner.py --dataset psm ... --no-promote
```

ถ้า AUUC ชนะ champion ปัจจุบัน → promote อัตโนมัติ → serving pipeline ใช้ทันที
