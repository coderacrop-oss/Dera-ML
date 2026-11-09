# Repo 1 — Phase Spec & Acceptance Criteria
> อัปเดต: พฤศจิกายน 2026

---

## Phase 1 — Anonymous Intelligence
> เป้า: engine ทำงานได้จาก 100% anonymous transaction data

### Deliverables
- `src/ingest.py` — MeriPOS adapter + source detection + quality gate
- `src/store_intelligence.py` — store-level analytics จาก anonymous data
- `src/stock_forecast.py` — weekly demand forecast per item

### Acceptance Criteria

#### AC-1.1 MeriPOS Item Parser
- [ ] `ingest.py` detect MeriPOS format จาก column `"Bill Opens"` อัตโนมัติ
- [ ] parse Items column `"1x Latte, 2x Croissant"` → rows `{item, qty, receipt_id, timestamp}`
- [ ] handle edge cases: ไม่มี qty prefix, items คั่นด้วย `/` หรือ `,`
- [ ] test: 50 rows จาก Transaction Report จริง → item_df ไม่มี null item_name

#### AC-1.2 Source Detection & Quality Gate
- [ ] `detect_source(df)` คืน `"meripos"` / `"dotdash"` / `"standard"` ถูกต้อง
- [ ] `quality_gate(df)` คืน `{"mode": "full"|"partial"|"anonymous", "linked_rate": float}`
- [ ] threshold: full ≥ 0.30, partial 0.05–0.30, anonymous < 0.05
- [ ] test: Transaction Report ร้านฟันฟู้ดส์ → mode = "anonymous" (linked_rate < 0.05)

#### AC-1.3 Store Intelligence
- [ ] `store_intelligence.py` รับ MeriPOS DataFrame (anonymous) → คืน `StoreProfile` dataclass
- [ ] `StoreProfile` มี fields: `peak_hours`, `top_items` (top 10), `basket_avg_trend`, `channel_mix`, `discount_depth`, `promo_lift_proxy`
- [ ] peak_hours ใช้ `peak_hours.py` ที่มีอยู่แล้ว (ไม่ duplicate)
- [ ] channel_mix: `{"grab": 0.35, "in_store": 0.65}` หรือ key ที่มีจริง
- [ ] test: ใส่ DataFrame ว่าง → คืน StoreProfile พร้อม defaults ไม่ crash

#### AC-1.4 Stock Forecast
- [ ] `stock_forecast.py` รับ item_df → คืน `forecast_df` per item per week
- [ ] algorithm: 7-day rolling avg × day-of-week multiplier
- [ ] `forecast_df` columns: `item_name, week_start, predicted_qty, lower_bound, upper_bound`
- [ ] alerts: `get_stock_alerts(forecast_df, history_df)` คืน list ของ `{item, alert_type, message_th}`
- [ ] alert_types: `"spike"` (+30%), `"slow_mover"` (-50% ติด 3 wk), `"low_stock_risk"`
- [ ] test: synthetic 90-day data → forecast week 13 อยู่ในช่วง ±20% ของ actual

#### AC-1.5 Integration
- [ ] `campaign_pipeline.py` เรียก `store_intelligence()` และแนบ `store_profile` ไปใน context
- [ ] `layer1_csv_segmentation()` ถ้า mode = "anonymous" → skip RFM clustering, คืน store_profile เท่านั้น
- [ ] ไม่มี regression: existing tests ผ่านทั้งหมด

---

## Phase 2 — New API Endpoints
> เป้า: ทุก endpoint ใน API_SPEC.md ตอบได้จริง

### Deliverables
- `src/crm_api/campaign.py` — router ใหม่ (plan, analyze, execute, record-outcome, history)
- `src/crm_api/health.py` — router ใหม่ (segment/health)
- `src/crm_api/content.py` — router ใหม่ (content/generate)
- `src/crm_api/store.py` — simple JSONL storage helper
- `src/crm_api/app.py` — register routers ใหม่

### Acceptance Criteria

#### AC-2.1 Storage Layer
- [ ] `store.py` มี: `append_plan()`, `get_plans()`, `append_outcome()`, `append_snapshot()`
- [ ] ใช้ JSONL append-only: `data/campaign_history.jsonl`, `data/outcomes.jsonl`, `data/segment_snapshots.jsonl`
- [ ] thread-safe: atomic write (ใช้ `atomic_io.py` ที่มีอยู่)
- [ ] `get_plans(store_id, limit)` คืน list เรียง latest first
- [ ] test: parallel writes 10 threads → ไม่มี corrupt lines

#### AC-2.2 POST /campaign/plan
- [ ] รับ `{store_id, pos_data (base64 CSV), mode, budget|target_profit, goal}`
- [ ] decode base64 → temp file → เรียก `campaign_planner.run()`
- [ ] คืน response ตาม API_SPEC.md spec ครบทุก field
- [ ] บันทึกลง `campaign_history.jsonl` พร้อม `plan_id` unique
- [ ] test: ส่ง Transaction Report จริง (base64) → ได้ response ไม่ error

#### AC-2.3 POST /campaign/analyze
- [ ] รับ `{plan_id, campaigns: [{name, segment, budget, promo_type, discount_pct}]}`
- [ ] เรียก `promo_advisor` → คืน 4 checks: profitability, overlap, segment_readiness, budget_efficiency
- [ ] `can_proceed` = true เสมอ (advisory only)
- [ ] test: ส่ง plan ที่ขาดทุน → `profitability.warning = true`

#### AC-2.4 GET /segment/health
- [ ] รับ `store_id` query param
- [ ] เรียก `campaign_pipeline` → คืน 10 segments พร้อม urgency_score, priority, trend_pct
- [ ] บันทึก snapshot → `segment_snapshots.jsonl`
- [ ] trend_pct: compare กับ snapshot ก่อนหน้า ±15 วัน (ถ้าไม่มี → null)
- [ ] segments ที่ไม่มีลูกค้า → count=0 ยังคืนมา (ไม่ตัดออก)
- [ ] test: เรียก 2 ครั้ง → ครั้งที่ 2 มี trend_pct

#### AC-2.5 GET /campaign/history
- [ ] รับ `store_id`, `limit` (default 20), `goal` (optional filter)
- [ ] อ่านจาก `campaign_history.jsonl` → คืน list เรียง latest first
- [ ] test: ไม่มี history → คืน `{"plans": [], "total": 0}`

#### AC-2.6 POST /model/retrain
- [ ] รับ `{store_id, pos_file, psm_mode}`
- [ ] เรียก `retrain_trigger.py` async (background task)
- [ ] คืน `{"status": "queued", "job_id": "..."}` ทันที ไม่ block
- [ ] test: mock retrain_trigger → ได้ job_id กลับมา

#### AC-2.7 POST /campaign/record-outcome
- [ ] รับ `{plan_id, actual_revenue, actual_customers_responded, measurement_period_days}`
- [ ] บันทึกลง `outcomes.jsonl`
- [ ] เรียก `get_calibration_stats()` → คืน calibration summary
- [ ] test: บันทึก 3 outcomes → calibration stats มี trend field

#### AC-2.8 POST /audience/export
- [ ] รับ `{plan_id, segment_ids[], format: "line_uid"|"phone"|"csv"}`
- [ ] เรียก `ad_targeting_export.py` → คืน file download หรือ JSON list
- [ ] test: export format "csv" → response content-type = text/csv

#### AC-2.9 POST /content/generate
- [ ] รับ `{brief_type: "ml_driven"|"brand_driven", brief: {...}, brand_kit: {...}}`
- [ ] เรียก `content_generator.generate_content()`
- [ ] คืน `{headline, body, cta, image_prompt, meta_primary_text}`
- [ ] ถ้าไม่มี ANTHROPIC_API_KEY → คืน 503 พร้อม message ชัดเจน
- [ ] test: mock Claude API → ได้ ContentOutput ครบ fields

#### AC-2.10 App Registration
- [ ] `app.py` register ทุก router ใหม่ under `/v1`
- [ ] `GET /health` คืน list endpoints ที่ available
- [ ] OpenAPI docs ครบทุก endpoint ใหม่
- [ ] ทุก endpoint ต้องการ `X-API-Key` (ยกเว้น /health)

---

## Phase 3 — DotDash Connector
> เป้า: ดึง member data จาก DotDash API มาเสริม anonymous data

### Deliverables
- `src/dotdash_connector.py` — API client + member data fetcher
- `src/ingest.py` — เพิ่ม DotDash source handler
- `src/campaign_pipeline.py` — Hybrid RFM builder

### Acceptance Criteria

#### AC-3.1 DotDash API Client
- [ ] `dotdash_connector.py` fetch all members via cursor pagination
- [ ] รับ `api_token` หรือ อ่านจาก env `DOTDASH_API_TOKEN`
- [ ] retry 3 ครั้ง ถ้า rate limit หรือ timeout
- [ ] map columns: `{จำนวนครั้ง → frequency, ยอดใช้จ่าย → monetary, ใช้บริการครั้งล่าสุด → last_visit, ชื่อเล่น → display_name}`
- [ ] test: mock API response (cursor pagination 3 pages) → ได้ DataFrame ครบทุก member

#### AC-3.2 LINE userId Extraction
- [ ] ถ้า API response มี `line_user_id` field → map เข้า `customer_id`
- [ ] ถ้าไม่มี → ใช้ internal DotDash ID แทน + log warning
- [ ] test: member ที่สมัครผ่าน LINE OA → `customer_id` = LINE userId

#### AC-3.3 Hybrid RFM Builder
- [ ] `build_hybrid_rfm(meripos_df, dotdash_df, quality_mode)` → คืน rfm_df มาตรฐาน
- [ ] `full` mode: join via phone → RFM จาก MeriPOS transactions
- [ ] `partial` mode: join บางส่วน + ใช้ DotDash F/M/R เติมที่ join ไม่ได้
- [ ] `anonymous` mode: RFM จาก DotDash โดยตรง (F/M/R จาก member data)
- [ ] output format เหมือนกันทุก mode → pipeline ต้องไม่รู้ว่า mode ไหน
- [ ] test: anonymous mode → rfm_df มี customer_id, recency, frequency, monetary ครบ

#### AC-3.4 Data Quality Report
- [ ] `get_quality_report(meripos_df, dotdash_df)` คืน dict:
  ```json
  {
    "total_transactions": 1240,
    "linked_transactions": 62,
    "linked_rate": 0.05,
    "total_members": 145,
    "members_with_spend": 23,
    "recommendation": "สอนพนักงานสแกน QR สมาชิกทุกบิล"
  }
  ```
- [ ] test: ใส่ Transaction Report ฟันฟู้ดส์ + Customer Report → ได้ report ถูกต้อง

#### AC-3.5 Test Data Filter
- [ ] `filter_test_data(df)` กรองออก: net_sales < 10, campaign_name contains "test" (case-insensitive)
- [ ] log จำนวน rows ที่ถูก filter
- [ ] test: 5 rows test + 95 rows จริง → ได้ 95 rows

---

## Phase 4 — Adaptive Model
> เป้า: model ใช้ anonymous signals เสริม customer-level features

### Deliverables
- `src/store_intelligence.py` — เพิ่ม `get_store_embedding()`
- `src/campaign_pipeline.py` — ใช้ store embedding เสริม RFM
- `src/propensity_score_matching.py` — PSM Mode C ที่ดีขึ้น
- `src/train_rlearner.py` — hierarchical model flag

### Acceptance Criteria

#### AC-4.1 Store Embedding
- [ ] `get_store_embedding(store_profile)` → numpy array 38-dim fixed
  - peak_hour_pattern: 24-dim (normalized hourly fraction)
  - day_of_week_pattern: 7-dim (normalized daily fraction)
  - top_category_mix: 5-dim (top 5 categories share)
  - scalar features: discount_depth, basket_avg_norm, promo_lift_proxy (2-dim)
- [ ] deterministic: input เดิม → output เดิมทุกครั้ง
- [ ] test: 2 stores ที่มี peak hour ต่างกัน → cosine similarity < 0.7

#### AC-4.2 Store Features ใน Campaign Pipeline
- [ ] `build_serving_context()` แนบ `store_embedding` เข้า context
- [ ] `uplift_engine.py` ใช้ store_embedding เป็น additional features ถ้ามี
- [ ] backward compatible: ถ้าไม่มี store_embedding → ทำงานปกติ (ไม่ crash)
- [ ] test: serving context มี store_embedding → predict uplift ได้ ไม่ error

#### AC-4.3 PSM Mode C Improved
- [ ] Mode C ใช้ทั้ง purchase frequency spike AND campaign_history (ถ้ามี)
- [ ] ถ้ามี `campaign_history.jsonl` → treated proxy = spike ในช่วง ±7 วัน ของ campaign date
- [ ] ถ้าไม่มี history → fallback เป็น spike-only (behavior เดิม)
- [ ] test: ใส่ campaign date → treated assignment แม่นขึ้น vs spike-only

#### AC-4.4 Hierarchical Model Flag
- [ ] `train_rlearner.py` รับ flag `--model-tier global|category|store`
- [ ] `global`: เทรนบน aggregate data ทุกร้าน
- [ ] `category`: filter ตาม `--category food_and_beverage|retail|beauty|services`
- [ ] `store`: เทรนเฉพาะร้าน (ต้องมี ≥12 เดือน data ไม่งั้น warn + fallback category)
- [ ] model manifest บันทึก `tier` และ `category` ไว้ด้วย
- [ ] test: `--model-tier store` กับ data 6 เดือน → warning + ใช้ category model แทน

---

## Phase 5 — Execution Layer
> เป้า: Codera push campaign ได้เองผ่าน LINE OA และ Meta

### Deliverables
- `src/line_oa_client.py` — LINE Messaging API client
- `src/meta_ads_client.py` — Meta Ads API client
- `src/crm_api/campaign.py` — เพิ่ม POST /campaign/execute endpoint

### Acceptance Criteria

#### AC-5.1 LINE OA Broadcast
- [ ] `line_oa_client.py` รับ `{user_ids[], message_template, channel_access_token}`
- [ ] ใช้ LINE Multicast API (max 500 per call → auto-batch)
- [ ] คืน `{sent: int, failed: int, failed_ids: []}`
- [ ] rate limit handling: 429 → exponential backoff
- [ ] test (mock): 1200 user_ids → 3 batches ถูกต้อง

#### AC-5.2 Meta Ads Pipeline
- [ ] `meta_ads_client.py` steps:
  1. `upload_custom_audience(phone_list)` → audience_id
  2. `create_lookalike(audience_id, country="TH", ratio=0.01)` → lookalike_id
  3. `create_campaign(name, objective, budget_thb)` → campaign_id
  4. `create_ad_set(campaign_id, audience_id, budget)` → ad_set_id
  5. `create_creative(image_url, headline, body, cta)` → creative_id
  6. `create_ad(ad_set_id, creative_id)` → ad_id
- [ ] `get_ad_insights(ad_id)` → `{impressions, clicks, ctr, conversions, spend_thb}`
- [ ] test (mock): full pipeline 6 steps → ad_id returned, no errors

#### AC-5.3 POST /campaign/execute
- [ ] รับ `{plan_id, channel: "line_oa"|"meta", content_id, ad_budget_thb?}`
- [ ] ดึง audience จาก plan_id → เรียก LINE หรือ Meta client ตาม channel
- [ ] อัป status ใน `campaign_history.jsonl`: executed_at, channel, delivery_stats
- [ ] คืน `{execution_id, status: "sent"|"queued"|"failed", delivery_summary}`
- [ ] test (mock clients): execute LINE → history อัปเดต status ถูกต้อง

#### AC-5.4 Outcome Auto-capture (30-day trigger)
- [ ] `retrain_scheduler.py` เพิ่ม job: หลัง 30 วันจาก executed_at → trigger outcome check
- [ ] outcome check: pull MeriPOS transaction → วัด lift ของ segment ที่ได้รับโปร vs holdout
- [ ] บันทึก outcome อัตโนมัติ → trigger drift check
- [ ] test: mock scheduler + mock MeriPOS pull → outcomes.jsonl อัปเดต

---

## Definition of Done (ทุก Phase)

- [ ] unit tests ผ่านทั้งหมด (`pytest src/ tests/`)
- [ ] ไม่มี regression: existing test suite ผ่านครบ
- [ ] ไม่มี hardcoded thresholds — ทุก param อยู่ใน `business_config.py` หรือ env
- [ ] ทุก function มี docstring อธิบาย input/output
- [ ] error ทุกตัวมี log message ชัดเจน ไม่ silent fail
