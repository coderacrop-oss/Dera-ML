# ML Campaign Engine — TODO

## ✅ แก้ไปแล้ว
- [x] layer4: model เทรนบน random labels → ใช้ RFM-based labels จริง
- [x] layer4: profit formula `prob × magic number` → Expected Value จริง
- [x] layer4: rule ordering conflict (low basket check ก่อน off-peak)
- [x] layer1_valuation: K-Means ทำแล้วไม่ได้ใช้ → dynamic k + silhouette จริง
- [x] layer2_preference: division by zero (Monetary = 0)
- [x] layer2_preference: NaN top_category/top_product ไหลไป layer3
- [x] layer2_preference: fallback กรณีไม่มี `category` column
- [x] layer3_campaign_matching: NaN top_cat ทำ offer string เสีย
- [x] layer3_campaign_matching: cluster ID hardcode → tier map จาก monetary จริง
- [x] csv_pipeline_engine: `i%500` invoice ID collision → proxy `customer_id + date`
- [x] layer5: model เปลี่ยนจาก qwen2.5:1.5b → llama3.2:3b
- [x] layer5: num_predict 500 → 3500
- [x] layer5: prompt เพิ่ม 5 sections รวม execution plan + post-campaign forecast
- [x] layer5: ส่ง Python dict ดิบ → pre-format เป็น text ก่อน
- [x] csv_pipeline_engine: คำนวณ retention rate + repeat purchase rate จากข้อมูลจริง

---

## 🟡 ค้างอยู่

### run_pipeline.py
- [x] **Baseline 81.5% hardcode** — แก้เป็น recency-based attrition: `min(avg_recency_days/365, 0.40)` ต่อ segment
- [x] **Fake customer IDs** — แก้เป็น `CUST_{i:06d}` (unique ทุก row) + invoice fallback ใช้ proxy `customer_id + date`
- [x] **Column mapping กิน column ผิด** — แก้เป็น pattern list exact match + duplicate conflict detection พร้อม warning
- [x] **Segment name มีแค่ 2 ชื่อ** — แก้เป็น monetary rank × recency quadrant: Champions / High Value (Dormant) / Potential Loyalists / Need Attention / New/Promising / At-Risk

### layer3_context.py
- [x] **Stock = 5000 ทุกสินค้า** — แก้เป็น `initial_stock_est = quantity_sold × 1.5` per product
- [x] **Timezone-naive peak detection** — แก้เป็น tz-localize/convert → Asia/Bangkok ก่อน extract hour

### layer4_uplift_margin.py
- [x] **Baseline 81.5% hardcode** — แก้เป็น time-split: baseline = avg spend ช่วง 75% แรก, current = 25% หลัง

### layer5_llm_summarizer.py
- [x] **LLM output ไม่ validate** — เพิ่ม `_validate_output()`: cross-check segment names, customer counts, profit values กับ payload → append warning block ถ้า mismatch

---

## 🟢 Nice to Have
- [x] **Configurable business parameters** — รวมทุก param ไว้ใน `src/business_config.py` → layer3/4, csv_pipeline_engine, run_pipeline.py import CONFIG แทน hardcode
- [ ] **Logging / observability** — ตอนนี้ failure เงียบ ไม่มี audit trail ระหว่าง layers
- [x] **layer1_rfm.py: hardcoded k=4** — แก้เป็น dynamic k via silhouette score (k=2..5)
