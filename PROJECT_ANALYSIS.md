# ML Campaign Engine — Project Analysis

## Overview

**Project:** ML Campaign Engine
**Purpose:** ระบบ MLOps แบบ multi-layer สำหรับ optimize campaign การตลาด e-commerce ผ่าน customer segmentation, product bundling, และ financial guardrails
**Business Context:** UK Online Gift & Home Decor Store (e-commerce retail)

---

## What This Project Does

ระบบนี้รับข้อมูล transaction แล้วประมวลผลผ่าน **5 layers** เพื่อสร้าง campaign action plan ที่ผ่าน CFO approval พร้อม A/B testing

```
Transaction CSV Data
    → Layer 1: RFM Segmentation (K-means)
    → Layer 2: Market Basket Analysis (Product Bundles)
    → Layer 3: Context Detection (Peak/Off-peak, Inventory)
    → Layer 4: Uplift ML Model + Validation Gate
    → Layer 5: LLM Report Generation
    → Executive Action Plan + Customer Assignments
```

---

## Directory Structure

```
ml-campaign-engine/
├── run_pipeline.py                  # Main entry point — CFO guardrail pipeline
├── update_engine.py                 # Engine update stub
├── generate_new_data.py             # Synthetic data generator
├── generate_stress_tests.py         # Stress test scenario generator
│
├── src/
│   ├── pipeline_engine.py           # In-memory RFM orchestrator
│   ├── csv_pipeline_engine.py       # CSV-driven orchestrator
│   ├── layer1_rfm.py                # Basic RFM clustering
│   ├── layer1_valuation.py          # Dual valuation + outlier filtering
│   ├── layer2_bundling.py           # Market basket analysis (Apriori)
│   ├── layer2_preference.py         # Customer category affinity profiling
│   ├── layer3_context.py            # Peak/off-peak + inventory flags
│   ├── layer3_campaign_matching.py  # Rule-based campaign assignment
│   ├── layer4_uplift_margin.py      # ML uplift model + validation gate
│   └── layer5_llm_summarizer.py     # Local LLM executive report
│
├── data/
│   ├── transactions.csv             # Main dataset (541,910 rows, real UK retail)
│   ├── raw_pos_transactions.csv     # Synthetic POS data (1,001 rows)
│   └── processed/                   # Layer outputs
│       ├── layer1_dual_valuation.csv
│       ├── layer2_top_bundles.csv
│       ├── layer3_context_flags.csv
│       └── final_campaign_engine_output.csv
│   └── stress_test/                 # 6 test scenarios
│
├── models/
│   ├── registry/
│   │   ├── champion_treat.pkl       # Production LGBMClassifier (treatment)
│   │   └── champion_ctrl.pkl        # Production LGBMClassifier (control)
│   └── archive/                     # Timestamped challenger versions
│
└── output/
    └── layer1_customer_segments.csv
```

---

## Pipeline Layers

### Layer 1 — Customer Segmentation (RFM)
- คำนวณ Recency, Frequency, Monetary ของแต่ละ customer
- ใช้ StandardScaler + K-means clustering (dynamic k=2-8 เลือกด้วย silhouette score)
- กรอง outlier ด้วย IQR method (3× upper quartile)
- Output: `rfm_decile_segment` (1-10), `kmeans_cluster` (0=low, 1=mid, 2=high)

### Layer 2 — Product Bundling & Preferences
- Apriori algorithm หา frequent itemsets (min_support=0.01)
- Association rules ด้วย lift metric
- Profile แต่ละ customer ว่าชอบ category อะไร
- Output: top 3 bundle rules, customer category affinity

### Layer 3 — Context & Campaign Matching
- ตรวจจับ peak hours (17:00–21:00) vs off-peak
- เช็ค inventory stock (flag ถ้า <5 units)
- Rule engine assign campaign ตาม cluster + category:
  - **Cluster 2 (Champions):** VIP Exclusive Rewards (free voucher + shipping)
  - **Cluster 1 (Regular):** Win-Back with 25% discount
  - **Cluster 0 (Base):** Loyalty Double Points

### Layer 4 — Uplift Modeling & Validation Gate
- Train LGBMClassifier แยก treatment vs control group
- คำนวณ uplift = prob_treatment − prob_control
- Decision rules: "VIP Reward", "Push Promotion", "Off-Peak Special", "Do Nothing"
- **Champion-Challenger**: รับ model ใหม่ถ้า profit growth ≥ 3% หรือ recession mode
- Output: `recommended_action`, `expected_profit` per customer

### Layer 5 — LLM Report Generation
- เรียก local Ollama server (model: qwen2.5:1.5b)
- สร้าง executive summary แบบ markdown
- Temperature 0.0 (deterministic), max 500 tokens

---

## Business Guardrails (CFO Rules)

```python
BUSINESS_PROFILE = {
    "cogs_rate": 0.50,               # ต้นทุนสินค้า 50%
    "min_allowable_net_margin": 0.15, # Net margin ต้องไม่ต่ำกว่า 15%
    "control_group_rate": 0.05,      # 5% ลูกค้าเป็น control group (ไม่รับ promo)
    "currency": "$"
}
```

- ทุก campaign ต้องผ่าน margin floor 15%
- 5% control group ไม่รับ promotion เพื่อวัด incremental lift จริง
- Discount cost ต้องมี uplift analysis รองรับ

---

## Key Algorithms

| Layer | Algorithm | Purpose |
|-------|-----------|---------|
| 1 | K-means + Silhouette | Customer segmentation |
| 1 | IQR outlier detection | Data quality |
| 1 | RFM decile ranking | Customer value tier |
| 2 | Apriori + Association Rules | Product bundles |
| 3 | Time/inventory rules | Context flags |
| 4 | LGBMClassifier | Uplift propensity |
| 4 | Champion-Challenger | Model versioning |
| 5 | qwen2.5:1.5b (ollama) | Report generation |

---

## Execution Entry Points

```bash
# Main pipeline — CFO guardrail + A/B testing
python run_pipeline.py

# In-memory engine (synthetic data, real-time)
python src/pipeline_engine.py

# CSV-driven engine (real data, batch)
python src/csv_pipeline_engine.py

# Generate synthetic data
python generate_new_data.py

# Generate stress test scenarios
python generate_stress_tests.py
```

---

## Technologies

- **Data:** pandas, numpy
- **ML:** scikit-learn (KMeans, StandardScaler), lightgbm (LGBMClassifier)
- **MBA:** mlxtend (apriori, association_rules)
- **LLM:** ollama (local, qwen2.5:1.5b)
- **Serialization:** pickle

---

## Stress Test Scenarios

| File | Scenario |
|------|----------|
| `mega_sale.csv` | Black Friday / Flash sale |
| `recession.csv` | Economic downturn |
| `bulk_shift.csv` | Bulk buyer behavior shift |
| `cold_start.csv` | New market entry |
| `off_peak.csv` | Off-peak hour optimization |
| `data_noise.csv` | Data quality robustness |

---

## Three Pipeline Modes

| Mode | File | Use Case |
|------|------|----------|
| In-memory | `pipeline_engine.py` | Real-time, monthly re-tuning |
| CSV-driven | `csv_pipeline_engine.py` | Batch, audit trail, reproducibility |
| CFO guardrail | `run_pipeline.py` | Financial compliance, exec reporting |

---

## Sample Output

```
MLOPS PROFIT-DRIVEN CAMPAIGN ENGINE - EXECUTIVE ACTION PLAN (A/B TESTING)
Planning Month: November 2026 | Engine: MLOps Production v6.0
- COGS Rate: 50% | Min Net Margin Floor: 15% | Control Group Split: 5%
- Total Transactions: 541,910 | Unique Customers: 600+

Champions (Cluster 2) — 33.3% of customers
- Last purchase 5.2 days ago | 12.3 transactions | $1,245 AOV
- A/B Split: 1,000 treatment | 53 control
- Campaign: VIP Exclusive Access
- Net Profit/Head: $187.50 | Margin: HEALTHY
```
