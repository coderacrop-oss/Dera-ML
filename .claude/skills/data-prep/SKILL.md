---
name: data-prep
description: Use this skill when the user asks to analyze, prepare, or validate a dataset for training or testing the ml-campaign-engine uplift model. Triggers on phrases like "วิเคราะห์ดาต้า", "prepare data", "ตรวจสอบ dataset", "data ready to train?", "เช็ค dataset", or when the user provides a CSV file path and asks if it can be used.
version: 1.0.0
---

# Data Preparation Guide — ml-campaign-engine

## Overview

This skill guides analysis and preparation of POS transaction data for two purposes:
1. **Training** — building or retraining an uplift model
2. **Serving / Testing** — scoring existing customers with the current champion

The pipeline accepts two input formats:
- **POS transactions** (raw format, proxy labels derived automatically)
- **POS transactions + labels file** (real treatment/conversion labels)

---

## Step 1: Inspect the Raw File

When given a CSV, immediately run:

```python
import pandas as pd
df = pd.read_csv(path)
ts = pd.to_datetime(df['timestamp'], errors='coerce')

print(df.shape, list(df.columns))
print(df.head(3).to_string())
print(df.dtypes)
print(df.isnull().sum())
print(f"Customers: {df['customer_id'].nunique()}")
print(f"Date range: {ts.min().date()} → {ts.max().date()} ({(ts.max()-ts.min()).days}d)")
```

---

## Step 2: Required Column Check

### Minimum (POS / proxy mode)
| Column | Type | Notes |
|--------|------|-------|
| `customer_id` | str | unique customer identifier |
| `timestamp` | datetime-parseable | transaction datetime |
| `transaction_id` | str | bill/invoice ID (multiple rows per bill OK) |
| `product_name` | str | for bundle analysis (Layer 2) |
| `quantity` | int | per line item |
| `unit_price` | numeric | per unit |
| `total_price` | numeric | quantity × unit_price |

### With real labels (training / real-labels mode)
Additional columns required in a **separate labels CSV**:
| Column | Type | Notes |
|--------|------|-------|
| `customer_id` | str | matches POS file |
| `is_treated` | 0/1 | did the customer receive a promotion? |
| `post_promo_visit` | 0/1 | did they return after the promotion? |

Alternatively, the labels can be embedded in the POS file itself (same columns inline).

---

## Step 3: Data Quality Checks

### 3.1 Temporal Window

```python
span_days = (ts.max() - ts.min()).days
```

| Span | Assessment |
|------|------------|
| < 14 days | ❌ Too short — RFM and temporal split unreliable |
| 14–29 days | ⚠️ Short window warning (pipeline auto-adjusts split) |
| 30–89 days | ✅ Acceptable for serving; marginal for training |
| ≥ 90 days | ✅ Good for training |

**Watch for:** missing months / data gaps. Check `df.groupby(ts.dt.month)['transaction_id'].nunique()`. A missing month mid-window can cause the temporal split to land in a gap, producing a misleading conversion signal.

### 3.2 Control Group Size (training only)

```python
cust = df.groupby('customer_id')['is_treated'].max()
ctrl_ratio = (cust == 0).mean()
```

| Ratio | Assessment |
|-------|------------|
| < 5% | ❌ Hard block — model cannot estimate uplift |
| 5–15% | ⚠️ Warning — high variance in estimates |
| ≥ 15% | ✅ Acceptable |
| ≥ 30% | ✅ Good |

### 3.3 Treatment Effect Direction (training only)

```python
cust_conv = df.groupby('customer_id').agg(
    treated=('is_treated', 'max'),
    converted=('post_promo_visit', 'max')
)
ate = cust_conv[cust_conv.treated==1].converted.mean() \
    - cust_conv[cust_conv.treated==0].converted.mean()
```

| ATE | Assessment |
|-----|------------|
| < -0.10 | ❌ Hard block — likely confounded control group |
| -0.10 to 0 | ⚠️ Borderline — verify control group composition |
| > 0 | ✅ Treatment increases conversion (expected) |

### 3.4 Single-Visit Customers

```python
freq = df.groupby('customer_id')['transaction_id'].nunique()
single_visit_pct = (freq == 1).mean()
```

- > 40% single-visit → warn: low-frequency customers have thin RFM signal; model will place them in lowest deciles by default
- These customers are still servable but uplift estimates are less reliable

### 3.5 Dead / Constant Columns

Drop any column where `nunique() == 1` before training. Common examples:
- `promo_applied` = 0 everywhere (no promotion tracking)
- `store_id` = same value for single-store data

### 3.6 Selection Bias Check (training only)

Compare frequency distribution between treated and control:

```python
treated_freq = freq[cust[freq.index] == 1]
ctrl_freq    = freq[cust[freq.index] == 0]
print(treated_freq.describe(), ctrl_freq.describe())
```

If treated customers have significantly higher frequency than control (e.g., treated p90 >> control p90), the model is likely seeing loyal customers vs occasional customers — not random assignment. The X-learner's propensity weighting partially corrects this, but note the limitation in results.

---

## Step 4: Common Issues and Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| `promo_applied` all zeros | `df['promo_applied'].nunique() == 1` | Drop column; it carries no signal |
| Missing month in window | Monthly volume has a 0 | Note it; temporal split may skip the gap naturally |
| Inconsistent `is_treated` per customer | Same customer has both 0 and 1 across bills | Take max per customer — a customer is treated if they received at least one promo |
| `total_price` ≠ `quantity × unit_price` | Discounts applied at line level | Use `total_price` as the authoritative spend column |
| Timestamps missing timezone | Comparison issues | Normalize: `pd.to_datetime(df['timestamp'], utc=True)` |

---

## Step 5: Serving vs Training Decision

```
Given a CSV, ask:
  1. Does it have is_treated + post_promo_visit columns?
       YES → can use real-labels mode (training or real-labels serving)
       NO  → proxy mode only (serving only, or training with proxy labels)

  2. ctrl_ratio ≥ 15% AND ATE > -0.10?
       YES → suitable for training
       NO  → suitable for serving only; do not train

  3. span_days ≥ 30?
       YES → good temporal signal
       NO  → serving OK, training marginal
```

---

## Step 6: Training Command

When data passes all checks:

```bash
python src/train_champion.py \
  --store-data <STORE_ID>:<pos_file.csv>[:<labels.csv>] \
  --baseline-data <pos_file.csv> \
  --models-root models
```

For multi-store pooling:
```bash
python src/train_champion.py \
  --store-data store-a:data/store_a.csv \
  --store-data store-b:data/store_b.csv \
  --baseline-data data/store_a.csv \
  --baseline-data data/store_b.csv \
  --models-root models
```

`--baseline-data` sets the feature distribution baseline for drift detection (PSI monitoring at serving time). Include all populations the model will be asked to score.

---

## Step 7: Serving / Testing Command

To score customers and get campaign recommendations:

```python
import campaign_pipeline as cp
out = cp.run_layer4('data/myfile.csv', labels_path=None, models_root='models')

# Key output columns
out[['customer_id', 'recommended_action', 'expected_profit', 'should_send']]
```

To test recommendation diversity across customer segments, run `scripts/compare_scenarios.py` or combine multiple CSV files representing different customer archetypes.

---

## Step 8: Quality Gates Summary

The pipeline automatically blocks candidates that fail these gates:

| Gate | Threshold | Type |
|------|-----------|------|
| Control ratio | < 5% | Hard block |
| ATE direction | < -0.10 | Hard block |
| Control ratio | < 15% | Warning |
| ATE bias | > 50% relative error | Warning |
| Data window | < 30 days | Warning |
| Profit improvement vs champion | < 3% | Soft block |

---

## Dessert Dataset (dessert-ml.csv) — Analysis Notes

*As of 2026-11-04. Re-run checks whenever the file is updated.*

| Check | Value | Status |
|-------|-------|--------|
| Customers | 2,409 | ✅ |
| Date span | 332 days (2025-11-01 → 2026-09-29) | ✅ |
| Control ratio | 29.3% (707/2409) | ✅ |
| Naive ATE | +0.163 | ✅ |
| Nulls | None | ✅ |
| Duplicates | None | ✅ |
| Missing month | October (month 10) absent | ⚠️ |
| `promo_applied` | All zeros — dead column | ⚠️ Drop before training |
| Single-visit customers | 28.4% (683 customers) | ⚠️ Thin RFM signal |
| Selection bias | Treated p90=25 visits vs control p90=10 | ⚠️ Non-random assignment |
| Avg basket | 236 THB (p10=65, p50=169, p90=513) | ✅ Good spread |
| Frequency | p50=4 visits, p90=23 | ✅ Adequate for RFM |

**Verdict: Ready for training.** Drop `promo_applied` column. Be aware of October gap when interpreting temporal split results. Selection bias is present but X-learner propensity weighting will partially correct.

**Training command:**
```bash
python src/train_champion.py \
  --store-data dessert:data/dessert-ml.csv \
  --baseline-data data/dessert-ml.csv \
  --models-root models
```
