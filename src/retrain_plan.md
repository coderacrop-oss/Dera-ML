# Uplift Model Retrain Plan — Dunnhumby Complete Journey

## Summary of Data Investigation Findings

### Dataset 1: `raw_pos_transactions.csv` (current)
- Treatment and outcome labels are **randomized / semi-synthetic** (is_treated assigned by RFM proxy, post_promo_visit defined as WEEK_NO > median).
- No real experiment: treated and control groups are statistically indistinguishable by construction.
- Expected AUUC ~0.05 is correct — null effect is the ground truth.

### Dataset 2: `marketing_campaign.csv`
- 2,240 customer-level rows, 29 columns.
- **No treatment/control split.** `Z_CostContact = 3` for every single row — all customers were contacted. `Response` is acceptance of the 6th campaign, not a binary contacted/not-contacted flag.
- Cannot be used directly for uplift training without an external holdout/control arm.
- Secondary use only: demographic features (Income, Recency, Mnt* spend columns) could enrich a Dunnhumby join if customer IDs aligned — they do not.

### Dataset 3: Dunnhumby — The Complete Journey
- **2,500 households**, ~2.6 M transaction rows, 711 days (~2 years).
- `campaign_table.csv` assigns 1,584 households to 30 campaigns (TypeA/B/C describe mailer format, not treatment vs. control tier).
- **916 households never appear in any campaign** — these are the natural holdout / control group.
- Pre-campaign spend (weeks 1–52): treated mean = $2,032 vs. control mean = $433. Strong selection bias — high-value shoppers were preferentially targeted.
- Campaign 26 (TypeA, days 224–264) showed: treated spend lift = +$21.94, control spend lift = +$14.82. Raw ATE ≈ +$7 incremental spend, ~5% incremental conversion rate. This is a detectable real effect, not null.
- Demographics available for 801 of 2,500 households (801 rows in `hh_demographic.csv`); only 41 of those 801 are in the control group — demographic features are sparse for controls.
- `causal_data.csv` (36.8 M rows) has product-level `display` and `mailer` flags that can serve as additional treatment indicators for store-level models.

---

## Recommended Dataset: Dunnhumby Complete Journey

Dunnhumby is the only dataset with a genuine observed control group (households never targeted by any campaign) and a real outcome (spend or purchase during a defined campaign window). The marketing_campaign.csv dataset is unusable for uplift without a control arm.

---

## Treatment Definition (T)

### Primary definition — household-level, per-campaign
```
T = 1  if household_key appears in campaign_table for a given CAMPAIGN
T = 0  if household_key never appears in campaign_table (pure holdout)
```

Build one row per household per campaign period. For households in multiple campaigns, each campaign episode is an independent observation (with appropriate de-duplication of overlapping windows).

### Alternative definition — all-campaigns pooled
```
T = 1  if household_key in campaign_table (any campaign)
T = 0  if household_key in transaction_data but NOT in campaign_table
```
Simpler, but confounds campaign type and timing. Use as a sanity check, not the primary definition.

---

## Outcome Definition (Y)

### Primary outcome — binary conversion during campaign window
```
Y = 1  if household made at least one purchase during [START_DAY, END_DAY]
Y = 0  otherwise
```
This gives a clean binary target. Campaign period boundaries come from `campaign_desc.csv`.

### Secondary outcome — continuous spend lift
```
Y_lift = spend_during_campaign - spend_pre_campaign
```
Where pre-campaign window = same length as the campaign, immediately preceding START_DAY. Use as a regression target for the tau learners inside the X-learner.

### Outcome for pooled model
Use a campaign-agnostic post-period: days 356–711 (second half of the study, past the mid-point at week 52). This matches the spirit of the existing `post_promo_visit` but uses the actual study midpoint rather than a per-row WEEK_NO comparison.

---

## Feature Engineering

### Step 1 — Pre-campaign RFM (computed from transaction_data, weeks BEFORE campaign start)
| Feature | Definition |
|---|---|
| `recency_pre` | Days since last purchase before campaign START_DAY |
| `frequency_pre` | Number of unique basket days before START_DAY |
| `monetary_pre` | Total SALES_VALUE before START_DAY |
| `avg_basket_pre` | monetary_pre / frequency_pre |
| `weekly_visit_rate` | frequency_pre / (START_DAY / 7) |

### Step 2 — Demographics (from hh_demographic.csv; 801/2500 households)
| Feature | Notes |
|---|---|
| `age_group` | Ordinal-encoded AGE_DESC (25-34, 35-44, …) |
| `income_band` | Ordinal-encoded INCOME_DESC |
| `household_size` | Ordinal-encoded HOUSEHOLD_SIZE_DESC |
| `has_kids` | Binary from KID_CATEGORY_DESC != "None/Unknown" |
| `homeowner` | Binary from HOMEOWNER_DESC |

Demographics are missing for 1,699 households. Impute with the mode per income band cluster, or use a LightGBM model that handles NaN natively (no explicit imputation needed).

### Step 3 — Coupon history (from coupon_redempt.csv)
| Feature | Definition |
|---|---|
| `n_coupons_redeemed_pre` | Count of coupon redemptions before campaign window |
| `coupon_redeemer` | Binary: ever redeemed a coupon before this campaign |

### Step 4 — Product category affinity
Aggregate transaction_data by commodity group before the campaign window:
| Feature | Definition |
|---|---|
| `top_dept_pct` | Share of spend in top department (e.g., GROCERY) |
| `n_departments_visited` | Count of distinct DEPARTMENT values visited |
| `avg_coupon_discount_rate` | mean(COUPON_DISC / SALES_VALUE) — proxy for price sensitivity |
| `avg_retail_discount_rate` | mean(abs(RETAIL_DISC) / SALES_VALUE) |

### Step 5 — Campaign features (for multi-campaign model only)
| Feature | Definition |
|---|---|
| `campaign_type` | One-hot or ordinal: TypeA=2, TypeB=1, TypeC=0 |
| `campaign_duration_days` | END_DAY - START_DAY |
| `n_prior_campaigns` | How many campaigns this household previously received |

### Step 6 — Treatment propensity features
The same features above are used to train the propensity model P(T=1|X). Because selection bias is severe (treated households spend 4.7x more pre-campaign), propensity weighting is critical.

---

## Training Procedure

### Data split
- Per-campaign model: for each of 30 campaigns, hold out 20% of households (stratified by T) for evaluation.
- Pooled model: hold out a random 20% of households before any feature engineering.

### Class balance
Treated/control ratio is 1,584:916 ≈ 63%:37%. This is acceptable for X-learner. No oversampling needed. LightGBM's `scale_pos_weight` can handle mild imbalance at the outcome level if conversion is rare in control (<5%).

### X-learner configuration
The existing `_train_xlearner` in `src/layer4_uplift_margin.py` is already correctly implemented. The only change needed is the input dataframe construction:

1. Replace `was_treated` source: use the campaign_table join instead of RFM proxy.
2. Replace `converted` source: use the campaign-window purchase indicator instead of WEEK_NO median split.
3. Expand features to the full set described above (currently only 5–8 features; target 15–20).

### Propensity model concern
With such a large pre-campaign spend gap (treated 4.7x higher than control), the propensity model will assign very high P(T=1) to high-spenders. This compresses the CATE estimate toward tau_1 (the treated-side estimator). Add propensity clipping at [0.1, 0.9] to prevent extreme weights.

---

## Expected AUUC Range

| Scenario | Expected AUUC | Rationale |
|---|---|---|
| Current (null data) | ~0.05 | Randomized labels; no real treatment effect |
| Dunnhumby pooled model | 0.55–0.65 | Real selection effect inflates apparent AUUC; needs IPW correction |
| Dunnhumby per-campaign with propensity weighting | 0.60–0.72 | Genuine heterogeneous treatment effect exists (Campaign 26 showed ~5% incremental conversion), rich features enable meaningful ranking |
| Upper bound (best realistic) | ~0.75 | Would require demographic coverage for all households and store-level causal_data features |

Note: AUUC > 0.7 on this dataset without propensity weighting likely indicates selection bias leakage (the model is ranking high-value shoppers, not genuinely persuadable ones). Target 0.60–0.68 with proper IPW as a healthy realistic goal.

---

## Risks

### 1. Selection bias (HIGH severity)
Treated households were pre-selected — they spend 4.7x more in the pre-campaign period. A naive X-learner will learn to score high-value shoppers as high-uplift, conflating "likely to buy anyway" with "responds to promotion." Mitigation: propensity clipping, doubly-robust AUUC evaluation via `compute_dr_auuc.py`.

### 2. Confounding by campaign type (MEDIUM)
TypeA, TypeB, TypeC mailers differ in format/value. A pooled model conflates these. If the goal is production deployment, train separate models per campaign type or include campaign dummies.

### 3. Control group sparsity for demographics (MEDIUM)
Only 41 of the 916 control households have demographic data. Income and age features will be missing for 96% of controls, forcing imputation. This can introduce bias in the propensity model. Mitigation: use transaction-derived features as the primary covariate set; treat demographics as supplementary.

### 4. Temporal leakage (MEDIUM)
If any feature (e.g., recency, frequency) is computed using data after START_DAY, the model sees the future. All pre-campaign features must be computed with a hard cutoff at START_DAY - 1. Use a time-based assertion in the feature engineering pipeline.

### 5. Overlap / SUTVA violation (LOW-MEDIUM)
Some households appear in multiple campaigns. If campaign effects persist (carry-over), the "control" observation for a later campaign may already be a treated unit from an earlier campaign. Mitigation: for the per-campaign model, exclude households that were treated in any prior campaign from the control pool for that campaign.

### 6. Sparse positive outcomes in campaign windows (LOW)
Short campaigns (e.g., 40 days) may have low visit counts for infrequent shoppers. Binary Y = "any purchase" is the most robust choice. Continuous spend lift has higher variance and needs winsorization at the 99th percentile.

---

## Implementation Checklist

1. Run `data/prepare_dunnhumby.py` with the corrected outcome definition (campaign-window purchase, not WEEK_NO > median).
2. Add per-campaign feature engineering: compute RFM with `DAY < START_DAY` cutoff per household-campaign pair.
3. Update `_load_real_labels` in `src/layer4_uplift_margin.py` to accept the Dunnhumby format with explicit campaign window dates.
4. Add propensity clipping (0.1, 0.9) in `_train_xlearner` before the CATE combination step.
5. Evaluate with `compute_dr_auuc.py` (doubly-robust) in addition to standard AUUC to detect and correct for selection bias.
6. Log feature importance for mu0, mu1, tau0, tau1 separately to diagnose which stage is driving predictions.
