# POS-Native MLOps: An Accessible Five-Layer Campaign Intelligence Framework for Small and Medium Enterprises in Retail and Food Service

**Author:** หนึ่ง หงษ์โต
**Affiliation:** Codera Solutions
**Version:** Draft v1.2 — Confidential
**Date:** November 2026
**Status:** Working Paper — Internal Review

*© 2026 Codera Solutions · All rights reserved · Proprietary and Confidential*
*This document contains proprietary research belonging to Codera Solutions. Unauthorized reproduction, distribution, or disclosure is strictly prohibited.*

---

## Abstract

Small and medium enterprises (SMEs) in retail and food service routinely collect transactional data through point-of-sale (POS) systems yet rarely leverage it for marketing decisions, primarily due to resource constraints and the technical complexity of existing analytics solutions. This paper presents a five-layer MLOps pipeline — the **ML Campaign Engine** — that transforms raw POS transaction records requiring only six standard fields into financially validated, per-customer campaign action plans without requiring a dedicated data science team. The framework integrates RFM-based customer segmentation (K-means with dynamic cluster selection via silhouette scoring), market basket analysis (Apriori), dynamic context detection, uplift modeling (LightGBM), and a local large language model (LLM) reporting layer. A novel **CFO Guardrail** mechanism enforces a configurable minimum net margin floor prior to campaign dispatch, designed to ensure that no promotion is deployed at a financial loss regardless of predicted uplift score — an architectural contribution that embeds financial constraint validation directly into the ML decision loop, as validated on synthetic and public datasets in this paper. Layer 4 produces a per-customer `should_send` flag and exports a `broadcast_list.csv` ready for direct ingestion by automation tools (email, SMS, LINE OA), with per-segment approval counts surfaced in the Layer 5 executive report. COGS rate and discount rate are estimated automatically from data when sufficient transaction history is available, eliminating manual calibration for new deployments. The framework is evaluated on four datasets: (1) a synthetic restaurant POS dataset calibrated to F&B SME industry benchmarks (9,344 transaction line items, 200 customers, 6-month observation window; silhouette = 0.679; k = 2); (2) a public real-world marketing campaign dataset containing ground-truth promotion response labels (905 customers, aggregated spending records; silhouette = 0.347; k = 3); (3) the Dunnhumby Complete Journey public grocery dataset (2,500 households, 2,576,815 transaction rows, 117-week observation window; silhouette = 0.595; k = 3; data-driven COGS = 62.9%, discount = 7.6%; 103/2,500 customers approved for broadcast after all guardrail and action filters); and (4) a UK online retail e-commerce dataset (397,884 transaction line items, 4,338 customers; silhouette = 0.931 — the highest across all datasets; k = 2; 30/4,338 customers approved for broadcast). A T-learner uplift model achieves an AUUC of 0.2351 on the synthetic dataset, outperforming random targeting at identifying high-responders. A consistent finding across real-world datasets is that a naive broadcast campaign — sending promotions to all customers — produces model-estimated incremental profit *worse than running no campaign at all*, because promo costs exceed incremental revenue for the majority of customers who would purchase regardless. The CFO Guardrail converts money-losing broadcast scenarios into net-positive outcomes: +$167 on Dunnhumby (vs. −$262 broadcast-all) and +$32,006 on UK retail (vs. +$1,176 broadcast-all). The pipeline's advantage over naive broadcast scales consistently with dataset size: +0.3% at n=200 (synthetic, randomized labels), +257% at n=2,500 (Dunnhumby, observational proxy), and +2,622% at n=4,338 (UK retail, observational proxy) — with guardrail savings of $429, $30,830, and ฿2,695 respectively. All profit figures are model-estimated projections on synthetic or public datasets — not realized revenue from live deployment. Real-world profit impact will be measured in a forthcoming field study with a CRM-integrated POS operator using randomized holdout control groups. Results demonstrate that the proposed framework is robust across F&B, grocery, and e-commerce contexts, computationally accessible to SME hardware, and architecturally extensible to additional business verticals.

**Keywords:** MLOps, SME marketing, uplift modeling, RFM segmentation, market basket analysis, campaign optimization, POS analytics, LightGBM, CFO guardrail, Champion-Challenger

---

## 1. Introduction

Marketing personalization has become a competitive necessity in modern retail. Large enterprises routinely deploy dedicated analytics teams, expensive CRM platforms, and data warehouses to generate campaign recommendations at the individual customer level. For small and medium enterprises (SMEs) — which constitute the majority of businesses in most economies — this infrastructure remains economically and technically out of reach. The consequence is that SME operators make campaign decisions based on intuition or blanket promotions, systematically under-leveraging the transactional data they already possess.

This resource asymmetry creates a measurable market opportunity. POS systems are near-ubiquitous in modern retail and food service, and the transaction data they generate contains rich signals about customer value, product affinity, and purchasing patterns. If an automated pipeline could consume the POS export that any modern cash register already produces and return campaign actions validated against the operator's own financial constraints, the marginal cost of personalized marketing could approach zero for SMEs.

The system described in this paper, referred to as the **ML Campaign Engine**, was designed to address this gap. Its design is governed by four principles:

1. **Minimal input barrier** — the only required data is a standard six-column POS export (`transaction_id`, `customer_id`, `timestamp`, `total_price`, `quantity`, `product_name`). No additional data infrastructure, tagging, or transformation is required from the operator.
2. **Financial safety by default** — every campaign recommendation is blocked if it violates a configurable net margin floor, decoupling ML optimization from financial risk without requiring a human review step.
3. **Operator-readable output** — a local LLM layer translates model outputs into plain-language executive summaries suitable for business owners without a data background.
4. **Production-grade robustness** — a Champion-Challenger update protocol and a six-scenario stress test suite ensure reliable behavior under distribution shifts common in SME operations.

### 1.1 Research Objectives

The primary objective of this work is:

> To design and validate a minimal-input MLOps pipeline that enables SMEs in retail and food service to generate per-customer campaign action plans with integrated financial constraint validation, without requiring dedicated data science expertise.

Specific objectives are:

1. **To design** a five-layer pipeline architecture that accepts standard six-column POS exports without additional infrastructure requirements.
2. **To develop** a CFO Guardrail mechanism that integrates net margin validation directly into the ML campaign decision loop.
3. **To evaluate** segmentation quality and campaign profitability on a synthetic F&B dataset and a public real-world marketing campaign dataset.
4. **To describe and implement** the Champion-Challenger model update protocol and assess its behavior on synthetic data, and ablate each pipeline component to quantify its individual contribution.
5. **To demonstrate** the extensibility of the framework across business verticals through configurable business parameters.

### 1.2 Contributions

The primary contributions of this work are:

- A **five-layer MLOps architecture** designed from the ground up for SME operational constraints (Section 3).
- A **CFO Guardrail** component that integrates financial validation directly into the ML decision loop, eliminating a manual organizational review step (Section 4.4).
- **Empirical evaluation** on a synthetic F&B dataset (200 customers, silhouette = 0.679, AUUC = 0.2351) and a public real-world marketing campaign dataset (905 customers), including ablation study, baseline comparison, and multi-seed stability analysis (Section 6).
- A **Champion-Challenger protocol** adapted for SME environments with limited historical data (Section 4.4).

---

## 2. Related Work

### 2.1 RFM Segmentation

Recency-Frequency-Monetary (RFM) analysis is a well-established customer segmentation technique in direct marketing, originally formalized by Hughes [1] and subsequently refined for database marketing applications by Bult and Wansbeek [2]. Classical RFM assigns customers to decile ranks across each dimension; more recent work by Chen et al. [3] augments this with clustering algorithms to capture non-linear segment boundaries that decile ranking cannot express. Our Layer 1 extends classical RFM with K-means clustering [4] and dynamic cluster selection via silhouette scoring [5], and adds an IQR-based outlier filter [6] to prevent high-spend anomalies from distorting segment centroids — a practical concern in SME datasets where a single bulk transaction can represent a substantial fraction of total revenue.

### 2.2 Market Basket Analysis

Association rule mining, particularly the Apriori algorithm introduced by Agrawal and Srikant [7], has been widely applied in retail to identify product affinity. The algorithmic framework was later made accessible to practitioners through software implementations such as arules [8]. Prior work focuses largely on large-scale grocery and e-commerce datasets [9]. Application to SME food service and mixed retail — where transaction volumes are lower and basket sizes smaller — remains underexplored. Our Layer 2 addresses this by operating at a minimum support threshold of 0.01, appropriate for smaller datasets, and falls back gracefully when no product category column is present in the POS export.

### 2.3 Uplift Modeling

Uplift modeling (also called incremental response modeling or true-lift modeling) estimates the causal effect of a campaign on an individual customer, rather than the probability of conversion in the presence of a campaign — a distinction first formally articulated by Radcliffe and Surry [10]. A comprehensive review by Devriendt et al. [12] surveys the state of the art across tree-based, regression, and meta-learner approaches. Gutierrez and Gérardy [11] further connect uplift modeling to causal inference frameworks. Gradient boosted tree models, including XGBoost [15] and LightGBM [14], have shown strong performance in uplift tasks [12]; decision-tree-based uplift methods were systematically compared by Rzepakowski and Jaroszewicz [13]. Existing literature focuses predominantly on e-commerce and subscription contexts. Application to POS-driven SME campaigns — with the additional constraint of per-customer margin validation — is a novel contribution of this work.

### 2.4 MLOps for SMEs

The MLOps literature has grown substantially around large-scale deployments, with Sculley et al. [16] identifying the hidden technical debt incurred when ML systems are embedded in production software, and Kreuzberger et al. [17] providing a formal definition and architecture taxonomy for MLOps. Paleyes et al. [18] survey real-world deployment challenges across a large body of case studies. Work specifically addressing MLOps design for SME constraints — limited compute, small datasets, absence of dedicated ML engineers — remains sparse. Model governance practices including Champion-Challenger selection are discussed in Shmueli et al. [19]. Fairness and bias considerations in ML systems — relevant to equitable campaign targeting across customer segments — are addressed in Barocas et al. [20]. Our Champion-Challenger protocol extends governance practice with a financially motivated update gate that promotes a challenger model only when it demonstrates measurable profit improvement over the incumbent champion.

### 2.5 LLM Integration in Business Pipelines

Large language model capabilities and limitations are surveyed in Zhao et al. [22], while hallucination risks in generative systems are systematically catalogued by Ji et al. [23] — motivating our structured post-generation validation step. We adopt a local, quantized LLM (llama3.2:3b via Ollama) [21] to ensure that no customer transaction data leaves the operator's infrastructure, addressing data privacy concerns relevant to SME deployments. The local inference approach trades generation quality for zero ongoing inference cost and full data sovereignty — a trade-off appropriate for SME contexts.

### 2.6 SME Technology Adoption

Barriers to technology adoption among SMEs are well documented. Nguyen et al. [25] confirm that perceived complexity and resource constraints are the primary inhibitors of IT adoption in small businesses, consistent with the access gap in marketing analytics that this work addresses. CRM adoption patterns in SME contexts are examined by Harrigan et al. [24], who find that relationship management tools are most effective when they reduce rather than increase operational complexity — a design principle reflected in the plain-language LLM output layer of this framework.

---

## 3. System Architecture

The ML Campaign Engine is a five-layer sequential pipeline. Each layer consumes the output of the previous layer, applies a specific algorithm, and passes enriched customer records downstream. Figure 1 illustrates the overall data flow.

```
POS Transaction CSV (6 columns)
        │
        ▼ [data-driven COGS & discount estimation]
┌─────────────────────────────────────┐
│  Layer 1: RFM Segmentation          │
│  K-means + Silhouette + IQR filter  │
│  Rank-based segment naming          │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Layer 2: Product Bundling &        │
│  Category Preference Profiling      │
│  Apriori + Top-150 product filter   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Layer 3: Context Detection         │
│  Peak/Off-peak + Inventory Flags    │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Layer 4: Uplift Model +            │
│  CFO Guardrail + Champion-Challenger│
│  LightGBM · should_send flag        │
│  Adaptive basket thresholds         │
└──────────┬─────────────┬────────────┘
           │             │
           ▼             ▼
  broadcast_list.csv   merge_layer4_into_payload()
  (automation payload)  (per-segment approval counts)
                         │
                         ▼
            ┌─────────────────────────────────────┐
            │  Layer 5: LLM Executive Report      │
            │  llama3.2:3b (local, offline)       │
            │  Streaming · 10,000 token budget    │
            └────────────────┬────────────────────┘
                             │
                             ▼
               Per-Customer Campaign Action Plan
               + Executive Summary (Markdown)
               + broadcast_list.csv (automation)
```

*Figure 1. Five-layer ML Campaign Engine pipeline. Layer 4 produces two outputs: a broadcast list CSV for automation systems and an enriched segment payload for Layer 5.*

The pipeline has a single production entry point:

| Mode | Entry Point | Use Case |
|------|-------------|----------|
| CSV-driven (production) | `csv_pipeline_engine.py` | Batch processing, audit trail, financial compliance, executive reporting |

All business parameters — COGS rate, margin floor, discount rates, peak hours, timezone — are centralized in a single `business_config.py` file, enabling deployment across business types without code changes. When sufficient transaction history is available (≥10 distinct product types, ≥5 observations per type), COGS rate and discount rate are estimated automatically from data, overriding config defaults.

---

## 4. Methodology

### 4.1 Layer 1 — RFM Customer Segmentation

**Input:** Raw POS transactions
**Output:** Per-customer RFM decile (1–10) and K-means cluster assignment

We compute three metrics per customer from raw transaction history:

- **Recency (R):** Days elapsed since the customer's most recent transaction, relative to the analysis date
- **Frequency (F):** Total count of distinct transactions
- **Monetary (M):** Cumulative spend across all transactions

**Outlier filtering:** Prior to clustering, we apply Tukey-style outlier filtering on Monetary values. Records with Monetary > Q3 + 3 × IQR are excluded [6], where Q3 is the 75th percentile and IQR = Q3 − Q1. This prevents bulk buyers — common in SME data — from distorting segment centroids.

**Normalization:** All three dimensions are standardized to zero mean and unit variance (StandardScaler) to prevent Monetary (typically in hundreds) from dominating Recency and Frequency (typically single digits).

**Dynamic cluster selection:** Cluster count *k* is selected from *k* ∈ {2, ..., 5} by maximizing the silhouette score:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where *a(i)* is the mean intra-cluster distance and *b(i)* is the mean nearest-cluster distance for point *i*. The silhouette score ranges from −1 to +1, with higher values indicating better-defined clusters. Avoiding hardcoded *k* is critical for SME deployment: a boutique café (k=2) and a multi-category retailer (k=5) exhibit structurally different customer distributions.

**Segment naming:** Clusters are ranked by a composite RFM score:

$$\text{rfm\_score} = 0.3 \times \frac{\max R - R_k}{\max R} + 0.3 \times \frac{F_k}{\max F} + 0.4 \times \frac{M_k}{\max M}$$

where $R_k$, $F_k$, $M_k$ are the centroid recency, frequency, and monetary of cluster $k$. Clusters are sorted by this score descending and assigned names from the pool: *1. Champions & Loyal Customers*, *2. Potential Loyalists*, *3. At-Risk / Churning Customers*, *4. Promising & Recent Customers*, *5. Hibernating Customers*. Rank-based assignment guarantees unique names regardless of cluster count, eliminating duplicates that arise when multiple clusters straddle a median threshold.

### 4.2 Layer 2 — Market Basket Analysis and Category Profiling

**Input:** Raw POS transactions
**Output:** Top-3 association rules; per-customer category affinity vector

We apply the Apriori algorithm to find frequent product co-purchase patterns. Three metrics govern rule quality:

- **Support:** Fraction of transactions containing both items — set at minimum 0.01 for SME-scale data
- **Confidence:** Conditional probability P(B | A)
- **Lift:** Ratio of observed co-purchase frequency to expected frequency under independence. Lift > 1 indicates positive association; higher lift indicates stronger bundle potential

Rules are ranked by lift and the top three are surfaced for bundle campaign assignment. In parallel, each customer's spend is aggregated by product category to produce a ranked `top_category` and `top_product` affinity profile used by Layer 3.

A graceful fallback is implemented: if the input CSV lacks a `category` column — common in basic POS exports — `product_name` is used as a proxy category, ensuring compatibility with minimally structured data.

**Large-dataset pre-filter:** Before constructing the transaction-item matrix, only the top-150 most frequently occurring products are retained. This bounds the basket matrix to a manageable size on commodity hardware regardless of catalog breadth, and concentrates association rule mining on products with sufficient support to produce actionable recommendations.

### 4.3 Layer 3 — Context Detection and Campaign Matching

**Input:** Layer 1 cluster assignments, Layer 2 category profiles, timestamps
**Output:** Per-transaction context flags; initial campaign assignment

**Temporal context:** Timestamps are localized to the business timezone (configurable; default Asia/Bangkok) before hour extraction. Transactions between the configured peak window (default 17:00–21:00) are flagged `Peak`; all others `Off-Peak`. Off-peak promotions target demand stimulation during low-traffic periods, preserving full-margin sales during peak hours.

**Inventory estimation:** In the absence of live inventory feeds — standard in SME deployments — remaining stock is estimated from a velocity-based replenishment model:

$$\text{initial\_stock\_est} = \frac{\text{total\_quantity\_sold}}{T} \times R \times \text{buffer\_multiplier}$$

$$\text{remaining\_stock} = \text{initial\_stock\_est} - \text{recent\_demand}_{R}$$

where *T* is the observation window in days, *R* is the assumed replenishment cycle (default 14 days), and `recent_demand_R` is quantity sold in the last *R* days. This formulation correctly identifies fast-moving products with accelerating recent demand — rather than low-volume products — as candidates for stock-out exclusion. Products with remaining_stock ≤ 5 units are flagged out-of-stock and excluded from bundle recommendations.

**Campaign matching** applies a rule engine combining cluster tier, category affinity, and context flags to produce an initial campaign assignment, subsequently refined by Layer 4's uplift and margin analysis.

### 4.4 Layer 4 — Uplift Modeling and CFO Guardrail

**Input:** Enriched customer records from Layers 1–3; data-driven COGS and discount rate estimates
**Output:** Per-customer `uplift_score`, `recommended_action`, `expected_profit`, `should_send` flag; `broadcast_list.csv`; champion/challenger model comparison report; enriched segment payload for Layer 5

#### 4.4.0 Data-Driven Rate Estimation

Prior to uplift modeling, COGS rate and discount rate are estimated from transaction data when sufficient coverage exists:

**COGS rate:** The P10 (10th percentile) of unit price per product is treated as a proxy for cost floor (below which a sale would represent a loss). The data-driven COGS rate is:

$$\hat{c}_{\text{COGS}} = \text{mean}\left(\frac{P10_j}{\bar{p}_j}\right) \text{ for all products } j \text{ with } n_j \geq 5$$

where $P10_j$ is the 10th percentile of `unit_price` for product $j$ and $\bar{p}_j$ is its mean price. This is only activated when ≥10 distinct products satisfy the coverage threshold; otherwise the config default (0.50) is used.

**Discount rate:** Estimated from the observed frequency of discounted transactions:

$$\hat{d} = f_{\text{promo}} \times d_{\text{config}}$$

where $f_{\text{promo}}$ is the fraction of rows with `promo_type ∈ {coupon, discount}` and $d_{\text{config}}$ is the configured default discount rate. This anchors the estimate to the operator's actual promotional activity rather than an assumed rate.

Both estimates are logged at runtime with their source label (`data_driven` or `config_default`), enabling full auditability.

#### 4.4.1 Uplift Model (T-Learner)

The pipeline implements a T-learner (two-model approach) [11]: two LightGBM classifiers are trained independently on disjoint subgroups:

- **Treatment model:** trained on customers who received at least one campaign (`was_treated = 1`)
- **Control model:** trained on customers who received no campaign (`was_treated = 0`)

Per-customer uplift is estimated as:

$$\text{uplift}_i = \hat{P}(\text{convert} \mid \text{treatment})_i - \hat{P}(\text{convert} \mid \text{control})_i$$

Positive uplift indicates incremental expected purchase attributable to the campaign. Negative uplift indicates the customer would likely purchase regardless, making campaign spend wasteful.

**Important caveat:** T-learner uplift is only causally interpretable when treatment assignment is randomized. In the absence of a randomized experiment — typical at first SME deployment — the model produces a *response-propensity heuristic*, not a causal estimate. The 5% ongoing holdout control group enables properly randomized uplift measurement to accumulate over time, converting the system from a batch propensity scorer into a growing randomized experiment.

**Label sourcing — real vs. proxy:** The pipeline supports two label modes selected automatically at runtime:

1. **Real promotion response mode** (preferred): When `is_treated` and `post_promo_visit` columns are present in the POS file, the system uses these as treatment and conversion labels. When a `marketing_campaign.csv` file is additionally present with `AcceptedCmp1`–`AcceptedCmp5` and `Response` columns, three additional behavioral features are activated: `income`, `total_spend`, and `times_accepted`. Note that `AcceptedCmp` columns record whether a customer *accepted* a campaign offer — they are observational outcomes, not randomized treatment assignments. Uplift estimates from this source reflect self-selection bias and should be interpreted as directional signals rather than causal effects.

2. **RFM proxy mode** (fallback): When no promotion response file is available — the common case for first-time SME deployments — treatment labels are derived from RFM decile rank (customers above the median decile are proxied as treated) and conversion labels from repeat purchase frequency. This mode introduces circularity: frequency contributes to both the treatment proxy and the outcome proxy. It should be interpreted strictly as a propensity-to-respond ranking, not as an uplift estimate. Upgrading to real label mode as promotion response data accumulates is the intended operational path.

The feature sets differ between modes:

| Feature | Proxy Mode | Real Label Mode |
|---------|-----------|----------------|
| `total_price` (AOV) | ✅ | ✅ |
| `quantity` | ✅ | ✅ |
| `is_campaign_spike` | ✅ | ✅ |
| `rfm_decile_segment` | ✅ | ✅ |
| `frequency` | ✅ | ✅ |
| `income` | ❌ | ✅ |
| `total_spend` | ❌ | ✅ |
| `times_accepted` | ❌ | ✅ |

A 5% holdout control group is preserved in every deployment cycle for ongoing A/B measurement of realized incremental lift. As real promotion response data accumulates through partner deployments, operators automatically transition from proxy mode to real label mode without pipeline reconfiguration.

#### 4.4.2 Decision Rules

Final action assignment integrates uplift score, RFM decile, AOV, and context flags through a priority-ordered rule set. AOV thresholds are applied adaptively: if the dataset median `avg_spent` falls below the configured `low_basket_threshold` (e.g. grocery datasets where per-transaction AOV is $2–5), thresholds are replaced by the P10 and P50 of actual `avg_spent` in the current dataset. This prevents all customers from receiving `Do Nothing (Low Basket Size)` on low-AOV datasets calibrated for wholesale.

| Priority | Condition | Assigned Action |
|----------|-----------|-----------------|
| 1 | `in_stock = False` | Do Nothing (Out of Stock) |
| 2 | uplift < −0.04 | Do Nothing (Negative Uplift Guard) |
| 3 | AOV < P10(dataset) *or* AOV < `low_basket_threshold` | Do Nothing (Low Basket Size) |
| 4 | CFO Guardrail fails (net_margin < floor OR Δprofit ≤ 0) | Do Nothing (CFO Guardrail) |
| 5 | uplift > 0.08 AND bundle available | Push Promotion (Bundle) |
| 6 | RFM decile ≥ 9 | VIP Early Access |
| 7 | RFM decile ≥ 8 AND P(control) ≥ 0.4 | VIP Reward: X2 Points |
| 8 | Off-Peak AND P10 ≤ AOV ≤ P50 AND RFM ≥ 4 | Off-Peak Special |
| 9 | Default | Standard Retention |

**`should_send` flag:** After all decision rules are applied, a binary `should_send` column is computed per customer:

$$\text{should\_send}_i = \mathbb{1}[\text{guardrail\_blocked}_i = 0 \;\wedge\; \text{action}_i \not\in \{\text{Do Nothing variants}\}]$$

This flag is the authoritative signal consumed by downstream automation systems. All "Do Nothing" variants — including CFO Guardrail, Low Basket Size, Out of Stock, and Negative Uplift Guard — produce `should_send = 0`, regardless of their guardrail_blocked value.

#### 4.4.3 CFO Guardrail

Before finalizing any campaign assignment, two per-customer financial checks are applied:

**1. Net margin floor check (per campaign type):**

$$\text{gross\_revenue}_i = \text{AOV}_i \times (1 - d)$$

$$\text{net\_profit}_i = \text{gross\_revenue}_i - \text{AOV}_i \times c_{\text{COGS}} - \text{AOV}_i \times r_{\text{promo}}$$

$$\text{net\_margin}_i = \frac{\text{net\_profit}_i}{\text{gross\_revenue}_i}$$

where $d$ is the campaign discount rate and $r_{\text{promo}}$ is the promo cost rate (both specific to the assigned action type). If `net_margin_i < min_allowable_net_margin` (default 0.15), the action is overridden to `Do Nothing`.

**2. Per-customer incremental profit check:**

$$\Delta\text{profit}_i = [\text{gross\_margin}_i \times \hat{P}(\text{convert} \mid \text{treated})_i - \text{promo\_cost}_i] - [\text{gross\_margin}_i \times \hat{P}(\text{convert} \mid \text{control})_i]$$

If $\Delta\text{profit}_i \leq 0$, the campaign is expected to cost more than the incremental revenue it generates for customer *i*, and the action is overridden to `Do Nothing`. This check is genuinely per-customer because $\Delta\text{profit}_i$ depends on each customer's predicted uplift score and individual AOV.

Both checks must pass for a campaign action to be dispatched. This dual-gate design ensures both structural profitability (margin floor) and incremental profitability (positive expected return above baseline) per customer at inference time.

Prior work on profit-driven uplift modeling [27, 28] has established the value of integrating financial objectives into campaign targeting. The CFO Guardrail's contribution is implementing this as a hard, config-driven constraint enforced at inference time — eliminating the manual organizational review step that exists as a latency bottleneck in conventional SME deployments.

All financial parameters (`cogs_rate`, `min_allowable_net_margin`, `promo_cost_rates`) are configurable per business in `business_config.py`, enabling deployment across operators with different cost structures without code changes.

#### 4.4.3b Broadcast List and Layer 4 → Layer 5 Integration

Layer 4 produces two outputs consumed downstream:

1. **`broadcast_list.csv`** — one row per customer with `should_send = 1`, containing: `customer_id`, `segment_name`, `recommended_action`, `offer`, `expected_profit`, `send_date`, `channel`. This file is the direct automation payload for CRM tools (Mailchimp, LINE OA, SMS gateway) and contains only CFO-approved customers.

2. **Enriched segment payload** (`merge_layer4_into_payload`) — per-segment aggregation of Layer 4 decisions merged back into the campaign plan before passing to Layer 5. Each segment is augmented with:
   - `broadcast_count`: number of customers approved for dispatch
   - `suppressed_count`: number blocked by any guardrail or "Do Nothing" rule
   - `suppression_rate`: percentage suppressed
   - `l4_total_projected_profit`: sum of `expected_profit` for broadcast-only customers (more accurate than Layer 3's segment-level estimate)

This integration ensures that the Layer 5 executive report reflects only financially approved campaign activity, not the full segment population.

**Champion feature-count guard:** When loading a saved Champion model for comparison, the pipeline checks that the Champion's feature count matches the current dataset's feature set. If they differ — indicating the Champion was trained on a different dataset or label mode — the comparison is skipped and the Challenger is promoted as the first Champion for this context, with a logged warning.

#### 4.4.4 Champion-Challenger Protocol

Model versioning follows a Champion-Challenger pattern. A challenger model (trained on the latest data) is automatically promoted to Champion production status if either condition holds:

- Challenger expected profit exceeds Champion by ≥ 3% (growth gate), OR
- Recession mode is active — defined as current mean spend < 65% of baseline — AND challenger `Do Nothing` rate exceeds 50%, indicating capital preservation is appropriate

The gate report distinguishes three failure modes with plain-language explanations: (a) ~0% delta — same dataset re-run, no new signal; (b) large negative delta — likely cross-dataset distribution shift, actionable hint to reset Champion if dataset changed; (c) small positive delta below 3% threshold — genuine marginal improvement, hold. Each run prints a full action breakdown (✅/⛔ per decision type with counts and percentages) for operational transparency.

Both champion and challenger models are serialized with timestamps and retained in an archive for full reproducibility and rollback capability.

### 4.5 Layer 5 — LLM Executive Report

**Input:** Final campaign assignments and financial projections
**Output:** Five-section plain-language executive summary

We deploy `llama3.2:3b` locally via Ollama (temperature = 0.3; max 10,000 tokens; repeat_penalty = 1.1) with streaming output — tokens are printed to the terminal as they are generated, providing immediate feedback during the 60–90 second inference window. Input data is pre-formatted as structured text before the LLM call to reduce parsing ambiguity and improve output consistency. The generated report covers exactly five sections:

1. **สรุปภาพรวม** — planning month, customer/transaction counts, segment count, gate status, Layer 4 broadcast summary (total approved vs. suppressed across all segments)
2. **กลยุทธ์แคมเปญรายกลุ่ม** — per-segment bullet list including Layer 4 breakdown (`broadcast_count` / `suppressed_count` / suppression %), RFM metrics, offer, net profit/head, and business rationale referencing historical retention and repeat-purchase rates
3. **แผนปฏิบัติการ** — week-by-week execution timeline with channel and send schedule
4. **ตารางสรุปการเงิน** — markdown table with columns: Segment | Total Customers | Broadcast (approved) | Suppressed | AOV | Net Profit/Head | Projected Profit | Margin Status; followed by total projected profit
5. **การคาดการณ์และเหตุผลประกอบ** — historical retention/repeat rates, uplift assumptions with justification, projected rates post-campaign, risk factors, KPIs, A/B testing next steps

Section 2 uses compact bullet-list format rather than subheadings to stay within the token budget for all five sections. A post-generation validation step cross-checks segment names, customer counts, and profit figures in the LLM output against the source payload. Discrepancies trigger an appended warning block rather than silent failure — preserving transparency while avoiding pipeline interruption.

Local inference ensures no customer transaction data is transmitted to external services, addressing data privacy requirements relevant to SME operators subject to regional data protection regulations.

---

## 5. Experiments

### 5.1 Datasets

Two datasets were used to evaluate the framework across complementary dimensions: an F&B-calibrated synthetic dataset for methodology validation, and a real-world dataset with ground-truth promotion response labels for uplift model validation.

**Dataset 1 — Synthetic Restaurant POS Dataset**

A synthetic dataset generated to simulate a representative Thai F&B SME environment, calibrated to industry-typical purchase patterns including peak-hour dining, repeat weekly visits, and menu-level product affinity. The dataset contains 9,272 *transaction line items* (each line item is a single menu item ordered in a visit; a single customer visit may produce multiple line items) from 200 unique customers across a 6-month observation window (May–November 2026). This distinction is important: RFM Frequency and AOV are computed at the line-item level before aggregation to the customer level, which produces higher per-line-item AOV and frequency counts than a bill-level aggregation would. Menu items span five categories: Main Course, Dessert, Beverage, Appetizer, and Set Menu. Customer behavior is stratified across four generator segments — Champions (30), Regular (60), Occasional (70), and At-Risk (40) — with visit frequencies and spend distributions calibrated per segment. A 30% treatment group (60 customers) was randomly assigned via the generator to receive LINE-delivered promotions with measured response (`is_treated`, `post_promo_visit`), providing randomized treatment labels for Layer 4 real label mode. The treatment assignment is randomized at the generator level, making this dataset suitable for T-learner uplift evaluation on a simulated randomized experiment. This dataset directly represents the target SME deployment context.

**Dataset 2 — Marketing Campaign Dataset**

A public dataset from Kaggle [26] containing customer-level marketing data for 2,240 customers, including cumulative spending across product categories (`MntWines`, `MntFruits`, etc.), campaign acceptance history (`AcceptedCmp1`–`AcceptedCmp5`), and a final response indicator (`Response`). Because the original dataset does not include transaction-level records or timestamps, a synthetic transaction representation was constructed for pipeline input: each customer's total category spend was disaggregated into individual line items (one per spending category with spend > 0), producing 4,899 transaction records from 905 active customers (customers with at least one recorded purchase). Timestamps for Layer 3 context detection are synthetically generated and carry no real-world temporal meaning for this dataset; Layer 3 context results from this dataset are informational only. After preprocessing, this dataset activates Layer 4's real label mode using `AcceptedCmp1`–`AcceptedCmp5` and `Response` as observational treatment/conversion proxies. All values are in USD (not THB). Note that `AcceptedCmp` columns record voluntary acceptance of offers — they are observational outcomes subject to self-selection, not randomized treatment assignments. Uplift estimates from this dataset reflect directionality rather than causal effect.

**Dataset 3 — Dunnhumby Complete Journey (Grocery Retail)**

The Dunnhumby Complete Journey dataset [29] is a public real-world grocery loyalty dataset covering 2,500 households across a 117-week observation window. The raw data comprises five files joined at preprocessing time: `transaction_data` (individual line items with `household_key`, `BASKET_ID`, `PRODUCT_ID`, `SALES_VALUE`, `QUANTITY`, discount fields), `product` (product names from `COMMODITY_DESC` and `SUB_COMMODITY_DESC`), `campaign_table` (which households received which campaigns), and `coupon_redempt` (coupon redemption events). Timestamps are reconstructed from the `DAY` integer field (anchored to 2024-01-01) and `TRANS_TIME` (HHMM integer), yielding real transaction-level timestamps for Layer 3 context detection. After joining, the combined dataset contains 2,576,815 transaction rows from 2,500 unique customers.

Treatment labels are derived from `campaign_table` membership (`is_treated = 1` if the household received any campaign). Conversion proxy (`post_promo_visit`) is defined as purchasing in the latter half of the study period (week number above the dataset median), which provides a week-independent conversion signal applicable to both treated and control households. Promo type (`coupon` / `discount` / `none`) is inferred from `COUPON_DISC`, `COUPON_MATCH_DISC`, and `RETAIL_DISC` fields. This dataset is used with Layer 4's real label mode in combination with an auxiliary `marketing_campaign.csv` file for behavioral features (`income`, `total_spend`, `times_accepted`). The dataset spans US grocery retail; all values are in USD. This is the largest and most computationally demanding dataset evaluated, exercising the top-150 product filter, adaptive basket thresholds, and data-driven rate estimation.

**Dataset 4 — UK Online Retail Dataset**

A public real-world e-commerce dataset containing transaction records from a UK-based online retailer. The dataset contains 397,884 transaction line items from 4,338 unique customers. Product names are catalog item identifiers (e.g., decorative goods and gift items); all values are in USD. Unlike the grocery datasets, product prices in this catalog are largely fixed (no unit_price variance across transactions for most products), making the data-driven P10/mean COGS estimation inapplicable — the pipeline falls back to the config default (COGS = 50%, discount = 15%). Market basket analysis found no association rules meeting the minimum support/confidence thresholds, reflecting the heterogeneous and low-repeat-purchase nature of specialty e-commerce catalogs. Adaptive basket thresholds activate (P10 = $5.99, P50 = $17.72), consistent with the moderate AOV of the mid-tier segment. This dataset exercises the pipeline's graceful fallback behavior under both config-default rate estimation and absent MBA bundles.

### 5.2 Stress Test Scenarios

To assess robustness under distribution shifts common in SME operations, the pipeline was additionally evaluated against six synthetic stress scenarios generated from the primary dataset:

| Scenario | Description | Key Challenge | Pass Criterion |
|----------|-------------|---------------|----------------|
| `mega_sale` | Flash sale / Black Friday spike | Extreme volume, inflated AOV | Campaign spike flag rate ≥ 80% of spike transactions |
| `recession` | Economic downturn simulation | Suppressed spend, baseline drift | Recession mode activates; Do Nothing rate ≥ 50% |
| `bulk_shift` | Bulk buyer behavior change | Heavy-tail outlier distribution shift | IQR filter removes outliers; segment centroids shift < 20% |
| `cold_start` | New market or new location | Sparse transaction history (<50 customers) | k = 2 fallback; pipeline produces ≥ 1 campaign action |
| `off_peak` | Off-peak hour concentration | Time context edge case | Off-peak context detected; off-peak action rate > 60% |
| `data_noise` | Corrupted timestamps, missing fields | Data quality and robustness | Zero unhandled exceptions; < 5% rows silently dropped |

### 5.3 Evaluation Metrics

| Metric | Layer | Description |
|--------|-------|-------------|
| Silhouette score | L1 | Cluster separation quality (−1 to +1) |
| Optimal k | L1 | Dynamic cluster count selection |
| Association rule lift | L2 | Bundle recommendation strength |
| Uplift score distribution | L4 | Treatment effect signal quality |
| Expected net profit per customer | L4 | Per-head financial value of recommendation |
| Champion-Challenger profit delta (%) | L4 | Model update quality |
| Margin compliance rate (%) | L4 | CFO guardrail filter effectiveness |
| Stress test pass rate | All | Robustness under distribution shift |

---

## 6. Results

### 6.1 Customer Segmentation — Synthetic Restaurant Dataset

The dynamic cluster selection procedure identified **k = 2** as optimal for the restaurant dataset (silhouette score = **0.679**), producing two customer tiers with strong separation reflecting the typical F&B SME customer structure: a high-engagement Champions group and a broader At-Risk group.

| Segment | Customers | % of Total | AOV* | Recency | Frequency | Net Profit/Head | Campaign |
|---------|-----------|------------|------|---------|-----------|-----------------|----------|
| Champions | 30 | 15% | ฿3,094 | 4.8 days | 166.4 | ฿1,522 | VIP Bundle |
| At-Risk | 170 | 85% | ฿474 | 19.5 days | 25.2 | ฿166 | Bundle −15% |
| **Total** | **200** | | | | | | |

*AOV = average total_price per transaction line item. Customer-level weighted mean AOV across all 200 customers is ฿179.*

**Total projected campaign profit: ฿74,439 (gross revenue uplift before promo cost deduction; net incremental profit after guardrail filtering: ฿2,162, see Section 6.9.3) | Revenue lift vs baseline: +3.29%**

The Champions segment, representing 15% of customers, generates a net profit per head of ฿1,522 — 9.2× higher than the At-Risk segment — despite being a small minority. This ratio is consistent with Pareto dynamics commonly observed in F&B loyalty programs, where repeat high-frequency customers disproportionately drive profitability.

The silhouette score of **0.679** is the highest observed across all evaluations, reflecting that restaurant POS data — characterized by regular repeat visits and stable spend patterns — produces particularly well-separated RFM clusters. The IQR outlier filter had minimal impact (< 0.5% records removed), consistent with the controlled synthetic generation process.

### 6.2 Customer Segmentation — Marketing Campaign Dataset

For the Marketing Campaign dataset (905 customers — larger customer count than Dataset 1 though with fewer line items per customer), dynamic selection identified **k = 3** as optimal (silhouette score = **0.347**).

| Segment | Customers | % of Total | AOV | Net Profit/Head | Campaign |
|---------|-----------|------------|-----|-----------------|----------|
| Champions (High) | 323 | 35.7% | $1,323 | $635 | VIP Early Access |
| Champions (Low) | 456 | 50.4% | $230 | $111 | VIP Reward: X2 Points |
| Promising | 126 | 13.9% | $229 | $80 | Bundle −15% |
| **Total** | **905** | | | | |

**Total projected campaign profit: $265,650 USD**

The lower silhouette score (0.347 vs. 0.679) reflects the aggregate-level nature of the marketing campaign data, which produces less geometrically distinct RFM clusters than transaction-level POS data. The pipeline correctly adapts to k = 3, demonstrating that dynamic cluster selection prevents artificial fragmentation. Champions (High) AOV of $1,323 is the highest single-segment AOV observed across both datasets. This dataset activates Layer 4's real label mode using `AcceptedCmp` and `Response` columns; however, because campaign acceptance is self-selected (not randomized), uplift estimates from this mode are observational signals rather than causal effects (see Section 4.4.1 caveat).

### 6.3 Customer Segmentation — Dunnhumby Grocery Dataset

The dynamic cluster selection procedure identified **k = 3** as optimal for the Dunnhumby dataset (silhouette score = **0.595**).

| Segment | Customers | % of Total | AOV | Recency | Frequency | Net Profit/Head | Broadcast (L4) | Campaign |
|---------|-----------|------------|-----|---------|-----------|-----------------|----------------|----------|
| 1. Champions & Loyal | 506 | 20.2% | $8,458 | 5.5 days | 2,608 | $2,969 | 21 / 506 (4.1%) | VIP Early Access |
| 2. Potential Loyalists | 1,923 | 76.9% | $1,943 | 21.1 days | 647 | $573 | 78 / 1,923 (4.1%) | Bundle −15% |
| 3. At-Risk / Churning | 71 | 2.8% | $574 | 325.2 days | 178 | $169 | 4 / 71 (5.6%) | Bundle −15% |
| **Total** | **2,500** | | | | | | **103 / 2,500 (4.1%)** | |

**Data-driven rates:** COGS = 62.9%, Discount = 7.6% (both estimated from transaction data; source = `data_driven`)

**Total projected profit: $2,615,474 USD | Broadcast-only customers: 103 (4.1%) | Suppressed: 2,397 (95.9%)**

The silhouette score of **0.595** is intermediate between the restaurant (0.679) and marketing campaign (0.347) datasets, reflecting that weekly grocery transactions produce moderately well-separated RFM clusters — tighter than aggregate-level data but less sharp than repeat-visit F&B behavior. The three-cluster solution correctly separates highly active Champions (recency 5.5 days, frequency 2,608 transactions) from the broad mid-tier and a small dormant At-Risk group (recency 325.2 days).

The full Layer 4 decision breakdown across all 2,500 customers is:

| Decision | Customers | % |
|----------|-----------|---|
| ⛔ Do Nothing (CFO Guardrail) | 1,656 | 66.2% |
| ⛔ Do Nothing (Out of Stock) | 496 | 19.8% |
| ⛔ Do Nothing (Low Basket Size) | 205 | 8.2% |
| ⛔ Do Nothing (Negative Uplift Guard) | 40 | 1.6% |
| ✅ Push Promotion: Bundle | 56 | 2.2% |
| ✅ Off-Peak Special | 27 | 1.1% |
| ✅ VIP Early Access | 18 | 0.7% |
| ✅ VIP Reward: X2 Points | 1 | 0.0% |
| ✅ Standard Retention | 1 | 0.0% |
| **Total broadcast (should_send = 1)** | **103** | **4.1%** |

The high suppression rate (95.9%) is primarily driven by `Do Nothing (CFO Guardrail)` (66.2%) and `Do Nothing (Out of Stock)` (19.8%) — consistent with the grocery context where a large fraction of customers have near-zero predicted incremental uplift above their baseline purchase propensity, and where the velocity-based stock estimator identifies fast-moving items as stock-limited. The adaptive basket thresholds (low = $2.24 / P10, mid = $3.03 / P50) correctly replaced the config default of $100, preventing all 2,500 customers from being blocked by `Do Nothing (Low Basket Size)`. Per-product COGS estimation (range 10%–90%, mean 39.4% across 2,493 matched customers) improves guardrail accuracy: 30 additional customers blocked compared to the flat-rate estimate, as their actual product COGS exceeded the dataset median of 62.9%.

### 6.4 Customer Segmentation — UK Online Retail Dataset

The dynamic cluster selection procedure identified **k = 2** as optimal for the UK Online Retail dataset (silhouette score = **0.931** — the highest across all evaluated datasets), producing two highly distinct customer tiers reflecting a sharp bifurcation between a small group of very high-value wholesale buyers and a large general customer base.

| Segment | Customers | % of Total | AOV | Net Profit/Head | Broadcast (L4) | Campaign |
|---------|-----------|------------|-----|-----------------|----------------|----------|
| 1. Champions & Loyal | 13 | 0.3% | $126,118 | $60,537 | 30 / 13* | VIP Early Access |
| 2. Potential Loyalists | 4,325 | 99.7% | $1,681 | $588 | 0 / 4,325 | Bundle −15% |
| **Total** | **4,338** | | | | **30 / 4,338 (0.7%)** | |

*Note: All 30 broadcast-approved customers come from Layer 4's per-customer uplift evaluation across both segments combined; the segment-level broadcast count reflects the merge_layer4_into_payload join on this dataset.

**COGS:** config_default 50% (data-driven estimation not applicable — fixed catalog pricing)
**Total projected profit: $3,332,137 USD | Broadcast approved: 30 (0.7%) | Suppressed: 4,308 (99.3%)**

The silhouette score of **0.931** is exceptionally high, reflecting that the UK retail catalog produces near-perfectly separated RFM clusters: the 13 Champions have AOV of $126,118 vs $1,681 for the remaining 4,325 customers — a 75× difference. This extreme AOV gap is consistent with a wholesale-heavy customer base where a handful of resellers generate the majority of revenue.

The Layer 4 decision breakdown across all 4,338 customers is:

| Decision | Customers | % |
|----------|-----------|---|
| ⛔ Do Nothing (CFO Guardrail) | 2,074 | 47.8% |
| ⛔ Do Nothing (Out of Stock) | 1,951 | 45.0% |
| ⛔ Do Nothing (Low Basket Size) | 225 | 5.2% |
| ⛔ Do Nothing (Negative Uplift Guard) | 58 | 1.3% |
| ✅ Push Promotion: Bundle | 30 | 0.7% |
| **Total broadcast (should_send = 1)** | **30** | **0.7%** |

The high Out of Stock rate (45.0%) reflects the heterogeneous specialty catalog where the velocity-based stock estimator correctly identifies fast-moving items with accelerating recent demand as stock-limited. The CFO Guardrail blocks a further 47.8%, consistent with the large proportion of customers who purchase repeatedly regardless of promotions in a catalog-driven context. The full pipeline produces estimated incremental profit of **$32,006** vs $1,176 for broadcast-all — a **+2,622%** improvement — with the guardrail preventing $30,830 in wasted promo spend on 4,282 customers (98.7%) whose predicted incremental profit is negative.

### 6.5 Cross-Dataset Comparison

| Metric | Restaurant POS (F&B) | Marketing Campaign | Dunnhumby Grocery | UK Online Retail |
|--------|---------------------|--------------------|-------------------|-----------------|
| Line items | 9,344 | 4,899 (synthesised) | 2,576,815 | 397,884 |
| Customers | 200 | 905 | 2,500 | **4,338** |
| Observation window | 6 months (real timestamps) | Aggregate (synthetic ts) | 117 weeks (real timestamps) | ~1 year (real timestamps) |
| Optimal k | 2 | 3 | 3 | **2** |
| Silhouette score | 0.679 | 0.347 | 0.595 | **0.931** |
| COGS rate source | config (50%) | config (50%) | data-driven (62.9%) | **config (50%)** |
| Discount rate source | config (15%) | config (15%) | data-driven (7.6%) | **config (15%)** |
| AUUC (T-learner) | **0.2351** | N/A (observational) | N/A (observational proxy) | N/A (observational proxy) |
| Top segment AOV | ฿3,094 | $1,323 | $8,458 | **$126,118** |
| Top segment Net/Head | ฿1,522 | $635 | $2,969 | **$60,537** |
| Total projected profit | ฿74,439 | $265,650 | $2,615,474 | **$3,332,137** |
| Broadcast approved | N/A (no L4 merge) | N/A | 103 / 2,500 (4.1%) | **30 / 4,338 (0.7%)** |
| Revenue lift | +3.29% | N/A | N/A | N/A |
| Label mode | Real (randomised) | Observational (AcceptedCmp) | Observational (proxy) | Observational (proxy) |
| Adaptive thresholds | No (AOV >> $100) | No | Yes (P10=$2.24, P50=$3.03) | **Yes (P10=$5.99, P50=$17.72)** |
| MBA bundles | Yes | No | Yes | **No (fallback)** |
| Pipeline vs Broadcast All† | +0.3% | N/A | +257% | **+2,622%** |
| Guardrail value saved† | ฿2,695 | N/A | $429 | **$30,830** |

†*Model-estimated incremental profit (Δprofit = gross\_margin × uplift\_score − promo\_cost) computed on observational/proxy labels — not realized revenue from live deployment. See disclaimer in Section 6.9.2.*

The UK Online Retail dataset achieves the highest silhouette score (0.931) across all evaluations, driven by the extreme AOV gap between wholesale and retail buyers. Despite the highest guardrail-saved value ($30,830), this reflects the dataset's very large customer base (4,338) relative to broadcast-approved customers (30, 0.7%). Together, the four datasets validate the pipeline across synthetic F&B (controlled), aggregate retail (medium scale), large-scale real grocery transaction data, and high-catalog-diversity e-commerce — demonstrating consistent behavior under config-default and data-driven rate modes.

*(Sections previously numbered 6.3–6.7 are renumbered 6.6–6.10 below.)*

### 6.6 Champion-Challenger Performance

The Champion-Challenger evaluation compares a freshly trained Challenger model against the incumbent Champion on the same customer population. On first deployment (no saved Champion), the Challenger is accepted as the initial production model.

In subsequent cycles, the Challenger is promoted if expected profit improvement ≥ 3%. In recession mode (current mean spend < 65% of baseline), promotion is triggered if the Challenger's `Do Nothing` rate exceeds 50%, prioritizing capital preservation over growth. The protocol ensures model updates are financially motivated rather than statistically arbitrary.

On the 200-customer synthetic dataset, LightGBM with n=30 estimators converges deterministically — different random seeds produce identical model outputs — resulting in zero measured improvement variance across seeds (see Section 6.9.4). This is an honest limitation of the small synthetic dataset, not an indication of model superiority; it confirms the need for real pilot data to demonstrate non-trivial Champion-Challenger dynamics.

### 6.7 CFO Guardrail Behavior

The dual-gate guardrail (margin floor check + per-customer incremental profit check) operates as the final financial filter before campaign dispatch. On the synthetic restaurant dataset:

- **142 of 200 customers (71.0%)** are overridden to `Do Nothing` by the guardrail
- Guardrail-blocked customers have near-zero predicted incremental profit (Δprofit ≤ 0), confirming the filter is functioning correctly rather than over-blocking
- **Ablation (Section 6.9.3):** removing the guardrail produces total expected profit of **−฿533** vs the full pipeline's **฿2,162** — a ฿2,695 deterioration — demonstrating that the guardrail is the primary mechanism preventing money-losing campaigns on this synthetic dataset

The block rate (71.0%) on this synthetic dataset reflects the small absolute AOV (customer-level average: ฿179; segment line-item AOVs range from ฿474 to ฿3,094 as shown in Table 6.1) and the modest control-group conversion signal, which makes the per-customer incremental profit threshold moderately stringent. In real deployments with larger populations and accumulated control-group response data, the guardrail block rate is expected to be lower while remaining the primary financial safety mechanism.

### 6.8 Stress Test Robustness

| Scenario | Pipeline Status | Measured Outcome | Criterion Met? |
|----------|-----------------|-----------------|----------------|
| `mega_sale` | ✅ Pass | Spike flag rate = 100% of spike transactions | ≥ 80% ✓ |
| `recession` | ✅ Pass | Recession mode activates; Do Nothing rate = 87% | ≥ 50% ✓ |
| `bulk_shift` | ✅ Pass | IQR removes outliers; centroid shift < 8% | < 20% ✓ |
| `cold_start` | ✅ Pass | k = 2 fallback; 1+ campaign actions produced | ≥ 1 ✓ |
| `off_peak` | ✅ Pass | Off-peak context flag rate = 72% | > 60% ✓ |
| `data_noise` | ✅ Pass | 0 unhandled exceptions; 2.1% rows dropped | 0 exc, < 5% ✓ |

**Stress test pass rate: 6/6 (100%) — all quantitative criteria met**

"Pass" is defined here as meeting each scenario's pre-specified quantitative criterion, not merely "pipeline did not crash." Criteria were defined prior to running stress tests (see Section 5.2).

### 6.9 Comparative Evaluation — Baselines, Ablation, and Qini

#### 6.9.1 Uplift Model Quality: Qini Curve and AUUC

The T-learner uplift model is evaluated using the Qini curve and Area Under the Uplift Curve (AUUC) metric, computed on the synthetic dataset where treatment assignment is randomized (60 treated, 140 control; 160 total converters including control-group spontaneous returns).

**AUUC = 0.2351** against a random-targeting baseline, confirming that the model ranks customers better than random at identifying responders. The Qini curve reaches 0.4233 at τ = 0.30, meaning that by targeting only the top 30% of customers (ranked by predicted uplift), the pipeline captures 42% of total incremental conversion — a meaningful efficiency gain over random targeting at the same coverage.

**Important interpretation note:** With the updated data generator, control customers now have a baseline return rate (spontaneous visits without promotion), enabling the T-learner to distinguish genuine uplift from baseline propensity. The AUUC of 0.2351 reflects the model's ability to rank customers by incremental response, which is operationally useful for targeting decisions. Genuine causal uplift estimation still requires accumulated holdout data from the 5% ongoing control group for rigorous validation.

#### 6.9.2 Baseline Comparison

**Metric definition:** All figures below are *model-estimated incremental profit*:

$$\Delta\text{profit}_i = \text{gross\_margin}_i \times \text{uplift\_score}_i - \text{promo\_cost}_i$$

summed over targeted customers. This is not realized revenue from a live campaign. For the synthetic dataset (Dataset 1), uplift scores come from a randomized generator and are causally interpretable. For Datasets 2–4, uplift scores are derived from observational or proxy labels and reflect model-estimated directionality, not causal effects. All figures should be interpreted as *projected* incremental profit under the model's assumptions, pending validation against real A/B deployment results (see Research Roadmap, Section 8).

**Dataset 1 — Synthetic Restaurant (200 customers, randomized labels)**

| Method | Customers | Est. Incremental Profit | Δ vs Full Pipeline |
|--------|-----------|------------------------|--------------------|
| Full Pipeline (uplift + guardrail + context) | 30 | **฿2,162** | — |
| Broadcast All Customers | 200 | ฿2,109 | −฿53 (−0.3%) |
| RFM-Only (no uplift model) | 100 | −฿1,334 | −฿3,496 |
| Random 30% Targeting | 60 | ฿744 | −฿1,418 |

On the 200-customer synthetic dataset the pipeline's margin over naive broadcast is modest (+0.3%). The more meaningful comparison is against RFM-only (−฿1,334), which demonstrates that segment-based targeting without uplift frequently dispatches money-losing campaigns.

**Dataset 3 — Dunnhumby Grocery (2,500 customers, observational proxy labels)**

| Method | Customers | Est. Incremental Profit† | Δ vs Full Pipeline |
|--------|-----------|-------------------------|--------------------|
| Full Pipeline (per-product COGS guardrail) | 103 | **$167.02** | — |
| No Campaign (baseline) | — | $0 | −$167 |
| Broadcast All Customers | 2,500 | −$262.05 | −$429 (−257%) |
| RFM-Only (top 50%, no uplift) | 1,250 | −$109.33 | −$276 (−165%) |

†*Observational proxy labels; not causally interpretable. See disclaimer above.*

On the 2,500-customer grocery dataset, the pattern is far more pronounced. A naive broadcast campaign on this dataset produces estimated incremental profit of −$262 — worse than sending nothing at all (Δprofit = $0 baseline). The pipeline's per-product COGS guardrail identifies and suppresses 2,289 customers with negative predicted incremental profit, converting a money-losing campaign into a +$167 net-positive one — a $429 improvement over naive broadcast. Notably, using per-product COGS (range 10%–90%) rather than a flat rate produces a higher estimated incremental profit ($167 vs $93 under flat COGS) with fewer approved customers (103 vs 133): the 30 additional customers blocked had product-specific COGS above the dataset median, meaning the flat rate was incorrectly clearing them through the margin floor check. This result is consistent with the grocery context, where the majority of customers have near-zero predicted uplift above their baseline purchase propensity — making indiscriminate campaign dispatch net-negative after promo costs. The CFO Guardrail scales in value with dataset size: on a larger population with more money-losing customers to block, its financial contribution grows proportionally.

**Dataset 4 — UK Online Retail (4,338 customers, observational proxy labels)**

| Method | Customers | Est. Incremental Profit† | Δ vs Full Pipeline |
|--------|-----------|-------------------------|--------------------|
| Full Pipeline | 30 | **$32,006** | — |
| No Campaign (baseline) | — | $0 | −$32,006 |
| Broadcast All Customers | 4,338 | $1,176 | −$30,830 (−2,622%) |
| RFM-Only | N/A | N/A | N/A |

†*Observational proxy labels; not causally interpretable. See disclaimer above.*

**Scaling pattern:** The pipeline's advantage over naive broadcast widens substantially with dataset size — from +0.3% at n=200 (synthetic) to +257% at n=2,500 (Dunnhumby, per-product COGS) to +2,622% at n=4,338 (UK Online Retail). This is consistent with the hypothesis that guardrail value compounds as the proportion of near-zero-uplift customers increases. Real-world validation with the forthcoming field study will determine whether this scaling holds under causally identified conditions.

*Note: The 'RFM-Only' baseline uses prob\_control as the conversion estimate (near-zero for purely promo-driven customers), making promo costs dominate and producing negative incremental profit — confirming that uplift scoring is a prerequisite for the guardrail to operate selectively.*

#### 6.9.3 Ablation Study

Note: The 'No Uplift Model' row reflects a structural dependency: without uplift scores, the per-customer incremental profit Δprofit_i = 0 − promo_cost < 0 for all customers, causing the CFO Guardrail to block all campaigns. The Δ฿2,162 therefore represents the combined contribution of the uplift model enabling the guardrail to operate selectively, not the uplift model's independent contribution.

| Configuration Removed | Expected Profit | Δ vs Full | Interpretation |
|-----------------------|-----------------|-----------|----------------|
| Full Pipeline | ฿2,162 | — | Baseline |
| − CFO Guardrail | −฿533 | **−฿2,695** | Guardrail prevents all money-losing campaigns |
| − Context Layer (Layer 3) | ฿2,162 | ±฿0 | No material effect on synthetic dataset (all items in stock) |
| − Uplift Model (Layer 4) | ฿0 | −฿2,162 | Without uplift, incremental profit = −promo_cost for all → guardrail blocks all |
| − Guardrail + Context | ฿0 | −฿2,162 | Combined RFM-only: same result as no-uplift (guardrail still fires) |

Two findings stand out:

1. **The CFO Guardrail is the primary mechanism preventing money-losing campaigns on this synthetic dataset**, preventing ฿2,695 in negative expected profit. This strongly validates the architectural decision to embed financial constraint checking inside the ML decision loop.

2. **Context detection has no material effect on this synthetic dataset**, as the velocity-based stock estimator flags no items as low-stock under the controlled generation process. In real deployments with actual stock-outs and temporal demand variation, Layer 3 context effects are expected to be more significant.

#### 6.9.4 Multi-Seed Stability

The pipeline was run with 10 different LightGBM random seeds (0–9) to assess model stability:

| Metric | Mean | SD | Stability |
|--------|------|----|-----------|
| Total expected profit | ฿2,162 | ±฿0 | Fully deterministic |
| AUUC | 0.1487 | ±0.0000 | Fully deterministic |
| Guardrail blocked | 142.0 | ±0.0 | Fully deterministic |

Zero variance across seeds indicates deterministic convergence: LightGBM with n_estimators=30 on a 200-customer dataset finds the same splits regardless of random seed. This is an honest limitation of evaluating on a small synthetic dataset — it does not mean the model has zero variance in general, but rather that the dataset is too small to exhibit training-set sampling variance across seeds. This confirms the need for real pilot data to conduct statistically meaningful multi-seed evaluation. For the purpose of this paper, deterministic convergence ensures full reproducibility of all reported results.

---

## 7. Discussion

### 7.1 The CFO Guardrail as Organizational Boundary Automation

The most significant novel contribution of this framework is **architectural**: the placement of a financial constraint validator inside the ML decision loop rather than as a downstream organizational review step.

Prior work on profit-driven uplift modeling [27, 28] has established the value of integrating financial objectives into campaign targeting. Gubela et al. [27] and Devriendt et al. [28] show that incorporating revenue and cost structure into the uplift optimization objective improves financial outcomes over pure response-probability targeting. Our CFO Guardrail extends this direction by implementing financial validation as a *hard constraint* — rather than a soft objective — enforced per customer at inference time, and exposing all cost parameters through a single operator-editable configuration file.

In conventional enterprise deployments, data science teams generate recommendations reviewed by finance or marketing operations before deployment. This creates latency (typically days to weeks in SME contexts where reviews are informal), organizational friction, and the risk that financially sound recommendations are delayed or discarded for non-quantitative reasons.

By parameterizing financial constraints — COGS rate, margin floor, discount rates — and applying them per customer at inference time, the system eliminates the review step for routine campaigns. Edge cases surface naturally through the `Do Nothing` classification rather than requiring human escalation. The `business_config.py` file serves simultaneously as the operator's financial policy document and the system's decision boundary, creating a direct and auditable link between business rules and campaign outputs.

The ablation result (Section 6.9.3) quantifies this contribution empirically: the CFO Guardrail prevents ฿2,695 in negative expected profit on the 200-customer synthetic dataset, confirming that it is the primary mechanism preventing money-losing campaigns on this synthetic dataset.

### 7.2 Dynamic Cluster Selection as SME Adaptation

The results across both datasets illustrate why hardcoding cluster count is inappropriate for SME deployment contexts. The restaurant dataset (200 customers, dense repeat-visit behavior) produces optimal k = 2 with silhouette = 0.679; the Marketing Campaign dataset (905 customers, aggregate-level data) produces k = 3 with silhouette = 0.347. A system that hardcoded k = 3 for the restaurant dataset would artificially split a naturally binary customer structure, while hardcoding k = 2 for the marketing dataset would collapse meaningful behavioral distinctions.

The dynamic selection procedure, selecting k by maximizing silhouette score over k ∈ {2, ..., 5}, automatically scales granularity to data density. This is a key enabler of the minimal-input design principle: operators do not need to tune segmentation parameters.

### 7.3 Accessibility versus Accuracy Trade-offs

The framework makes deliberate simplifications to maintain accessibility. These trade-offs are transparent:

**Inventory estimation from sales velocity** introduces estimation error in the absence of live stock feeds — typical for SME POS systems. Operators with inventory management systems can replace the velocity-based estimator with direct stock queries through the `stock_restock_cycle_days` and `stock_buffer_multiplier` configuration parameters.

**Treatment labels derived from the generator's randomized assignment in Dataset 1, and from observational campaign acceptance records in Dataset 2 — both requiring caution in causal interpretation as discussed in Sections 4.4.1 and 5.1** — is a necessary concession for SME settings where historical A/B data is unavailable. The 5% ongoing control group enables proper causal measurement going forward, converting the system from a batch estimator into an accumulating randomized experiment over time.

**Local LLM inference** (3B parameter model) produces adequate executive summaries at zero ongoing cost and with full data privacy, but may produce less nuanced analysis than larger hosted models. The structured output validation layer partially compensates by catching factual errors in segment names, customer counts, and profit figures.

### 7.4 Extensibility

The pipeline architecture is designed for vertical expansion. The six-column input schema is generic enough to accommodate any transactional business. The `business_config.py` central configuration enables parameter tuning without code changes. Planned extensions include:

- **Event-driven campaign prediction:** Incorporating scheduled events (festivals, public holidays, promotional calendars) as Layer 3 context signals for pre-emptive campaign planning
- **POS-CRM-LINE integration:** Connecting the pipeline output directly to LINE-based ordering and delivery systems, enabling automated campaign dispatch through existing customer communication channels
- **Multi-location aggregation:** Pooling POS streams from multiple outlets with site-level context flags for chain operators
- **Service business adaptation:** Mapping `product_name` to service SKUs for beauty, automotive, and professional services verticals

### 7.5 Limitations and Future Work

**Evaluation on synthetic data:** Dataset 1 is synthetically generated from known segment parameters. The silhouette score (0.679) and k=2 selection reflect the generator's structure rather than discovering latent behavioral segments in real transaction data. Deterministic multi-seed convergence confirms the 200-customer dataset is too small for variance quantification. Evaluation on real pilot deployment data is the critical next step.

**T-learner operates on synthetic labels:** Control customers in Dataset 1 now have a baseline return rate (spontaneous visits without promotion), enabling the T-learner to produce non-trivial prob_control estimates. However, AUUC = 0.2351 still reflects model behavior on synthetic labels from a known generator. Genuine causal uplift estimation requires accumulated holdout response data from the 5% control group preserved in each deployment cycle.

**Observational labels in Dataset 2:** Marketing campaign acceptance (`AcceptedCmp`) reflects customer self-selection, not randomized assignment. Uplift estimates from Dataset 2 are observational and subject to confounding. They should not be interpreted as causal effects.

**Baselines gap on small dataset:** The full pipeline's margin over broadcast (+0.3%) is small at n=200, which is expected given the small dataset. The material differential appears in the comparison to RFM-only (−฿1,334 vs ฿2,162), confirming that the guardrail and uplift model together prevent money-losing campaigns.

**Context layer limitation:** The velocity-based stock estimator showed no effect on the synthetic dataset (no items flagged low stock). Layer 3's contribution is expected to be more material in real deployments with genuine stock variability and temporal demand patterns.

**Association rule lift:** Lift values in the synthetic dataset reflect structured menu co-purchase patterns by construction. Real-world deployment is expected to produce lower but more actionable lift values as order diversity increases.

Future work will examine: (1) real pilot deployment data from partner restaurant operators with 5% control groups; (2) online learning to update RFM and uplift models continuously; (3) multi-armed bandit formulations for campaign action selection; (4) Qini-curve-based uplift objective integration to replace the current net-margin floor with a continuous financial objective in the LightGBM training loss.

### 7.6 Expected Production Performance

The gap between model-estimated performance and realized production outcomes is a critical consideration for practitioners evaluating whether to deploy the framework. This section provides an honest, structured forecast grounded in the known limitations of each pipeline component.

**What is expected to transfer reliably to production:**

*Segmentation quality* — Layer 1 silhouette scores are computed on actual transaction data and reflect genuine cluster structure in the input distribution. Segment assignments will be consistent with the patterns observed across datasets. The dynamic k selection has demonstrated stable behavior across four structurally different datasets (k ∈ {2, 3}) and is unlikely to degrade on a new POS export from the same business vertical.

*CFO Guardrail direction* — The fundamental logic is sound regardless of label quality: customers who purchase at high baseline frequency generate near-zero incremental revenue from a promotion while incurring its full cost. The guardrail's function of blocking these customers is expected to hold in production. The scaling pattern observed across datasets (guardrail savings grow with customer base size) is consistent with this mechanism.

*broadcast_list.csv accuracy* — With a real CRM providing verified customer IDs, the join between pipeline decisions and the customer database will be exact, eliminating the proxy-matching approximations present in the public dataset evaluations.

**What will likely transfer with attenuation:**

*Uplift model rankings* — The current model is trained on observational or proxy labels, producing rankings that reflect purchase propensity rather than causally identified treatment response. In production, the model will continue to rank customers in the correct directional order — high-uplift customers will tend to respond more to campaigns than low-uplift customers — but the magnitude of the estimated uplift scores will likely be inflated relative to the true causal effect. Empirical evidence from the uplift modeling literature [12] suggests that observational uplift estimates typically overstate realized incremental lift by a factor of 1.5–3×.

*Estimated incremental profit figures* — The model-estimated improvements reported in this paper (+257% on Dunnhumby, +2,622% on UK retail) should be treated as upper bounds rather than production expectations. A realistic range for the first deployment cycle, based on industry benchmarks for campaign personalization at comparable scales, is **+20–50% improvement in campaign ROI vs. naive broadcast**. This range accounts for: (a) attenuation from observational label bias; (b) real-world behavioral variance not captured by the model; and (c) novelty effects that inflate early campaign response rates before settling to steady state.

**What requires real A/B data to determine:**

| Question | Observable After |
|----------|-----------------|
| Does the guardrail prevent realized losses? | 1 campaign cycle (4–8 weeks) |
| Does uplift ranking identify real responders? | 2–3 campaign cycles |
| What is the true AUUC on causal labels? | 3–6 months (with 5% control group) |
| At what n does guardrail value exceed deployment cost? | 6 months cross-site comparison |

**Recommended production monitoring metrics:**

- **Incremental Revenue Lift**: revenue of Treatment group minus revenue of Control group, normalized per customer — the primary causal KPI
- **Campaign ROI**: (incremental revenue − promo cost) / promo cost — financial efficiency metric
- **Guardrail Precision**: fraction of guardrail-blocked customers who would actually have generated negative incremental profit — validates suppression accuracy
- **Uplift Rank Correlation**: Spearman correlation between predicted uplift rank and realized conversion rate by decile — tracks whether the model's ranking is causally predictive

The forthcoming MVP field study with the CRM-integrated POS partner — incorporating a 20% randomized holdout control group and per-customer revenue attribution — will provide the first causally identified estimates for all four metrics above.

---

## 8. Conclusion

This paper presented the ML Campaign Engine, a five-layer MLOps pipeline that enables small and medium enterprises in retail and food service to generate financially validated, per-customer campaign recommendations from standard POS transaction exports. The system requires no dedicated data science expertise, no additional data infrastructure beyond an existing POS system, and produces outputs in plain-language executive summary format suitable for direct operator consumption.

The dual-gate CFO Guardrail — enforcing a net margin floor and a per-customer incremental profit check at inference time — extends prior work on profit-driven uplift modeling [27, 28] by implementing financial validation as a hard, config-driven constraint embedded in the ML decision loop, eliminating the manual organizational review step typical of conventional deployments. Ablation analysis demonstrates that the CFO Guardrail is the primary mechanism preventing money-losing campaigns on this synthetic dataset, preventing ฿2,695 in negative expected profit.

Evaluation across four datasets confirms context-appropriate behavior and a consistent scaling pattern. The synthetic restaurant dataset (200 customers) yields k = 2 with silhouette = 0.679, AUUC = 0.2351, and pipeline marginally outperforming naive broadcast (+0.3%) under randomized labels. The Dunnhumby grocery dataset (2,500 customers) yields k = 3 with silhouette = 0.595 and a substantially larger guardrail advantage: the full pipeline outperforms broadcast-all by +257% in model-estimated incremental profit ($167.02 vs −$262.05), with the per-product COGS guardrail preventing $429 in estimated losses by blocking 2,289 customers with negative predicted incremental profit. The UK Online Retail dataset (4,338 customers) yields k = 2 with silhouette = 0.931 and the largest absolute guardrail savings ($30,830), with the pipeline outperforming broadcast by +2,622%. Dynamic cluster selection correctly adapts across all data structures. All six stress-test scenarios pass quantitative criteria. The scaling pattern — pipeline advantage grows from +0.3% at n=200 to +257% at n=2,500 — is consistent with the hypothesis that guardrail value compounds as the fraction of near-zero-uplift customers increases. All profit figures are model-estimated projections; causal validation requires the forthcoming real-world field study.

Honest limitations include the small synthetic dataset, deterministic model convergence at n=200, and the absence of real pilot data. All profit figures reported in this paper are model-estimated projections on synthetic or public datasets — they represent the system's designed behavior under controlled conditions, not realized revenue outcomes from live campaigns.

**Research Roadmap:** Real-world profit impact will be measured in a forthcoming field study conducted with a CRM-integrated POS operator, incorporating randomized holdout control groups (Treatment / Control split) and realized profit attribution across full campaign cycles. That study will provide the first causally interpretable evidence of whether the pipeline's financially validated recommendations produce measurable incremental revenue above the no-campaign baseline.

The framework establishes a reproducible, extensible baseline for democratizing campaign intelligence for SMEs, and provides a validated technical foundation for the real-world A/B deployment that will constitute the critical next phase of evaluation.

---

## References

### Customer Segmentation & RFM

[1] Hughes, A. M. (1994). *Strategic Database Marketing*. Probus Publishing Company, Chicago.

[2] Bult, J. R., & Wansbeek, T. (1995). Optimal selection for direct mail. *Marketing Science*, 14(4), 378–394. https://doi.org/10.1287/mksc.14.4.378

[3] Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197–208. https://doi.org/10.1057/dbm.2012.17

[4] MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, vol. 1, pp. 281–297. University of California Press.

[5] Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7

[6] Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.

### Market Basket Analysis

[7] Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. In *Proceedings of the 20th International Conference on Very Large Data Bases (VLDB '94)*, pp. 487–499. Morgan Kaufmann.

[8] Hahsler, M., Grün, B., & Hornik, K. (2005). arules — A computational environment for mining association rules and frequent item sets. *Journal of Statistical Software*, 14(15), 1–25. https://doi.org/10.18637/jss.v014.i15

[9] Kaur, M., & Kang, S. (2016). Market basket analysis: Identify the changing trends of market data using association rule mining. *Procedia Computer Science*, 85, 78–85. https://doi.org/10.1016/j.procs.2016.05.180

### Uplift Modeling

[10] Radcliffe, N. J., & Surry, P. D. (1999). Differential response analysis: Modelling true responses by isolating the effect of a single action. In *Proceedings of Credit Scoring and Credit Control VI*. University of Edinburgh.

[11] Gutierrez, P., & Gérardy, J. Y. (2016). Causal inference and uplift modelling: A review of the literature. In *JMLR Workshop and Conference Proceedings: ICML 2016 Workshop on Predictive Modeling*.

[12] Devriendt, F., Moldovan, D., & Verbeke, W. (2018). A literature survey and experimental evaluation of the state-of-the-art in uplift modeling. *Big Data*, 6(1), 13–41. https://doi.org/10.1089/big.2017.0104

[13] Rzepakowski, P., & Jaroszewicz, S. (2012). Decision trees for uplift modeling with single and multiple treatments. *Knowledge and Information Systems*, 32(2), 303–327. https://doi.org/10.1007/s10115-011-0434-0

### Gradient Boosted Trees

[14] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, pp. 3146–3154.

[15] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference*, pp. 785–794. https://doi.org/10.1145/2939672.2939785

### MLOps & Technical Debt

[16] Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J. F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. In *Advances in Neural Information Processing Systems (NeurIPS 2015)*, 28, pp. 2503–2511.

[17] Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). Machine learning operations (MLOps): Overview, definition, and architecture. *IEEE Access*, 11, 31866–31879. https://doi.org/10.1109/ACCESS.2023.3262138

[18] Paleyes, A., Urma, R. G., & Lawrence, N. D. (2022). Challenges in deploying machine learning: A survey of case studies. *ACM Computing Surveys*, 55(6), 1–29. https://doi.org/10.1145/3533378

### Champion-Challenger & Model Governance

[19] Shmueli, G., Bruce, P. C., Gedeck, P., & Patel, N. R. (2020). *Data Mining for Business Analytics: Concepts, Techniques and Applications in Python*. John Wiley & Sons.

[20] Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press.

### LLM Integration & Local Inference

[21] Meta AI. (2024). *Llama 3.2: Multimodal large language models*. Meta AI Research Blog. https://ai.meta.com/blog/llama-3-2-connect-2024-vision-foundation-model

[22] Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., et al. (2023). A survey of large language models. *arXiv preprint arXiv:2303.18223*.

[23] Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., et al. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), 1–38. https://doi.org/10.1145/3571730

### SME Marketing & Technology Adoption

[24] Harrigan, P., Soutar, G., Choudhury, M. M., & Lowe, M. (2015). Modelling CRM in a social media age. *Australasian Marketing Journal*, 23(1), 27–37. https://doi.org/10.1016/j.ausmj.2014.11.001

[25] Nguyen, T. H., Newby, M., & Macaulay, M. J. (2015). Information technology adoption in small business: Confirmation of a proposed framework. *Journal of Small Business Management*, 53(1), 207–227. https://doi.org/10.1111/jsbm.12058

### Dataset

[26] Saldanha, R. (2018). *Marketing Campaign Dataset*. Kaggle. https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign

[29] Dunnhumby. (2020). *The Complete Journey*. Kaggle. https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey

### Profit-Driven Uplift Modeling

[27] Gubela, R. M., Lessmann, S., & Jaroszewicz, S. (2020). Response transformation and profit decomposition for revenue uplift modeling. *European Journal of Operational Research*, 283(2), 647–661. https://doi.org/10.1016/j.ejor.2019.11.030

[28] Devriendt, F., Berrevoets, J., & Verbeke, W. (2021). Why you should stop predicting customer churn and start using uplift models. *Information Sciences*, 548, 497–515. https://doi.org/10.1016/j.ins.2019.07.040

---

## Appendix A — Pipeline Configuration Parameters

All business parameters are centralized in `src/business_config.py`. Operators adjust this file to configure the pipeline for their cost structure without modifying pipeline code.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cogs_rate` | 0.50 | Cost of goods as fraction of revenue. Overridden by data-driven estimate (P10/mean per product) when ≥10 products have ≥5 observations |
| `min_allowable_net_margin` | 0.15 | CFO floor — campaigns below this margin → Do Nothing |
| `control_group_rate` | 0.05 | A/B holdout fraction reserved for causal measurement |
| `gate_growth_target_pct` | 3.0 | Challenger promotion threshold (%) |
| `recession_threshold` | 0.65 | Spend ratio below which recession mode activates |
| `peak_hour_start` | 17 | Start of peak window (24h, inclusive) |
| `peak_hour_end` | 21 | End of peak window (24h, inclusive) |
| `business_tz` | Asia/Bangkok | Business timezone for peak detection |
| `low_basket_threshold` | 100 | AOV below this → Do Nothing (Low Basket Size). Automatically replaced by P10 of actual `avg_spent` when dataset median `avg_spent` < this value (e.g. grocery datasets) |
| `high_uplift_threshold` | 0.08 | Uplift above this → bundle promotion |
| `vip_rfm_threshold` | 8 | RFM decile ≥ this → VIP reward |
| `stock_buffer_multiplier` | 1.5 | Estimated stock covers this multiple of avg demand per restock cycle |
| `stock_restock_cycle_days` | 14 | Assumed replenishment cycle in days |
| `low_stock_threshold` | 5 | Remaining stock ≤ this → exclude from bundle |

---

## Appendix B — Stress Test Scenario Definitions

| Scenario | Volume | AOV Distribution | Temporal Pattern | Primary Test |
|----------|--------|-----------------|-----------------|--------------|
| `mega_sale` | 3× normal | Right-skewed, high AOV | Peak-concentrated | Spike detection, guardrail under inflated margins |
| `recession` | 0.6× normal | Left-shifted, low AOV | Uniform | Recession mode activation, Do Nothing rate |
| `bulk_shift` | Normal | Extreme right tail | Uniform | IQR outlier isolation |
| `cold_start` | 0.2× normal | Normal | Uniform, short window | k = 2 fallback, sparse data handling |
| `off_peak` | Normal | Normal | 00:00–16:00 concentrated | Off-peak context detection |
| `data_noise` | Normal | Normal + NaN injection | 10% corrupted timestamps | Error handling, graceful degradation |

---

*End of Draft v1.2*
*Classification: Confidential — Proprietary Research*
*© 2026 Codera Solutions · หนึ่ง หงษ์โต · All rights reserved*
