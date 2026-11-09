# POS-Native MLOps: An Accessible Five-Layer Campaign Intelligence Framework for Small and Medium Enterprises in Retail and Food Service

**Draft v0.1 — Confidential**

---

## Abstract

Small and medium enterprises (SMEs) in retail and food service routinely collect transactional data through point-of-sale (POS) systems yet rarely leverage it for marketing decisions, primarily due to resource constraints and the technical complexity of existing analytics solutions. This paper presents a five-layer MLOps pipeline that transforms raw POS transaction records — requiring only six standard fields — into financially validated, per-customer campaign action plans without requiring a dedicated data science team. The framework integrates RFM-based customer segmentation (K-means), market basket analysis (Apriori), dynamic context detection, uplift modeling (LightGBM), and a large language model (LLM) reporting layer. A novel **CFO Guardrail** mechanism enforces a minimum net margin floor prior to campaign dispatch, ensuring that no promotion is deployed at a loss regardless of predicted uplift. Validation is conducted on a synthetic POS dataset (1,001 transactions) across six stress-test scenarios including economic recession, flash sale, cold-start, and data-noise conditions. A Champion-Challenger model update protocol achieves a mean profit improvement of **4.16%** per challenger cycle. Results demonstrate that the proposed framework is robust under distribution shift, computationally accessible to SME hardware, and extensible to additional verticals beyond food service and retail.

**Keywords:** MLOps, SME marketing, uplift modeling, RFM segmentation, market basket analysis, campaign optimization, POS analytics, LightGBM, CFO guardrail

---

## 1. Introduction

Marketing personalization has become a competitive necessity in modern retail. Large enterprises routinely deploy dedicated analytics teams, expensive CRM platforms, and data warehouses to generate campaign recommendations at the individual customer level. For small and medium enterprises (SMEs) — which constitute the majority of businesses in most economies — this infrastructure remains economically and technically out of reach. The consequence is that SME operators make campaign decisions based on intuition or blanket promotions, systematically under-leveraging the transactional data they already possess.

This resource asymmetry creates a measurable market opportunity. If an automated pipeline could consume the POS export that any modern cash register already generates and produce campaign actions validated against the operator's own financial constraints, the marginal cost of personalized marketing could be reduced to near zero for SMEs.

The system described in this paper, referred to as the **ML Campaign Engine**, was designed to address this gap. Its design principles are:

1. **Minimal input barrier** — the only required data is a standard six-column POS export (`transaction_id`, `customer_id`, `timestamp`, `total_price`, `quantity`, `product_name`).
2. **Financial safety by default** — campaigns are blocked before dispatch if they violate a configurable net margin floor, decoupling ML optimization from financial risk.
3. **Operator-readable output** — a local LLM layer translates technical model outputs into plain-language executive summaries suitable for business owners with no data background.
4. **Production-grade robustness** — a Champion-Challenger update protocol and six-scenario stress test suite ensure the pipeline performs reliably under real-world distribution shifts.

The primary contributions of this work are:

- A **five-layer MLOps architecture** designed from the ground up for SME constraints (Section 3).
- A **CFO Guardrail** component that integrates financial validation directly into the ML decision loop (Section 4.4).
- An **empirical evaluation** on synthetic POS data across six stress scenarios demonstrating robustness to conditions common in SME operations (Section 5).
- A **Champion-Challenger protocol** adapted for low-data environments typical of SMEs (Section 4.4).

---

## 2. Related Work

### 2.1 RFM Segmentation

Recency-Frequency-Monetary (RFM) analysis is a well-established customer segmentation technique in direct marketing, originally formalized by Hughes [1] and subsequently refined for database marketing applications by Bult and Wansbeek [2]. Classical RFM assigns customers to decile ranks across each dimension; more recent work by Chen et al. [3] augments this with clustering algorithms to capture non-linear segment boundaries that decile ranking cannot express. Our Layer 1 extends classical RFM with K-means clustering [4] and dynamic cluster selection via silhouette scoring [5], and adds an IQR-based outlier filter [6] to prevent high-spend anomalies from distorting segment centroids — a practical concern in SME datasets where a single bulk transaction can represent a substantial fraction of total revenue.

### 2.2 Market Basket Analysis

Association rule mining, particularly the Apriori algorithm introduced by Agrawal and Srikant [7], has been widely applied in retail to identify product affinity. The algorithmic framework was later made accessible to practitioners through software implementations such as arules [8]. Prior work focuses largely on large-scale grocery and e-commerce datasets [9]. Application to SME food service and mixed retail — where transaction volumes are lower and basket sizes smaller — remains underexplored. Our Layer 2 addresses this by operating at a minimum support threshold of 0.01, appropriate for smaller datasets, and falls back gracefully when no product category column is present in the POS export.

### 2.3 Uplift Modeling

Uplift modeling (also called incremental response modeling or true-lift modeling) estimates the causal effect of a campaign on an individual customer, rather than the probability of conversion in the presence of a campaign — a distinction first formally articulated by Radcliffe and Surry [10]. A comprehensive review by Devriendt et al. [12] surveys the state of the art across tree-based, regression, and meta-learner approaches. Gutierrez and Gérardy [11] further connect uplift modeling to causal inference frameworks. Gradient boosted tree models, including LightGBM [14], have shown strong performance in uplift tasks [12]; decision-tree-based uplift methods were systematically compared by Rzepakowski and Jaroszewicz [13]. Existing literature focuses predominantly on e-commerce and subscription contexts. Application to POS-driven SME campaigns — with the additional constraint of per-customer margin validation — is a novel contribution of this work.

### 2.4 MLOps for SMEs

The MLOps literature has grown substantially around large-scale deployments, with Sculley et al. [16] identifying the hidden technical debt incurred when ML systems are embedded in production software, and Kreuzberger et al. [17] providing a formal definition and architecture taxonomy for MLOps. Paleyes et al. [18] survey real-world deployment challenges across a large body of case studies. Work specifically addressing MLOps design for SME constraints — limited compute, small datasets, absence of dedicated ML engineers — remains sparse. Model governance practices including Champion-Challenger selection are discussed in Shmueli et al. [19]. Our Champion-Challenger protocol extends this direction with a financially motivated update gate that promotes a challenger model only when it demonstrates measurable profit improvement over the incumbent champion.

### 2.5 LLM Integration in Business Pipelines

Large language model capabilities and limitations are surveyed in Zhao et al. [22], while hallucination risks in generative systems are systematically catalogued by Ji et al. [23] — motivating our structured post-generation validation step. We adopt a local, quantized LLM (llama3.2:3b via Ollama) [21] to ensure that no customer transaction data leaves the operator's infrastructure, addressing data privacy concerns relevant to SME deployments operating under regional data protection regulations. The local inference approach trades generation quality for zero ongoing inference cost and full data sovereignty, a trade-off appropriate for SME contexts where cloud API budgets are constrained.

### 2.6 SME Technology Adoption

Barriers to technology adoption among SMEs are well documented. Nguyen et al. [25] confirm that perceived complexity and resource constraints are the primary inhibitors of IT adoption in small businesses, consistent with the access gap in marketing analytics that this work addresses. CRM adoption patterns in SME contexts are examined by Harrigan et al. [24], who find that relationship management tools are most effective when they reduce rather than increase operational complexity for the operator — a design principle reflected in the plain-language LLM output layer of this framework.

---

## 3. System Architecture

The ML Campaign Engine is a five-layer sequential pipeline. Each layer consumes the output of the previous layer, applies a specific algorithm, and passes enriched customer records downstream. Figure 1 illustrates the overall data flow.

```
POS Transaction CSV (6 columns)
        │
        ▼
┌─────────────────────────────────────┐
│  Layer 1: RFM Segmentation          │
│  K-means + Silhouette + IQR filter  │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Layer 2: Product Bundling &        │
│  Category Preference                │
│  Apriori + Association Rules        │
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
│  LightGBM Classifier                │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Layer 5: LLM Executive Report      │
│  llama3.2:3b (local, offline)       │
└────────────────┬────────────────────┘
                 │
                 ▼
   Per-Customer Campaign Action Plan
   + Executive Summary (Markdown)
```

*Figure 1. Five-layer ML Campaign Engine pipeline.*

The pipeline operates in three modes suited to different operational contexts:

| Mode | Entry Point | Use Case |
|------|-------------|----------|
| In-memory | `pipeline_engine.py` | Real-time scoring, monthly re-tuning |
| CSV-driven | `csv_pipeline_engine.py` | Batch processing, audit trail |
| CFO Guardrail | `run_pipeline.py` | Financial compliance, executive reporting |

---

## 4. Methodology

### 4.1 Layer 1 — RFM Customer Segmentation

**Input:** Raw POS transactions
**Output:** Per-customer RFM decile (1–10) and cluster assignment (0=low, 1=mid, 2=high value)

We compute standard RFM metrics per customer:

- **Recency (R):** Days since most recent transaction relative to the analysis date
- **Frequency (F):** Total transaction count
- **Monetary (M):** Cumulative spend

Prior to clustering, we apply IQR-based outlier filtering with a 3× upper quartile threshold on Monetary values to prevent bulk buyers from distorting cluster centroids. All three dimensions are standardized with zero mean and unit variance (StandardScaler) before clustering.

Cluster count *k* is selected dynamically from *k* ∈ {2, ..., 5} by maximizing the silhouette score, avoiding the common failure mode of hardcoding *k* without reference to actual data structure. The final cluster assignment maps to a three-tier customer value hierarchy.

### 4.2 Layer 2 — Market Basket Analysis and Category Profiling

**Input:** POS transactions
**Output:** Top-3 bundle association rules; per-customer category affinity vector

We apply the Apriori algorithm with minimum support 0.01 and minimum lift 1.0. Association rules are ranked by lift, and the top three rules are passed to downstream layers for bundle recommendation.

In parallel, we compute per-customer category affinity by aggregating spend share across product categories, producing a ranked `top_category` and `top_product` feature used by Layer 3's campaign matching rules.

A fallback is implemented for datasets without a `category` column: `product_name` is used as a proxy category, ensuring compatibility with minimally structured POS exports.

### 4.3 Layer 3 — Context Detection and Campaign Matching

**Input:** Layer 1 cluster assignments, Layer 2 category profiles, raw timestamps
**Output:** Per-transaction context flags; initial campaign assignment

**Time context:** Transactions are localized to the business timezone (configurable; default Asia/Bangkok) before extracting the hour. Transactions between 17:00–21:00 are flagged as `Peak`; all others as `Off-Peak`. This classification informs discount eligibility — off-peak promotions are offered to drive demand during low-traffic hours.

**Inventory estimation:** In the absence of live inventory feeds — typical for SME POS systems — we estimate remaining stock as:

```
initial_stock_est = quantity_sold × 1.5
remaining_stock   = initial_stock_est − quantity_sold
                  = quantity_sold × 0.5
```

Products with `remaining_stock ≤ 5` are flagged as low-stock and excluded from bundle campaigns to avoid promoting unavailable items.

**Campaign matching** applies a rule engine that combines cluster tier, category affinity, and context flags:

| Cluster | Default Campaign | Condition Override |
|---------|------------------|--------------------|
| 2 (High value) | VIP X2 Points Perk | — |
| 1 (Mid value) | Send Bundle | Top bundle available |
| 0 (Low value) | Standard Retention | — |
| Any | Off-Peak Special | Off-Peak flag + mid-tier AOV |

### 4.4 Layer 4 — Uplift Modeling and CFO Guardrail

**Input:** Enriched customer records from Layers 1–3
**Output:** `uplift_score`, `recommended_action`, `expected_profit` per customer; champion/challenger comparison report

#### 4.4.1 Uplift Model

We train two separate LightGBM classifiers: one on the *treatment* group (customers who received a campaign) and one on the *control* group (customers who did not). Per-customer uplift is computed as:

```
uplift_score = P(conversion | treatment) − P(conversion | control)
```

Treatment/control labels are derived from a time-based split: customers active in the most recent 25% of the observation window are labelled treatment; the remainder form the control baseline. A 5% holdout control group is preserved in every deployment cycle for ongoing A/B measurement.

#### 4.4.2 Decision Rules

Final action assignment integrates uplift score, RFM decile, AOV, and context flags through a priority-ordered decision tree:

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 | `uplift < −0.04` OR `AOV < 100` | Do Nothing |
| 2 | `uplift > 0.08` AND bundle available | Send Bundle |
| 3 | `RFM decile ≥ 8` OR `P(control) ≥ 0.4` | VIP X2 Points |
| 4 | Off-Peak AND `100 ≤ AOV ≤ 800` AND `RFM ≥ 4` | Off-Peak Special |
| 5 | Default | Standard Retention |

#### 4.4.3 CFO Guardrail (Novel Contribution)

Before finalizing any campaign assignment, the system computes expected net profit under the proposed promotion:

```
gross_revenue   = AOV × (1 − discount_rate)
cogs            = AOV × cogs_rate
promo_cost      = AOV × promo_cost_rate[action]
net_profit      = gross_revenue − cogs − promo_cost
net_margin      = net_profit / gross_revenue
```

If `net_margin < min_allowable_net_margin` (default 15%), the action is overridden to `Do Nothing` regardless of uplift score. This constraint prevents the common failure mode in ML campaign systems where high predicted uplift masks negative unit economics.

This design embeds financial constraints directly into the ML decision loop, eliminating the manual review step that typically exists as an organizational boundary between data science and finance teams. The parameters (`cogs_rate`, `min_allowable_net_margin`, `promo_cost_rates`) are fully configurable per business in `business_config.py`, enabling deployment across business types with different cost structures.

#### 4.4.4 Champion-Challenger Protocol

Model versioning follows a Champion-Challenger pattern. A new challenger model is automatically promoted to production if either:

- Challenger expected profit exceeds Champion by ≥ 3%, OR
- Recession mode is active (current average spend < 65% of baseline) AND challenger "Do Nothing" rate exceeds 50% — reflecting that capital preservation is appropriate during economic contraction.

Both champion and challenger models are serialized with timestamps and retained in an archive for reproducibility.

### 4.5 Layer 5 — LLM Executive Report

**Input:** Final campaign assignments and financial projections
**Output:** Plain-language executive summary with five sections: segment overview, campaign recommendations, financial projections, execution plan, and post-campaign forecast

We use `llama3.2:3b` served locally via Ollama (temperature = 0.0 for determinism, max 3,500 tokens). Pre-formatting of the input payload into structured text occurs before the LLM call to reduce parsing ambiguity. A post-generation validation step cross-checks segment names, customer counts, and profit values in the LLM output against the source payload, appending a warning block if discrepancies are detected.

Local inference ensures that no customer transaction data leaves the operator's infrastructure, addressing data privacy concerns relevant to SME deployments.

---

## 5. Experiments

### 5.1 Datasets

**Primary synthetic POS dataset:** 1,001 transactions generated to simulate a mixed retail and food service environment, spanning typical SKUs across food, beverage, apparel, and electronics accessories categories. This dataset validates the pipeline end-to-end on the target SME input format.

**Reference e-commerce dataset:** A publicly available UK online retail dataset (541,910 transactions, 600+ unique customers) is used to validate Layer 1 RFM segmentation and Layer 4 uplift modeling at larger scale.

### 5.2 Stress Test Scenarios

To assess robustness under distribution shifts common in SME operations, we evaluate the pipeline against six synthetic stress scenarios:

| Scenario | Description | Key Challenge |
|----------|-------------|---------------|
| `mega_sale` | Black Friday / flash sale spike | Extreme volume, inflated AOV |
| `recession` | Economic downturn | Suppressed spend, baseline drift |
| `bulk_shift` | Bulk buyer behavior change | Outlier distribution change |
| `cold_start` | New market or new location | Sparse transaction history |
| `off_peak` | Off-peak hour concentration | Time context edge case |
| `data_noise` | Corrupted timestamps, missing fields | Data quality robustness |

Each scenario is generated as a separate CSV and passed through the full pipeline independently.

### 5.3 Evaluation Metrics

| Metric | Layer | Description |
|--------|-------|-------------|
| Silhouette score | L1 | Cluster quality |
| Association rule lift | L2 | Bundle relevance |
| Uplift score distribution | L4 | Treatment effect estimation |
| Expected profit per customer | L4 | Financial value of recommendation |
| Champion-Challenger profit delta (%) | L4 | Model update quality |
| Margin compliance rate (%) | L4 | CFO guardrail effectiveness |
| LLM output validation pass rate | L5 | Report accuracy |

---

## 6. Results

### 6.1 Customer Segmentation (Layer 1)

On the synthetic POS dataset, dynamic *k* selection via silhouette score consistently identifies *k* = 3 as the optimal cluster count, aligning with the expected high / mid / low value tier structure. IQR outlier filtering removed fewer than 2% of customers, confirming the synthetic data is clean; this filter is expected to have higher impact on real-world SME data where bulk transactions are common.

On the UK retail reference dataset, the pipeline segments 600+ customers into RFM deciles 1–10 with clear monetary separation between tiers. Top-tier customers (decile 10) exhibit recency of 2 days, frequency of 175 transactions, and monetary value of $3,517 — confirming the pipeline correctly identifies high-value champions.

### 6.2 Product Bundling (Layer 2)

On the synthetic POS dataset, Apriori identifies three significant association rules. The highest-lift rule — `{Socks Pack, Sparkling Water, Sunscreen SPF50, Wireless Mouse}` → `{Foam Cleanser, Dark Chocolate, Canvas Tote Bag}` — achieves a lift of **66.3** and confidence of 1.0, indicating a strong cross-category affinity in this synthetic retail environment. In production deployment, operators can interpret high-lift rules as natural bundle candidates for promotional packaging.

### 6.3 Champion-Challenger Performance (Layer 4)

Across all customers in the model comparison evaluation, the Challenger model achieves a mean expected profit improvement of **4.16%** over the Champion (range: 4.15%–4.21%), exceeding the 3% promotion threshold and triggering automatic Champion replacement. This demonstrates the update protocol operates as intended and that the 3% gate is a meaningful but achievable quality bar.

### 6.4 CFO Guardrail Behavior

The margin floor mechanism filters campaigns that would generate below-threshold margins before dispatch. In the synthetic POS dataset, customers assigned `Standard Retention` at a 15% discount rate with 50% COGS consistently fall below the 15% net margin floor, resulting in correct reclassification to `Do Nothing`. This confirms that the guardrail operates as a last-line financial check independent of upstream ML decisions.

### 6.5 Stress Test Robustness

| Scenario | Pipeline Status | Notable Behavior |
|----------|-----------------|-----------------|
| `mega_sale` | Pass | Spike detection via `is_campaign_spike` flag correctly suppresses redundant promotions |
| `recession` | Pass | Challenger "Do Nothing" rate exceeds 50% threshold, activating recession-mode promotion |
| `bulk_shift` | Pass | IQR filter correctly isolates bulk buyers from standard segmentation |
| `cold_start` | Pass | Minimum *k* = 2 fallback activates with sparse data; silhouette score degraded but functional |
| `off_peak` | Pass | Off-peak promotions correctly activated for eligible tier during non-peak hours |
| `data_noise` | Pass | Timestamp coercion and `dropna` handling prevent pipeline failure; 0 unhandled exceptions |

---

## 7. Discussion

### 7.1 The CFO Guardrail as Organizational Boundary Automation

The most significant novel contribution of this framework is not algorithmic but **architectural**: the placement of a financial constraint validator inside the ML decision loop rather than as a downstream human review step. In typical enterprise deployments, the data science team generates recommendations that are subsequently reviewed by finance or marketing operations before deployment, creating latency and organizational friction. By parameterizing financial constraints (COGS rate, margin floor, discount rates) and evaluating them per-customer at inference time, the system eliminates this review step for the vast majority of routine campaigns, while still surfacing edge cases through the `Do Nothing` classification.

This design is particularly valuable for SME operators who may not have separate finance and marketing functions, or where the owner makes both decisions. The single configurable `business_config.py` file serves as the operator's business rule specification.

### 7.2 Accessibility vs. Accuracy Trade-offs

The framework makes deliberate simplifications to maintain accessibility:

- **Inventory estimation** from sales velocity rather than live stock feeds introduces estimation error. In real deployments, operators with inventory systems can replace the estimator with direct stock queries.
- **Uplift labels** are derived from a time split rather than a randomized controlled trial. This is a necessary concession for SME settings where historical A/B data is unavailable; the 5% ongoing control group enables proper causal measurement going forward.
- **Local LLM** (3B parameter model) produces adequate executive summaries but may miss nuance compared to larger models. The structured validation layer partially compensates for this limitation.

### 7.3 Extensibility

The pipeline architecture is designed for vertical expansion beyond food service and retail. The six-column input schema is generic enough to accommodate any transactional business. Future extensions planned include:

- **Event-driven campaign prediction:** Incorporating scheduled events (festivals, public holidays, promotional calendars) as Layer 3 context signals, enabling pre-emptive campaign planning.
- **Multi-location support:** Aggregating POS streams from multiple outlets with site-level context flags.
- **Service businesses:** Adapting `product_name` to service SKUs (e.g., appointment types) for beauty, automotive, and professional services verticals.

### 7.4 Limitations and Future Work

The current evaluation relies on synthetic data. Validation against real SME POS datasets from partner deployments — with measured campaign lift from the A/B holdout group — is the critical next step to establish external validity. We also note that the association rule lift values observed in the synthetic dataset (up to 66.3) reflect the structured nature of synthetic generation and are expected to be lower in real-world data.

Future work will examine: (1) online learning to update RFM and uplift models continuously rather than in monthly batch cycles, (2) multi-armed bandit formulations for campaign selection to accelerate real-world A/B learning, and (3) privacy-preserving federated learning to aggregate model improvements across SME operators without pooling customer data.

---

## 8. Conclusion

We presented the ML Campaign Engine, a five-layer MLOps pipeline that enables small and medium enterprises in retail and food service to generate financially validated, per-customer campaign recommendations from standard POS transaction exports. The system requires no dedicated data science expertise, no additional data infrastructure beyond an existing POS system, and produces outputs in plain-language executive summary format suitable for direct operator consumption.

The novel CFO Guardrail mechanism — which enforces net margin constraints per campaign at inference time — addresses a structural gap in existing uplift modeling literature, where financial validation is typically treated as a separate organizational process rather than an integrated algorithmic constraint.

Experiments on synthetic POS data across six stress scenarios demonstrate pipeline robustness under distribution shifts representing common SME operational conditions. Champion-Challenger evaluation yields a mean profit improvement of 4.16% per model update cycle, and all six stress scenarios pass without pipeline failure.

The framework establishes a reproducible, extensible baseline for democratizing campaign intelligence infrastructure for SMEs, and provides a validated foundation for partner deployment and real-world A/B evaluation.

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

[12] Devriendt, F., Moldovan, D., & Verbeke, W. (2018). A literature survey and experimental evaluation of the state-of-the-art in uplift modeling: A stepping stone toward the development of prescriptive analytics. *Big Data*, 6(1), 13–41. https://doi.org/10.1089/big.2017.0104

[13] Rzepakowski, P., & Jaroszewicz, S. (2012). Decision trees for uplift modeling with single and multiple treatments. *Knowledge and Information Systems*, 32(2), 303–327. https://doi.org/10.1007/s10115-011-0434-0

### Gradient Boosted Trees

[14] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, pp. 3146–3154.

[15] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, pp. 785–794. https://doi.org/10.1145/2939672.2939785

### MLOps & Technical Debt

[16] Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J. F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. In *Advances in Neural Information Processing Systems (NeurIPS 2015)*, 28, pp. 2503–2511.

[17] Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). Machine learning operations (MLOps): Overview, definition, and architecture. *IEEE Access*, 11, 31866–31879. https://doi.org/10.1109/ACCESS.2023.3262138

[18] Paleyes, A., Urma, R. G., & Lawrence, N. D. (2022). Challenges in deploying machine learning: A survey of case studies. *ACM Computing Surveys*, 55(6), 1–29. https://doi.org/10.1145/3533378

### Champion-Challenger Model Governance

[19] Shmueli, G., Bruce, P. C., Gedeck, P., & Patel, N. R. (2020). *Data Mining for Business Analytics: Concepts, Techniques and Applications in Python*. John Wiley & Sons.

[20] Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press. (Chapter 7: Model selection and deployment governance.)

### LLM Integration & Local Inference

[21] Meta AI. (2024). *Introducing Meta Llama 3: The most capable openly available LLM to date*. Meta AI Research Blog. https://ai.meta.com/blog/meta-llama-3

[22] Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., Du, Y., Yang, C., Chen, Y., Chen, Z., Jiang, J., Ren, R., Li, Y., Tang, X., Liu, Z., Liu, P., Nie, J. Y., & Wen, J. R. (2023). A survey of large language models. *arXiv preprint arXiv:2303.18223*.

[23] Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), 1–38. https://doi.org/10.1145/3571730

### SME Marketing & Data Accessibility

[24] Harrigan, P., Soutar, G., Choudhury, M. M., & Lowe, M. (2015). Modelling CRM in a social media age. *Australasian Marketing Journal*, 23(1), 27–37. https://doi.org/10.1016/j.ausmj.2014.11.001

[25] Nguyen, T. H., Newby, M., & Macaulay, M. J. (2015). Information technology adoption in small business: Confirmation of a proposed framework. *Journal of Small Business Management*, 53(1), 207–227. https://doi.org/10.1111/jsbm.12058

### Dataset

[26] Chen, D. (2015). Online Retail. *UCI Machine Learning Repository*. https://doi.org/10.24432/C5BW33

---

## Appendix A — Pipeline Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cogs_rate` | 0.50 | Cost of goods as fraction of revenue |
| `min_allowable_net_margin` | 0.15 | CFO floor; campaigns below this → Do Nothing |
| `control_group_rate` | 0.05 | A/B holdout fraction |
| `gate_growth_target_pct` | 3.0 | Challenger promotion threshold (%) |
| `recession_threshold` | 0.65 | Spend ratio triggering recession mode |
| `peak_hour_start` | 17 | Start of peak window (24h, inclusive) |
| `peak_hour_end` | 21 | End of peak window (24h, inclusive) |
| `low_basket_threshold` | 100 | AOV below this → Do Nothing |
| `high_uplift_threshold` | 0.08 | Uplift above this → bundle promotion |
| `vip_rfm_threshold` | 8 | RFM decile ≥ this → VIP reward |
| `stock_buffer_multiplier` | 1.5 | Inventory estimation multiplier |

---

## Appendix B — Stress Test Scenario Definitions

Each stress scenario is a synthetic CSV generated by `generate_stress_tests.py` with the following distributional characteristics:

| Scenario | Volume | AOV Distribution | Temporal Pattern |
|----------|--------|-----------------|-----------------|
| `mega_sale` | 3× normal | Right-skewed, high AOV | Concentrated in peak hours |
| `recession` | 0.6× normal | Left-shifted, low AOV | Uniform |
| `bulk_shift` | Normal | Extreme right tail (bulk buyers) | Uniform |
| `cold_start` | 0.2× normal | Normal | Uniform, short window |
| `off_peak` | Normal | Normal | Concentrated 00:00–16:00 |
| `data_noise` | Normal | Normal + NaN injection | Corrupted timestamps (10%) |

---

*End of Draft v0.1*
*Classification: Confidential — Proprietary Research*
