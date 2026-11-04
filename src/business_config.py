# =============================================================
# Business Configuration — edit here to tune pipeline params
# =============================================================
import os as _os
REGISTRY_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models", "registry")

CONFIG = {
    # === Identity ===
    "business_name":    "UK Online Gift & Home Decor Store",
    "business_type":    "ECommerce_Retail",
    "currency":         "$",

    # === Financial ===
    "cogs_rate":                  0.50,   # Cost of Goods Sold as % of revenue
    "min_allowable_net_margin":   0.15,   # CFO floor: reject campaigns below this margin
    "control_group_rate":         0.05,   # A/B holdout split

    # === Promo cost rates (as % of AOV per action type) ===
    # Off-peak rates are higher — justified because store is quiet and needs traffic.
    # Peak rates are lower — customers are already coming, just retain/upsell.
    "promo_cost_rates": {
        "vip_points":         0.02,   # Double-points reward (peak)
        "bundle":             0.15,   # Bundle discount (peak)
        "off_peak":           0.15,   # Off-peak discount (legacy)
        "festival":           0.05,   # Festival incentive
        "standard":           0.05,   # Standard retention offer (peak)
        # Context-aware rates
        "off_peak_traffic":   0.22,   # Aggressive off-peak traffic driver — heavier discount ok
        "peak_upsell":        0.08,   # Light touch during peak — customers already there
        "off_peak_vip":       0.05,   # VIP points during off-peak — cheap way to pull them in
        "bundle_upsell_low":  0.03,   # Light nudge for low-basket customers — suggest adding 1 bundle item, no heavy discount
        "milestone_reward":   0.03,   # Points reward for reaching milestone basket (cost as % of milestone target)
        "new_customer_welcome": 0.10, # Welcome offer — 1st return visit incentive
        "churn_rescue":         0.12, # Early warning retention — before full churn
        "vip_winback":          0.15, # Aggressive win-back for high-value drifting VIP
    },

    # === Milestone Reward (low-basket Champions) ===
    "milestone_basket":  150,   # THB target order size to earn reward
    "milestone_points":   50,   # points awarded on reaching milestone

    # === Send window: when to push the notification per context ===
    # Format: "HH:MM" — send notification this many hours BEFORE the target window
    "send_window": {
        "off_peak": "pre_off_peak",   # send 30-60 min before off-peak window opens
        "peak":     "pre_peak",       # send 30-60 min before peak window opens
    },

    # === Default campaign discount ===
    "default_discount_rate": 0.15,

    # === Post-campaign uplift assumptions ===
    "retention_uplift_champion": 0.05,   # retention improvement for Champions
    "retention_uplift_atrisk":   0.12,   # retention improvement for At-Risk
    "repeat_uplift_champion":    0.05,
    "repeat_uplift_atrisk":      0.10,

    # === Model validation gate ===
    "gate_growth_target_pct":         3.0,    # Challenger must beat Champion by this %
    "recession_threshold":            0.65,   # current_avg < baseline_avg * this → recession
    "do_nothing_recession_threshold": 50.0,   # %Do-Nothing required to pass gate in recession

    # === Uplift decision rules ===
    "low_basket_threshold":   100,    # AOV below this → Do Nothing
    "mid_basket_threshold":   800,    # AOV below this → off-peak promo
    "negative_uplift_guard":  -0.04,  # uplift below this → Do Nothing
    "rescue_uplift_guard":    -0.10,  # relaxed threshold for churn_risk / cant_lose / hibernating
    "high_uplift_threshold":   0.08,  # uplift above this → push bundle promo
    "vip_rfm_threshold":       8,     # RFM decile ≥ this → VIP reward
    "zero_cost_rfm_threshold": 7,     # RFM decile ≥ this → eligible for zero-cost notification
    "vip_prob_threshold":      0.4,   # prob_control ≥ this → VIP reward
    "spike_rfm_threshold":     7,     # RFM decile for campaign-spike guard
    "mid_rfm_threshold":       4,     # RFM decile for mid-tier off-peak eligibility

    # Uplift group classification thresholds
    "lost_cause_uplift_threshold": 0.02,   # uplift < this AND prob_control < lost_cause_prob → Lost Cause
    "lost_cause_prob_threshold":   0.10,   # prob_control < this (combined with uplift) → Lost Cause
    "sure_thing_prob_threshold":   0.50,   # prob_control >= this → Sure Thing (converts regardless)

    # === Campaign type bands (uplift percentile → promo type) ===
    # Used by UpliftInference.target() to assign campaign actions.
    # Keys are percentile thresholds (pct_rank >= key → value), ordered high → low.
    "campaign_type_bands": {
        0.90: "bundle_discount",    # top decile
        0.40: "loyalty_points",     # mid (40th–90th pct)
        0.00: "off_peak_discount",  # lower 40th pct
    },

    # === Inventory estimation ===
    "stock_buffer_multiplier":   1.5,   # estimated stock covers this multiple of avg demand per cycle
    "low_stock_threshold":       5,     # remaining_stock ≤ this → out of stock
    "stock_restock_cycle_days":  14,    # assumed replenishment cycle (days)

    # === Rule-based segment flag thresholds ===
    # Applied AFTER K-Means — override tier for critical states K-Means may miss
    "new_customer_max_days":     30,   # recency < this + freq ≤ 2 → new customer
    "new_customer_max_freq":      2,
    "churn_risk_min_days":       45,   # recency > this + freq ≥ threshold → churn risk
    "churn_risk_min_freq":        3,
    "cant_lose_min_days":        90,   # recency > this + high F/M → can't lose (VIP drifting)
    "cant_lose_min_freq":         5,
    "cant_lose_min_rfm_decile":   7,   # top 30%
    "about_to_sleep_min_days":   25,   # recency > this → starting to drift
    "about_to_sleep_max_days":   45,   # recency ≤ this → not yet churn_risk
    "about_to_sleep_max_freq":    4,   # max frequency to qualify (not a frequent buyer drifting)

    # === Timing & timezone ===
    "business_tz":     "Asia/Bangkok",
    "peak_hour_start": 17,   # 5 pm (inclusive)
    "peak_hour_end":   21,   # 9 pm (inclusive)

    # === Cluster-tier promo rates ===
    # Maps cluster tier name → rate key from promo_cost_rates
    "cluster_promo_rates": {
        "champions":   "vip_points",   # no % discount — loyalty/VIP only
        "at_risk":     "standard",
        "hibernating": "off_peak",     # deeper discount for reactivation
        "mid":         "bundle",
    },
    "hibernating_recency_days": 90,    # recency > this → hibernating/lost_cause threshold
}
