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
    "promo_cost_rates": {
        "vip_points": 0.02,   # Double-points reward
        "bundle":     0.15,   # Bundle discount
        "off_peak":   0.15,   # Off-peak discount
        "festival":   0.05,   # Festival incentive
        "standard":   0.05,   # Standard retention offer
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
    "high_uplift_threshold":   0.08,  # uplift above this → push bundle promo
    "vip_rfm_threshold":       8,     # RFM decile ≥ this → VIP reward
    "vip_prob_threshold":      0.4,   # prob_control ≥ this → VIP reward
    "spike_rfm_threshold":     7,     # RFM decile for campaign-spike guard
    "mid_rfm_threshold":       4,     # RFM decile for mid-tier off-peak eligibility

    # === Inventory estimation ===
    "stock_buffer_multiplier":   1.5,   # estimated stock covers this multiple of avg demand per cycle
    "low_stock_threshold":       5,     # remaining_stock ≤ this → out of stock
    "stock_restock_cycle_days":  14,    # assumed replenishment cycle (days)

    # === Timing & timezone ===
    "business_tz":     "Asia/Bangkok",
    "peak_hour_start": 17,   # 5 pm (inclusive)
    "peak_hour_end":   21,   # 9 pm (inclusive)
}
