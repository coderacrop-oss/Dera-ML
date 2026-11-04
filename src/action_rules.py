"""Declarative action rules for the 2-step campaign decision engine.

HOW EACH LAYER FEEDS THE FINAL ACTION
======================================

  Layer 1 — RFM + K-Means
    → cluster_tier        which rule set applies (champions / at_risk / hibernating / mid)
    → rfm_decile_segment  VIP eligibility within a tier (rfm_min conditions on rules)

  Layer 2 — Market Basket Analysis
    → bundle_name         bundle partner injected into action name templates
                          e.g. "VIP Bundle: ชาไทย + ทาร์ตไข่"
                          falls back to top_bundle_name if Layer 2 found no rules

  Layer 3 — Peak / Off-Peak + Inventory
    → time_context        selects peak vs off_peak rate keys and action variants
    → is_campaign_spike   enables Festival Guard for mid-tier high-RFM customers
    → in_stock            hard stop — Out of Stock → no action regardless of uplift

  Layer 4 — X-Learner Uplift Model
    → uplift_score + prob_control → uplift_group
        "do_not_disturb"  uplift < neg_guard (-0.04)        hard stop
        "lost_cause"      uplift < 0.02 AND prob < 0.10     hard stop
        "sure_thing"      prob_control >= 0.50              loyalty-only rules
        "persuadable"     otherwise                         full action menu
    → uplift_score        also used directly as uplift_min condition (Bundle Upsell rule)

RULE SCHEMA
===========
Both SURE_THING_RULES and PERSUADABLE_RULES share the same field schema:

  id            str         unique rule identifier
  tiers         list[str]   cluster tiers that match; omit = all tiers
  time_context  list[str]   ["peak"] | ["off_peak"]; omit = both
  rfm_min       int|str     minimum rfm_decile_segment (1–10); str = CONFIG key to resolve
  is_spike_min  float       minimum is_campaign_spike (0–1)
  uplift_min    float|str   minimum uplift_score; str = CONFIG key to resolve
  rate_key      str         key into CONFIG["promo_cost_rates"]
  action        str         action name; {bundle} is replaced with the customer's bundle

EVALUATION STRATEGY
===================
  SURE_THING_RULES  — first_viable: rules evaluated in list order, first profitable wins.
  PERSUADABLE_RULES — best_profit:  all matching viable rules collected, max profit wins.
"""

from business_config import CONFIG as _CONFIG


def _resolve_threshold(val):
    """Resolve a CONFIG key reference (str) or return numeric value as-is."""
    if isinstance(val, str):
        return _CONFIG[val]
    return val


def cluster_tier(segment_name: str) -> str:
    """Infer cluster tier from segment label.

    Checks rule-based flag names first (set by _apply_segment_flags), then
    IBM centroid-based K-Means labels. Uses startswith so suffixed duplicates
    like "Champions (2)" still match correctly.
    """
    n = segment_name.lower()
    # Rule-based flags (priority overrides)
    if n.startswith("can't lose") or n.startswith("cant lose"):
        return "cant_lose"
    if n.startswith("churn risk") or n.startswith("churn_risk"):
        return "churn_risk"
    if n.startswith("new customer") or n.startswith("new_customer"):
        return "new_customer"
    if n.startswith("about to sleep"):
        return "about_to_sleep"
    # IBM K-Means labels
    if n.startswith("champion"):
        return "champions"
    if n.startswith("loyal"):
        return "loyal"
    if n.startswith("potential loyalist"):
        return "potential_loyalist"
    if n.startswith("promising"):
        return "promising"
    if n.startswith("need attention"):
        return "need_attention"
    if "at-risk" in n or "at_risk" in n:
        return "at_risk"
    if n.startswith("hibernat") or n.startswith("lost"):
        return "hibernating"
    return "mid"


def matches(rule: dict, tier: str, is_off_peak: bool,
            rfm_seg: float, is_spike: float, uplift: float) -> bool:
    """Return True if the customer's context satisfies all conditions in `rule`."""
    if "tiers" in rule and tier not in rule["tiers"]:
        return False
    if "time_context" in rule:
        ctx = "off_peak" if is_off_peak else "peak"
        if ctx not in rule["time_context"]:
            return False
    rfm_min = _resolve_threshold(rule.get("rfm_min"))
    if rfm_min is not None and rfm_seg < rfm_min:
        return False
    if "is_spike_min" in rule and is_spike < rule["is_spike_min"]:
        return False
    uplift_min = _resolve_threshold(rule.get("uplift_min"))
    if uplift_min is not None and uplift < uplift_min:
        return False
    return True


# ─── Sure-Thing rules ────────────────────────────────────────────────────────
# Evaluated in order; the first profitable rule wins.
# Sure Things convert regardless — loyalty and VIP only, never % discount.

SURE_THING_RULES = [
    # Tier-specific overrides for critical states — evaluated before generic rules
    {
        "id":      "st_cantlose_winback",
        "tiers":   ["cant_lose"],
        "rate_key": "vip_winback",
        "action":  "VIP Win-Back: กลับมาเป็น VIP อีกครั้ง",
    },
    {
        "id":      "st_newcust_welcome",
        "tiers":   ["new_customer"],
        "rate_key": "new_customer_welcome",
        "action":  "Welcome: ยินดีต้อนรับ กลับมาอีกนะ!",
    },
    # General rules — apply to all tiers
    {
        "id":           "st_vip_reward_offpeak",
        "time_context": ["off_peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "vip_points",
        "action":       "VIP Reward: แจกแต้ม X2",
    },
    {
        "id":           "st_vip_early_access_peak",
        "time_context": ["peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "peak_upsell",
        "action":       "VIP Early Access",
    },
    {
        "id":       "st_loyalty_nurture",
        "rate_key": "vip_points",
        "action":   "Loyalty Nurture: แต้มสะสม",
    },
]


# ─── Persuadable rules ───────────────────────────────────────────────────────
# All matching viable rules are collected; the one with highest expected profit wins.

PERSUADABLE_RULES = [

    # ── Loyal Customers: high F/M, medium R — treat well, light discount ok ──
    {
        "id":    "loyal_vip_bundle",
        "tiers": ["loyal"],
        "rate_key": "vip_points",
        "action": "Loyalty Reward: {bundle} + แต้มพิเศษ",
    },
    {
        "id":           "loyal_vip_reward",
        "tiers":        ["loyal"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "vip_points",
        "action":       "VIP Reward: แจกแต้ม X2",
    },
    {
        "id":           "loyal_standard",
        "tiers":        ["loyal"],
        "rate_key":     "standard",
        "action":       "Loyal Offer: {bundle}",
    },

    # ── Potential Loyalist: high R, medium F — build the habit ───────────────
    {
        "id":    "potloyal_habit_builder",
        "tiers": ["potential_loyalist"],
        "rate_key": "vip_points",
        "action": "Habit Builder: สะสมแต้ม เพิ่มความถี่",
    },
    {
        "id":           "potloyal_offpeak_vip",
        "tiers":        ["potential_loyalist"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak_vip",
        "action":       "Loyalty Builder Off-Peak: แต้มพิเศษช่วงเงียบ",
    },

    # ── Promising: high R, low F, medium M — encourage return visit ──────────
    {
        "id":    "promising_reengage",
        "tiers": ["promising"],
        "rate_key": "standard",
        "action": "Re-engage: {bundle} พิเศษสำหรับคุณ",
    },
    {
        "id":           "promising_offpeak",
        "tiers":        ["promising"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak",
        "action":       "Re-engage Off-Peak: ส่วนลดพิเศษกลับมาหาเรา",
    },

    # ── Need Attention: medium all, starting to drift ─────────────────────────
    {
        "id":    "needattn_bundle",
        "tiers": ["need_attention"],
        "rate_key": "standard",
        "action": "เราเป็นห่วง: {bundle} + ส่วนลดพิเศษ",
    },
    {
        "id":           "needattn_offpeak_traffic",
        "tiers":        ["need_attention"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak_traffic",
        "action":       "Re-engage: ส่วนลดพิเศษช่วงเงียบสำหรับคุณ",
    },

    # ── About to Sleep: recency growing, low-medium F — urgent before dark ───
    {
        "id":    "aboutsleep_urgent",
        "tiers": ["about_to_sleep"],
        "rate_key": "off_peak",
        "action": "เราคิดถึงคุณ: {bundle} ก่อนที่จะไม่ได้เจอกัน",
    },
    {
        "id":           "aboutsleep_traffic_driver",
        "tiers":        ["about_to_sleep"],
        "rate_key":     "off_peak_traffic",
        "action":       "Last Chance: ส่วนลดพิเศษ กลับมาหาเราเลย",
    },

    # ── New Customer: onboarding — light welcome offer to earn 2nd visit ────────
    {
        "id":           "newcust_welcome_peak",
        "tiers":        ["new_customer"],
        "time_context": ["peak"],
        "rate_key":     "new_customer_welcome",
        "action":       "Welcome: สั่งครั้งที่ 2 รับส่วนลด 10%",
    },
    {
        "id":           "newcust_welcome_offpeak",
        "tiers":        ["new_customer"],
        "time_context": ["off_peak"],
        "rate_key":     "new_customer_welcome",
        "action":       "Welcome Off-Peak: กลับมาช้อปรับส่วนลดพิเศษ",
    },

    # ── Churn Risk: early warning — act before the customer fully drifts ──────
    {
        "id":    "churnrisk_rescue_bundle",
        "tiers": ["churn_risk"],
        "rate_key": "churn_rescue",
        "action": "Early Win-Back: {bundle} ก่อนที่จะไม่กลับมา",
    },
    {
        "id":           "churnrisk_offpeak_traffic",
        "tiers":        ["churn_risk"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak_traffic",
        "action":       "Churn Rescue: ส่วนลดพิเศษช่วงเงียบ เราคิดถึงคุณ",
    },

    # ── Can't Lose Them: VIP drifting — aggressive win-back ───────────────────
    {
        "id":    "cantlose_vip_winback",
        "tiers": ["cant_lose"],
        "rate_key": "vip_winback",
        "action": "VIP Win-Back: {bundle} + แต้มพิเศษ X3",
    },
    {
        "id":           "cantlose_aggressive_offpeak",
        "tiers":        ["cant_lose"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak_traffic",
        "action":       "VIP Win-Back: กลับมาช้อปรับส่วนลดพิเศษ VIP",
    },

    # ── Champions: VIP / loyalty only — never a % discount ───────────────────
    {
        "id":    "champ_vip_bundle",
        "tiers": ["champions"],
        "rate_key": "vip_points",
        "action": "VIP Bundle: {bundle}",
    },
    {
        "id":           "champ_early_access_peak",
        "tiers":        ["champions"],
        "time_context": ["peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "peak_upsell",
        "action":       "VIP Early Access",
    },
    {
        "id":           "champ_early_access_offpeak",
        "tiers":        ["champions"],
        "time_context": ["off_peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "vip_points",
        "action":       "VIP Early Access",
    },
    {
        "id":      "champ_vip_reward",
        "tiers":   ["champions"],
        "rfm_min": "vip_rfm_threshold",
        "rate_key": "vip_points",
        "action":  "VIP Reward: แจกแต้ม X2",
    },

    # ── At-Risk: discount bundle first, then standard retention ──────────────
    {
        "id":    "atrisk_discount_bundle",
        "tiers": ["at_risk"],
        "rate_key": "standard",
        "action": "Discount 15%: ซื้อ {bundle}",
    },
    {
        "id":           "atrisk_standard_offpeak",
        "tiers":        ["at_risk"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak",
        "action":       "Standard Retention",
    },
    {
        "id":           "atrisk_standard_peak",
        "tiers":        ["at_risk"],
        "time_context": ["peak"],
        "rate_key":     "standard",
        "action":       "Standard Retention",
    },
    {
        "id":           "atrisk_offpeak_traffic",
        "tiers":        ["at_risk"],
        "time_context": ["off_peak"],
        "rfm_min":      "mid_rfm_threshold",
        "rate_key":     "off_peak_traffic",
        "action":       "Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ",
    },

    # ── Hibernating: reactivation incentive ──────────────────────────────────
    {
        "id":    "hiber_reactivation_deep",
        "tiers": ["hibernating"],
        "rate_key": "off_peak",
        "action": "กลับมาช้อป: {bundle} ลด 20%",
    },
    {
        "id":    "hiber_reactivation_traffic",
        "tiers": ["hibernating"],
        "rate_key": "off_peak_traffic",
        "action": "Reactivation: {bundle}",
    },

    # ── Mid: contextual upsell ladder ─────────────────────────────────────────
    {
        "id":           "mid_festival_guard",
        "tiers":        ["mid"],
        "rfm_min":      "spike_rfm_threshold",
        "is_spike_min": 0.5,
        "rate_key":     "festival",
        "action":       "Standard Retention (Festival Guard)",
    },
    {
        "id":           "mid_offpeak_traffic",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "rfm_min":      "mid_rfm_threshold",
        "rate_key":     "off_peak_traffic",
        "action":       "Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ",
    },
    {
        "id":           "mid_offpeak_vip_points",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "rfm_min":      "mid_rfm_threshold",
        "rate_key":     "off_peak_vip",
        "action":       "Off-Peak VIP Points: ดึงลูกค้าช่วงเงียบ (แต้มพิเศษ)",
    },
    {
        "id":           "mid_vip_reward_peak",
        "tiers":        ["mid"],
        "time_context": ["peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "peak_upsell",
        "action":       "VIP Reward: แจกแต้ม X2",
    },
    {
        "id":           "mid_vip_reward_offpeak",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "vip_points",
        "action":       "VIP Reward: แจกแต้ม X2",
    },
    {
        "id":           "mid_vip_early_access_peak",
        "tiers":        ["mid"],
        "time_context": ["peak"],
        "rfm_min":      9,   # rfm >= 9 (stricter than vip_rfm=8)
        "rate_key":     "peak_upsell",
        "action":       "VIP Early Access",
    },
    {
        "id":           "mid_vip_early_access_offpeak",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "rfm_min":      9,
        "rate_key":     "vip_points",
        "action":       "VIP Early Access",
    },
    {
        "id":           "mid_bundle_upsell_peak",
        "tiers":        ["mid"],
        "time_context": ["peak"],
        "uplift_min":   "high_uplift_threshold",
        "rate_key":     "peak_upsell",
        "action":       "Bundle Upsell: {bundle}",
    },
    {
        "id":           "mid_bundle_upsell_offpeak",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "uplift_min":   "high_uplift_threshold",
        "rate_key":     "bundle",
        "action":       "Bundle Upsell: {bundle}",
    },
    {
        "id":           "mid_standard_peak",
        "tiers":        ["mid"],
        "time_context": ["peak"],
        "rate_key":     "standard",
        "action":       "Standard Retention",
    },
    {
        "id":           "mid_standard_offpeak",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak",
        "action":       "Standard Retention",
    },
]
