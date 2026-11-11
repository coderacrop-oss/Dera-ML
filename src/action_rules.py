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

  id              str         unique rule identifier
  tiers           list[str]   cluster tiers that match; omit = all tiers
  time_context    list[str]   ["peak"] | ["off_peak"]; omit = both
  rfm_min         int|str     minimum rfm_decile_segment (1–10); str = CONFIG key to resolve
  is_spike_min    float       minimum is_campaign_spike (0–1)
  uplift_min      float|str   minimum uplift_score; str = CONFIG key to resolve
  rate_key        str         key into CONFIG["promo_cost_rates"]
  action          str         action name; {bundle} is replaced with the customer's bundle
  dotdash_offer   dict        DotDash campaign output spec (see below)

DOTDASH OFFER SCHEMA
====================
Each rule carries a ``dotdash_offer`` dict that tells DotDash exactly how to
create the draft campaign.  DotDash supports two output types for AI plans
(แต้มตัวคูณใช้ไม่ได้กับแผน AI ที่ส่งครั้งเดียว):

  offer_type      "points" | "coupon"
  points_amount   int     — points to award (offer_type = "points" only)
  coupon_type     "fixed" | "percent"  (offer_type = "coupon" only)
  coupon_value    float   — THB or % off  (offer_type = "coupon" only)
  min_spend       float   — minimum spend to trigger offer (0 = no minimum)
  expiry_days     int     — days until offer expires after receipt

Helper
------
  get_dotdash_offer(rule_id_or_rule, aov=None) → dict
      Resolves the offer for a rule, optionally scaling coupon_value
      proportionally when aov is provided (future use).

EVALUATION STRATEGY
===================
  SURE_THING_RULES  — first_viable: rules evaluated in list order, first profitable wins.
  PERSUADABLE_RULES — best_profit:  all matching viable rules collected, max profit wins.
"""

import logging

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
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 15.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":      "st_newcust_welcome",
        "tiers":   ["new_customer"],
        "rate_key": "new_customer_welcome",
        "action":  "Welcome: ยินดีต้อนรับ กลับมาอีกนะ!",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 10.0, "min_spend": 0, "expiry_days": 30},
    },
    # General rules — apply to all tiers
    {
        "id":           "st_vip_reward_offpeak",
        "time_context": ["off_peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "vip_points",
        "action":       "VIP Reward: แจกแต้ม X2",
        "dotdash_offer": {"offer_type": "points", "points_amount": 100,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "st_vip_early_access_peak",
        "time_context": ["peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "peak_upsell",
        "action":       "VIP Early Access",
        "dotdash_offer": {"offer_type": "points", "points_amount": 50,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":       "st_loyalty_nurture",
        "rate_key": "vip_points",
        "action":   "Loyalty Nurture: แต้มสะสม",
        "dotdash_offer": {"offer_type": "points", "points_amount": 50,
                          "min_spend": 0, "expiry_days": 30},
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
        "dotdash_offer": {"offer_type": "points", "points_amount": 80,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "loyal_vip_reward",
        "tiers":        ["loyal"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "vip_points",
        "action":       "VIP Reward: แจกแต้ม X2",
        "dotdash_offer": {"offer_type": "points", "points_amount": 100,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "loyal_standard",
        "tiers":        ["loyal"],
        "rate_key":     "standard",
        "action":       "Loyal Offer: {bundle}",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 5.0, "min_spend": 0, "expiry_days": 30},
    },

    # ── Potential Loyalist: high R, medium F — build the habit ───────────────
    {
        "id":    "potloyal_habit_builder",
        "tiers": ["potential_loyalist"],
        "rate_key": "vip_points",
        "action": "Habit Builder: สะสมแต้ม เพิ่มความถี่",
        "dotdash_offer": {"offer_type": "points", "points_amount": 60,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "potloyal_offpeak_vip",
        "tiers":        ["potential_loyalist"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak_vip",
        "action":       "Loyalty Builder Off-Peak: แต้มพิเศษช่วงเงียบ",
        "dotdash_offer": {"offer_type": "points", "points_amount": 60,
                          "min_spend": 0, "expiry_days": 30},
    },

    # ── Promising: high R, low F, medium M — encourage return visit ──────────
    {
        "id":    "promising_reengage",
        "tiers": ["promising"],
        "rate_key": "standard",
        "action": "Re-engage: {bundle} พิเศษสำหรับคุณ",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 5.0, "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "promising_offpeak",
        "tiers":        ["promising"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak",
        "action":       "Re-engage Off-Peak: ส่วนลดพิเศษกลับมาหาเรา",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 15.0, "min_spend": 0, "expiry_days": 14},
    },

    # ── Need Attention: medium all, starting to drift ─────────────────────────
    {
        "id":    "needattn_bundle",
        "tiers": ["need_attention"],
        "rate_key": "standard",
        "action": "เราเป็นห่วง: {bundle} + ส่วนลดพิเศษ",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 5.0, "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "needattn_offpeak_traffic",
        "tiers":        ["need_attention"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak_traffic",
        "action":       "Re-engage: ส่วนลดพิเศษช่วงเงียบสำหรับคุณ",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 22.0, "min_spend": 0, "expiry_days": 14},
    },

    # ── About to Sleep: recency growing, low-medium F — urgent before dark ───
    {
        "id":    "aboutsleep_urgent",
        "tiers": ["about_to_sleep"],
        "rate_key": "off_peak",
        "action": "เราคิดถึงคุณ: {bundle} ก่อนที่จะไม่ได้เจอกัน",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 15.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":           "aboutsleep_traffic_driver",
        "tiers":        ["about_to_sleep"],
        "rate_key":     "off_peak_traffic",
        "action":       "Last Chance: ส่วนลดพิเศษ กลับมาหาเราเลย",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 22.0, "min_spend": 0, "expiry_days": 7},
    },

    # ── New Customer: onboarding — light welcome offer to earn 2nd visit ────────
    {
        "id":           "newcust_welcome_peak",
        "tiers":        ["new_customer"],
        "time_context": ["peak"],
        "rate_key":     "new_customer_welcome",
        "action":       "Welcome: สั่งครั้งที่ 2 รับส่วนลด 10%",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 10.0, "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "newcust_welcome_offpeak",
        "tiers":        ["new_customer"],
        "time_context": ["off_peak"],
        "rate_key":     "new_customer_welcome",
        "action":       "Welcome Off-Peak: กลับมาช้อปรับส่วนลดพิเศษ",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 10.0, "min_spend": 0, "expiry_days": 30},
    },

    # ── Churn Risk: early warning — act before the customer fully drifts ──────
    {
        "id":    "churnrisk_rescue_bundle",
        "tiers": ["churn_risk"],
        "rate_key": "churn_rescue",
        "action": "Early Win-Back: {bundle} ก่อนที่จะไม่กลับมา",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 12.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":           "churnrisk_offpeak_traffic",
        "tiers":        ["churn_risk"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak_traffic",
        "action":       "Churn Rescue: ส่วนลดพิเศษช่วงเงียบ เราคิดถึงคุณ",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 22.0, "min_spend": 0, "expiry_days": 14},
    },

    # ── Can't Lose Them: VIP drifting — aggressive win-back ───────────────────
    {
        "id":    "cantlose_vip_winback",
        "tiers": ["cant_lose"],
        "rate_key": "vip_winback",
        "action": "VIP Win-Back: {bundle} + แต้มพิเศษ X3",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 15.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":           "cantlose_aggressive_offpeak",
        "tiers":        ["cant_lose"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak_traffic",
        "action":       "VIP Win-Back: กลับมาช้อปรับส่วนลดพิเศษ VIP",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 22.0, "min_spend": 0, "expiry_days": 7},
    },

    # ── Champions: VIP / loyalty only — never a % discount ───────────────────
    {
        "id":    "champ_vip_bundle",
        "tiers": ["champions"],
        "rate_key": "vip_points",
        "action": "VIP Bundle: {bundle}",
        "dotdash_offer": {"offer_type": "points", "points_amount": 150,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "champ_early_access_peak",
        "tiers":        ["champions"],
        "time_context": ["peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "peak_upsell",
        "action":       "VIP Early Access",
        "dotdash_offer": {"offer_type": "points", "points_amount": 80,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "champ_early_access_offpeak",
        "tiers":        ["champions"],
        "time_context": ["off_peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "vip_points",
        "action":       "VIP Early Access",
        "dotdash_offer": {"offer_type": "points", "points_amount": 100,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":      "champ_vip_reward",
        "tiers":   ["champions"],
        "rfm_min": "vip_rfm_threshold",
        "rate_key": "vip_points",
        "action":  "VIP Reward: แจกแต้ม X2",
        "dotdash_offer": {"offer_type": "points", "points_amount": 100,
                          "min_spend": 0, "expiry_days": 30},
    },

    # ── At-Risk: discount bundle first, then standard retention ──────────────
    {
        "id":    "atrisk_discount_bundle",
        "tiers": ["at_risk"],
        "rate_key": "standard",
        "action": "Discount 15%: ซื้อ {bundle}",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 15.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":           "atrisk_standard_offpeak",
        "tiers":        ["at_risk"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak",
        "action":       "Standard Retention",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 15.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":           "atrisk_standard_peak",
        "tiers":        ["at_risk"],
        "time_context": ["peak"],
        "rate_key":     "standard",
        "action":       "Standard Retention",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 5.0, "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "atrisk_offpeak_traffic",
        "tiers":        ["at_risk"],
        "time_context": ["off_peak"],
        "rfm_min":      "mid_rfm_threshold",
        "rate_key":     "off_peak_traffic",
        "action":       "Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 22.0, "min_spend": 0, "expiry_days": 14},
    },

    # ── Hibernating: reactivation incentive ──────────────────────────────────
    {
        "id":    "hiber_reactivation_deep",
        "tiers": ["hibernating"],
        "rate_key": "off_peak",
        "action": "กลับมาช้อป: {bundle} ลด 20%",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 20.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":    "hiber_reactivation_traffic",
        "tiers": ["hibernating"],
        "rate_key": "off_peak_traffic",
        "action": "Reactivation: {bundle}",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 22.0, "min_spend": 0, "expiry_days": 14},
    },

    # ── Mid: contextual upsell ladder ─────────────────────────────────────────
    {
        "id":           "mid_festival_guard",
        "tiers":        ["mid"],
        "rfm_min":      "spike_rfm_threshold",
        "is_spike_min": 0.5,
        "rate_key":     "festival",
        "action":       "Standard Retention (Festival Guard)",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 5.0, "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "mid_offpeak_traffic",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "rfm_min":      "mid_rfm_threshold",
        "rate_key":     "off_peak_traffic",
        "action":       "Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 22.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":           "mid_offpeak_vip_points",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "rfm_min":      "mid_rfm_threshold",
        "rate_key":     "off_peak_vip",
        "action":       "Off-Peak VIP Points: ดึงลูกค้าช่วงเงียบ (แต้มพิเศษ)",
        "dotdash_offer": {"offer_type": "points", "points_amount": 60,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "mid_vip_reward_peak",
        "tiers":        ["mid"],
        "time_context": ["peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "peak_upsell",
        "action":       "VIP Reward: แจกแต้ม X2",
        "dotdash_offer": {"offer_type": "points", "points_amount": 80,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "mid_vip_reward_offpeak",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "rfm_min":      "vip_rfm_threshold",
        "rate_key":     "vip_points",
        "action":       "VIP Reward: แจกแต้ม X2",
        "dotdash_offer": {"offer_type": "points", "points_amount": 100,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "mid_vip_early_access_peak",
        "tiers":        ["mid"],
        "time_context": ["peak"],
        "rfm_min":      9,   # rfm >= 9 (stricter than vip_rfm=8)
        "rate_key":     "peak_upsell",
        "action":       "VIP Early Access",
        "dotdash_offer": {"offer_type": "points", "points_amount": 80,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "mid_vip_early_access_offpeak",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "rfm_min":      9,
        "rate_key":     "vip_points",
        "action":       "VIP Early Access",
        "dotdash_offer": {"offer_type": "points", "points_amount": 100,
                          "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "mid_bundle_upsell_peak",
        "tiers":        ["mid"],
        "time_context": ["peak"],
        "uplift_min":   "high_uplift_threshold",
        "rate_key":     "peak_upsell",
        "action":       "Bundle Upsell: {bundle}",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 8.0, "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "mid_bundle_upsell_offpeak",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "uplift_min":   "high_uplift_threshold",
        "rate_key":     "bundle",
        "action":       "Bundle Upsell: {bundle}",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 15.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":           "mid_standard_peak",
        "tiers":        ["mid"],
        "time_context": ["peak"],
        "rate_key":     "standard",
        "action":       "Standard Retention",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 5.0, "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":           "mid_standard_offpeak",
        "tiers":        ["mid"],
        "time_context": ["off_peak"],
        "rate_key":     "off_peak",
        "action":       "Standard Retention",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 15.0, "min_spend": 0, "expiry_days": 14},
    },
]


# ─── Low-basket rules ────────────────────────────────────────────────────────
# Actions the engine picks directly for customers whose average basket is below
# CONFIG["low_basket_threshold"] (uplift_engine.calculate_action_and_profit).  They are not
# matched through matches()/menus — they exist here so every action has a stable id and an offer.
# The offer's size follows the cost rate the engine charges for the action (rate_key).

LOW_BASKET_RULES = [
    {
        "id":       "low_milestone",
        "tiers":    ["champions"],
        "rate_key": "milestone_reward",
        "action":   "Milestone Reward: สั่งครบ {basket:.0f} บาท รับ {pts} แต้ม",
        # points and the basket target come from CONFIG at call time (never a copy)
        "dotdash_offer_fn": lambda: {"offer_type": "points",
                                     "points_amount": _CONFIG["milestone_points"],
                                     "min_spend": _CONFIG["milestone_basket"], "expiry_days": 30},
    },
    {
        "id":       "low_welcome",
        "tiers":    ["new_customer"],
        "rate_key": "new_customer_welcome",
        "action":   "Welcome: สั่งครั้งที่ 2 รับส่วนลด 10%",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 10.0, "min_spend": 0, "expiry_days": 30},
    },
    {
        "id":       "low_abouttosleep_miss_you",
        "tiers":    ["about_to_sleep"],
        "rate_key": "standard",
        "action":   "เราคิดถึงคุณ: กลับมาหาเราก่อนที่จะไม่ได้เจอกัน",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 5.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":       "low_hibernating_nudge",
        "tiers":    ["hibernating"],
        "rate_key": "bundle_upsell_low",
        "action":   "Reactivation Nudge: กลับมาช้อปรับส่วนลดพิเศษ",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 3.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":       "low_churn_winback",
        "tiers":    ["churn_risk", "cant_lose"],
        "rate_key": "churn_rescue",
        "action":   "Early Win-Back: กลับมาหาเรา รับส่วนลดพิเศษ",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 12.0, "min_spend": 0, "expiry_days": 14},
    },
    {
        "id":       "low_bundle_upsell",
        "rate_key": "bundle_upsell_low",
        "action":   "Bundle Upsell: เพิ่ม {bundle}",
        "dotdash_offer": {"offer_type": "coupon", "coupon_type": "percent",
                          "coupon_value": 3.0, "min_spend": 0, "expiry_days": 30},
    },
]

# Stable action_key -> rule.  The key is the rule id: names may change, keys never do.
ACTION_KEYS = {r["id"]: r for r in SURE_THING_RULES + PERSUADABLE_RULES + LOW_BASKET_RULES}


# ---------------------------------------------------------------------------
# DotDash offer helper
# ---------------------------------------------------------------------------

def get_dotdash_offer(rule: dict) -> dict:
    """Return the DotDash offer spec for a rule, with safe defaults.

    Parameters
    ----------
    rule : dict
        A rule dict from SURE_THING_RULES or PERSUADABLE_RULES.

    Returns
    -------
    dict
        ``{offer_type, points_amount?, coupon_type?, coupon_value?,
           min_spend, expiry_days}``
    """
    fn = rule.get("dotdash_offer_fn")
    offer = dict(fn() if fn else (rule.get("dotdash_offer") or {}))
    if not offer:
        # Fallback: infer from rate_key
        rk = rule.get("rate_key", "")
        if "points" in rk or "vip" in rk:
            offer = {"offer_type": "points", "points_amount": 50,
                     "min_spend": 0, "expiry_days": 30}
        else:
            offer = {"offer_type": "coupon", "coupon_type": "percent",
                     "coupon_value": 10.0, "min_spend": 0, "expiry_days": 30}
    return offer


def get_dotdash_offer_by_id(rule_id: str) -> dict:
    """Return the DotDash offer of the rule with this id (any rule list).

    Raises ValueError for an unknown id — a guessed offer would silently reward the wrong thing.
    """
    rule = ACTION_KEYS.get(rule_id)
    if rule is None:
        raise ValueError(f"unknown action_key: {rule_id!r}")
    return get_dotdash_offer(rule)


_NO_PROMOTION_PREFIXES = ("Do Nothing", "Zero-Cost")


def infer_dotdash_offer(action_key: str | None, action: str = "") -> dict:
    """The DotDash offer for one decision — the public, stable entry point (used by ml-service).

    ``action_key`` (the id of the rule that produced the action) is authoritative: an unknown key
    raises ValueError.  Without a key (rows produced before action_key existed) the action *text*
    is matched against the rule templates — a legacy path that logs a warning and may guess wrong.
    Returns ``{offer_type, points_amount?, coupon_type?, coupon_value?, min_spend, expiry_days,
    action_key}``.
    """
    if action_key is not None:
        return {**get_dotdash_offer_by_id(action_key), "action_key": action_key}

    if not action or action.startswith(_NO_PROMOTION_PREFIXES):
        return {"offer_type": "points", "points_amount": 0, "min_spend": 0,
                "expiry_days": 0, "action_key": None}

    logging.getLogger(__name__).warning(
        "infer_dotdash_offer called without action_key; matching action text %r (legacy, may be wrong)", action)
    return {**_offer_from_action_text(action), "action_key": None}


def _offer_from_action_text(action: str) -> dict:
    """Legacy: rule-template prefix match, then keyword fallback."""
    for rule in SURE_THING_RULES + PERSUADABLE_RULES:
        prefix = rule["action"].split("{bundle}")[0].strip().rstrip(":").strip()
        if prefix and prefix in action:
            return get_dotdash_offer(rule)
    a = action.lower()
    if any(w in a for w in ["แต้ม", "vip reward", "loyalty", "points", "milestone", "early access", "habit"]):
        return {"offer_type": "points", "points_amount": 50, "min_spend": 0, "expiry_days": 30}
    return {"offer_type": "coupon", "coupon_type": "percent",
            "coupon_value": 15.0, "min_spend": 0, "expiry_days": 14}
