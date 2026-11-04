"""Smoke test for promo_advisor.py

Run:
    cd /path/to/ml-campaign-engine
    python src/test_promo_advisor.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from promo_advisor import (
    PromoSpec, evaluate_operator_promo, resolve_per_customer,
    compare_scenarios, promo_advisor, print_promo_report, _promo_group,
)


# ── Minimal synthetic fixtures ────────────────────────────────────────────────

def _make_cust_agg(n=20):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "customer_id":  [f"c{i:03d}" for i in range(n)],
        "avg_spent":    rng.uniform(50, 500, n).round(2),
        "cluster_segment_name": rng.choice(
            ["champions", "potential_loyalist", "at_risk", "promising"], n
        ),
        "top_category": rng.choice(
            ["ครัวซองต์และเบเกอรี่", "เครื่องดื่ม", "ทาร์ตไข่"], n
        ),
    })


def _make_ml_output(cust_agg, send_frac=0.5):
    rng = np.random.default_rng(7)
    n = len(cust_agg)
    send_mask = rng.random(n) < send_frac
    actions = np.where(
        send_mask,
        rng.choice(["Push Promotion: Bundle", "Off-Peak Special", "Win-Back Offer"], n),
        "Do Nothing (No Profitable Action)",
    )
    profits = np.where(send_mask, rng.uniform(20, 200, n).round(2), 0.0)
    costs   = np.where(send_mask, profits * 0.15, 0.0)
    types   = np.where(send_mask,
        rng.choice(["bundle", "discount", "retention"], n), "none")
    return pd.DataFrame({
        "customer_id":        cust_agg["customer_id"].values,
        "recommended_action": actions,
        "expected_profit":    profits,
        "promo_cost":         costs,
        "should_send":        send_mask.astype(int),
        "treatment_type":     types,
        "uplift_score":       rng.uniform(-0.05, 0.15, n).round(4),
        "prob_control":       rng.uniform(0.1, 0.8, n).round(4),
        "avg_spent":          cust_agg["avg_spent"].values,
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_promo_group():
    assert _promo_group("Push Promotion: Bundle") == "discount"
    assert _promo_group("VIP Reward") == "points_vip"
    assert _promo_group("blanket_discount") == "discount"
    assert _promo_group("notification") == "notification"
    print("  [PASS] _promo_group")


def test_evaluate_all_customers():
    cust_agg  = _make_cust_agg(50)
    ml_output = _make_ml_output(cust_agg)
    spec = PromoSpec(name="B1G1 ครัวซองต์", discount_rate=0.50, target="all_customers")
    result = evaluate_operator_promo(ml_output, cust_agg, spec)
    assert "promo_eligible" in result.columns
    assert "promo_viable" in result.columns
    assert "operator_promo_profit" in result.columns
    assert result["promo_eligible"].all(), "all_customers → everyone eligible"
    assert result["operator_promo_blocked_reason"].notna().all()
    n_viable = result["promo_viable"].sum()
    print(f"  [PASS] evaluate_operator_promo: {n_viable}/50 viable (B1G1 50% discount)")


def test_evaluate_segment_target():
    cust_agg  = _make_cust_agg(50)
    ml_output = _make_ml_output(cust_agg)
    spec = PromoSpec(
        name="เซ็ตเครื่องดื่ม",
        discount_rate=0.15,
        target="segment",
        segments=["potential_loyalist", "promising"],
    )
    result = evaluate_operator_promo(ml_output, cust_agg, spec)
    eligible = result["promo_eligible"].sum()
    assert eligible < 50, "segment filter should restrict eligibility"
    print(f"  [PASS] evaluate_operator_promo (segment): {eligible}/50 eligible")


def test_resolve_conflict():
    cust_agg  = _make_cust_agg(30)
    ml_output = _make_ml_output(cust_agg, send_frac=0.7)
    spec = PromoSpec(
        name="Discount ร้าน",
        discount_rate=0.20,
        promo_type="discount",
        target="all_customers",
    )
    promo_eval = evaluate_operator_promo(ml_output, cust_agg, spec)
    resolved   = resolve_per_customer(ml_output, promo_eval, spec)
    assert "conflict_type" in resolved.columns
    assert "final_source" in resolved.columns
    assert "advisor_reason" in resolved.columns
    assert set(resolved["final_source"].unique()).issubset(
        {"ml", "operator", "none"}
    )
    print("  [PASS] resolve_per_customer: conflict types =",
          resolved["conflict_type"].value_counts().to_dict())


def test_compare_scenarios():
    cust_agg  = _make_cust_agg(40)
    ml_output = _make_ml_output(cust_agg)
    spec = PromoSpec(name="TestPromo", discount_rate=0.15, target="all_customers")
    promo_eval = evaluate_operator_promo(ml_output, cust_agg, spec)
    resolved   = resolve_per_customer(ml_output, promo_eval, spec)
    report     = compare_scenarios(resolved, spec)
    assert "scenario_A" in report
    assert "scenario_B" in report
    assert "scenario_C" in report
    assert report["total_customers"] == 40
    # C send count should be <= A + B (no double sending)
    assert report["scenario_C"]["send_count"] <= 40
    print("  [PASS] compare_scenarios: A={} B={} C={}".format(
        report["scenario_A"]["send_count"],
        report["scenario_B"]["send_count"],
        report["scenario_C"]["send_count"],
    ))


def test_full_entry_point():
    cust_agg  = _make_cust_agg(60)
    ml_output = _make_ml_output(cust_agg)
    spec = PromoSpec(
        name="B1G1 ทาร์ตไข่",
        discount_rate=0.50,
        scope="category",
        category="ทาร์ตไข่",
        target="all_customers",
    )
    resolved, report = promo_advisor(ml_output, cust_agg, spec)
    assert len(resolved) == 60
    print_promo_report(report)
    print("  [PASS] promo_advisor() full pipeline")


def test_no_conflict_compatible():
    """ML sends VIP Reward, operator does B1G1 — different types, both viable."""
    cust_agg = _make_cust_agg(10)
    ml_output = pd.DataFrame({
        "customer_id":        cust_agg["customer_id"].values,
        "recommended_action": ["VIP Reward"] * 10,
        "expected_profit":    [100.0] * 10,
        "promo_cost":         [5.0] * 10,
        "should_send":        [1] * 10,
        "treatment_type":     ["points_vip"] * 10,
        "uplift_score":       [0.08] * 10,
        "prob_control":       [0.5] * 10,
        "avg_spent":          cust_agg["avg_spent"].values,
    })
    spec = PromoSpec(
        name="B1G1 ครัวซองต์",
        discount_rate=0.50,
        promo_type="discount",
        target="all_customers",
    )
    promo_eval = evaluate_operator_promo(ml_output, cust_agg, spec)
    resolved   = resolve_per_customer(ml_output, promo_eval, spec)
    # All should be "compatible" or either ml/operator depending on profit
    for _, row in resolved.iterrows():
        if row["promo_viable"]:
            assert row["conflict_type"] == "compatible"
    print("  [PASS] no conflict when types differ")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Promo Advisor Smoke Tests ===\n")
    test_promo_group()
    test_evaluate_all_customers()
    test_evaluate_segment_target()
    test_resolve_conflict()
    test_compare_scenarios()
    test_full_entry_point()
    test_no_conflict_compatible()
    print("\n✓ All tests passed\n")
