"""Budget-constrained campaign targeting.

Given the full Layer 4 output and a campaign budget ceiling, selects the
subset of approved customers that maximises total expected incremental
profit subject to:

    sum(promo_cost_i  for i in selected) <= budget

Algorithm: greedy bang-per-buck with single-item fallback.
  - Sort approved customers by expected_profit / promo_cost (descending).
  - Greedily add customers until budget is exhausted.
  - Compare greedy total profit against the single highest-profit customer
    that fits within budget; return whichever is larger.
  - This max(greedy, best_single) comparison achieves a 1/2-approximation
    guarantee for the 0/1 knapsack (Sahni 1975).
  - Optimal for the fractional knapsack (pure greedy suffices there).
  - O(n log n) — fast enough for any realistic customer list.

Typical usage
-------------
    from budget_optimizer import select_within_budget, budget_summary

    # all_customers: full decide_for_customers() output (should_send=0 and 1)
    result = select_within_budget(all_customers, budget=3000)

    # Only send to budget-approved customers
    final_send = result[result["selected_for_budget"]]
    print(budget_summary(result, budget=3000))
"""

from __future__ import annotations

import pandas as pd


def select_within_budget(
    df: pd.DataFrame,
    budget: float,
    profit_col: str = "expected_profit",
    cost_col: str = "promo_cost",
) -> pd.DataFrame:
    """Annotate df with budget-selection columns.

    Parameters
    ----------
    df : DataFrame
        Full Layer 4 output from decide_for_customers().
        Must contain ``should_send``, ``expected_profit``, and ``promo_cost``.
    budget : float
        Total campaign spend ceiling (same currency as promo_cost).
        Pass 0 or negative to disable — all approved customers stay approved.
    profit_col, cost_col : str
        Column names for profit and cost (rarely need to change).

    Returns
    -------
    DataFrame — original df with four new columns appended:
        budget_rank        int|None   priority order (1 = best bang-per-buck)
        bang_per_buck      float|None expected_profit / promo_cost
        cumulative_cost    float|None running spend total at this rank
        selected_for_budget bool      True if customer fits within budget
    """
    result = df.copy()

    # No budget constraint — all approved customers are selected
    if budget <= 0:
        result["budget_rank"]         = None
        result["bang_per_buck"]       = None
        result["cumulative_cost"]     = None
        result["selected_for_budget"] = result["should_send"].astype(bool)
        return result

    approved_mask = result["should_send"] == 1
    approved = result[approved_mask].copy()
    not_approved = result[~approved_mask].copy()

    if approved.empty:
        result["budget_rank"]         = None
        result["bang_per_buck"]       = None
        result["cumulative_cost"]     = None
        result["selected_for_budget"] = False
        return result

    # bang-per-buck: clip cost to tiny positive to avoid div-by-zero for zero-cost notifications
    approved["bang_per_buck"] = (
        approved[profit_col] / approved[cost_col].clip(lower=0.01)
    ).round(4)

    approved = approved.sort_values("bang_per_buck", ascending=False).reset_index(drop=True)
    approved["budget_rank"]     = range(1, len(approved) + 1)
    approved["cumulative_cost"] = approved[cost_col].cumsum().round(2)
    approved["selected_for_budget"] = approved["cumulative_cost"] <= budget

    # 1/2-approximation: compare greedy set to the single highest-profit customer
    # that fits within budget (Sahni 1975).  Replace greedy if best-single wins.
    greedy_profit = float(approved.loc[approved["selected_for_budget"], profit_col].sum())
    fits_mask = approved[cost_col] <= budget
    if fits_mask.any():
        best_single_row = approved[fits_mask].loc[
            approved[fits_mask][profit_col].idxmax()
        ]
        best_single_profit = float(best_single_row[profit_col])
        if best_single_profit > greedy_profit:
            approved["selected_for_budget"] = False
            approved.loc[best_single_row.name, "selected_for_budget"] = True

    not_approved["bang_per_buck"]       = None
    not_approved["budget_rank"]         = None
    not_approved["cumulative_cost"]     = None
    not_approved["selected_for_budget"] = False

    return pd.concat([approved, not_approved], ignore_index=True)


def budget_summary(df: pd.DataFrame, budget: float) -> dict:
    """Return a plain-dict summary suitable for logging or executive reporting.

    Parameters
    ----------
    df      : DataFrame returned by select_within_budget()
    budget  : float — same value passed to select_within_budget()
    """
    if "selected_for_budget" not in df.columns:
        raise ValueError("Run select_within_budget() before calling budget_summary().")

    selected    = df[df["selected_for_budget"] == True]
    total_appvd = int((df["should_send"] == 1).sum())
    excluded    = total_appvd - len(selected)

    total_cost   = round(float(selected["promo_cost"].sum()),    2) if len(selected) else 0.0
    total_profit = round(float(selected["expected_profit"].sum()), 2) if len(selected) else 0.0
    utilisation  = round(total_cost / budget * 100, 1) if budget > 0 else 0.0

    return {
        "budget_ceiling":          budget,
        "customers_selected":      len(selected),
        "customers_excluded_by_budget": excluded,
        "total_promo_cost":        total_cost,
        "total_expected_profit":   total_profit,
        "budget_utilisation_pct":  utilisation,
        "avg_profit_per_baht":     round(total_profit / total_cost, 4) if total_cost > 0 else None,
    }


def print_budget_report(df: pd.DataFrame, budget: float) -> None:
    """Pretty-print budget selection summary to stdout."""
    s = budget_summary(df, budget)
    currency = "฿"
    print(f"\n💰 [Budget] ceiling={currency}{s['budget_ceiling']:,.0f}")
    print(f"   ✅ Selected : {s['customers_selected']:,} customers  "
          f"(cost={currency}{s['total_promo_cost']:,.2f}, "
          f"profit={currency}{s['total_expected_profit']:,.2f})")
    print(f"   ⛔ Excluded : {s['customers_excluded_by_budget']:,} customers (budget exhausted)")
    print(f"   📊 Utilisation: {s['budget_utilisation_pct']:.1f}%  |  "
          f"ROI: {s['avg_profit_per_baht']:.2f}x" if s['avg_profit_per_baht'] else "")
