"""Per-customer product pairs and the pair a DotDash group should be offered.

Pure functions over a POS frame — nothing is written to disk (unlike layer2_csv_mba's preference file).
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import pandas as pd


def personal_pairs(raw: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """customer_id -> the two products the customer most often bought on the same bill.

    Ties go to the alphabetically smallest pair so results are deterministic.  Customers who never
    bought two different products on one bill are absent.  A frame without product_name gives {}.
    """
    needed = {"customer_id", "transaction_id", "product_name"}
    if raw is None or raw.empty or not needed <= set(raw.columns):
        return {}
    bills = (raw.dropna(subset=["product_name"])
                .assign(customer_id=lambda d: d["customer_id"].astype(str))
                .groupby(["customer_id", "transaction_id"])["product_name"]
                .apply(lambda s: sorted(set(map(str, s)))))
    counts: dict[str, Counter] = {}
    for (cid, _tx), products in bills.items():
        if len(products) >= 2:
            counts.setdefault(cid, Counter()).update(combinations(products, 2))
    return {cid: min(c, key=lambda pair: (-c[pair], pair)) for cid, c in counts.items()}


def group_bundle(customer_ids: list[str], pairs: dict[str, tuple[str, str]]) -> tuple[list[str] | None, float]:
    """(most common personal pair among the group's customers, share of the group that has it).

    The share's denominator is the whole group — customers with no pair count against coverage.
    Ties go to the alphabetically smallest pair.
    """
    if not customer_ids:
        return None, 0.0
    tally = Counter(pairs[c] for c in map(str, customer_ids) if c in pairs)
    if not tally:
        return None, 0.0
    best = min(tally, key=lambda pair: (-tally[pair], pair))
    return list(best), round(tally[best] / len(customer_ids), 4)
