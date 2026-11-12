"""Customers that Layer 1 drops from the RFM table (e.g. every line has total_price <= 0)
must still get a decision instead of crashing with "'float' object has no attribute 'lower'"."""
import pandas as pd
import pytest

import uplift_engine as l4
from fixtures_pos import make_pos_csv


@pytest.fixture(scope="module")
def decided(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rfm_missing")
    csv = make_pos_csv(tmp, n_customers=300, seed=11)
    df = pd.read_csv(csv)
    extra = df[df["customer_id"] == "c000"].copy()
    extra["customer_id"] = "zero_price_cust"
    extra["total_price"] = 0.0          # Layer 1 keeps only total_price > 0 -> absent from RFM
    pd.concat([df, extra]).to_csv(csv, index=False)
    ctx = l4.build_serving_context(csv, None)
    assert "zero_price_cust" not in set(ctx.rfm_df["customer_id"]), "premise: customer must be missing from RFM"
    model = l4._train_xlearner(ctx.raw_df[ctx.raw_df["was_treated"] == 1],
                               ctx.raw_df[ctx.raw_df["was_treated"] == 0], ctx.features)
    return l4.decide_for_customers(model, ctx)[1]


def test_customer_missing_from_rfm_does_not_crash_and_has_string_segment(decided):
    row = decided[decided["customer_id"] == "zero_price_cust"]
    assert len(row) == 1
    assert isinstance(row.iloc[0]["cluster_segment"], str)
