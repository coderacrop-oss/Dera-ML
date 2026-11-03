"""Layer 4 turns each customer's post_promo_visit into a 0/1 'converted' label. It must not destroy a label that is already 0/1."""
import pandas as pd

from uplift_engine import _load_real_labels


def _pos(tmp_path, per_customer):
    """per_customer: list of lists of post_promo_visit values, one list per customer (one row per value)."""
    rows = []
    for i, values in enumerate(per_customer):
        for j, v in enumerate(values):
            rows.append(dict(customer_id=f"c{i:03d}", timestamp=f"2026-01-{1 + j % 28:02d} 10:00:00", total_price=5.0,
                             is_treated=i % 2, post_promo_visit=v, promo_type="none"))
    path = tmp_path / "pos.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_binary_labels_survive_when_most_customers_came_back(tmp_path):
    # 70 of 100 customers came back: the median rate is 1, and 'rate > median' used to turn EVERY label into 0
    out = _load_real_labels(_pos(tmp_path, [[1, 1]] * 70 + [[0, 0]] * 30))
    assert int(out["converted"].sum()) == 70


def test_binary_labels_survive_when_few_customers_came_back(tmp_path):
    out = _load_real_labels(_pos(tmp_path, [[1, 1]] * 20 + [[0, 0]] * 80))
    assert int(out["converted"].sum()) == 20


def test_binary_labels_survive_at_exactly_half(tmp_path):
    out = _load_real_labels(_pos(tmp_path, [[1]] * 50 + [[0]] * 50))
    assert int(out["converted"].sum()) == 50


def test_fractions_between_0_and_1_keep_the_median_rule(tmp_path):
    # customers with a MIX of visits (rates 0, .25, .5, .75, 1): no 0/1 label to keep, so 'above the median rate' still applies
    per_customer = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
    out = _load_real_labels(_pos(tmp_path, per_customer)).sort_values("customer_id")
    assert out["converted"].tolist() == [0, 0, 0, 1, 1]


def test_one_customer_without_any_visit_value_does_not_bring_back_the_median_rule(tmp_path):
    # a customer whose post_promo_visit cells are all blank has a NaN rate; the other 99 are 0/1 and must stay as they are
    path = _pos(tmp_path, [[1, 1]] * 70 + [[0, 0]] * 29 + [[None, None]])
    out = _load_real_labels(path)
    assert int(out["converted"].sum()) == 70
