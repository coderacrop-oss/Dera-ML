"""Layer 2: the top bundles must be different pairs, and the pair counts must count bills."""
import pandas as pd

from campaign_pipeline import layer2_csv_mba
from uplift_engine import _run_layer2 as run_layer2


def _bundle_csv(path):
    """300 bills: three product pairs bought together (30, 25 and 20 bills) plus single-item bills."""
    rows, n = [], 0

    def bill(*items):
        nonlocal n
        n += 1
        rows.extend(dict(transaction_id=f"T{n}", customer_id=f"C{n % 9}", product_name=p, quantity=1, total_price=5.0) for p in items)

    for pair, times in ((("tea", "scone"), 30), (("cola", "chips"), 25), (("latte", "tart"), 20)):
        for _ in range(times):
            bill(*pair)
    for i in range(225):
        bill(("x1", "x2", "x3", "x4", "x5", "x6")[i % 6])
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_top_bundles_are_three_different_pairs_not_a_pair_and_its_reverse(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    top = run_layer2(_bundle_csv(tmp_path / "pos.csv"))
    assert len(top) == 3
    pairs = {frozenset((a, b)) for a, b in zip(top["antecedents"], top["consequents"])}
    assert len(pairs) == 3, f"expected 3 different pairs, got {pairs}"
    assert pairs == {frozenset(("tea", "scone")), frozenset(("cola", "chips")), frozenset(("latte", "tart"))}


def _engine_inputs():
    """c1 bought Bun+Coke in 4 bills, c2 in 1 bill, c3 bought Apple+Pie once: Bun+Coke is together in 5 bills, Apple+Pie in 1."""
    lines = []
    for bill, cust, items in (("b1", "c1", ("Bun", "Coke")), ("b2", "c1", ("Bun", "Coke")), ("b3", "c1", ("Bun", "Coke")), ("b4", "c1", ("Bun", "Coke")),
                              ("b5", "c2", ("Bun", "Coke")), ("b6", "c3", ("Apple", "Pie"))):
        for item in items:
            lines.append(dict(customer_id=cust, invoice_id=bill, product_name=item, amount=10.0, transaction_date=pd.Timestamp("2026-01-01")))
    df = pd.DataFrame(lines)
    df_rfm = pd.DataFrame({"customer_id": ["c1", "c2", "c3"], "cluster": [0, 0, 0]})
    segments = [{"cluster_id": 0, "segment_name": "2. Potential Loyalists", "customer_count": 3}]
    return df, segments, df_rfm


def test_segment_pair_is_the_one_bought_together_most_often_not_the_first_in_the_alphabet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df, segments, df_rfm = _engine_inputs()
    seg, _ = layer2_csv_mba(df, segments, df_rfm)
    assert "[Bun]" in seg[0]["recommended_promo"] and "[Coke]" in seg[0]["recommended_promo"], seg[0]["recommended_promo"]
    assert "5 txns" in seg[0]["rationale"], seg[0]["rationale"]


def test_personal_pair_count_is_the_number_of_bills(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df, segments, df_rfm = _engine_inputs()
    layer2_csv_mba(df, segments, df_rfm)
    prefs = pd.read_csv("data/processed/layer2_customer_preferences.csv").set_index("customer_id")
    assert prefs.loc["c1", "personal_pair_times"] == 4
    assert {prefs.loc["c1", "personal_pair_a"], prefs.loc["c1", "personal_pair_b"]} == {"Bun", "Coke"}
