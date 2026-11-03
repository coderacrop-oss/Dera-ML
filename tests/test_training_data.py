import pandas as pd
import pytest

from fixtures_pos import make_pos_csv
from training_data import (PooledData, StoreInput, TrainingDataError, build_pooled, parse_store_data,
                           split_customers, training_frames)


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # Layers 1-3 write under ./data/processed


def test_parse_store_data_with_and_without_labels():
    assert parse_store_data("shop-a:/x/a.csv") == StoreInput("shop-a", "/x/a.csv", None)
    assert parse_store_data("shop-a:/x/a.csv:/x/l.csv") == StoreInput("shop-a", "/x/a.csv", "/x/l.csv")


@pytest.mark.parametrize("bad", ["..:a.csv", "A/b:a.csv", "a" * 64 + ":a.csv", "-a:a.csv", "shop-a", "shop-a:", ":a.csv"])
def test_parse_rejects_bad_store_id_or_shape(bad):
    with pytest.raises(TrainingDataError):
        parse_store_data(bad)


def test_split_is_deterministic_and_disjoint_and_sized():
    ids = [f"c{i}" for i in range(100)]
    tr, ho = split_customers(ids, 0.2, 42)
    tr2, ho2 = split_customers(list(reversed(ids)), 0.2, 42)
    assert (tr, ho) == (tr2, ho2)
    assert tr | ho == set(ids) and not (tr & ho) and len(ho) == 20
    assert split_customers(ids, 0.2, 7)[1] != ho


def _two_shops(tmp_path):
    a = make_pos_csv(tmp_path, n_customers=60, seed=1, name="a.csv")
    b = make_pos_csv(tmp_path, n_customers=60, seed=2, name="b.csv")  # same ids c000.. as shop a
    return [StoreInput("shop-a", a, None), StoreInput("shop-b", b, None)]


def test_two_stores_with_same_customer_ids_are_not_merged(tmp_path):
    pooled = build_pooled(_two_shops(tmp_path), holdout_frac=0.2, seed=42)
    assert isinstance(pooled, PooledData) and pooled.label_mode == "proxy"
    treat, ctrl = training_frames(pooled)
    keys = pd.concat([treat, ctrl])["customer_key"]
    assert keys.str.startswith(("shop-a|", "shop-b|")).all()
    per_store = keys.str.split("|").str[0].value_counts()
    assert set(per_store.index) == {"shop-a", "shop-b"}
    assert pooled.train_ids["shop-a"] | pooled.holdout_ids["shop-a"] == set(pooled.contexts["shop-a"].cust_agg["customer_id"])


def test_no_holdout_customer_in_training_frames(tmp_path):
    pooled = build_pooled(_two_shops(tmp_path), holdout_frac=0.25, seed=3)
    treat, ctrl = training_frames(pooled)
    used = set(pd.concat([treat, ctrl])["customer_key"])
    held = {f"{s}|{c}" for s, ids in pooled.holdout_ids.items() for c in ids}
    assert held and not (used & held)


def test_mixed_label_modes_across_stores_raise_naming_stores(tmp_path):
    lab = make_pos_csv(tmp_path, seed=1, with_labels=True, name="lab.csv")
    plain = make_pos_csv(tmp_path, seed=2, name="plain.csv")
    with pytest.raises(TrainingDataError) as e:
        build_pooled([StoreInput("shop-a", lab, lab), StoreInput("shop-b", plain, None)])
    assert "shop-a" in str(e.value) and "shop-b" in str(e.value)


def test_store_without_treated_rows_in_training_raises_naming_store(tmp_path):
    p = make_pos_csv(tmp_path, seed=1, with_labels=True, name="none_treated.csv")
    df = pd.read_csv(p)
    df["is_treated"] = 0
    df.to_csv(p, index=False)
    good = make_pos_csv(tmp_path, seed=2, with_labels=True, name="good.csv")
    pooled = build_pooled([StoreInput("shop-a", p, p), StoreInput("shop-b", good, good)])
    with pytest.raises(TrainingDataError, match="shop-a"):
        training_frames(pooled)


def test_empty_inputs_raise():
    with pytest.raises(TrainingDataError):
        build_pooled([])


def test_duplicate_store_ids_raise(tmp_path):
    a = make_pos_csv(tmp_path, n_customers=30, name="a.csv")
    with pytest.raises(TrainingDataError, match="shop-a"):
        build_pooled([StoreInput("shop-a", a, None), StoreInput("shop-a", a, None)])
