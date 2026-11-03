import os
import pickle

import pandas as pd
import pytest

import layer4_uplift_margin as l4
from candidates import (candidate_model_bytes, check_token, create_candidate, evaluate_candidate, load_candidate,
                        mark_candidate)
from fixtures_pos import make_pos_csv
from model_registry import ChampionRef, RegistryError
from training_data import StoreInput, build_pooled, training_frames


def _pooled(tmp_path, n=60, frac=0.25):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ins = [StoreInput("shop-a", make_pos_csv(tmp_path, n, seed=1, name="a.csv"), None),
               StoreInput("shop-b", make_pos_csv(tmp_path, n, seed=2, name="b.csv"), None)]
        return build_pooled(ins, holdout_frac=frac, seed=42)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cand")
    pooled = _pooled(tmp)
    treat, ctrl = training_frames(pooled)
    challenger = l4._train_xlearner(treat, ctrl, pooled.features)
    champion = l4._train_xlearner(treat.sample(frac=0.5, random_state=1), ctrl.sample(frac=0.5, random_state=1),
                                  pooled.features)
    ref = ChampionRef("v3", "x", "abc", "simulated")
    meta = {"files": ["a.csv", "b.csv"], "seed": 42, "holdout_frac": 0.25}
    return dict(pooled=pooled, challenger=challenger, champion=champion, ref=ref, meta=meta, tmp=tmp)


def _recompute(model, pooled):
    frames = [l4.decide_for_customers(model, pooled.contexts[s], pooled.holdout_ids[s])[1] for s in pooled.contexts]
    df = pd.concat(frames)
    return dict(profit=float(df["expected_profit"].sum()), send_count=int(df["should_send"].sum()),
                blocked_count=int(df["guardrail_blocked"].sum()),
                do_nothing_pct=float(df["recommended_action"].str.startswith("Do Nothing").mean() * 100))


def test_report_numbers_equal_recomputation_on_holdout(world):
    r = evaluate_candidate(world["pooled"], world["challenger"], world["champion"], world["ref"], world["meta"])
    for key, model in (("challenger", world["challenger"]), ("champion", world["champion"])):
        want = _recompute(model, world["pooled"])
        for k, v in want.items():
            assert r[key][k] == pytest.approx(v), (key, k)


def test_report_has_all_keys_and_label_mode(world):
    r = evaluate_candidate(world["pooled"], world["challenger"], world["champion"], world["ref"], world["meta"])
    for k in ("created_at", "compared_against_version", "label_mode", "provenance", "holdout_size", "holdout_warning",
              "stores", "champion", "challenger", "per_store", "gate", "training"):
        assert k in r
    assert r["compared_against_version"] == "v3" and r["label_mode"] == "proxy" and r["provenance"] == "proxy-labels"
    assert {s["store_id"] for s in r["stores"]} == {"shop-a", "shop-b"}
    assert set(r["per_store"]) == {"shop-a", "shop-b"}
    assert r["holdout_size"] == sum(s["holdout_customers"] for s in r["stores"])
    assert set(r["gate"]) == {"passed", "reason", "growth_pct"}
    assert r["training"] == world["meta"]


def test_first_champion_report_has_none_champion_and_gate_passes(world):
    r = evaluate_candidate(world["pooled"], world["challenger"], None, None, world["meta"])
    assert r["champion"] is None and r["compared_against_version"] is None
    assert r["gate"]["passed"] is True and r["gate"]["growth_pct"] is None
    assert all(v["champion"] is None for v in r["per_store"].values())


def test_holdout_warning_below_100_and_not_above(world, tmp_path):
    small = evaluate_candidate(world["pooled"], world["challenger"], None, None, {})
    assert small["holdout_size"] < 100 and small["holdout_warning"] is True
    big = _pooled(tmp_path, n=120, frac=0.5)
    r = evaluate_candidate(big, world["challenger"], None, None, {})
    assert r["holdout_size"] >= 100 and r["holdout_warning"] is False


@pytest.fixture
def made(world, tmp_path):
    r = evaluate_candidate(world["pooled"], world["challenger"], None, None, world["meta"])
    cid, token = create_candidate(str(tmp_path), world["challenger"], r)
    return str(tmp_path), cid, token


def test_create_candidate_stores_only_token_hash(made):
    root, cid, token = made
    for dp, _d, files in os.walk(os.path.join(root, "candidates", cid)):
        for f in files:
            assert token.encode() not in open(os.path.join(dp, f), "rb").read()
    c = load_candidate(root, cid)
    assert c["status"]["status"] == "pending" and c["report"]["candidate_id"] == cid
    assert len(c["status"]["token_sha256"]) == 64


def test_candidate_model_bytes_roundtrip(made, world):
    root, cid, _ = made
    assert pickle.loads(candidate_model_bytes(root, cid)).feature_names_ == world["pooled"].features


def test_check_token_wrong_and_empty(made):
    root, cid, token = made
    assert check_token(root, cid, token) is True
    assert check_token(root, cid, token + "x") is False
    assert check_token(root, cid, "") is False


def test_mark_candidate_twice_refused(made):
    root, cid, _ = made
    mark_candidate(root, cid, "approved", "tester")
    assert load_candidate(root, cid)["status"]["approver"] == "tester"
    with pytest.raises(RegistryError):
        mark_candidate(root, cid, "rejected", "tester")


@pytest.mark.parametrize("bad", ["../x", "", "20260101T000000Z-zzzzzzzz", "nope"])
def test_malformed_or_unknown_candidate_id_raises_keyerror(tmp_path, bad):
    with pytest.raises(KeyError):
        load_candidate(str(tmp_path), bad)
    assert not check_token(str(tmp_path), bad, "t")
