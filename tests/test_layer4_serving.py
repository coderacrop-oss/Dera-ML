"""Serving mode: run_layer4 only scores with the approved champion; it never trains or writes the registry."""
import hashlib
import json
import os
import pickle

import pandas as pd
import pytest

import uplift_engine as l4
import campaign_pipeline as cp
from fixtures_pos import make_pos_csv
from model_registry import ModelRegistry, RegistryError

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "layer4_basic.csv")
COLS = ["customer_id", "recommended_action", "expected_profit", "should_send"]
META = {"approved_by": "t", "approved_at": "x", "provenance": "proxy-labels", "metrics": None, "created_at": "x"}


def _tree_hash(root):
    h = hashlib.sha256()
    for dp, _d, files in sorted(os.walk(root)):
        for n in sorted(files):
            if n != ".lock":
                p = os.path.join(dp, n)
                h.update(os.path.relpath(p, root).encode())
                h.update(open(p, "rb").read())
    return h.hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    csv = make_pos_csv(tmp_path, n_customers=120, seed=0)
    ctx = l4.build_serving_context(csv, None)
    model = l4._train_xlearner(ctx.raw_df[ctx.raw_df["was_treated"] == 1],
                               ctx.raw_df[ctx.raw_df["was_treated"] == 0], ctx.features)
    return dict(tmp=tmp_path, models=str(models), csv=csv, ctx=ctx, model=model)


def _register(env, model=None):
    reg = ModelRegistry(env["models"])
    reg.promote(pickle.dumps(model or env["model"]), None, META)
    return reg


def test_serving_leaves_models_dir_byte_identical(env):
    _register(env)
    before = _tree_hash(env["models"])
    cp.run_layer4(env["csv"], None, models_root=env["models"])
    assert _tree_hash(env["models"]) == before


def test_serving_never_calls_fit(env, monkeypatch):
    _register(env)

    def boom(*a, **k):
        raise AssertionError("serving must not fit")

    monkeypatch.setattr(l4, "_train_xlearner", boom)
    import lightgbm
    monkeypatch.setattr(lightgbm.LGBMClassifier, "fit", boom)
    monkeypatch.setattr(lightgbm.LGBMRegressor, "fit", boom)
    out = cp.run_layer4(env["csv"], None, models_root=env["models"])
    assert len(out) == 120


def test_serving_output_equals_golden_when_champion_is_that_model(env):
    _register(env)
    out = cp.run_layer4(env["csv"], None, models_root=env["models"])
    out = out[COLS].sort_values("customer_id").reset_index(drop=True)
    golden = pd.read_csv(GOLDEN, dtype={"customer_id": str})
    pd.testing.assert_frame_equal(out, golden, check_dtype=False, rtol=1e-6)


def test_serving_writes_model_used_json_with_version(env):
    ref = _register(env).current()
    cp.run_layer4(env["csv"], None, models_root=env["models"])
    used = json.loads((env["tmp"] / "data" / "processed" / "model_used.json").read_text())
    assert used["champion_version"] == "v1" and used["sha256"] == ref.sha256
    assert (env["tmp"] / "data" / "processed" / "final_campaign_engine_output.csv").exists()


def test_no_champion_raises_registryerror_and_writes_nothing(env):
    with pytest.raises(RegistryError, match="train_champion.py"):
        cp.run_layer4(env["csv"], None, models_root=env["models"])
    assert not (env["tmp"] / "data" / "processed" / "final_campaign_engine_output.csv").exists()
    assert not (env["tmp"] / "data" / "processed" / "model_used.json").exists()
    assert not os.path.exists(env["models"])


def test_feature_name_mismatch_raises_naming_both_lists(env):
    env["model"].feature_names_ = ["total_price", "quantity", "something_else"]
    _register(env)
    with pytest.raises(RegistryError) as e:
        cp.run_layer4(env["csv"], None, models_root=env["models"])
    assert "something_else" in str(e.value) and "is_campaign_spike" in str(e.value)
    assert not (env["tmp"] / "data" / "processed" / "final_campaign_engine_output.csv").exists()


def test_same_count_different_order_raises(env):
    env["model"].feature_names_ = list(reversed(env["ctx"].features))
    _register(env)
    with pytest.raises(RegistryError):
        cp.run_layer4(env["csv"], None, models_root=env["models"])


def test_legacy_champion_without_feature_names_is_accepted_on_count(env):
    del env["model"].feature_names_
    _register(env)
    out = cp.run_layer4(env["csv"], None, models_root=env["models"])
    assert len(out) == 120


def test_legacy_champion_with_wrong_feature_count_is_refused(env):
    del env["model"].feature_names_
    # Force legacy model to report wrong feature count (simulate champion trained on 4 features)
    env["model"].mu0.n_features_in_ = 4
    _register(env)
    with pytest.raises(RegistryError, match="features"):
        cp.run_layer4(env["csv"], None, models_root=env["models"])


def test_modified_champion_file_refused(env):
    _register(env)
    with open(os.path.join(env["models"], "registry", "champion_xlearner.pkl"), "ab") as f:
        f.write(b"x")
    with pytest.raises(RegistryError, match="sha256"):
        cp.run_layer4(env["csv"], None, models_root=env["models"])
