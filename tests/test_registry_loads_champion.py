"""CI guard: the champion pkl in models/registry/ must be loadable with the current src/.

Purpose
-------
If a new champion is pickled from a class that lives outside src/ (e.g. a
training helper like rlearner_wrapper.py that was never git-added), this test
will fail with ModuleNotFoundError on the next `pytest` run, catching the
issue before it reaches production.

What it checks
--------------
1. ModelRegistry.load_current() completes without error.
2. The loaded object satisfies the serving contract:
   - has predict_uplift(X) → shape (n,)
   - has n_features_       → positive int
3. predict_uplift runs end-to-end on synthetic data with the right number
   of features, returning finite floats.

This test reads from the REAL models/registry/ directory.  It is therefore
sensitive to the champion pkl and manifest being consistent — which is exactly
the invariant we want to protect.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_ROOT = os.path.join(_REPO_ROOT, "models")


def _models_exist() -> bool:
    manifest = os.path.join(_MODELS_ROOT, "registry", "manifest.json")
    champion = os.path.join(_MODELS_ROOT, "registry", "champion_xlearner.pkl")
    return os.path.exists(manifest) and os.path.exists(champion)


skip_if_no_models = pytest.mark.skipif(
    not _models_exist(),
    reason="models/registry/ not present (fresh clone without LFS or model files)",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@skip_if_no_models
def test_champion_loads_without_error():
    """ModelRegistry.load_current() must complete without raising."""
    from model_registry import ModelRegistry
    reg = ModelRegistry(_MODELS_ROOT)
    model, ref = reg.load_current()
    assert model is not None
    assert ref is not None
    assert ref.version  # non-empty version string


@skip_if_no_models
def test_champion_satisfies_serving_contract():
    """Loaded champion must expose predict_uplift and n_features_."""
    from model_registry import ModelRegistry
    reg = ModelRegistry(_MODELS_ROOT)
    model, ref = reg.load_current()

    assert hasattr(model, "predict_uplift"), (
        f"champion {ref.version} ({type(model).__name__}) is missing predict_uplift — "
        "serving pipeline will break"
    )
    assert hasattr(model, "n_features_"), (
        f"champion {ref.version} ({type(model).__name__}) is missing n_features_ — "
        "serving pipeline will break"
    )
    assert isinstance(model.n_features_, int) and model.n_features_ > 0, (
        f"n_features_ must be a positive int, got {model.n_features_!r}"
    )


@skip_if_no_models
def test_champion_predict_uplift_returns_finite_floats():
    """predict_uplift(X) must return a finite float array of shape (n,)."""
    from model_registry import ModelRegistry
    reg = ModelRegistry(_MODELS_ROOT)
    model, ref = reg.load_current()

    n_features = model.n_features_
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, n_features)).astype(np.float32)

    uplift = model.predict_uplift(X)

    assert uplift.shape == (10,), (
        f"predict_uplift must return shape (n,), got {uplift.shape}"
    )
    assert np.all(np.isfinite(uplift)), (
        f"predict_uplift returned non-finite values: {uplift}"
    )
