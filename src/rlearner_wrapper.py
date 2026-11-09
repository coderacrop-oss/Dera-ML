"""rlearner_wrapper.py — Thin adapter so econml estimators can serve as registry champion.

The ModelRegistry and serving pipeline expect champion.predict_uplift(X)
and champion.n_features_. RLearnerWrapper satisfies both while storing the
underlying CausalForestDML or LinearDRLearner model.

Keeping this in its own module ensures pickle.loads() can resolve the class
regardless of which script promoted the model.
"""

from __future__ import annotations

import numpy as np


class RLearnerWrapper:
    """Adapter around a fitted econml estimator (CausalForestDML / LinearDRLearner).

    Exposes the two methods the serving pipeline requires:
      - predict_uplift(X)         → CATE estimates
      - predict_control_proba(X)  → baseline conversion probability (control group)

    Attributes
    ----------
    model                : fitted econml estimator with .effect(X) method
    feature_cols         : list of feature column names used during training
    n_features_          : len(feature_cols) — checked by _check_features
    baseline_control_prob: mean outcome in control group at training time
                           used as a constant estimate for predict_control_proba
    model_name           : "R-Learner" | "DR-Learner"
    dataset              : dataset tag, e.g. "psm", "hillstrom"
    auuc                 : AUUC on test set at time of promotion
    """

    def __init__(
        self,
        model,
        feature_cols: list,
        model_name: str,
        dataset: str,
        auuc: float,
        baseline_control_prob: float = 0.5,
    ):
        self.model                 = model
        self.feature_cols          = list(feature_cols)
        self.n_features_           = len(feature_cols)
        self.model_name            = model_name
        self.dataset               = dataset
        self.auuc                  = auuc
        self.baseline_control_prob = float(baseline_control_prob)

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Return CATE estimates shaped (n_samples,)."""
        return self.model.effect(X).ravel()

    def predict_control_proba(self, X: np.ndarray) -> np.ndarray:
        """Return baseline conversion probability for each customer.

        CausalForestDML does not expose a separate control outcome model,
        so we return the mean control-group outcome observed at training time
        as a constant array. This is used by action_rules for expected-profit
        calculations — a reasonable approximation until real causal labels arrive.
        """
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, self.baseline_control_prob)

    def __repr__(self) -> str:
        return (
            f"RLearnerWrapper({self.model_name}, dataset={self.dataset}, "
            f"n_features={self.n_features_}, auuc={self.auuc:.4f}, "
            f"baseline_ctrl={self.baseline_control_prob:.3f})"
        )
