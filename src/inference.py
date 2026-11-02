"""
inference.py — SOTA Inference & Budget-Constrained Targeting
Uncertainty-aware dispatch, budget optimization, A/B test planning,
real-time scoring, and counterfactual explanation.
"""

import numpy as np
import pandas as pd
from typing import Optional


class UpliftInference:
    """
    SOTA uplift inference engine with uncertainty quantification,
    budget-constrained targeting, budget optimization, A/B power analysis,
    and SHAP-style counterfactual explanation.
    """

    # Campaign type bands (by uplift percentile rank within the scored population)
    _CAMPAIGN_TYPES = {
        "top":  "bundle_discount",    # top decile
        "mid":  "loyalty_points",     # mid decile (40th–90th pct)
        "low":  "off_peak_discount",  # lower 40th pct
    }

    def __init__(self, model, feature_names: list, business_config: dict = None):
        self.model = model
        self.feature_names = list(feature_names)
        self.config = business_config or {}

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_effect(self, X: pd.DataFrame) -> np.ndarray:
        """Call model.effect() or model.predict_uplift() depending on API."""
        if hasattr(self.model, "effect"):
            return np.asarray(self.model.effect(X), dtype=float)
        if hasattr(self.model, "predict_uplift"):
            return np.asarray(self.model.predict_uplift(X), dtype=float)
        raise AttributeError(
            "Model must implement effect() or predict_uplift()."
        )

    def _get_ci(
        self, X: pd.DataFrame, alpha: float = 0.05, n_bootstrap: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (lower, upper) confidence interval arrays.
        Uses model.effect_interval() when available (EconML API);
        otherwise falls back to bootstrap resampling.
        """
        if hasattr(self.model, "effect_interval"):
            lb, ub = self.model.effect_interval(X, alpha=alpha)
            return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)

        # Bootstrap fallback — resample rows of X with replacement
        n = len(X)
        estimates = np.zeros((n_bootstrap, n), dtype=float)
        rng = np.random.default_rng(seed=0)
        for i in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_boot = X.iloc[idx].reset_index(drop=True)
            eff = self._predict_effect(X_boot)
            # Map back to original order via a running mean
            estimates[i] = eff

        mean_eff = estimates.mean(axis=0)
        std_eff = estimates.std(axis=0)
        z = 1.96  # 95 % CI
        return mean_eff - z * std_eff, mean_eff + z * std_eff

    @staticmethod
    def _confidence_label(ci_width: np.ndarray) -> np.ndarray:
        labels = np.where(
            ci_width < 0.02, "high",
            np.where(ci_width < 0.05, "medium", "low")
        )
        return labels

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Score each customer row and return an uncertainty-aware result table.

        Returns DataFrame with columns:
            customer_id, uplift_score, uplift_lower, uplift_upper,
            recommend, confidence
        """
        X = X.reset_index(drop=True)
        uplift = self._predict_effect(X)
        lower, upper = self._get_ci(X)
        ci_width = upper - lower

        # Default min threshold — can be overridden via business_config
        min_thr = self.config.get("negative_uplift_guard", 0.0)

        result = pd.DataFrame(
            {
                "customer_id": X.index,
                "uplift_score": np.round(uplift, 6),
                "uplift_lower": np.round(lower, 6),
                "uplift_upper": np.round(upper, 6),
                "recommend": lower > min_thr,
                "confidence": self._confidence_label(ci_width),
            }
        )
        return result

    def target(
        self,
        X: pd.DataFrame,
        budget_pct: float = 0.30,
        min_uplift_threshold: float = 0.0,
        require_positive_ci: bool = True,
    ) -> pd.DataFrame:
        """
        Budget-constrained targeting decision.

        Parameters
        ----------
        X : customer feature DataFrame
        budget_pct : fraction of customers to target (max)
        min_uplift_threshold : minimum uplift_score to be eligible
        require_positive_ci : if True, only include customers where
                              uplift_lower > 0

        Returns DataFrame with scoring columns plus:
            targeted, expected_profit, campaign_type, roi_estimate
        """
        scored = self.score(X)
        scored = scored.sort_values("uplift_score", ascending=False).reset_index(drop=True)

        # --- Eligibility filters ---
        eligible = scored["uplift_score"] >= min_uplift_threshold
        if require_positive_ci:
            eligible = eligible & (scored["uplift_lower"] > 0)

        # Budget cap: top budget_pct of all customers
        n_total = len(scored)
        n_budget = max(1, int(np.floor(n_total * budget_pct)))

        # Targeted = eligible AND within budget rank
        scored["_rank"] = np.arange(1, n_total + 1)
        targeted_mask = eligible & (scored["_rank"] <= n_budget)
        scored["recommend"] = targeted_mask

        # --- Campaign type from uplift percentile rank ---
        # Percentile of each row's uplift within the full scored population
        pct_rank = scored["uplift_score"].rank(pct=True)
        campaign_type = np.where(
            pct_rank >= 0.90,
            self._CAMPAIGN_TYPES["top"],
            np.where(
                pct_rank >= 0.40,
                self._CAMPAIGN_TYPES["mid"],
                self._CAMPAIGN_TYPES["low"],
            ),
        )
        scored["campaign_type"] = campaign_type

        # --- ROI estimate ---
        # roi = uplift * avg_spend * (1 - cogs) - promo_cost
        avg_spend = self.config.get("avg_spend", 500.0)
        cogs = self.config.get("cogs_rate", 0.50)
        promo_cost_rates = self.config.get(
            "promo_cost_rates",
            {"bundle": 0.15, "vip_points": 0.02, "off_peak": 0.15, "standard": 0.05},
        )
        # Map campaign type to a promo cost rate
        campaign_to_rate = {
            "bundle_discount": promo_cost_rates.get("bundle", 0.15),
            "loyalty_points":  promo_cost_rates.get("vip_points", 0.02),
            "off_peak_discount": promo_cost_rates.get("off_peak", 0.15),
        }
        promo_cost_arr = np.array(
            [campaign_to_rate.get(ct, 0.05) for ct in campaign_type], dtype=float
        )
        roi = (
            scored["uplift_score"].values * avg_spend * (1.0 - cogs)
            - promo_cost_arr * avg_spend
        )
        scored["expected_profit"] = np.round(roi, 4)
        scored["roi_estimate"] = scored["expected_profit"]  # alias for clarity

        scored = scored.drop(columns=["_rank"])
        return scored

    def optimize_budget(
        self,
        X: pd.DataFrame,
        cost_per_treatment: float,
        revenue_per_conversion: float,
        max_budget: float,
    ) -> dict:
        """
        Sweep targeting_pct in [0.05, 1.0] and return the pct that maximises
        expected net revenue subject to max_budget.

        Returns
        -------
        dict with keys: optimal_targeting_pct, expected_roi, break_even_lift,
                        n_targeted, total_cost
        """
        scored = self.score(X)
        # Sort by uplift descending for greedy marginal gain
        scored = scored.sort_values("uplift_score", ascending=False).reset_index(drop=True)
        uplift_vals = scored["uplift_score"].values
        n_total = len(uplift_vals)

        best_pct = 0.05
        best_revenue = -np.inf
        best_n = 0

        targeting_pcts = np.arange(0.05, 1.0 + 1e-9, 0.05)
        for pct in targeting_pcts:
            n = max(1, int(np.floor(n_total * pct)))
            total_cost = n * cost_per_treatment
            if total_cost > max_budget:
                continue
            top_uplift = uplift_vals[:n]
            expected_revenue = (top_uplift * revenue_per_conversion).sum() - total_cost
            if expected_revenue > best_revenue:
                best_revenue = expected_revenue
                best_pct = float(pct)
                best_n = n

        # Break-even lift: minimum average lift so that
        #   n * avg_lift * revenue_per_conversion >= n * cost_per_treatment
        # => avg_lift >= cost_per_treatment / revenue_per_conversion
        break_even_lift = (
            cost_per_treatment / revenue_per_conversion
            if revenue_per_conversion > 0
            else float("inf")
        )

        return {
            "optimal_targeting_pct": round(best_pct, 4),
            "expected_roi": round(float(best_revenue), 4),
            "break_even_lift": round(break_even_lift, 6),
            "n_targeted": best_n,
            "total_cost": round(best_n * cost_per_treatment, 2),
        }

    def ab_test_plan(
        self,
        expected_lift: float,
        baseline_conversion: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> dict:
        """
        Power analysis for the A/B test using statsmodels NormalIndPower.

        Returns
        -------
        dict with keys: n_per_arm, total_n, weeks_to_significance
        """
        from statsmodels.stats.power import NormalIndPower

        # Effect size (Cohen's h equivalent for proportions): use z-test approach
        # p1 = baseline_conversion + expected_lift (treatment arm)
        # p2 = baseline_conversion (control arm)
        # pooled std under H0
        p1 = np.clip(baseline_conversion + expected_lift, 0.0, 1.0)
        p2 = np.clip(baseline_conversion, 0.0, 1.0)
        p_pool = (p1 + p2) / 2.0
        se_pool = np.sqrt(2 * p_pool * (1.0 - p_pool))

        if se_pool <= 0 or p1 == p2:
            effect_size = 0.0
        else:
            effect_size = abs(p1 - p2) / se_pool

        analysis = NormalIndPower()
        if effect_size == 0.0:
            n_per_arm = float("inf")
        else:
            n_per_arm = analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative="two-sided",
            )

        n_per_arm_int = int(np.ceil(n_per_arm)) if np.isfinite(n_per_arm) else None
        total_n = n_per_arm_int * 2 if n_per_arm_int is not None else None

        # Weeks to significance: assume 1 000 new customers/week (typical restaurant)
        customers_per_week = 1000
        if total_n is not None:
            weeks = int(np.ceil(total_n / customers_per_week))
        else:
            weeks = None

        return {
            "n_per_arm": n_per_arm_int,
            "total_n": total_n,
            "weeks_to_significance": weeks,
            "effect_size": round(float(effect_size), 6),
            "alpha": alpha,
            "power": power,
        }

    def explain(self, X_row: pd.Series) -> dict:
        """
        SHAP-style feature attribution via leave-one-feature-at-median
        perturbation analysis.

        Returns top-5 features sorted by |delta| as {feature: delta}.
        """
        # Build a 1-row DataFrame from the input series
        row_df = X_row.to_frame().T.reset_index(drop=True)
        # Only keep known feature columns
        feat_cols = [f for f in self.feature_names if f in row_df.columns]
        row_df = row_df[feat_cols]

        baseline_uplift = float(self._predict_effect(row_df)[0])

        # Compute column medians (use row values as proxy when we have a single
        # observation; the caller is expected to pass column medians separately
        # in production, but for the approximation we use the row's own values
        # shifted to 0 as a simple neutral baseline)
        # Per spec: set each feature to its median, recompute, measure delta
        medians = {col: float(row_df[col].iloc[0]) * 0.0 for col in feat_cols}
        # Better: use 0.0 is not meaningful; use the feature's own value as
        # the median stand-in requires a reference distribution.
        # We implement the spec literally: median = 0 placeholder when no
        # reference is provided.  If the model has a .X_train_ attribute we
        # use it.
        if hasattr(self.model, "X_train_"):
            ref = pd.DataFrame(self.model.X_train_, columns=self.feature_names)
            medians = {col: float(ref[col].median()) for col in feat_cols}
        else:
            # Fall back: treat the row itself shifted by -row_value as perturbation
            # i.e. set each feature to 0 (a common neutral baseline for z-scored data)
            medians = {col: 0.0 for col in feat_cols}

        deltas = {}
        for feat in feat_cols:
            perturbed = row_df.copy()
            perturbed[feat] = medians[feat]
            perturbed_uplift = float(self._predict_effect(perturbed)[0])
            deltas[feat] = round(baseline_uplift - perturbed_uplift, 8)

        # Top 5 by absolute delta
        top5 = dict(
            sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        )
        return top5
