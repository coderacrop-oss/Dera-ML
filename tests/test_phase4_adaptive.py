"""Phase 4 — Adaptive Model tests.

Covers AC-4.1 through AC-4.4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# AC-4.1 — Store Embedding (38-dim deterministic)
# ===========================================================================

class TestGetStoreEmbedding:
    def _make_profile(self, peak_hours=None, off_peak_hours=None, top_items=None,
                      discount_depth=0.1, promo_lift_proxy=0.05):
        from store_intelligence import StoreProfile
        return StoreProfile(
            peak_hours=peak_hours or [11, 12, 13],
            off_peak_hours=off_peak_hours or [8, 9, 15, 16],
            top_items=top_items or [
                {"item_name": "Latte",     "avg_daily_qty": 20, "share_pct": 30.0},
                {"item_name": "Croissant", "avg_daily_qty": 15, "share_pct": 20.0},
            ],
            basket_avg=150.0,
            basket_avg_trend=0.05,
            channel_mix={"in_store": 0.7, "grab": 0.3},
            discount_depth=discount_depth,
            promo_lift_proxy=promo_lift_proxy,
            n_transactions=500,
            date_range_days=90,
        )

    def test_returns_38_dim_array(self):
        from store_intelligence import get_store_embedding

        profile = self._make_profile()
        emb = get_store_embedding(profile)

        assert emb.shape == (38,)
        assert emb.dtype == np.float32

    def test_deterministic_same_input(self):
        """Same StoreProfile → identical embedding every call."""
        from store_intelligence import get_store_embedding

        profile = self._make_profile()
        emb1 = get_store_embedding(profile)
        emb2 = get_store_embedding(profile)

        np.testing.assert_array_equal(emb1, emb2)

    def test_peak_hour_pattern_sums_to_one(self):
        """Dims [0:24] should sum to 1.0 (normalised hourly fraction)."""
        from store_intelligence import get_store_embedding

        profile = self._make_profile(peak_hours=[11, 12, 13])
        emb = get_store_embedding(profile)

        assert abs(emb[0:24].sum() - 1.0) < 1e-5

    def test_dow_pattern_sums_to_one(self):
        """Dims [24:31] should sum to 1.0."""
        from store_intelligence import get_store_embedding

        profile = self._make_profile()
        emb = get_store_embedding(profile)

        assert abs(emb[24:31].sum() - 1.0) < 1e-5

    def test_discount_depth_in_range(self):
        """Dim [36] should be clipped to [0, 1]."""
        from store_intelligence import get_store_embedding

        profile = self._make_profile(discount_depth=1.5)  # over 1 → clipped
        emb = get_store_embedding(profile)

        assert 0.0 <= emb[36] <= 1.0

    def test_promo_lift_clipped(self):
        """Dim [37] clipped to [-1, +1]."""
        from store_intelligence import get_store_embedding

        profile_high = self._make_profile(promo_lift_proxy=5.0)
        profile_low  = self._make_profile(promo_lift_proxy=-5.0)

        emb_h = get_store_embedding(profile_high)
        emb_l = get_store_embedding(profile_low)

        assert emb_h[37] == pytest.approx(1.0, abs=1e-5)
        assert emb_l[37] == pytest.approx(-1.0, abs=1e-5)

    def test_different_peak_hours_differ(self):
        """Two stores with different peak hours → different peak-hour dims."""
        from store_intelligence import get_store_embedding

        morning = self._make_profile(peak_hours=[7, 8, 9],  off_peak_hours=[14, 15])
        evening = self._make_profile(peak_hours=[18, 19, 20], off_peak_hours=[10, 11])

        emb_m = get_store_embedding(morning)
        emb_e = get_store_embedding(evening)

        # Peak-hour dims should differ
        assert not np.allclose(emb_m[0:24], emb_e[0:24])

    def test_cosine_similarity_different_stores(self):
        """AC-4.1: 2 stores with very different peak hours → cosine similarity < 0.7."""
        from store_intelligence import get_store_embedding

        morning = self._make_profile(
            peak_hours=[7, 8, 9],     off_peak_hours=[13, 14, 15],
            discount_depth=0.05, promo_lift_proxy=0.0,
        )
        evening = self._make_profile(
            peak_hours=[18, 19, 20],  off_peak_hours=[10, 11, 12],
            discount_depth=0.05, promo_lift_proxy=0.0,
        )

        emb_m = get_store_embedding(morning).astype(np.float64)
        emb_e = get_store_embedding(evening).astype(np.float64)

        cos_sim = np.dot(emb_m, emb_e) / (np.linalg.norm(emb_m) * np.linalg.norm(emb_e))
        assert cos_sim < 0.7, f"Expected cosine sim < 0.7, got {cos_sim:.3f}"

    def test_top_5_items_captured(self):
        """With 5+ items, dims [31:36] should have non-zero values."""
        from store_intelligence import get_store_embedding

        items = [
            {"item_name": f"Item{i}", "avg_daily_qty": 10, "share_pct": 10.0}
            for i in range(7)
        ]
        profile = self._make_profile(top_items=items)
        emb = get_store_embedding(profile)

        # At least some of [31:36] non-zero
        assert emb[31:36].sum() > 0

    def test_empty_peak_hours_no_crash(self):
        """Empty peak_hours → no crash, returns valid 38-dim array."""
        from store_intelligence import get_store_embedding

        profile = self._make_profile(peak_hours=[], off_peak_hours=[])
        emb = get_store_embedding(profile)

        assert emb.shape == (38,)
        assert np.isfinite(emb).all()


# ===========================================================================
# AC-4.2 — store_embedding attached to Layer4Context
# ===========================================================================

class TestStoreEmbeddingInContext:
    def test_layer4context_has_store_embedding_field(self):
        """Layer4Context must have a store_embedding field (default None)."""
        import dataclasses
        from uplift_engine import Layer4Context

        fields = {f.name for f in dataclasses.fields(Layer4Context)}
        assert "store_embedding" in fields

    def test_store_embedding_defaults_to_none(self):
        """store_embedding field defaults to None — no regression."""
        from uplift_engine import Layer4Context
        import pandas as pd

        # Minimal construction — just check the field exists and defaults
        ctx = Layer4Context(
            raw_df=pd.DataFrame(),
            cust_agg=pd.DataFrame(),
            features=[],
            rfm_df=pd.DataFrame(),
            context_df=pd.DataFrame(),
            top_bundle_name="",
            product_cogs={},
            default_cogs=0.0,
            label_mode="proxy",
        )
        assert ctx.store_embedding is None

    def test_store_embedding_accepts_numpy_array(self):
        """store_embedding field can hold a numpy array."""
        from uplift_engine import Layer4Context
        import pandas as pd

        emb = np.zeros(38, dtype=np.float32)
        ctx = Layer4Context(
            raw_df=pd.DataFrame(),
            cust_agg=pd.DataFrame(),
            features=[],
            rfm_df=pd.DataFrame(),
            context_df=pd.DataFrame(),
            top_bundle_name="",
            product_cogs={},
            default_cogs=0.0,
            label_mode="proxy",
            store_embedding=emb,
        )
        assert ctx.store_embedding is not None
        assert ctx.store_embedding.shape == (38,)


# ===========================================================================
# AC-4.3 — PSM Mode C improved (campaign history awareness)
# ===========================================================================

class TestPsmModeC:
    def _make_pos_df(self, campaign_month="2026-10") -> pd.DataFrame:
        """Build synthetic POS data with spikes in campaign_month."""
        rows = []
        base_date = pd.Timestamp("2026-01-01")

        for cid in range(1, 11):
            # Normal months: 2 visits each
            for m in range(10):
                for _ in range(2):
                    rows.append({
                        "customer_id":    str(cid),
                        "transaction_id": f"T{cid}_{m}_{_}",
                        "timestamp":      base_date + pd.DateOffset(months=m, days=_),
                        "total_price":    100.0,
                    })
            # Spike in campaign_month (10 visits) — only for even-numbered customers
            if cid % 2 == 0:
                spike_date = pd.Timestamp(campaign_month + "-01")
                for v in range(10):
                    rows.append({
                        "customer_id":    str(cid),
                        "transaction_id": f"T{cid}_spike_{v}",
                        "timestamp":      spike_date + pd.Timedelta(days=v),
                        "total_price":    100.0,
                    })

        return pd.DataFrame(rows)

    def test_spike_only_detects_treated(self):
        """Without campaign_dates, spike-only mode detects spiky customers."""
        from propensity_score_matching import _infer_treatment_mode_c

        pos_df = self._make_pos_df()
        treated, warning = _infer_treatment_mode_c(pos_df)

        # Even-numbered customers (2,4,6,8,10) should be detected
        treated_ids = set(treated[treated["is_treated"] == 1]["customer_id"].astype(str))
        assert len(treated_ids) >= 3  # at least some detected
        assert "PSM Mode C" in warning
        assert "proxy" in warning.lower()

    def test_history_mode_fewer_false_positives(self):
        """With campaign_dates, treated count <= spike-only count (more precise)."""
        from propensity_score_matching import _infer_treatment_mode_c

        pos_df = self._make_pos_df(campaign_month="2026-10")

        treated_spike, _ = _infer_treatment_mode_c(pos_df, campaign_dates=None)
        treated_hist,  _ = _infer_treatment_mode_c(
            pos_df,
            campaign_dates=["2026-10-15"],
            campaign_window_days=7,
        )

        spike_count = int(treated_spike["is_treated"].sum())
        hist_count  = int(treated_hist["is_treated"].sum())

        # History-based should be <= spike-only (more selective)
        assert hist_count <= spike_count

    def test_history_warning_mentions_campaign_dates(self):
        """Warning message should mention campaign dates used."""
        from propensity_score_matching import _infer_treatment_mode_c

        pos_df = self._make_pos_df()
        _, warning = _infer_treatment_mode_c(
            pos_df,
            campaign_dates=["2026-10-01"],
        )
        assert "history" in warning.lower() or "campaign date" in warning.lower()

    def test_no_dates_fallback_to_spike(self):
        """campaign_dates=[] (empty list) → falls back to spike-only."""
        from propensity_score_matching import _infer_treatment_mode_c

        pos_df = self._make_pos_df()
        treated_none,  _ = _infer_treatment_mode_c(pos_df, campaign_dates=None)
        treated_empty, _ = _infer_treatment_mode_c(pos_df, campaign_dates=[])

        # Empty list → no dates parsed → same as spike-only
        pd.testing.assert_frame_equal(
            treated_none.reset_index(drop=True),
            treated_empty.reset_index(drop=True),
        )

    def test_match_treated_control_passes_campaign_dates(self):
        """match_treated_control exposes campaign_dates kwarg for Mode C."""
        import inspect
        from propensity_score_matching import match_treated_control

        sig = inspect.signature(match_treated_control)
        assert "campaign_dates" in sig.parameters
        assert "campaign_window_days" in sig.parameters


# ===========================================================================
# AC-4.4 — Hierarchical model tier validation
# ===========================================================================

class TestModelTierValidation:
    def test_global_tier_no_warning(self):
        from train_rlearner import _validate_model_tier

        tier, warning = _validate_model_tier("global", None, None, None)
        assert tier == "global"
        assert warning is None

    def test_category_tier_requires_category(self):
        from train_rlearner import _validate_model_tier

        tier, warning = _validate_model_tier("category", category=None, store_id=None, pos_file=None)
        # Falls back to global + warning
        assert tier == "global"
        assert warning is not None
        assert "category" in warning.lower()

    def test_category_tier_with_category_ok(self):
        from train_rlearner import _validate_model_tier

        tier, warning = _validate_model_tier(
            "category", category="food_and_beverage", store_id=None, pos_file=None
        )
        assert tier == "category"
        assert warning is None

    def test_store_tier_requires_store_id(self):
        from train_rlearner import _validate_model_tier

        tier, warning = _validate_model_tier("store", None, store_id=None, pos_file=None)
        # Falls back — no store_id
        assert tier in ("global", "category")
        assert warning is not None
        assert "store-id" in warning.lower() or "store_id" in warning.lower()

    def test_store_tier_missing_pos_file_fallback(self, tmp_path):
        from train_rlearner import _validate_model_tier

        tier, warning = _validate_model_tier(
            "store", category=None, store_id="store_001",
            pos_file=str(tmp_path / "nonexistent.csv"),
        )
        assert tier in ("global", "category")
        assert warning is not None

    def test_store_tier_insufficient_data_fallback(self, tmp_path):
        """6 months data < min 12 months → fallback to category tier."""
        from train_rlearner import _validate_model_tier

        # Write POS file with only 6 months of data
        pos = tmp_path / "pos.csv"
        dates = pd.date_range("2026-01-01", "2026-06-30", freq="W")
        df = pd.DataFrame({
            "customer_id":    "C001",
            "timestamp":      dates,
            "total_price":    100.0,
            "transaction_id": range(len(dates)),
        })
        df.to_csv(pos, index=False)

        tier, warning = _validate_model_tier(
            "store", category="food_and_beverage", store_id="store_001",
            pos_file=str(pos), min_store_months=12,
        )
        assert tier == "category"
        assert warning is not None
        assert "6" in warning or "month" in warning.lower()

    def test_store_tier_sufficient_data_ok(self, tmp_path):
        """13 months data ≥ min 12 months → store tier OK."""
        from train_rlearner import _validate_model_tier

        pos = tmp_path / "pos.csv"
        dates = pd.date_range("2025-01-01", "2026-02-28", freq="W")
        df = pd.DataFrame({
            "customer_id":    "C001",
            "timestamp":      dates,
            "total_price":    100.0,
            "transaction_id": range(len(dates)),
        })
        df.to_csv(pos, index=False)

        tier, warning = _validate_model_tier(
            "store", category=None, store_id="store_001",
            pos_file=str(pos), min_store_months=12,
        )
        assert tier == "store"
        assert warning is None

    def test_argparse_has_model_tier_flag(self):
        """train_rlearner.py CLI must support --model-tier argument."""
        import subprocess, sys

        result = subprocess.run(
            [sys.executable, "src/train_rlearner.py", "--help"],
            capture_output=True, text=True,
            cwd="/Users/ninakim/Documents/ml-campaign-engine",
        )
        assert "--model-tier" in result.stdout

    def test_argparse_has_category_flag(self):
        import subprocess, sys

        result = subprocess.run(
            [sys.executable, "src/train_rlearner.py", "--help"],
            capture_output=True, text=True,
            cwd="/Users/ninakim/Documents/ml-campaign-engine",
        )
        assert "--category" in result.stdout


# ===========================================================================
# Integration smoke test
# ===========================================================================

class TestPhase4Integration:
    def test_embedding_pipeline_no_crash(self):
        """StoreProfile → get_store_embedding → 38-dim float32 — full path."""
        from store_intelligence import StoreProfile, get_store_embedding

        profile = StoreProfile(
            peak_hours=[11, 12, 13, 14],
            off_peak_hours=[8, 9, 15, 16, 17],
            top_items=[
                {"item_name": "Coffee", "avg_daily_qty": 50, "share_pct": 40.0},
                {"item_name": "Cake",   "avg_daily_qty": 20, "share_pct": 15.0},
            ],
            basket_avg=120.0,
            basket_avg_trend=0.02,
            channel_mix={"in_store": 0.6, "grab": 0.4},
            discount_depth=0.08,
            promo_lift_proxy=0.12,
            n_transactions=1200,
            date_range_days=180,
        )
        emb = get_store_embedding(profile)

        assert emb.shape == (38,)
        assert np.isfinite(emb).all()
        assert emb.dtype == np.float32
