"""Tests for content_performance_tracker.py."""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_content_output(
    segment="At-Risk", action="Win-Back", headline="เราคิดถึงคุณ",
    cta="รับสิทธิ์เลย", emotion="nostalgia + urgency",
    tone_used="friendly", key_message="กลับมาหาเรา", mode="ml_driven", channel="meta",
):
    obj = MagicMock()
    obj.segment        = segment
    obj.action         = action
    obj.headline       = headline
    obj.body           = "รับส่วนลด 15% วันนี้"
    obj.cta            = cta
    obj.emotion        = emotion
    obj.tone_used      = tone_used
    obj.key_message    = key_message
    obj.meta_primary_text = "copy ยาว"
    obj.mode           = mode
    obj.channel        = channel
    return obj


def _write_registry(data_dir, records):
    path = os.path.join(str(data_dir), "content_registry.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_outcomes(data_dir, records):
    path = os.path.join(str(data_dir), "content_outcomes.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ===========================================================================
# record_content
# ===========================================================================

class TestRecordContent:

    def test_returns_content_id(self, tmp_path):
        from content_performance_tracker import record_content
        out = _make_content_output()
        cid = record_content(str(tmp_path), out, plan_id="plan_abc", store_id="store_001")
        assert cid.startswith("content_")

    def test_writes_to_registry(self, tmp_path):
        from content_performance_tracker import record_content
        out = _make_content_output()
        cid = record_content(str(tmp_path), out)
        records = _read_jsonl(str(tmp_path / "content_registry.jsonl"))
        assert len(records) == 1
        assert records[0]["content_id"] == cid

    def test_stores_content_metadata(self, tmp_path):
        from content_performance_tracker import record_content
        out = _make_content_output(headline="เมนูใหม่มาแล้ว", emotion="excitement")
        record_content(str(tmp_path), out, plan_id="plan_x")
        records = _read_jsonl(str(tmp_path / "content_registry.jsonl"))
        r = records[0]
        assert r["headline"] == "เมนูใหม่มาแล้ว"
        assert r["emotion"]  == "excitement"
        assert r["plan_id"]  == "plan_x"
        assert r["ad_id"]    is None     # not linked yet

    def test_multiple_records_appended(self, tmp_path):
        from content_performance_tracker import record_content
        for i in range(3):
            record_content(str(tmp_path), _make_content_output(headline=f"headline {i}"))
        records = _read_jsonl(str(tmp_path / "content_registry.jsonl"))
        assert len(records) == 3

    def test_stores_segment_and_action(self, tmp_path):
        from content_performance_tracker import record_content
        out = _make_content_output(segment="Champions", action="VIP Reward")
        record_content(str(tmp_path), out)
        records = _read_jsonl(str(tmp_path / "content_registry.jsonl"))
        assert records[0]["segment"] == "Champions"
        assert records[0]["action"]  == "VIP Reward"


# ===========================================================================
# link_content_to_ad
# ===========================================================================

class TestLinkContentToAd:

    def test_links_ad_id(self, tmp_path):
        from content_performance_tracker import record_content, link_content_to_ad
        cid = record_content(str(tmp_path), _make_content_output())
        result = link_content_to_ad(str(tmp_path), cid, "ad_123456")
        assert result is True
        records = _read_jsonl(str(tmp_path / "content_registry.jsonl"))
        assert records[0]["ad_id"] == "ad_123456"

    def test_returns_false_for_unknown_content_id(self, tmp_path):
        from content_performance_tracker import record_content, link_content_to_ad
        record_content(str(tmp_path), _make_content_output())
        result = link_content_to_ad(str(tmp_path), "content_nonexistent", "ad_xxx")
        assert result is False

    def test_returns_false_when_no_registry(self, tmp_path):
        from content_performance_tracker import link_content_to_ad
        result = link_content_to_ad(str(tmp_path), "content_xxx", "ad_yyy")
        assert result is False

    def test_does_not_double_link(self, tmp_path):
        """Already-linked record should not be overwritten."""
        from content_performance_tracker import record_content, link_content_to_ad
        cid = record_content(str(tmp_path), _make_content_output())
        link_content_to_ad(str(tmp_path), cid, "ad_first")
        link_content_to_ad(str(tmp_path), cid, "ad_second")  # already has ad_id
        records = _read_jsonl(str(tmp_path / "content_registry.jsonl"))
        assert records[0]["ad_id"] == "ad_first"


# ===========================================================================
# track_meta_performance
# ===========================================================================

class TestTrackMetaPerformance:

    def test_dry_run_tracks_all_linked(self, tmp_path):
        _write_registry(tmp_path, [
            {"content_id": "c1", "ad_id": "ad_1", "segment": "At-Risk", "action": "Win-Back",
             "headline": "h1", "cta": "c", "emotion": "e", "key_message": "k",
             "mode": "ml_driven", "channel": "meta", "plan_id": "", "store_id": "", "execution_id": ""},
            {"content_id": "c2", "ad_id": "ad_2", "segment": "Champions", "action": "VIP Reward",
             "headline": "h2", "cta": "c", "emotion": "e", "key_message": "k",
             "mode": "ml_driven", "channel": "meta", "plan_id": "", "store_id": "", "execution_id": ""},
        ])
        from content_performance_tracker import track_meta_performance
        result = track_meta_performance(str(tmp_path), dry_run=True)
        assert result["tracked"] == 2
        assert result["skipped_already_done"] == 0

    def test_dry_run_writes_outcomes(self, tmp_path):
        _write_registry(tmp_path, [
            {"content_id": "c1", "ad_id": "ad_1", "segment": "At-Risk", "action": "Win-Back",
             "headline": "h1", "cta": "c", "emotion": "e", "key_message": "k",
             "mode": "ml_driven", "channel": "meta", "plan_id": "", "store_id": "", "execution_id": ""},
        ])
        from content_performance_tracker import track_meta_performance
        track_meta_performance(str(tmp_path), dry_run=True)
        outcomes = _read_jsonl(str(tmp_path / "content_outcomes.jsonl"))
        assert len(outcomes) == 1
        assert outcomes[0]["content_id"] == "c1"
        assert outcomes[0]["ctr"] > 0

    def test_skips_already_tracked(self, tmp_path):
        _write_registry(tmp_path, [
            {"content_id": "c1", "ad_id": "ad_1", "segment": "At-Risk", "action": "Win-Back",
             "headline": "h1", "cta": "c", "emotion": "e", "key_message": "k",
             "mode": "ml_driven", "channel": "meta", "plan_id": "", "store_id": "", "execution_id": ""},
        ])
        _write_outcomes(tmp_path, [
            {"content_id": "c1", "impressions": 500, "clicks": 15, "ctr": 3.0,
             "conversions": 2, "spend_thb": 50.0},
        ])
        from content_performance_tracker import track_meta_performance
        result = track_meta_performance(str(tmp_path), dry_run=True)
        assert result["skipped_already_done"] == 1
        assert result["tracked"] == 0

    def test_skips_records_without_ad_id(self, tmp_path):
        _write_registry(tmp_path, [
            {"content_id": "c1", "ad_id": None, "segment": "At-Risk",
             "headline": "h", "cta": "c", "emotion": "e", "key_message": "k",
             "mode": "ml_driven", "channel": "meta", "plan_id": "", "store_id": "", "execution_id": ""},
        ])
        from content_performance_tracker import track_meta_performance
        result = track_meta_performance(str(tmp_path), dry_run=True)
        assert result["tracked"] == 0

    def test_empty_registry_returns_zeros(self, tmp_path):
        from content_performance_tracker import track_meta_performance
        result = track_meta_performance(str(tmp_path), dry_run=True)
        assert result == {"tracked": 0, "skipped_already_done": 0,
                          "skipped_low_signal": 0, "failed": 0}

    def test_live_pulls_meta_insights(self, tmp_path):
        _write_registry(tmp_path, [
            {"content_id": "c1", "ad_id": "ad_live", "segment": "At-Risk", "action": "Win-Back",
             "headline": "h1", "cta": "c", "emotion": "e", "key_message": "k",
             "mode": "ml_driven", "channel": "meta", "plan_id": "", "store_id": "", "execution_id": ""},
        ])
        mock_insights = {"impressions": 5000, "clicks": 160, "ctr": 3.2,
                         "conversions": 23, "spend_thb": 450.0}
        mock_client = MagicMock()
        mock_client.get_ad_insights.return_value = mock_insights

        from content_performance_tracker import track_meta_performance
        with patch("content_performance_tracker.MetaAdsClient", return_value=mock_client, create=True):
            with patch.dict(sys.modules, {"meta_ads_client": MagicMock(MetaAdsClient=lambda: mock_client)}):
                result = track_meta_performance(str(tmp_path), dry_run=False)

        # Verify outcome written
        outcomes = _read_jsonl(str(tmp_path / "content_outcomes.jsonl"))
        assert len(outcomes) == 1
        assert outcomes[0]["ctr"] == 3.2
        assert outcomes[0]["conversions"] == 23


# ===========================================================================
# get_top_performers
# ===========================================================================

class TestGetTopPerformers:

    @pytest.fixture
    def populated_data_dir(self, tmp_path):
        _write_outcomes(tmp_path, [
            {"content_id": "c1", "segment": "At-Risk", "action": "Win-Back",
             "headline": "เราคิดถึงคุณ", "cta": "รับสิทธิ์", "emotion": "nostalgia",
             "key_message": "k1", "impressions": 5000, "clicks": 160, "ctr": 3.2, "conversions": 23},
            {"content_id": "c2", "segment": "At-Risk", "action": "Win-Back",
             "headline": "กลับมาหาเรา", "cta": "คลิกเลย", "emotion": "urgency",
             "key_message": "k2", "impressions": 4000, "clicks": 44, "ctr": 1.1, "conversions": 5},
            {"content_id": "c3", "segment": "Champions", "action": "VIP Reward",
             "headline": "สิทธิพิเศษสำหรับคุณ", "cta": "ดูเลย", "emotion": "exclusivity",
             "key_message": "k3", "impressions": 3000, "clicks": 150, "ctr": 5.0, "conversions": 40},
        ])
        return tmp_path

    def test_returns_top_n(self, populated_data_dir):
        from content_performance_tracker import get_top_performers
        results = get_top_performers(str(populated_data_dir), top_n=2)
        assert len(results) == 2

    def test_sorted_by_ctr_descending(self, populated_data_dir):
        from content_performance_tracker import get_top_performers
        results = get_top_performers(str(populated_data_dir))
        ctrs = [r["ctr"] for r in results]
        assert ctrs == sorted(ctrs, reverse=True)

    def test_filter_by_segment(self, populated_data_dir):
        from content_performance_tracker import get_top_performers
        results = get_top_performers(str(populated_data_dir), segment="At-Risk")
        assert all(r["segment"] == "At-Risk" for r in results)
        assert len(results) == 2

    def test_filter_by_action(self, populated_data_dir):
        from content_performance_tracker import get_top_performers
        results = get_top_performers(str(populated_data_dir), action="Win-Back")
        assert all("Win-Back" in r["action"] for r in results)

    def test_no_data_returns_empty(self, tmp_path):
        from content_performance_tracker import get_top_performers
        results = get_top_performers(str(tmp_path))
        assert results == []

    def test_result_has_required_fields(self, populated_data_dir):
        from content_performance_tracker import get_top_performers
        results = get_top_performers(str(populated_data_dir), top_n=1)
        for field in ("headline", "cta", "emotion", "ctr", "conversions", "segment", "action"):
            assert field in results[0]

    def test_filters_low_impressions(self, tmp_path):
        _write_outcomes(tmp_path, [
            {"content_id": "c_low", "segment": "At-Risk", "action": "Win-Back",
             "headline": "h", "cta": "c", "emotion": "e", "key_message": "k",
             "impressions": 50, "clicks": 10, "ctr": 20.0, "conversions": 5},  # too few impressions
        ])
        from content_performance_tracker import get_top_performers
        results = get_top_performers(str(tmp_path))
        assert results == []


# ===========================================================================
# enrich_brand_kit_examples
# ===========================================================================

class TestEnrichBrandKitExamples:

    def test_adds_reference_examples(self, tmp_path):
        _write_outcomes(tmp_path, [
            {"content_id": "c1", "segment": "At-Risk", "action": "Win-Back",
             "headline": "เราคิดถึงคุณ", "cta": "รับสิทธิ์", "emotion": "nostalgia",
             "key_message": "k", "impressions": 5000, "ctr": 3.2, "conversions": 10},
        ])
        from content_generator import BrandKit
        from content_performance_tracker import enrich_brand_kit_examples
        kit = BrandKit(brand_name="ร้านทดสอบ")
        enriched = enrich_brand_kit_examples(str(tmp_path), kit, segment="At-Risk")
        assert len(enriched.reference_examples) > 0
        assert "เราคิดถึงคุณ" in enriched.reference_examples[0]
        assert "3.2%" in enriched.reference_examples[0]

    def test_preserves_existing_examples(self, tmp_path):
        _write_outcomes(tmp_path, [
            {"content_id": "c1", "segment": "At-Risk", "action": "Win-Back",
             "headline": "h new", "cta": "c", "emotion": "e", "key_message": "k",
             "impressions": 5000, "ctr": 2.5, "conversions": 8},
        ])
        from content_generator import BrandKit
        from content_performance_tracker import enrich_brand_kit_examples
        kit = BrandKit(brand_name="ร้าน", reference_examples=["existing example"])
        enriched = enrich_brand_kit_examples(str(tmp_path), kit)
        assert "existing example" in enriched.reference_examples

    def test_no_data_returns_kit_unchanged(self, tmp_path):
        from content_generator import BrandKit
        from content_performance_tracker import enrich_brand_kit_examples
        kit = BrandKit(brand_name="ร้าน")
        enriched = enrich_brand_kit_examples(str(tmp_path), kit)
        assert enriched.reference_examples == []

    def test_does_not_duplicate_examples(self, tmp_path):
        _write_outcomes(tmp_path, [
            {"content_id": "c1", "segment": "At-Risk", "action": "Win-Back",
             "headline": "h dup", "cta": "c", "emotion": "e", "key_message": "k",
             "impressions": 5000, "ctr": 3.2, "conversions": 10},
        ])
        from content_generator import BrandKit
        from content_performance_tracker import enrich_brand_kit_examples
        kit = BrandKit(brand_name="ร้าน")
        # Enrich twice
        enriched  = enrich_brand_kit_examples(str(tmp_path), kit)
        enriched2 = enrich_brand_kit_examples(str(tmp_path), enriched)
        count = enriched2.reference_examples.count(enriched2.reference_examples[0])
        assert count == 1
