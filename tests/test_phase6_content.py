"""Phase 6 — Content Generator tests (AC-2.9).

Covers:
  TestContentGeneratorCore    — unit tests for content_generator.py
  TestContentGenerateEndpoint — API tests for POST /content/generate
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ---------------------------------------------------------------------------
# Shared mock Claude JSON response
# ---------------------------------------------------------------------------

_MOCK_CLAUDE_JSON = {
    "key_message": "กลับมาหาเราสิ — มีส่วนลดพิเศษรอคุณอยู่",
    "emotion": "nostalgia + urgency",
    "tone_used": "friendly",
    "visual_concept": "ภาพกาแฟอุ่นๆ บนโต๊ะไม้ใกล้หน้าต่าง ให้ความรู้สึกคุ้นเคยและอบอุ่น",
    "composition": "hero center",
    "mood": "warm golden hour",
    "color_direction": "น้ำตาลทองและครีม — warm palette ที่เชิญชวน",
    "hero_subject": "แก้วลาเต้กับไอน้ำลอยขึ้น",
    "headline": "เราคิดถึงคุณ",
    "body": "นานไม่ได้เจอ รับส่วนลด 15% สำหรับสมาชิกพิเศษ วันนี้ถึงสิ้นเดือน",
    "cta": "รับสิทธิ์เลย",
    "meta_primary_text": "คุณยังอยู่ในใจเราเสมอ ร้านคาเฟ่ริมน้ำมีส่วนลด 15% รอคุณอยู่",
    "image_prompt": (
        "A warm, inviting café scene with a latte cup on a wooden table near a sunlit window. "
        "Soft golden hour light, steam rising from cup, shallow depth of field, lifestyle photography style, "
        "warm brown and cream tones, cozy atmosphere. No text in image."
    ),
}


# ===========================================================================
# TestContentGeneratorCore
# ===========================================================================

class TestContentGeneratorCore:
    """Unit tests for content_generator.py pure functions and classes."""

    def test_import(self):
        import content_generator  # noqa: F401

    def test_brand_kit_defaults(self):
        from content_generator import BrandKit
        kit = BrandKit(brand_name="ร้านทดสอบ")
        assert kit.category == "food_and_beverage"
        assert kit.tone == "friendly"
        assert kit.forbidden_words == []
        assert kit.preferred_words == []
        assert kit.reference_examples == []

    def test_ml_driven_brief_defaults(self):
        from content_generator import MLDrivenBrief
        brief = MLDrivenBrief(segment="At-Risk", action="Win-Back", promo_type="discount_15")
        assert brief.top_product == ""
        assert brief.channel == "meta"
        assert brief.promo_cost == 0.0

    def test_brand_driven_brief_instantiation(self):
        from content_generator import BrandDrivenBrief
        brief = BrandDrivenBrief(content_type="new_menu", topic="Matcha Latte ใหม่")
        assert brief.content_type == "new_menu"
        assert brief.channel == "meta"

    def test_match_action_pattern_win_back(self):
        from content_generator import _match_action_pattern
        pattern = _match_action_pattern("Win-Back")
        assert "nostalgia" in pattern["emotion"]
        assert "win_back" not in pattern  # returns the dict, not the key

    def test_match_action_pattern_rescue(self):
        from content_generator import _match_action_pattern
        pattern = _match_action_pattern("Rescue Offer")
        assert "urgency" in pattern["emotion"]

    def test_match_action_pattern_milestone(self):
        from content_generator import _match_action_pattern
        # "Milestone Reward" would hit vip_reward first (contains "reward") —
        # use Thai keyword "สั่งครบ" which is unique to milestone_reward
        pattern = _match_action_pattern("สั่งครบ")
        assert "achievement" in pattern["emotion"]

    def test_match_action_pattern_unknown_returns_default(self):
        from content_generator import _match_action_pattern
        pattern = _match_action_pattern("some_unknown_action_xyz")
        assert "friendly" in pattern["emotion"]

    def test_promo_label_known_codes(self):
        from content_generator import _promo_label
        assert _promo_label("discount_15") == "ส่วนลด 15%"
        assert _promo_label("points_x2") == "แต้มสะสม 2 เท่า"
        assert _promo_label("free_item") == "ของแถมฟรี"
        assert _promo_label("free_drink") == "เครื่องดื่มฟรี 1 แก้ว"
        assert _promo_label("bundle") == "ซื้อ 2 แถม 1"

    def test_promo_label_unknown_converts_underscores(self):
        from content_generator import _promo_label
        assert _promo_label("custom_promo_code") == "custom promo code"

    def test_build_forbidden_instruction_empty(self):
        from content_generator import BrandKit, _build_forbidden_instruction
        kit = BrandKit(brand_name="ร้าน")
        assert _build_forbidden_instruction(kit) == ""

    def test_build_forbidden_instruction_with_words(self):
        from content_generator import BrandKit, _build_forbidden_instruction
        kit = BrandKit(brand_name="ร้าน", forbidden_words=["ถูก", "ลดราคา"])
        result = _build_forbidden_instruction(kit)
        assert "ถูก" in result
        assert "ลดราคา" in result
        assert "ห้ามใช้" in result

    def test_build_preferred_instruction_empty(self):
        from content_generator import BrandKit, _build_preferred_instruction
        kit = BrandKit(brand_name="ร้าน")
        assert _build_preferred_instruction(kit) == ""

    def test_build_preferred_instruction_with_words(self):
        from content_generator import BrandKit, _build_preferred_instruction
        kit = BrandKit(brand_name="ร้าน", preferred_words=["สมาชิก", "พิเศษ"])
        result = _build_preferred_instruction(kit)
        assert "สมาชิก" in result

    def test_build_reference_instruction_truncates_to_3(self):
        from content_generator import BrandKit, _build_reference_instruction
        examples = [f"example {i}" for i in range(5)]
        kit = BrandKit(brand_name="ร้าน", reference_examples=examples)
        result = _build_reference_instruction(kit)
        assert "example 3" not in result  # only first 3 included
        assert "example 0" in result

    def test_build_ml_driven_prompt_returns_tuple(self):
        from content_generator import BrandKit, MLDrivenBrief, _build_ml_driven_prompt
        kit = BrandKit(brand_name="คาเฟ่ริมน้ำ")
        brief = MLDrivenBrief(segment="At-Risk", action="Win-Back", promo_type="discount_15",
                               top_product="Latte", channel="meta")
        system, user = _build_ml_driven_prompt(brief, kit)
        assert "creative director" in system
        assert "Win-Back" in user or "win-back" in user.lower()
        assert "คาเฟ่ริมน้ำ" in user
        assert "JSON" in system

    def test_build_brand_driven_prompt_returns_tuple(self):
        from content_generator import BrandKit, BrandDrivenBrief, _build_brand_driven_prompt
        kit = BrandKit(brand_name="ร้านทดสอบ")
        brief = BrandDrivenBrief(content_type="new_menu", topic="เมนูใหม่", channel="line_oa")
        system, user = _build_brand_driven_prompt(brief, kit)
        assert "awareness content" in system
        assert "new_menu" in user
        assert "ร้านทดสอบ" in user

    def test_call_claude_raises_without_api_key(self):
        """_call_claude raises ValueError when API key absent (mocking anthropic import)."""
        from content_generator import _call_claude
        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("ANTHROPIC_API_KEY", None)
                with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                    _call_claude("system", "user", api_key=None)

    def test_generate_content_ml_driven_returns_content_output(self):
        """generate_content with mocked _call_claude returns fully populated ContentOutput."""
        from content_generator import BrandKit, MLDrivenBrief, generate_content, ContentOutput

        kit = BrandKit(brand_name="คาเฟ่ริมน้ำ", tone="friendly")
        brief = MLDrivenBrief(segment="At-Risk", action="Win-Back", promo_type="discount_15",
                               top_product="Latte", channel="meta")

        with patch("content_generator._call_claude", return_value=_MOCK_CLAUDE_JSON):
            out = generate_content(brief, kit)

        assert isinstance(out, ContentOutput)
        assert out.mode == "ml_driven"
        assert out.channel == "meta"
        assert out.segment == "At-Risk"
        assert out.action == "Win-Back"
        assert out.headline == "เราคิดถึงคุณ"
        assert out.body != ""
        assert out.cta == "รับสิทธิ์เลย"
        assert out.image_prompt != ""
        assert out.key_message != ""
        assert out.visual_concept != ""

    def test_generate_content_brand_driven_mode(self):
        """Brand-driven mode sets mode='brand_driven' and action=content_type."""
        from content_generator import BrandKit, BrandDrivenBrief, generate_content

        kit = BrandKit(brand_name="ร้านทดสอบ")
        brief = BrandDrivenBrief(content_type="seasonal", topic="ต้อนรับปีใหม่", channel="line_oa")

        with patch("content_generator._call_claude", return_value=_MOCK_CLAUDE_JSON):
            out = generate_content(brief, kit)

        assert out.mode == "brand_driven"
        assert out.action == "seasonal"
        assert out.segment == ""
        assert out.channel == "line_oa"

    def test_generate_content_warnings_empty_by_default(self):
        from content_generator import BrandKit, MLDrivenBrief, generate_content

        kit = BrandKit(brand_name="ร้าน")
        brief = MLDrivenBrief(segment="Champions", action="VIP Reward", promo_type="points_x2")

        with patch("content_generator._call_claude", return_value=_MOCK_CLAUDE_JSON):
            out = generate_content(brief, kit)

        assert out.warnings == []

    def test_generate_campaign_content_calls_per_action(self):
        """generate_campaign_content returns one ContentOutput per action."""
        from content_generator import BrandKit, generate_campaign_content

        kit = BrandKit(brand_name="ร้านทดสอบ")
        actions = [
            {"segment": "At-Risk", "action": "Win-Back", "promo_type": "discount_15",
             "top_product": "Latte", "top_category": "Coffee", "send_window": "17:00-19:00"},
            {"segment": "Champions", "action": "VIP Reward", "promo_type": "points_x2",
             "top_product": "Croissant", "top_category": "Bakery", "send_window": "09:00-11:00"},
        ]

        with patch("content_generator._call_claude", return_value=_MOCK_CLAUDE_JSON):
            results = generate_campaign_content(actions, kit)

        assert len(results) == 2
        assert results[0].segment == "At-Risk"
        assert results[1].segment == "Champions"

    def test_generate_campaign_content_uses_channel(self):
        from content_generator import BrandKit, generate_campaign_content

        kit = BrandKit(brand_name="ร้านทดสอบ")
        actions = [{"segment": "At-Risk", "action": "Win-Back", "promo_type": "discount_10"}]

        with patch("content_generator._call_claude", return_value=_MOCK_CLAUDE_JSON):
            results = generate_campaign_content(actions, kit, channel="line_oa")

        assert results[0].channel == "line_oa"

    def test_content_output_meta_primary_text(self):
        from content_generator import BrandKit, MLDrivenBrief, generate_content

        kit = BrandKit(brand_name="ร้านทดสอบ")
        brief = MLDrivenBrief(segment="At-Risk", action="Rescue", promo_type="discount_25")

        with patch("content_generator._call_claude", return_value=_MOCK_CLAUDE_JSON):
            out = generate_content(brief, kit)

        assert out.meta_primary_text != ""


# ===========================================================================
# TestContentGenerateEndpoint
# ===========================================================================

_ML_DRIVEN_REQUEST = {
    "brief_type": "ml_driven",
    "brief": {
        "segment": "At-Risk",
        "action": "Win-Back",
        "promo_type": "discount_15",
        "top_product": "Latte",
        "top_category": "Coffee",
        "send_window": "17:00-19:00",
        "channel": "meta",
    },
    "brand_kit": {
        "brand_name": "คาเฟ่ริมน้ำ",
        "category": "food_and_beverage",
        "tone": "friendly",
        "photo_style": "warm lifestyle, natural light",
    },
}

_BRAND_DRIVEN_REQUEST = {
    "brief_type": "brand_driven",
    "brief": {
        "content_type": "new_menu",
        "topic": "Matcha Latte สูตรใหม่",
        "top_product": "Matcha Latte",
        "channel": "meta",
    },
    "brand_kit": {
        "brand_name": "คาเฟ่ริมน้ำ",
        "category": "food_and_beverage",
        "tone": "playful",
    },
}


class TestContentGenerateEndpoint:
    """API tests for POST /v1/content/generate."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        import sys, os
        # Add crm_api to path
        crm_src = os.path.join(os.path.dirname(__file__), "..", "src")
        if crm_src not in sys.path:
            sys.path.insert(0, crm_src)

        from fastapi.testclient import TestClient
        from crm_api.app import create_app
        from crm_api.settings import Settings

        self.client = TestClient(create_app(Settings(data_dir=tmp_path, api_key="test-key")))
        self.headers = {"X-API-Key": "test-key"}

    def test_requires_auth(self):
        resp = self.client.post("/v1/content/generate", json=_ML_DRIVEN_REQUEST)
        assert resp.status_code == 401

    def test_no_api_key_returns_503(self):
        """No ANTHROPIC_API_KEY → 503 with clear error."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            resp = self.client.post("/v1/content/generate",
                                    json=_ML_DRIVEN_REQUEST,
                                    headers=self.headers)
        assert resp.status_code == 503
        body = resp.json()
        # Error shape: {"error": {"code": "...", "message": "..."}}
        assert "ANTHROPIC_API_KEY" in body.get("error", {}).get("message", "")

    def test_invalid_brief_type_returns_400(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("content_generator._call_claude", return_value=_MOCK_CLAUDE_JSON):
                resp = self.client.post(
                    "/v1/content/generate",
                    json={**_ML_DRIVEN_REQUEST, "brief_type": "unsupported_mode"},
                    headers=self.headers,
                )
        assert resp.status_code == 400

    def test_ml_driven_happy_path_returns_200(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("content_generator._call_claude", return_value=_MOCK_CLAUDE_JSON):
                resp = self.client.post("/v1/content/generate",
                                        json=_ML_DRIVEN_REQUEST,
                                        headers=self.headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "headline" in body
        assert "body" in body
        assert "cta" in body
        assert "image_prompt" in body
        assert "meta_primary_text" in body

    def test_ml_driven_response_has_strategy(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("content_generator._call_claude", return_value=_MOCK_CLAUDE_JSON):
                resp = self.client.post("/v1/content/generate",
                                        json=_ML_DRIVEN_REQUEST,
                                        headers=self.headers)
        body = resp.json()
        assert "strategy" in body
        assert "key_message" in body["strategy"]
        assert "emotion" in body["strategy"]
        assert "tone_used" in body["strategy"]

    def test_ml_driven_response_has_art_direction(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("content_generator._call_claude", return_value=_MOCK_CLAUDE_JSON):
                resp = self.client.post("/v1/content/generate",
                                        json=_ML_DRIVEN_REQUEST,
                                        headers=self.headers)
        body = resp.json()
        assert "art_direction" in body
        ad = body["art_direction"]
        for field in ("visual_concept", "composition", "mood", "color_direction", "hero_subject"):
            assert field in ad, f"art_direction missing: {field}"

    def test_brand_driven_happy_path_returns_200(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("content_generator._call_claude", return_value=_MOCK_CLAUDE_JSON):
                resp = self.client.post("/v1/content/generate",
                                        json=_BRAND_DRIVEN_REQUEST,
                                        headers=self.headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["headline"] == "เราคิดถึงคุณ"
        assert body["cta"] == "รับสิทธิ์เลย"

    def test_response_values_come_from_claude_output(self):
        """Verify response maps Claude JSON → ContentOutput → API response correctly."""
        custom_json = {**_MOCK_CLAUDE_JSON, "headline": "เมนูใหม่มาแล้ว!"}
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("content_generator._call_claude", return_value=custom_json):
                resp = self.client.post("/v1/content/generate",
                                        json=_BRAND_DRIVEN_REQUEST,
                                        headers=self.headers)
        assert resp.json()["headline"] == "เมนูใหม่มาแล้ว!"
