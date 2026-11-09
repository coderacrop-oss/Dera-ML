"""Tests for margin_config.py — per-category margin lookup."""
from __future__ import annotations

import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ---------------------------------------------------------------------------
# Shared test configs
# ---------------------------------------------------------------------------

_CFG_WITH_CATEGORIES = {
    "cogs_rate":               0.50,
    "min_allowable_net_margin": 0.15,
    "category_margins": {
        "mass":    {"cogs_rate": 0.60, "min_margin": 0.10},
        "premium": {"cogs_rate": 0.40, "min_margin": 0.20},
        "rental":  {"cogs_rate": 0.20, "min_margin": 0.30},
    },
}

_CFG_EMPTY_CATEGORIES = {
    "cogs_rate":               0.50,
    "min_allowable_net_margin": 0.15,
    "category_margins": {},
}

_CFG_NO_CATEGORY_KEY = {
    "cogs_rate":               0.55,
    "min_allowable_net_margin": 0.12,
}


# ===========================================================================
# get_margin_for_category
# ===========================================================================

class TestGetMarginForCategory:

    def test_known_category_returns_configured_values(self):
        from margin_config import get_margin_for_category
        cogs, min_m = get_margin_for_category("premium", config=_CFG_WITH_CATEGORIES)
        assert cogs  == 0.40
        assert min_m == 0.20

    def test_mass_category(self):
        from margin_config import get_margin_for_category
        cogs, min_m = get_margin_for_category("mass", config=_CFG_WITH_CATEGORIES)
        assert cogs  == 0.60
        assert min_m == 0.10

    def test_rental_category(self):
        from margin_config import get_margin_for_category
        cogs, min_m = get_margin_for_category("rental", config=_CFG_WITH_CATEGORIES)
        assert cogs  == 0.20
        assert min_m == 0.30

    def test_case_insensitive(self):
        from margin_config import get_margin_for_category
        cogs, min_m = get_margin_for_category("PREMIUM", config=_CFG_WITH_CATEGORIES)
        assert cogs == 0.40

    def test_unknown_category_falls_back_to_global(self):
        from margin_config import get_margin_for_category
        cogs, min_m = get_margin_for_category("unknown_cat", config=_CFG_WITH_CATEGORIES)
        assert cogs  == 0.50
        assert min_m == 0.15

    def test_empty_category_name_falls_back(self):
        from margin_config import get_margin_for_category
        cogs, min_m = get_margin_for_category("", config=_CFG_WITH_CATEGORIES)
        assert cogs == 0.50

    def test_no_category_margins_key_falls_back(self):
        from margin_config import get_margin_for_category
        cogs, min_m = get_margin_for_category("premium", config=_CFG_NO_CATEGORY_KEY)
        assert cogs  == 0.55
        assert min_m == 0.12

    def test_empty_category_margins_falls_back(self):
        from margin_config import get_margin_for_category
        cogs, min_m = get_margin_for_category("premium", config=_CFG_EMPTY_CATEGORIES)
        assert cogs == 0.50


# ===========================================================================
# get_margin_for_product
# ===========================================================================

class TestGetMarginForProduct:

    def test_product_name_contains_category_keyword(self):
        from margin_config import get_margin_for_product
        cogs, min_m = get_margin_for_product("premium AI Camera 4K", config=_CFG_WITH_CATEGORIES)
        assert cogs  == 0.40
        assert min_m == 0.20

    def test_product_name_contains_mass_keyword(self):
        from margin_config import get_margin_for_product
        cogs, min_m = get_margin_for_product("mass market CCTV kit", config=_CFG_WITH_CATEGORIES)
        assert cogs == 0.60

    def test_product_name_contains_rental_keyword(self):
        from margin_config import get_margin_for_product
        cogs, min_m = get_margin_for_product("Monthly Rental Service Basic", config=_CFG_WITH_CATEGORIES)
        assert cogs == 0.20

    def test_no_keyword_match_falls_back_to_global(self):
        from margin_config import get_margin_for_product
        cogs, min_m = get_margin_for_product("CCTV Standard Kit", config=_CFG_WITH_CATEGORIES)
        assert cogs  == 0.50
        assert min_m == 0.15

    def test_case_insensitive_keyword_match(self):
        from margin_config import get_margin_for_product
        cogs, _ = get_margin_for_product("PREMIUM Bundle Set", config=_CFG_WITH_CATEGORIES)
        assert cogs == 0.40

    def test_empty_product_name_falls_back(self):
        from margin_config import get_margin_for_product
        cogs, min_m = get_margin_for_product("", config=_CFG_WITH_CATEGORIES)
        assert cogs == 0.50

    def test_no_category_margins_configured_falls_back(self):
        from margin_config import get_margin_for_product
        cogs, min_m = get_margin_for_product("AI Camera Pro", config=_CFG_EMPTY_CATEGORIES)
        assert cogs  == 0.50
        assert min_m == 0.15


# ===========================================================================
# build_product_margin_table
# ===========================================================================

class TestBuildProductMarginTable:

    def test_returns_dict_per_product(self):
        from margin_config import build_product_margin_table
        products = ["CCTV Basic Kit", "premium AI Camera 4K", "Monthly Rental Basic"]
        table = build_product_margin_table(products, config=_CFG_WITH_CATEGORIES)
        assert set(table.keys()) == set(products)

    def test_values_are_tuples(self):
        from margin_config import build_product_margin_table
        table = build_product_margin_table(["Item A"], config=_CFG_WITH_CATEGORIES)
        assert isinstance(table["Item A"], tuple)
        assert len(table["Item A"]) == 2

    def test_known_product_gets_category_margin(self):
        from margin_config import build_product_margin_table
        table = build_product_margin_table(["rental service"], config=_CFG_WITH_CATEGORIES)
        cogs, min_m = table["rental service"]
        assert cogs  == 0.20
        assert min_m == 0.30

    def test_empty_list_returns_empty_dict(self):
        from margin_config import build_product_margin_table
        assert build_product_margin_table([], config=_CFG_WITH_CATEGORIES) == {}


# ===========================================================================
# get_all_configured_categories
# ===========================================================================

class TestGetAllConfiguredCategories:

    def test_returns_all_category_names(self):
        from margin_config import get_all_configured_categories
        cats = get_all_configured_categories(config=_CFG_WITH_CATEGORIES)
        assert set(cats) == {"mass", "premium", "rental"}

    def test_empty_config_returns_empty(self):
        from margin_config import get_all_configured_categories
        assert get_all_configured_categories(config=_CFG_EMPTY_CATEGORIES) == []

    def test_no_key_returns_empty(self):
        from margin_config import get_all_configured_categories
        assert get_all_configured_categories(config=_CFG_NO_CATEGORY_KEY) == []


# ===========================================================================
# validate_category_margins
# ===========================================================================

class TestValidateCategoryMargins:

    def test_valid_config_returns_no_errors(self):
        from margin_config import validate_category_margins
        errors = validate_category_margins(config=_CFG_WITH_CATEGORIES)
        assert errors == []

    def test_missing_cogs_rate_is_error(self):
        from margin_config import validate_category_margins
        cfg = {"category_margins": {"bad_cat": {"min_margin": 0.10}}}
        errors = validate_category_margins(config=cfg)
        assert any("cogs_rate" in e for e in errors)

    def test_missing_min_margin_is_error(self):
        from margin_config import validate_category_margins
        cfg = {"category_margins": {"bad_cat": {"cogs_rate": 0.50}}}
        errors = validate_category_margins(config=cfg)
        assert any("min_margin" in e for e in errors)

    def test_cogs_rate_out_of_range_is_error(self):
        from margin_config import validate_category_margins
        cfg = {"category_margins": {"bad": {"cogs_rate": 1.50, "min_margin": 0.10}}}
        errors = validate_category_margins(config=cfg)
        assert any("cogs_rate" in e for e in errors)

    def test_cogs_plus_margin_gte_1_is_error(self):
        from margin_config import validate_category_margins
        cfg = {"category_margins": {"impossible": {"cogs_rate": 0.80, "min_margin": 0.30}}}
        errors = validate_category_margins(config=cfg)
        assert any("ไม่มีกำไร" in e or ">= 1.0" in e for e in errors)

    def test_empty_category_margins_no_errors(self):
        from margin_config import validate_category_margins
        errors = validate_category_margins(config=_CFG_EMPTY_CATEGORIES)
        assert errors == []

    def test_min_margin_zero_is_valid(self):
        """cogs=0.50 + min_margin=0.0 < 1.0 → valid (zero margin allowed for analysis)."""
        from margin_config import validate_category_margins
        cfg = {"category_margins": {"zero_margin": {"cogs_rate": 0.50, "min_margin": 0.0}}}
        errors = validate_category_margins(config=cfg)
        assert errors == []
