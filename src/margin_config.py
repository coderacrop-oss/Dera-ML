"""margin_config.py — Per-category margin lookup for the ML pipeline.

Extends business_config.py to support stores with multiple product types
that have different COGS and minimum margin requirements.

Product types expected for the camera/security store:
  mass     — standard CCTV / accessories sold at market price (high volume, thin margin)
  premium  — high-end AI cameras sold outright (low volume, higher margin required)
  rental   — subscription / monthly rental service (recurring, low COGS)

Usage
-----
    from margin_config import get_margin_for_category, get_margin_for_product

    # By product category tag
    cogs, min_margin = get_margin_for_category("premium")
    # → (0.40, 0.20) if configured, else global fallback

    # By product name — matches via keyword in category_margins keys
    cogs, min_margin = get_margin_for_product("AI Camera Pro 4K")
    # → looks for "camera" in keys, falls back to global

    # Build a lookup table for a full product list
    from margin_config import build_product_margin_table
    table = build_product_margin_table(["CCTV Kit", "AI Cam Pro", "Monthly Rental Basic"])
    # → {"CCTV Kit": (0.60, 0.10), "AI Cam Pro": (0.40, 0.20), ...}
"""
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: load config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        import sys, os
        src = os.path.dirname(__file__)
        if src not in sys.path:
            sys.path.insert(0, src)
        from business_config import CONFIG
        return CONFIG
    except Exception:
        return {}


def _global_margins(config: dict) -> tuple[float, float]:
    return (
        float(config.get("cogs_rate",               0.50)),
        float(config.get("min_allowable_net_margin", 0.15)),
    )


def _category_margins_map(config: dict) -> dict[str, dict]:
    """Return normalised {category_lower: {cogs_rate, min_margin}} dict."""
    raw = config.get("category_margins", {})
    return {k.lower(): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_margin_for_category(
    category: str,
    config: dict | None = None,
) -> tuple[float, float]:
    """Return (cogs_rate, min_margin) for a product category.

    Parameters
    ----------
    category : str
        Category name e.g. "mass", "premium", "rental", "food_and_beverage".
        Case-insensitive.
    config : dict, optional
        Override for testing. Defaults to loaded business_config.CONFIG.

    Returns
    -------
    tuple[float, float]
        (cogs_rate, min_margin). Falls back to global config if category not found.
    """
    cfg    = config if config is not None else _load_config()
    cat_map = _category_margins_map(cfg)
    global_cogs, global_min = _global_margins(cfg)

    key = category.lower().strip() if category else ""
    if key in cat_map:
        entry = cat_map[key]
        cogs      = float(entry.get("cogs_rate",  global_cogs))
        min_margin = float(entry.get("min_margin", global_min))
        _logger.debug("get_margin_for_category(%s): cogs=%.2f min_margin=%.2f", category, cogs, min_margin)
        return cogs, min_margin

    _logger.debug("get_margin_for_category(%s): not found — using global fallback", category)
    return global_cogs, global_min


def get_margin_for_product(
    product_name: str,
    config: dict | None = None,
) -> tuple[float, float]:
    """Return (cogs_rate, min_margin) for a product, matched by keyword.

    Tries each category key as a keyword in product_name (case-insensitive).
    Falls back to global if no keyword matches.

    Parameters
    ----------
    product_name : str
        Product name e.g. "AI Camera Pro 4K", "CCTV Basic Kit".
    config : dict, optional

    Returns
    -------
    tuple[float, float]
        (cogs_rate, min_margin).
    """
    cfg     = config if config is not None else _load_config()
    cat_map = _category_margins_map(cfg)

    name_lower = product_name.lower().strip() if product_name else ""

    for key, entry in cat_map.items():
        if key in name_lower:
            cogs       = float(entry.get("cogs_rate",  _global_margins(cfg)[0]))
            min_margin = float(entry.get("min_margin", _global_margins(cfg)[1]))
            _logger.debug("get_margin_for_product(%s) → matched key '%s': cogs=%.2f min_margin=%.2f",
                          product_name, key, cogs, min_margin)
            return cogs, min_margin

    _logger.debug("get_margin_for_product(%s): no keyword match — using global", product_name)
    return _global_margins(cfg)


def build_product_margin_table(
    product_names: list[str],
    config: dict | None = None,
) -> dict[str, tuple[float, float]]:
    """Build a {product_name: (cogs_rate, min_margin)} lookup for a list of products.

    Useful for pre-computing margins before running the CFO Guardrail loop.

    Parameters
    ----------
    product_names : list[str]
    config : dict, optional

    Returns
    -------
    dict[str, tuple[float, float]]
    """
    cfg = config if config is not None else _load_config()
    return {p: get_margin_for_product(p, config=cfg) for p in product_names}


def get_all_configured_categories(config: dict | None = None) -> list[str]:
    """Return list of all category names that have explicit margin config."""
    cfg = config if config is not None else _load_config()
    return list(_category_margins_map(cfg).keys())


def validate_category_margins(config: dict | None = None) -> list[str]:
    """Validate category_margins config. Returns list of error strings (empty = OK)."""
    cfg    = config if config is not None else _load_config()
    errors: list[str] = []
    cat_map = _category_margins_map(cfg)

    for category, entry in cat_map.items():
        cogs = entry.get("cogs_rate")
        min_m = entry.get("min_margin")

        if cogs is None:
            errors.append(f"category '{category}': missing cogs_rate")
        elif not (0.0 <= float(cogs) <= 1.0):
            errors.append(f"category '{category}': cogs_rate {cogs} must be 0.0–1.0")

        if min_m is None:
            errors.append(f"category '{category}': missing min_margin")
        elif not (0.0 <= float(min_m) <= 1.0):
            errors.append(f"category '{category}': min_margin {min_m} must be 0.0–1.0")

        if cogs is not None and min_m is not None:
            if float(cogs) + float(min_m) >= 1.0:
                errors.append(
                    f"category '{category}': cogs_rate ({cogs}) + min_margin ({min_m}) >= 1.0 "
                    f"— ไม่มีกำไรเลย"
                )

    return errors
