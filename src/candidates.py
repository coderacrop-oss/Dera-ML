"""Candidate store (models/candidates/<id>/) and holdout evaluation of a challenger against the champion."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import re
import secrets
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from atomic_io import atomic_write_bytes, atomic_write_text
from layer4_uplift_margin import decide_for_customers, gate_decision, market_is_recession
from model_registry import RegistryError
from uplift_metrics import compute_uplift_metrics

CANDIDATE_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")
MIN_HOLDOUT = 100


def _metrics(frames: list) -> dict:
    df = pd.concat(frames) if frames else pd.DataFrame(columns=["expected_profit", "should_send",
                                                               "guardrail_blocked", "recommended_action"])
    n = len(df)
    return {"profit": float(df["expected_profit"].sum()),
            "send_count": int(df["should_send"].sum()),
            "blocked_count": int(df["guardrail_blocked"].sum()),
            "do_nothing_pct": float(df["recommended_action"].str.startswith("Do Nothing").mean() * 100) if n else 0.0}


def _uplift_metrics_for_holdout(model, ctx, holdout_ids: set) -> dict:
    """Compute AUUC/Qini/ECE on held-out customers using actual labels from ctx."""
    holdout_cust = ctx.cust_agg[
        ctx.cust_agg["customer_id"].astype(str).isin(holdout_ids)
    ]
    if len(holdout_cust) < 10 or "was_treated" not in holdout_cust.columns or \
       "converted" not in holdout_cust.columns:
        return {"error": "insufficient holdout labels"}
    X = holdout_cust[ctx.features].fillna(0).values
    T = holdout_cust["was_treated"].fillna(0).values
    Y = holdout_cust["converted"].fillna(0).values
    try:
        return compute_uplift_metrics(model, X, T, Y)
    except Exception as exc:
        return {"error": str(exc)}


def _shap_importance(model, ctx, holdout_ids: set) -> dict:
    """Top-5 feature importances from SHAP on the CATE regressors (tau0/tau1).

    Returns {"feature": importance, ...} sorted descending, or {"error": ...} on failure.
    SHAP is computed on tau0 (control-side CATE regressor) which is a raw LGBMRegressor.
    """
    try:
        import shap  # optional — only installed in training environment
    except ImportError:
        return {"error": "shap not installed"}
    try:
        tau0 = getattr(model, "tau0", None)
        if tau0 is None:
            return {"error": "model has no tau0 (not an XLearnerModel)"}

        holdout_cust = ctx.cust_agg[ctx.cust_agg["customer_id"].astype(str).isin(holdout_ids)]
        if len(holdout_cust) < 5:
            return {"error": "insufficient holdout for SHAP"}

        X = holdout_cust[ctx.features].fillna(0)
        explainer = shap.TreeExplainer(tau0)
        shap_vals = explainer.shap_values(X)      # shape (n, p)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        importance = dict(zip(ctx.features, mean_abs.tolist()))
        # Return top-5 sorted descending
        top5 = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:5])
        return {k: round(v, 6) for k, v in top5.items()}
    except Exception as exc:
        return {"error": str(exc)}


def evaluate_candidate(pooled, challenger, champion, champion_ref, training_meta: dict) -> dict:
    """Score both models on the held-out customers of every shop (same decision code as serving)."""
    ch_frames, cp_frames, per_store, stores, holdout_raw = [], [], {}, [], []
    for s, ctx in pooled.contexts.items():
        ids = pooled.holdout_ids[s]
        _, ch = decide_for_customers(challenger, ctx, ids)
        ch_frames.append(ch)
        entry = {"challenger": _metrics([ch]), "champion": None}
        if champion is not None:
            _, cp = decide_for_customers(champion, ctx, ids)
            cp_frames.append(cp)
            entry["champion"] = _metrics([cp])
        per_store[s] = entry
        stores.append({"store_id": s, "rows": int(len(ctx.raw_df)), "customers": int(len(ctx.cust_agg)),
                       "holdout_customers": int(len(ids))})
        holdout_raw.append(ctx.raw_df[ctx.raw_df["customer_id"].astype(str).isin(ids)][["timestamp", "total_price"]])
    challenger_m = _metrics(ch_frames)
    champion_m   = _metrics(cp_frames) if champion is not None else None
    is_rec, current_avg = market_is_recession(pd.concat(holdout_raw))
    gate = gate_decision(challenger_m["profit"], champion_m["profit"] if champion_m else None,
                         challenger_m["do_nothing_pct"], is_rec, current_avg)
    holdout_size = sum(s["holdout_customers"] for s in stores)

    # Uplift quality metrics (AUUC, Qini, ECE) — pooled across all stores
    uplift_m = {}
    if pooled.label_mode == "real":
        # Only meaningful with real labels; proxy mode has random treatment → Qini ≈ 0 by design
        all_ctx_items = list(pooled.contexts.items())
        if len(all_ctx_items) == 1:
            s, ctx = all_ctx_items[0]
            uplift_m = _uplift_metrics_for_holdout(challenger, ctx, pooled.holdout_ids[s])
        else:
            # Multi-store: compute on first store with enough holdout
            for s, ctx in all_ctx_items:
                if len(pooled.holdout_ids[s]) >= 30:
                    uplift_m = _uplift_metrics_for_holdout(challenger, ctx, pooled.holdout_ids[s])
                    uplift_m["note"] = f"computed on store {s!r} (first store with ≥30 holdout customers)"
                    break
    else:
        uplift_m = {"note": "proxy mode — Qini≈0 by design (random treatment assignment); skipped"}

    challenger_m["uplift_metrics"] = uplift_m

    # Also snapshot champion uplift metrics on the same holdout (for manifest comparison)
    champion_uplift_m = {}
    if champion is not None and pooled.label_mode == "real":
        all_ctx_items = list(pooled.contexts.items())
        if len(all_ctx_items) == 1:
            s, ctx = all_ctx_items[0]
            try:
                champion_uplift_m = _uplift_metrics_for_holdout(champion, ctx, pooled.holdout_ids[s])
            except Exception as exc:
                champion_uplift_m = {"error": str(exc)}
        else:
            for s, ctx in all_ctx_items:
                if len(pooled.holdout_ids[s]) >= 30:
                    try:
                        champion_uplift_m = _uplift_metrics_for_holdout(champion, ctx, pooled.holdout_ids[s])
                        champion_uplift_m["note"] = f"computed on store {s!r} (first store with ≥30 holdout customers)"
                    except Exception as exc:
                        champion_uplift_m = {"error": str(exc)}
                    break
    if champion_m is not None:
        champion_m["uplift_metrics"] = champion_uplift_m

    # SHAP feature importance for CATE (tau0 regressor)
    shap_importance: dict = {}
    if pooled.label_mode == "real":
        all_ctx_items = list(pooled.contexts.items())
        for s, ctx in all_ctx_items:
            if len(pooled.holdout_ids[s]) >= 10:
                shap_importance = _shap_importance(challenger, ctx, pooled.holdout_ids[s])
                break
    challenger_m["shap_importance"] = shap_importance

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "compared_against_version": champion_ref.version if champion_ref else None,
        "label_mode": pooled.label_mode,
        "provenance": "real-labels" if pooled.label_mode == "real" else "proxy-labels",
        "holdout_size": holdout_size,
        "holdout_warning": holdout_size < MIN_HOLDOUT,
        "stores": stores,
        "champion": champion_m,
        "challenger": challenger_m,
        "per_store": per_store,
        "gate": {"passed": bool(gate.passed), "reason": gate.reason,
                 "growth_pct": None if gate.growth_pct is None else float(gate.growth_pct)},
        "training": training_meta,
    }


def _dir(models_root: str, candidate_id: str) -> str:
    if not isinstance(candidate_id, str) or not CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise KeyError(candidate_id)
    path = os.path.join(models_root, "candidates", candidate_id)
    if not os.path.isdir(path):
        raise KeyError(candidate_id)
    return path


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_candidate(models_root: str, challenger, report: dict) -> tuple[str, str]:
    cid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
    token = secrets.token_urlsafe(24)
    path = os.path.join(models_root, "candidates", cid)
    os.makedirs(path)
    atomic_write_bytes(os.path.join(path, "model.pkl"), pickle.dumps(challenger))
    atomic_write_text(os.path.join(path, "report.json"),
                      json.dumps({**report, "candidate_id": cid}, indent=2, ensure_ascii=False))
    atomic_write_text(os.path.join(path, "status.json"), json.dumps(
        {"status": "pending", "token_sha256": _token_hash(token), "approver": None, "decided_at": None}, indent=2))
    return cid, token


def load_candidate(models_root: str, candidate_id: str) -> dict:
    path = _dir(models_root, candidate_id)
    with open(os.path.join(path, "report.json"), encoding="utf-8") as f:
        report = json.load(f)
    with open(os.path.join(path, "status.json"), encoding="utf-8") as f:
        status = json.load(f)
    return {"report": report, "status": status}


def check_token(models_root: str, candidate_id: str, token: str) -> bool:
    try:
        status = load_candidate(models_root, candidate_id)["status"]
    except KeyError:
        return False
    return bool(token) and hmac.compare_digest(_token_hash(token), status["token_sha256"])


def candidate_model_bytes(models_root: str, candidate_id: str) -> bytes:
    with open(os.path.join(_dir(models_root, candidate_id), "model.pkl"), "rb") as f:
        return f.read()


def mark_candidate(models_root: str, candidate_id: str, status: str, approver: str) -> None:
    if status not in ("approved", "rejected"):
        raise ValueError(status)
    path = _dir(models_root, candidate_id)
    current = load_candidate(models_root, candidate_id)["status"]
    if current["status"] != "pending":
        raise RegistryError(f"candidate {candidate_id} is already {current['status']}")
    current.update(status=status, approver=approver, decided_at=datetime.now(timezone.utc).isoformat())
    atomic_write_text(os.path.join(path, "status.json"), json.dumps(current, indent=2))
