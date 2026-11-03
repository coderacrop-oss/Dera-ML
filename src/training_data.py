"""Pooled multi-shop training data: per-shop Layer 4 contexts, a customer-level holdout split, training frames."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from layer4_uplift_margin import Layer4Context, build_layer4_context

STORE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


class TrainingDataError(Exception):
    pass


@dataclass(frozen=True)
class StoreInput:
    store_id: str
    csv_path: str
    labels_path: str | None = None


def parse_store_data(arg: str) -> StoreInput:
    """`<store_id>:<csv>[:<labels>]`"""
    parts = arg.split(":", 2)
    if len(parts) < 2 or not parts[1]:
        raise TrainingDataError(f"expected <store_id>:<csv>[:<labels>], got {arg!r}")
    store_id, csv_path = parts[0], parts[1]
    labels = parts[2] if len(parts) == 3 and parts[2] else None
    if not STORE_ID_RE.fullmatch(store_id):
        raise TrainingDataError(f"invalid store id {store_id!r} (lowercase letters, digits, hyphen; max 63)")
    return StoreInput(store_id, csv_path, labels)


def split_customers(customer_ids: Iterable[str], holdout_frac: float, seed: int) -> tuple[set, set]:
    """Deterministic customer-level split: (train_ids, holdout_ids)."""
    ids = sorted(set(map(str, customer_ids)))
    perm = np.random.RandomState(seed).permutation(len(ids))
    n_hold = int(round(holdout_frac * len(ids)))
    holdout = {ids[i] for i in perm[:n_hold]}
    return set(ids) - holdout, holdout


@dataclass
class PooledData:
    contexts: dict
    train_ids: dict
    holdout_ids: dict
    features: list
    label_mode: str


def build_pooled(inputs: list, holdout_frac: float = 0.2, seed: int = 42) -> PooledData:
    if not inputs:
        raise TrainingDataError("no shop data given")
    seen = set()
    for i in inputs:
        if i.store_id in seen:
            raise TrainingDataError(f"store {i.store_id!r} given more than once")
        seen.add(i.store_id)
    contexts: dict[str, Layer4Context] = {}
    for i in inputs:
        if not os.path.exists(i.csv_path):
            raise TrainingDataError(f"store {i.store_id}: file not found: {i.csv_path}")
        contexts[i.store_id] = build_layer4_context(i.csv_path, i.labels_path)
    modes = {s: c.label_mode for s, c in contexts.items()}
    if len(set(modes.values())) > 1:
        raise TrainingDataError(f"shops mix real and proxy labels ({modes}); give a labels file for all shops or none")
    feats = {s: tuple(c.features) for s, c in contexts.items()}
    if len(set(feats.values())) > 1:
        raise TrainingDataError(f"shops produce different feature lists: {feats}")
    train_ids, holdout_ids = {}, {}
    for s, c in contexts.items():
        train_ids[s], holdout_ids[s] = split_customers(c.cust_agg["customer_id"], holdout_frac, seed)
    if not any(holdout_ids.values()):
        raise TrainingDataError("holdout is empty; give more customers or a larger --holdout-frac")
    first = next(iter(contexts.values()))
    return PooledData(contexts, train_ids, holdout_ids, list(first.features), first.label_mode)


def training_frames(pooled: PooledData):
    """(treated rows, control rows) from **training customers only**, with `customer_key` = store|customer."""
    parts = []
    for s, ctx in pooled.contexts.items():
        rows = ctx.raw_df[ctx.raw_df["customer_id"].astype(str).isin(pooled.train_ids[s])].copy()
        rows["customer_key"] = s + "|" + rows["customer_id"].astype(str)
        rows["store_id"] = s
        n_t, n_c = int((rows["was_treated"] == 1).sum()), int((rows["was_treated"] == 0).sum())
        if n_t == 0 or n_c == 0:
            raise TrainingDataError(f"store {s}: training split has {n_t} treated and {n_c} control rows; "
                                    f"the model needs both (leave this shop out or add promotion labels)")
        parts.append(rows)
    allrows = pd.concat(parts, ignore_index=True)
    return allrows[allrows["was_treated"] == 1], allrows[allrows["was_treated"] == 0]
