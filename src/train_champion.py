"""Training command: build a Challenger from pooled shop data and compare it with the champion on held-out customers.

It never changes the champion. A person approves the candidate on the page whose link is printed at the end.

  python src/train_champion.py --store-data shop-a:data/a.csv[:labels.csv] --store-data shop-b:data/b.csv
  python src/train_champion.py --import-legacy        # one-time: adopt the existing champion file as v1
  python src/train_champion.py --rollback v1
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager

import pandas as pd

from business_config import REGISTRY_PATH
from candidates import create_candidate, evaluate_candidate
from data_quality import DataQualityError, print_dq_report, validate_training_data
from layer4_uplift_margin import _train_xlearner
from model_registry import ModelRegistry, RegistryError
from pipeline_logger import log
from training_data import StoreInput, TrainingDataError, build_pooled, parse_store_data, training_frames


def _load_champion(registry: ModelRegistry):
    """(model, ref) or (None, None) when there is no champion at all. A champion file without a manifest is an error."""
    if registry.current() is None and not os.path.exists(registry.champion_path):
        return None, None
    return registry.load_current()


def _training_meta(inputs, pooled, holdout_frac, seed) -> dict:
    ts = pd.concat([pd.to_datetime(c.raw_df["timestamp"], errors="coerce") for c in pooled.contexts.values()])
    return {"files": [{"store_id": i.store_id, "csv": os.path.basename(i.csv_path),
                       "labels": os.path.basename(i.labels_path) if i.labels_path else None} for i in inputs],
            "date_range": [str(ts.min()), str(ts.max())], "seed": seed, "holdout_frac": holdout_frac}


@contextmanager
def _scratch_cwd():
    """Layers 1-3 write intermediate files under ./data/processed (the serving shop's folder): do that in a throwaway dir."""
    old, tmp = os.getcwd(), tempfile.mkdtemp(prefix="train-champion-")
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)


def _compatible(champion, features) -> bool:
    names = getattr(champion, "feature_names_", None)
    return list(names) == list(features) if names is not None else champion.n_features_ == len(features)


def _train(args) -> int:
    inputs = [StoreInput(i.store_id, os.path.abspath(i.csv_path),
                         os.path.abspath(i.labels_path) if i.labels_path else None)
              for i in (parse_store_data(a) for a in args.store_data)]
    args.models_root = os.path.abspath(args.models_root)
    registry = ModelRegistry(args.models_root)
    champion, ref = _load_champion(registry)
    with _scratch_cwd():
        pooled = build_pooled(inputs, args.holdout_frac, args.seed)

        # ── P0: Data quality validation ───────────────────────────────────────
        print("\n🔍 [DQ] Validating training data quality...")
        dq_report = validate_training_data(pooled)   # raises DataQualityError if blocking
        print_dq_report(dq_report)

        treat, ctrl = training_frames(pooled)
        challenger = _train_xlearner(treat, ctrl, pooled.features)

        # ── P0: Compute customer-level feature baseline for drift detection ────
        try:
            from drift_detector import compute_baseline
            train_custs = []
            for s, ctx in pooled.contexts.items():
                mask = ctx.cust_agg["customer_id"].astype(str).isin(pooled.train_ids[s])
                train_custs.append(ctx.cust_agg[mask][[f for f in pooled.features if f in ctx.cust_agg.columns]])
            all_train_cust = pd.concat(train_custs, ignore_index=True)
            challenger.feature_baseline_stats = compute_baseline(all_train_cust, pooled.features)
            print(f"📐 [Drift] Feature baseline computed ({len(pooled.features)} features, customer-level)")
        except Exception as _e:
            print(f"⚠️  [Drift] Could not compute baseline: {_e}")

        incompatible = champion is not None and not _compatible(champion, pooled.features)
        report = evaluate_candidate(pooled, challenger, None if incompatible else champion, ref,
                                    _training_meta(inputs, pooled, args.holdout_frac, args.seed))

    if incompatible:
        report["gate"] = {"passed": False, "growth_pct": None,
                          "reason": "Champion เดิมใช้ฟีเจอร์ชุดอื่น จึงเทียบกับโมเดลใหม่ไม่ได้ — ให้ผู้อนุมัติพิจารณาเอง"}

    # ── P0: Print uplift quality metrics ─────────────────────────────────────
    um = report["challenger"].get("uplift_metrics", {})
    if um and "error" not in um and "note" not in um:
        print(f"\n📊 [Model Quality] Holdout uplift metrics (n={um.get('n_holdout','?')}):")
        print(f"   Qini coeff : {um.get('qini', 'N/A')}  ({um.get('interpretation','?')})")
        print(f"   AUUC       : {um.get('auuc', 'N/A')}  (avg incremental conversion rate)")
        print(f"   ATE pred   : {um.get('ate_pred', 'N/A')}  |  actual: {um.get('ate_actual', 'N/A')}")
        print(f"   Baseline ECE: {um.get('baseline_ece', 'N/A')}  (calibration error of outcome model)")
        log("L4", "train_uplift_metrics", **{k: v for k, v in um.items()
                                              if k not in ("interpretation", "note")})
    elif um.get("note"):
        print(f"\n📊 [Model Quality] {um['note']}")

    # ── P1: Print SHAP feature importance ────────────────────────────────────
    shap_imp = report["challenger"].get("shap_importance", {})
    if shap_imp and "error" not in shap_imp:
        print(f"\n🔍 [SHAP] Top-{len(shap_imp)} features driving treatment heterogeneity (tau0):")
        for feat, imp in shap_imp.items():
            print(f"   {feat:<35} {imp:.6f}")
    elif shap_imp.get("error"):
        print(f"\n🔍 [SHAP] Skipped: {shap_imp['error']}")

    cid, token = create_candidate(args.models_root, challenger, report)
    log("L4", "train_candidate_created", candidate_id=cid, stores=[i.store_id for i in inputs],
        compared_against=report["compared_against_version"], gate_passed=report["gate"]["passed"])
    base = (os.environ.get("CRM_API_PUBLIC_URL") or "http://localhost:8000").rstrip("/")
    print(f"\n✅ สร้างโมเดลผู้สมัคร {cid} แล้ว (ยังไม่ได้เปลี่ยน Champion)")
    print(f"   ผลเทียบ: {report['gate']['reason']}")
    print(f"👉 เปิดหน้าเปรียบเทียบเพื่ออนุมัติ: {base}/v1/models/candidates/{cid}?token={token}")
    print("   (ต้องเปิด CRM API ไว้ จึงจะเปิดลิงก์ได้)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store-data", action="append", default=[], metavar="STORE_ID:CSV[:LABELS]")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models-root", default=os.path.dirname(REGISTRY_PATH))
    p.add_argument("--import-legacy", action="store_true")
    p.add_argument("--rollback", metavar="VERSION")
    args = p.parse_args(argv)
    modes = [bool(args.store_data), args.import_legacy, bool(args.rollback)]
    if sum(modes) != 1:
        p.error("choose exactly one of: --store-data (train), --import-legacy, --rollback VERSION")
    try:
        if args.import_legacy:
            v = ModelRegistry(args.models_root).import_legacy()
            log("L4", "champion_imported_legacy", version=v)
            print(f"✅ นำ Champion เดิมเข้าทะเบียนเป็น {v}")
            return 0
        if args.rollback:
            ModelRegistry(args.models_root).rollback(args.rollback)
            log("L4", "champion_rolled_back", version=args.rollback)
            print(f"✅ ย้อนกลับไปใช้ Champion {args.rollback}")
            return 0
        return _train(args)
    except (TrainingDataError, RegistryError, DataQualityError) as exc:
        print(f"❌ {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
