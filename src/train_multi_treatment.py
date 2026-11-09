"""Multi-treatment uplift training.

Requires labels CSV with column `treatment_type` (discount/points_vip/bundle/reactivation).
Each treatment type trains its own X-learner: treatment = "received this type" vs "received any other".

Usage:
  python src/train_multi_treatment.py \\
    --store-data shop:data/transactions.csv:data/treatment_labels.csv

Labels CSV columns required:
  customer_id, treatment_type, converted
  (was_treated is derived: 1 if treatment_type == current_arm, 0 otherwise)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import shutil
from contextlib import contextmanager

import pandas as pd

from business_config import REGISTRY_PATH
from candidates import create_candidate, evaluate_candidate
from data_quality import DataQualityError, print_dq_report, validate_training_data
from model_registry import ModelRegistry, RegistryError
from pipeline_logger import log
from training_data import StoreInput, TrainingDataError, build_pooled, parse_store_data, training_frames
from uplift_engine import MultiTreatmentUpliftModel, _train_xlearner, _with_names

TREATMENT_TYPES = MultiTreatmentUpliftModel.TREATMENT_TYPES


@contextmanager
def _scratch_cwd():
    old, tmp = os.getcwd(), tempfile.mkdtemp(prefix="train-multi-treatment-")
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)


def _train_multi(args) -> int:
    inputs = [StoreInput(i.store_id, os.path.abspath(i.csv_path),
                         os.path.abspath(i.labels_path) if i.labels_path else None)
              for i in (parse_store_data(a) for a in args.store_data)]

    if not any(i.labels_path for i in inputs):
        print("❌ ต้องระบุ labels CSV ที่มีคอลัมน์ treatment_type")
        print("   format: --store-data STORE:transactions.csv:treatment_labels.csv")
        return 1

    args.models_root = os.path.abspath(args.models_root)
    registry = ModelRegistry(args.models_root)

    with _scratch_cwd():
        pooled = build_pooled(inputs, args.holdout_frac, args.seed)

        # Validate data quality
        print("\n🔍 [DQ] Validating training data...")
        try:
            dq_report = validate_training_data(pooled)
            print_dq_report(dq_report)
        except DataQualityError as exc:
            print(f"❌ {exc}")
            return 1

        treat_full, ctrl_full = training_frames(pooled)
        all_data = pd.concat([treat_full, ctrl_full], ignore_index=True)

        # Check treatment_type column exists
        if 'treatment_type' not in all_data.columns:
            print("❌ labels CSV ไม่มีคอลัมน์ treatment_type")
            print(f"   columns ที่มี: {list(all_data.columns)}")
            return 1

        available_types = [t for t in TREATMENT_TYPES if (all_data['treatment_type'] == t).sum() >= 50]
        if not available_types:
            print(f"❌ ไม่มี treatment type ไหนมีข้อมูลเพียงพอ (ต้องการ ≥ 50 ราย)")
            print(f"   treatment_type distribution:\n{all_data['treatment_type'].value_counts()}")
            return 1

        print(f"\n🎯 Training {len(available_types)} treatment models: {available_types}")
        models = {}
        for ttype in available_types:
            print(f"\n{'='*50}")
            print(f"🔬 Training: {ttype}")
            # treatment = received this type, control = received something else or nothing
            arm_treat = all_data[all_data['treatment_type'] == ttype].copy()
            arm_ctrl  = all_data[all_data['treatment_type'] != ttype].copy()
            arm_treat['was_treated'] = 1
            arm_ctrl['was_treated']  = 0
            model = _train_xlearner(arm_treat, arm_ctrl, pooled.features)
            models[ttype] = model
            log("L4", f"multi_treatment_trained_{ttype}",
                n_treat=len(arm_treat), n_ctrl=len(arm_ctrl))

        multi_model = MultiTreatmentUpliftModel(models)
        _with_names(multi_model, pooled.features)

        # Evaluate vs champion
        champion, ref = (None, None)
        try:
            champion, ref = registry.load_current()
        except RegistryError:
            pass

        report = evaluate_candidate(pooled, multi_model, champion, ref,
                                    {"type": "multi_treatment", "treatment_types": available_types,
                                     "seed": args.seed, "holdout_frac": args.holdout_frac})

    cid, token = create_candidate(args.models_root, multi_model, report)
    log("L4", "multi_treatment_candidate_created", candidate_id=cid,
        treatment_types=available_types, gate_passed=report["gate"]["passed"])
    base = (os.environ.get("CRM_API_PUBLIC_URL") or "http://localhost:8000").rstrip("/")
    print(f"\n✅ สร้าง Multi-Treatment model candidate {cid}")
    print(f"   Treatment types: {available_types}")
    print(f"   ผลเทียบ: {report['gate']['reason']}")
    print(f"👉 อนุมัติที่: {base}/v1/models/candidates/{cid}?token={token}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store-data", action="append", default=[], metavar="STORE_ID:CSV:LABELS")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models-root", default=os.path.dirname(REGISTRY_PATH))
    args = p.parse_args(argv)
    if not args.store_data:
        p.error("ต้องระบุ --store-data อย่างน้อย 1 ร้าน")
    try:
        return _train_multi(args)
    except (TrainingDataError, RegistryError) as exc:
        print(f"❌ {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
