"""Serve champion model on all synthetic scenarios and print summary per scenario."""
import sys, os
sys.path.insert(0, "src")
import pandas as pd
import campaign_pipeline as cp

SCENARIOS = {
    "reactivation":  "data/scenarios/reactivation.csv",
    "new_customers": "data/scenarios/new_customers.csv",
    "low_season":    "data/scenarios/low_season.csv",
    "b2b_mix":       "data/scenarios/b2b_mix.csv",
    "ab_experiment": "data/scenarios/ab_experiment.csv",
}

results = {}
for name, csv in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print('='*60)
    try:
        out = cp.run_layer4(csv, None)
        send = out["should_send"].astype(bool)
        results[name] = out
        print(f"\n>>> SUMMARY: {len(out)} customers | send={send.sum()} ({send.mean():.1%})")
        if send.sum() > 0:
            print(out[send]["recommended_action"].value_counts().head(8).to_string())
        print(f"Uplift groups: {out['uplift_group'].value_counts().to_string()}" if "uplift_group" in out.columns else "")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results[name] = None

# Cross-scenario comparison
print("\n" + "="*60)
print("CROSS-SCENARIO COMPARISON")
print("="*60)
for name, out in results.items():
    if out is None:
        print(f"  {name:<20} ❌ failed")
        continue
    send = out["should_send"].astype(bool)
    top_action = out[send]["recommended_action"].value_counts().index[0] if send.sum() > 0 else "none"
    print(f"  {name:<20} send={send.mean():.1%}  top_action={top_action[:40]}")
