"""
RecoPulse — training entrypoint
===============================
Reproducible pipeline (replaces the notebook's main()):
  1. Load real data/train.csv
  2. Split BY USER (no leakage) into train / validation
  3. Fit the hybrid, validate, compute baselines + tier mix  -> honest metrics.json
  4. Retrain on ALL data, serialize the model -> artifacts/model.joblib

Usage:
  python ml/train.py                      # full validation (slower, most honest)
  python ml/train.py --val-sample 60000   # cap validation rows for a fast first table
  python ml/train.py --no-refit           # skip the full-data refit (metrics only)
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

# allow `python ml/train.py` from repo root or from ml/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import HybridPredictor          # noqa: E402
import evaluate as ev                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "train.csv")
ARTIFACTS = os.path.join(HERE, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS, "model.joblib")
METRICS_PATH = os.path.join(ARTIFACTS, "metrics.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-sample", type=int, default=0, help="cap validation rows (0 = all)")
    ap.add_argument("--no-refit", action="store_true", help="skip full-data refit + save")
    ap.add_argument("--split", choices=["by-interaction", "by-user"], default="by-interaction",
                    help="by-interaction mirrors the real test set (all users known, ~9%% cold items); "
                         "by-user is a harder cold-USER stress test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(ARTIFACTS, exist_ok=True)
    print("=" * 70)
    print("RecoPulse — hybrid recommender training")
    print("=" * 70)

    df = pd.read_csv(DATA)
    print(f"Loaded {len(df):,} ratings | {df.user_id.nunique():,} users | "
          f"{df.product_id.nunique():,} products")

    # --- train/validation split ---
    rng = np.random.RandomState(args.seed)
    if args.split == "by-user":
        # Cold-USER stress test: held-out users are 100% unseen (pessimistic for this
        # competition, whose real test set has 0% cold users). Kept as a stress test.
        users = df["user_id"].unique()
        rng.shuffle(users)
        train_users = set(users[: int(len(users) * 0.85)])
        mask = df["user_id"].isin(train_users)
        tr = df[mask].reset_index(drop=True)
        va = df[~mask].reset_index(drop=True)
    else:
        # by-interaction: mirrors the real test set — every user is known (median 219
        # ratings), and holding out random rows naturally leaves ~9% of val items cold
        # (median item has 1 rating). This is the deployment-realistic protocol.
        idx = rng.permutation(len(df))
        cut = int(len(df) * 0.85)
        tr = df.iloc[idx[:cut]].reset_index(drop=True)
        va = df.iloc[idx[cut:]].reset_index(drop=True)
    print(f"Split [{args.split}] -> train {len(tr):,} rows / val {len(va):,} rows")
    cold_users = (~va["user_id"].isin(set(tr["user_id"]))).mean() * 100
    cold_items = (~va["product_id"].isin(set(tr["product_id"]))).mean() * 100
    print(f"Val composition: {cold_users:.1f}% cold users, {cold_items:.1f}% cold items")

    if args.val_sample and args.val_sample < len(va):
        va = va.sample(args.val_sample, random_state=args.seed).reset_index(drop=True)
        print(f"Validation capped to {len(va):,} rows for speed")

    # --- fit on train split, validate ---
    t0 = time.time()
    model = HybridPredictor().fit(tr)
    print(f"Fit (train split) in {time.time() - t0:.1f}s")

    print("Predicting on validation set...")
    t0 = time.time()
    preds, tiers = ev.batch_predict(model, va)
    print(f"Validation predicted in {time.time() - t0:.1f}s")

    report = ev.full_report(model, tr, va, preds, tiers)
    report["split"] = args.split
    report["val_cold_users_pct"] = round(float(cold_users), 2)
    report["val_cold_items_pct"] = round(float(cold_items), 2)
    report["trained_at_unix"] = int(time.time())
    report["dataset"] = {"total_ratings": int(len(df)), "users": int(df.user_id.nunique()),
                         "products": int(df.product_id.nunique())}
    report["hyperparams"] = {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in model.hp.items()}

    _print_summary(report)
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nMetrics written -> {METRICS_PATH}")

    # --- refit on ALL data + serialize ---
    if not args.no_refit:
        print("\nRefitting on ALL data for the served model...")
        t0 = time.time()
        final = HybridPredictor().fit(df)
        final.save(MODEL_PATH)
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print(f"Saved model -> {MODEL_PATH} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")


def _print_summary(r):
    print("\n" + "=" * 70)
    print("HONEST VALIDATION RESULTS  (regenerated, not quoted from the notebook)")
    print("=" * 70)
    h = r["hybrid"]
    print(f"HYBRID     RMSE {h['rmse']:.4f} | MAE {h['mae']:.4f} | R2 {h['r2']:.4f}   "
          f"(n={r['validation_rows']:,})")
    print("-" * 70)
    print("Baselines it must beat:")
    for name, m in r["baselines"].items():
        delta = m["rmse"] - h["rmse"]
        print(f"  {name:12s} RMSE {m['rmse']:.4f} | MAE {m['mae']:.4f} | R2 {m['r2']:.4f}"
              f"   (hybrid better by {delta:+.4f})")
    print("-" * 70)
    print("Tier mix (share of predictions):")
    for t, pct in r["tier_mix_pct"].items():
        pt = r["per_tier_rmse"].get(t, {})
        extra = f" | tier RMSE {pt['rmse']:.4f}" if "rmse" in pt else ""
        print(f"  {t:20s} {pct:5.1f}%{extra}")


if __name__ == "__main__":
    main()
