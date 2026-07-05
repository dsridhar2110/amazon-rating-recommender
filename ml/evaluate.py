"""
RecoPulse — evaluation utilities
================================
Honest metrics we regenerate ourselves (nothing is quoted from the old notebook):
  - RMSE / MAE / R^2 for the hybrid
  - Tier mix (how much traffic each strategy serves) + coverage (incl. cold-start)
  - Calibration (predicted vs actual by bucket)
  - Baselines the hybrid must beat: global-mean, user-mean, item-mean
"""
from __future__ import annotations
import time
import numpy as np

from model import clean_product_name, TIER_SVD, TIER_STATS, TIER_CONTENT


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def _r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def batch_predict(model, df, progress_every=10000):
    """Row-by-row hybrid prediction with tier tracking. Returns (preds, tiers)."""
    preds = np.empty(len(df), dtype=float)
    tiers = []
    t0 = time.time()
    users = df["user_id"].values
    names = df["product_name"].values
    for i in range(len(df)):
        preds[i] = model.predict(users[i], names[i])
        tiers.append(model.tier_of(users[i], names[i]))
        if progress_every and i and i % progress_every == 0:
            rate = i / (time.time() - t0)
            print(f"  predicted {i:,}/{len(df):,} ({rate:.0f}/s)", flush=True)
    return preds, tiers


def baselines(train_df, val_df):
    """Simple, defensible baselines computed on the same holdout."""
    gm = float(train_df["rating"].mean())
    user_mean = train_df.groupby("user_id")["rating"].mean().to_dict()

    tr = train_df.copy()
    tr["clean"] = tr["product_name"].apply(clean_product_name)
    item_mean = tr.groupby("clean")["rating"].mean().to_dict()

    y = val_df["rating"].values
    out = {}
    # global mean
    out["global_mean"] = _metrics(y, np.full(len(val_df), gm))
    # user mean (fallback global)
    p_user = val_df["user_id"].map(lambda u: user_mean.get(u, gm)).values
    out["user_mean"] = _metrics(y, p_user)
    # item mean (fallback global)
    clean_val = val_df["product_name"].apply(clean_product_name)
    p_item = clean_val.map(lambda c: item_mean.get(c, gm)).values
    out["item_mean"] = _metrics(y, p_item)
    return out


def _metrics(y, p):
    return {"rmse": _rmse(y, p), "mae": _mae(y, p), "r2": _r2(y, p)}


def full_report(model, train_df, val_df, preds, tiers):
    """Assemble the metrics.json payload."""
    y = val_df["rating"].values
    tiers = np.array(tiers)
    n = len(val_df)

    tier_mix = {t: int(np.sum(tiers == t)) for t in [TIER_SVD, TIER_STATS, TIER_CONTENT]}
    tier_mix_pct = {t: round(100 * c / n, 2) for t, c in tier_mix.items()}

    # per-tier RMSE — where does the model do well / struggle?
    per_tier = {}
    for t in [TIER_SVD, TIER_STATS, TIER_CONTENT]:
        mask = tiers == t
        if mask.sum() > 0:
            per_tier[t] = {"n": int(mask.sum()), **_metrics(y[mask], preds[mask])}

    # calibration: mean actual vs mean predicted, bucketed by predicted rating
    buckets = []
    edges = [1, 2, 3, 4, 4.5, 5.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (preds >= lo) & (preds < hi)
        if m.sum() > 0:
            buckets.append({"bucket": f"[{lo},{hi})", "n": int(m.sum()),
                            "mean_pred": round(float(preds[m].mean()), 3),
                            "mean_actual": round(float(y[m].mean()), 3)})

    return {
        "validation_rows": n,
        "hybrid": _metrics(y, preds),
        "baselines": baselines(train_df, val_df),
        "tier_mix_counts": tier_mix,
        "tier_mix_pct": tier_mix_pct,
        "per_tier_rmse": per_tier,
        "coverage_pct": 100.0,  # every row gets a bounded prediction by construction
        "calibration": buckets,
        "prediction_distribution": {
            "mean": round(float(preds.mean()), 3),
            "std": round(float(preds.std()), 3),
            "min": round(float(preds.min()), 3),
            "max": round(float(preds.max()), 3),
            "pct_below_3": round(float(100 * np.mean(preds < 3.0)), 2),
            "pct_above_4_5": round(float(100 * np.mean(preds > 4.5)), 2),
        },
    }
