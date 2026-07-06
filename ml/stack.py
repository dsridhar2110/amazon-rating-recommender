"""
RecoPulse — Stage 1 experiment: stacked ensemble
=================================================
Pushes past the shipped hybrid (RMSE ~0.81) by stacking diverse base models.
Honest, apples-to-apples: same by-interaction split (seed 42), same 50k holdout.

Base models (each gets LEAK-FREE out-of-fold predictions via 3-fold on train):
  - SVD (tuned)            collaborative, matrix factorization
  - SVD++                  MF + implicit "what they rated at all" signal
  - KNNBaseline (user)     neighbourhood model with baselines
  - NMF                    non-negative matrix factorization
  - LightGBM               gradient-boosted trees on engineered stats/content features

Meta-learner (LightGBM) blends the 5 base predictions + item/user counts, so it
LEARNS the routing the hand-built hybrid does by hand.

Honesty guardrails baked in:
  - votes/helpful_votes used ONLY as an aggregate item-level signal (never per-row:
    they don't exist before a rating is made — that would be leakage).
  - LightGBM features for OOF are recomputed from each fold's own train portion.

Usage:  python ml/stack.py           (full run, ~15-30 min)
        python ml/stack.py --smoke   (tiny subset, ~1 min, to check it runs)
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import clean_product_name, HybridPredictor  # noqa: E402

from surprise import Dataset, Reader, SVD, SVDpp, KNNBaseline, NMF  # noqa: E402
import lightgbm as lgb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "train.csv")
OUT = os.path.join(HERE, "artifacts", "stack_leaderboard.json")
SEED = 42
N_FOLDS = 3


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


# --------------------------------------------------------------------------- #
# Feature engineering (train-only stats; leak-free per fold)
# --------------------------------------------------------------------------- #
def compute_stats(df):
    gm = float(df["rating"].mean())
    u = df.groupby("user_id")["rating"].agg(["mean", "std", "count"])
    i = df.groupby("clean")["rating"].agg(["mean", "std", "count"])
    # aggregate item helpfulness (item-level, NOT per-row → not leakage)
    if {"votes", "helpful_votes"}.issubset(df.columns):
        h = df.groupby("clean").agg(v=("votes", "sum"), hv=("helpful_votes", "sum"))
        help_ratio = ((h["hv"] + 1) / (h["v"] + 2)).to_dict()
    else:
        help_ratio = {}
    return {"gm": gm, "u": u.to_dict("index"), "i": i.to_dict("index"), "help": help_ratio}


def make_features(df, s):
    gm = s["gm"]; U = s["u"]; I = s["i"]; H = s["help"]
    rows = []
    for uid, c in zip(df["user_id"].values, df["clean"].values):
        u = U.get(uid); it = I.get(c)
        rows.append((
            u["mean"] if u else gm,
            u["count"] if u else 0,
            u["std"] if u and u["std"] == u["std"] else 0.0,
            it["mean"] if it else np.nan,
            it["count"] if it else 0,
            it["std"] if it and it["std"] == it["std"] else np.nan,
            H.get(c, np.nan),
            len(str(c).split()),
            gm,
        ))
    cols = ["u_mean", "u_count", "u_std", "i_mean", "i_count", "i_std", "i_help", "name_words", "gmean"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Surprise helpers
# --------------------------------------------------------------------------- #
def surprise_fit_predict(algo_factory, tr_df, pred_df):
    reader = Reader(rating_scale=(1, 5))
    ts = Dataset.load_from_df(tr_df[["user_id", "clean", "rating"]], reader).build_full_trainset()
    algo = algo_factory()
    algo.fit(ts)
    return np.array([algo.predict(u, c).est for u, c in zip(pred_df["user_id"].values, pred_df["clean"].values)])


def oof_and_holdout(name, algo_factory, train, holdout, folds):
    """Return (oof_preds_on_train, holdout_preds). Leak-free OOF via K folds."""
    t0 = time.time()
    oof = np.zeros(len(train))
    for k, (tr_idx, va_idx) in enumerate(folds):
        oof[va_idx] = surprise_fit_predict(algo_factory, train.iloc[tr_idx], train.iloc[va_idx])
    hold = surprise_fit_predict(algo_factory, train, holdout)
    print(f"  [{name}] oof+holdout in {time.time()-t0:.0f}s | holdout RMSE {rmse(holdout['rating'], hold):.4f}", flush=True)
    return oof, hold


def lgb_oof_and_holdout(train, holdout, folds):
    """LightGBM on engineered features — stats recomputed per fold (leak-free)."""
    t0 = time.time()
    oof = np.zeros(len(train))
    params = dict(objective="regression", metric="rmse", num_leaves=63, learning_rate=0.05,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED)
    for tr_idx, va_idx in folds:
        tf, vf = train.iloc[tr_idx], train.iloc[va_idx]
        s = compute_stats(tf)
        Xt, Xv = make_features(tf, s), make_features(vf, s)
        m = lgb.train(params, lgb.Dataset(Xt, tf["rating"].values), num_boost_round=300)
        oof[va_idx] = np.clip(m.predict(Xv), 1, 5)   # ratings are bounded
    s = compute_stats(train)
    m = lgb.train(params, lgb.Dataset(make_features(train, s), train["rating"].values), num_boost_round=300)
    hold = np.clip(m.predict(make_features(holdout, s)), 1, 5)
    print(f"  [LightGBM] oof+holdout in {time.time()-t0:.0f}s | holdout RMSE {rmse(holdout['rating'], hold):.4f}", flush=True)
    return oof, hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    df["clean"] = df["product_name"].apply(clean_product_name)
    if args.smoke:
        df = df.sample(30000, random_state=SEED).reset_index(drop=True)

    # reproduce the shipped by-interaction split + 50k holdout sample (seed 42)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.85)
    train = df.iloc[idx[:cut]].reset_index(drop=True)
    holdout = df.iloc[idx[cut:]].reset_index(drop=True)
    if not args.smoke and len(holdout) > 50000:
        holdout = holdout.sample(50000, random_state=SEED).reset_index(drop=True)
    print(f"train {len(train):,} | holdout {len(holdout):,}", flush=True)

    # fold indices on train (shared across base models)
    perm = np.random.RandomState(SEED).permutation(len(train))
    folds = []
    for k in range(N_FOLDS):
        va = perm[k::N_FOLDS]
        tr = np.setdiff1d(perm, va, assume_unique=False)
        folds.append((tr, va))

    y_hold = holdout["rating"].values
    board = {}          # name -> holdout RMSE
    oof_mat, hold_mat, base_names = [], [], []

    ep = 8 if args.smoke else 30
    # NOTE: SVD++ was tried but dropped — its implicit-feedback pass is ~40x slower
    # here for negligible gain over SVD. Instead we add a 2nd SVD config for cheap
    # diversity (more factors, lighter regularization) so the stack still sees variety.
    base_specs = [
        ("SVD", lambda: SVD(n_factors=85, reg_all=0.02, lr_all=0.01, n_epochs=ep, random_state=SEED)),
        ("SVD-wide", lambda: SVD(n_factors=160, reg_all=0.05, lr_all=0.008, n_epochs=ep, random_state=SEED)),
        ("KNNBaseline", lambda: KNNBaseline(k=40, sim_options={"user_based": True, "name": "pearson_baseline"}, verbose=False)),
        ("NMF", lambda: NMF(n_factors=30, n_epochs=ep, random_state=SEED)),
    ]
    for name, fac in base_specs:
        oof, hold = oof_and_holdout(name, fac, train, holdout, folds)
        board[name] = rmse(y_hold, hold)
        oof_mat.append(oof); hold_mat.append(hold); base_names.append(name)

    # LightGBM base
    oof, hold = lgb_oof_and_holdout(train, holdout, folds)
    board["LightGBM"] = rmse(y_hold, hold)
    oof_mat.append(oof); hold_mat.append(hold); base_names.append("LightGBM")

    # --- meta features: base preds + item/user counts (lets meta learn routing) ---
    s_full = compute_stats(train)
    feat_tr = make_features(train, s_full)[["i_count", "u_count"]].values
    feat_ho = make_features(holdout, s_full)[["i_count", "u_count"]].values
    X_oof = np.column_stack(oof_mat + [feat_tr[:, 0], feat_tr[:, 1]])
    X_hold = np.column_stack(hold_mat + [feat_ho[:, 0], feat_ho[:, 1]])
    y_tr = train["rating"].values

    # simple average of base models (honest cheap baseline)
    board["Average (bases)"] = rmse(y_hold, np.mean(hold_mat, axis=0))

    # meta-learner: LightGBM
    mparams = dict(objective="regression", metric="rmse", num_leaves=31, learning_rate=0.05, verbose=-1, seed=SEED)
    meta = lgb.train(mparams, lgb.Dataset(X_oof, y_tr), num_boost_round=200)
    board["STACKED (meta-LGBM)"] = rmse(y_hold, np.clip(meta.predict(X_hold), 1, 5))

    # reference rows: shipped hybrid + lazy baselines
    print("  [reference] fitting shipped hybrid…", flush=True)
    hyb = HybridPredictor().fit(train.rename(columns={}), verbose=False)
    hpred = np.array([hyb.predict(u, n) for u, n in zip(holdout["user_id"].values, holdout["product_name"].values)])
    board["— shipped hybrid"] = rmse(y_hold, hpred)
    board["— baseline: global mean"] = rmse(y_hold, np.full(len(holdout), train["rating"].mean()))

    # --- leaderboard ---
    print("\n" + "=" * 56 + "\nSTAGE 1 LEADERBOARD  (holdout RMSE, lower is better)\n" + "=" * 56)
    for name, r in sorted(board.items(), key=lambda kv: kv[1]):
        star = "  <<< best" if r == min(board.values()) else ""
        print(f"  {name:28s} {r:.4f}{star}")

    result = {"holdout_rows": len(holdout), "n_folds": N_FOLDS, "smoke": args.smoke,
              "leaderboard": board, "base_models": base_names}
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
