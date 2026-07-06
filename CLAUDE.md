# RecoPulse — Project Context (CLAUDE.md)

> Single source of truth for this project. Read this first.
> Built by **Deekshita Sridhar** ([`dsridhar2110`](https://github.com/dsridhar2110)) as an
> interview case study for the **Siemens Healthineers Data Scientist** role (customer-service
> predictive analytics). Expands the FIT5212 `amazon-rating-recommender` notebook into a
> deployable, explainable, monitored full-stack ML product.

---

## 0. What this is (in one paragraph)

**RecoPulse** is an explainable hybrid recommendation engine that predicts how a user will
rate an item (1–5) — even for items almost no one has rated before — and serves that
prediction through a clickable web product with a live **prediction console** and a
**model-monitoring cockpit**. The hard problem it solves is the *long tail*: in the real
data, the median product has been rated exactly **once**, and ~**9%** of items at prediction
time have never been seen at all. A naïve collaborative-filtering model collapses on that tail;
RecoPulse routes every prediction through the right strategy (collaborative → statistical →
content-similarity) and *tells you which one fired and why*. The same machine that ranks
rarely-seen Amazon titles is the machine a service organisation needs to rank rarely-seen
**spare parts, error codes, and knowledge-base articles** and to predict **service
satisfaction** — which is why this is the right case study for a Healthineers service-analytics role.

## 1. Why this exists (objective + locked decisions)

**Objective:** give Deekshita one project she can screen-share and defend end-to-end to three
audiences — the **technical panel** (she defends the ML), the **business stakeholder** (clear
use case + measurable success criteria), and a **CXO** (one-screen value story). It must be
something she can *click, deploy, and monitor*, not a notebook.

**Locked decisions (do not relitigate mid-build):**

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | **Domain framing** | **Dual** — real Amazon engine as the honest core, plus a clearly-labelled "service-analytics lens" mapping the same techniques to Siemens use cases | Honest (real data, real metrics) *and* relevant to the JD. Avoids the defensibility risk of fabricated healthcare data. |
| 2 | **Scope of v1** | **API + prediction UI + monitoring cockpit** | Directly hits the JD's "develop… deployment, monitoring" and "creating dashboards" language. |
| 3 | **Hosting & stack** | **Next.js on Vercel** (frontend) + **Python FastAPI** model service | Matches the fleet-pulse pattern and the workspace's Vercel tooling; fast to a public URL. |
| 4 | **Data** | **Real Kaggle FIT5212 data** to train (`train.csv`/`test.csv`, already in `data/`), tiny committed **sample** so the app clones-and-runs | Real metrics for defensibility; reproducible demo without committing 40 MB. |

**Open questions (for the coordinator / Deekshita):**
- **Serving packaging** — `scikit-surprise` is a heavy C-extension. Recommended resolution:
  export the *learned* SVD factors + stats to a **numpy-only** inference module so the API is
  light and deployable anywhere (incl. serverless). Confirm during Phase 2.
- **Model-service host** — Vercel Python functions vs. a small container (Render/Railway/HF).
  Decide once artifact size is known (Phase 2).
- **Repo strategy** — expand this repo in place, or fork to a clean `recopulse` repo under
  `dsridhar2110` for a tidy shopfront? (Recommend a clean repo; keep `code.ipynb` as provenance.)

## 2. The business problem

**Amazon framing (the real one):** A catalogue of 200k+ items, most rated once or never.
Predict the rating a given user would give a given item so you can rank, recommend, and flag —
*without* falling apart on the cold-start tail.

**Service-analytics lens (the JD bridge — clearly labelled as an analogy, not fabricated data):**
- **User → service engineer / customer site.** **Item → spare part / error code / KB article / service action.**
- **Predicted rating → predicted relevance / priority / satisfaction (CSAT)** of that action for that site.
- **The long tail → the real service catalogue:** thousands of rare parts and error signatures
  the model has barely seen. Cold-start handling is not academic here — it is the whole job.
- **Unstructured text (item names → machine-log / notification text)** is handled by the same
  TF-IDF/NLP tier, mapping straight to the JD's "Machine Logs… Unstructured data, NLP".

**Who has the pain / what it costs:** service operations that can't prioritise the long tail of
rare-but-critical events waste engineer time and miss preventable dissatisfaction. Measurable
success = lower rating-prediction error (RMSE/MAE) and **100% coverage including cold-start**.

## 3. Repository layout (target)

```
amazon-rating-recommender/            (may be renamed → recopulse)
├── CLAUDE.md                 ← this file (single source of truth)
├── README.md                 ← refreshed at case-study stage
├── code.ipynb                ← original FIT5212 notebook, KEPT as provenance
├── Final_predictions.csv     ← original Kaggle submission, KEPT as provenance
├── data/
│   ├── train.csv, test.csv   ← real data (GITIGNORED, 40 MB / 11 MB)
│   └── sample_train.csv …     ← tiny committed sample for clone-and-run
├── ml/                       ← Python: training + serving
│   ├── train.py              ← notebook refactored into a script → writes artifacts
│   ├── model.py              ← HybridPredictor (numpy-only inference at serve time)
│   ├── evaluate.py           ← RMSE / MAE / R² + tier-mix + coverage report
│   ├── artifacts/            ← serialized model + metrics.json (gitignored / LFS)
│   └── api.py                ← FastAPI: /predict, /metrics, /health, /explain
├── web/                      ← Next.js app on Vercel
│   └── app/                  ← prediction console + monitoring cockpit
├── requirements.txt          ← Python deps (training + serving)
└── .gitignore
```

## 4. Architecture & data flow

```
        TRAIN (offline, ml/train.py)                 SERVE (ml/api.py, FastAPI)
  data/train.csv ─► clean names ─► calc user/product ─► artifacts/ ─► load numpy-only ─► /predict
                     (regex)        stats + SVD +         (model      HybridPredictor      │
                                    TF-IDF matrix          .pkl/.npz) routes to a tier     ▼
                                                                                      web/ (Next.js, Vercel)
  evaluate.py ─► metrics.json ──────────────────────────────────────────────────►  ├─ Prediction console
   (RMSE/MAE/R², tier mix, coverage, calibration)                                   └─ Monitoring cockpit
```

**Prediction router (the defensible core — 3 tiers):**
1. **Case 1 — Enhanced SVD** (collaborative): item has enough ratings → matrix-factorisation
   prediction, optionally quality/helpfulness-adjusted. (~85% of predictions in the original run.)
2. **Case 2 — Hybrid statistics**: item seen but thin → confidence-weighted blend of user bias,
   product bias, and global mean. (~7%.)
3. **Case 3 — Content similarity** (cold-start): item unseen → TF-IDF over item names, borrow
   ratings from the most similar known items. (~8%.) **This is the tier that maps to NLP on logs.**

Every response returns **which tier fired + the human-readable reason** → this is what makes the
product *explainable* to a non-technical stakeholder.

## 5. Data — touchpoints & honest facts

Computed directly from the real `train.csv` (745,889 ratings) / `test.csv` (223,553 cases):

- **Schema (train):** `user_id, product_id, product_name, rating, votes, helpful_votes, ID`.
  Test drops `rating` (the target) and the vote columns.
- **Scale:** 2,000 users · **201,325 products** · 745,889 ratings. Items are Amazon **movie/DVD** titles.
- **The long tail:** **median ratings per product = 1**; median per user = 219. This is the whole story.
- **Cold-start at test time:** **8.9% of products unseen**, **0% of users unseen**.
- **Rating skew:** mean **4.24**, and **55.8% are 5-star** (1:3.9% 2:4.8% 3:10.7% 4:24.8% 5:55.8%).
  → talk about calibration and why RMSE (not accuracy) is the right metric.
- **Cleanliness:** zero missing values across all columns.
- `votes` / `helpful_votes` → the "quality/helpfulness" signal the SVD tier can lean on.

**Design decisions:** split train/validation **by user** (not by row) to avoid leakage; clean
product names with regex before grouping; treat the tail as the design driver, not an afterthought.

## 6. Model / analysis — spec & honest results

- **Metric of record:** RMSE (primary), MAE, R². Original notebook targeted **sub-0.87 RMSE**.
- **⚠️ Regenerate, don't trust:** all metrics in the old README (85/7/8 tier split, RMSE, distribution)
  came from the academic run. `ml/evaluate.py` recomputes them on our pipeline; we quote only what we reproduce.

### Honest regenerated results (Phase 1, 50k held-out ratings, seed 42)

**The evaluation-design story (this is the strongest interview point):** how you split decides
the number. This competition's real test set has **0% cold users, ~9% cold items**, so the
deployment-realistic protocol is **by-interaction**, not by-user.

| Split | What it simulates | Hybrid RMSE | MAE | R² |
|---|---|---|---|---|
| **by-interaction** (headline) | Real deployment — every user known, ~16% val items cold | **0.8105** | 0.529 | 0.429 |
| by-user (stress test) | A 100%-cold-user world that never occurs here | 1.0800 | 0.835 | 0.053 |

The **0.27 RMSE gap** = the measurable value of knowing a user's history. Business line:
"a returning customer is predicted well; a brand-new one is our weakest case — so capturing
history early is the highest-leverage data investment."

**Hybrid vs baselines (by-interaction):** global-mean 1.073 · user-mean 0.981 · item-mean 1.054
→ hybrid **0.811** beats all by 0.17–0.26 RMSE.

**Tier mix & per-tier RMSE (by-interaction):**
- `ENHANCED_SVD` 77.2% @ RMSE **0.754** (collaborative core, strongest)
- `HYBRID_STATISTICS` 9.2% @ 0.947 (thin items)
- `CONTENT_SIMILARITY` 13.6% @ 1.000 (cold-start tail — hardest, exactly as expected)

100% coverage; served model refit on all 745k ratings → `ml/artifacts/model.joblib` (70 MB).
- **Hyperparameters (starting point, from the notebook):** SVD 85 factors, lr 0.01, reg 0.01,
  40 epochs; TF-IDF 1,500 features; confidence divisor 5; user/product weights (0.65, 0.35);
  similarity thresholds 0.2 (case 2) / 0.3 (case 3).
- **Baselines to beat (for the interview):** global-mean predictor, user-mean, item-mean, pure-SVD.
  Showing the hybrid beating each baseline on RMSE is the single most defensible slide.

## 7. The product (what she clicks and shows)

1. **Prediction console** — pick/enter a user + item → predicted rating, **which tier fired**,
   a plain-English "why", and (for cold-start) the similar items it borrowed from. This is the
   "explain your work to people who don't understand the mechanics" JD line, made literal.
2. **Monitoring cockpit** — RMSE/MAE/R² headline tiles, **tier-mix** donut (how much of traffic
   each strategy serves), **coverage** (incl. cold-start), **calibration** (predicted vs actual),
   and a rating-distribution view. This is the "deployment, monitoring" + "dashboards" JD line.
3. **A one-screen "value story"** framed for a CXO: the tail problem, the coverage win, the transfer.

## 8. The plan (touchpoint per phase)

- **Phase 0 — Kickoff & lock (DONE):** dig into repo, verify data, lock the 4 decisions, scaffold this CLAUDE.md.
- **Phase 1 — Reproducible ML core:** refactor notebook → `ml/train.py` + `ml/model.py`; run on real
  data; `ml/evaluate.py` produces honest `metrics.json` (hybrid vs baselines). *Touchpoint: real RMSE table.*
- **Phase 2 — Serving:** export numpy-only artifacts; `ml/api.py` FastAPI with `/predict`,
  `/explain`, `/metrics`, `/health`; resolve packaging/host open questions. *Touchpoint: live API call.*
- **Phase 3 — Product:** Next.js prediction console + monitoring cockpit on Vercel. *Touchpoint: public URL.*
- **Phase 4 — Case study:** refresh README, service-lens narrative, portfolio card (`case-study` skill),
  interview Q&A prep (defend every choice). *Touchpoint: screen-share rehearsal.*

## 9. How to run (filled in as we build)

```bash
# ML (Phase 1+)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python ml/train.py            # trains on data/train.csv → ml/artifacts/
python ml/evaluate.py         # writes ml/artifacts/metrics.json
uvicorn ml.api:app --reload   # serves the model (Phase 2)
# Web (Phase 3)
cd web && npm install && npm run dev
```

## 10. Key facts & guardrails

- **Honesty is the moat.** The Amazon engine is real; the service lens is an explicitly-labelled
  *analogy*, never presented as real healthcare data. Every quoted metric is regenerated by us.
- **Git:** public repo under **`dsridhar2110`**. Never commit `data/*.csv`, `.env`, keys, or large artifacts.
- **Keep provenance:** `code.ipynb` and `Final_predictions.csv` stay as evidence of the original work.
- **Defend every choice:** why hybrid over pure SVD (the tail), why RMSE (skew), why by-user split
  (leakage), why TF-IDF for cold-start (NLP on sparse text), why numpy-only serving (deployability).

## 11. Current status & next steps

- ✅ Phase 0 complete: repo understood, real data verified & profiled, 4 decisions locked, CLAUDE.md scaffolded.
- ✅ **Phase 1 complete:** notebook refactored into `ml/{model,evaluate,train}.py`; venv + pinned deps
  (incl. scikit-surprise) installed; honest metrics regenerated (**RMSE 0.81 deployment-realistic**,
  beats all baselines); served model saved; committed clone-and-run sample.
- ✅ **Phase 2 complete:** FastAPI service (`ml/api.py`) with `/predict`, `/explain`, `/metrics`,
  `/users`, `/health`; verified serving the 70 MB model locally (known + cold-start paths).
- ✅ **Phase 3 complete:** Next.js 14 product in `web/` on Vercel-ready App Router. Three routes —
  **Prediction Console** (`/`, live API + offline-snapshot fallback, cold-start explainer),
  **Monitoring Cockpit** (`/monitor`, reads real metrics with static fallback),
  **Service Lens** (`/about`, the Siemens bridge). `npm run build` clean; verified end-to-end.
  One-command runner: `./run_local.sh`.
- ✅ **Deployed to Vercel:** **https://amazon-rating-recommender.vercel.app** (public).
  Rebranded **StarSense — Amazon Rating Recommender**; **light/white theme** matching the
  `presentation/interview-walkthrough.html`. Home has a plain-English intro + data snapshot;
  console uses non-technical labels ("Shopper", "Product name"); cockpit has inline explainers
  for every concept (collaborative / statistical / NLP / calibration / distribution). Console
  runs in **offline-snapshot mode** on Vercel (preset demos incl. cold-start work; custom free-text
  needs the local model — the 70 MB scikit-surprise model doesn't fit Vercel's Python runtime).
  Vercel project `amazon-rating-recommender` (framework pinned via `web/vercel.json`), on the
  coordinator's account `shamanths-projects`. ⚠️ Old preview `web-mu-steel-25` now superseded.
- ✅ **Interview study material:** `presentation/interview-walkthrough.html` — mirrors the
  MarketLens walkthrough (purpose → plain 5-step story → data snapshot → spoken answers → glossary
  → 6 beats → click path → Q&A → **JD-mapping table** → **"How far I pushed it" leaderboard** →
  Siemens bridge → cold-start one-pager), in Deekshita's plain-English voice. Includes "explain
  further" toggles (honest-testing breakdown, **data-drift vs concept-drift**).
- ✅ **Stage 1 experiment — stacked ensemble (`ml/stack.py`):** an honest "how far can I push it"
  study. Same 50k by-interaction holdout. Leaderboard: **STACKED (meta-LGBM) 0.767** > SVD 0.798 >
  SVD-wide 0.806 > shipped hybrid 0.811 > KNNBaseline 0.835 > avg-of-bases 0.870 > NMF 1.050 >
  baseline 1.073 > LightGBM 1.580. SVD++ dropped (unusably slow for negligible gain).
  **Decision (locked): SHIP the explainable hybrid (0.81); the 0.767 ensemble is a documented
  experiment, NOT deployed** — the app/monitoring cockpit stay at the honest 0.81. Stage 2
  (transformer embeddings) deferred to the NLP-classification project.
- ▶️ **Next: Phase 4** — (a) for a fully-live public console: numpy-only model export as a Vercel
  Python function OR a small container (Render/Railway); (b) refresh README + `case-study` skill;
  (c) move to **`dsridhar2110`** — push to her GitHub, redeploy under her Vercel, rename project `recopulse`.
