"""
RecoPulse — HybridPredictor
============================
A faithful, refactored port of the FIT5212 hybrid recommender (code.ipynb).
Predicts a 1-5 rating for a (user, item) pair by routing through three tiers:

    Tier 1  ENHANCED_SVD          collaborative filtering (surprise SVD)   — item well-rated
    Tier 2  HYBRID_STATISTICS     user/product bias + confidence blend     — item seen but thin
    Tier 3  CONTENT_SIMILARITY    TF-IDF over item names (cold-start / NLP) — item unseen

Every prediction is explainable: `predict(..., explain=True)` returns which tier fired,
a plain-English reason, and (for cold-start) the similar items it borrowed from. That is
what makes this defensible to a non-technical stakeholder.
"""
from __future__ import annotations
import re
import time
import numpy as np
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------------------- #
# Hyperparameters (defaults from the notebook's proven configuration)
# ----------------------------------------------------------------------------- #
DEFAULT_HYPERPARAMS = {
    "similarity_threshold_case2": 0.2,
    "similarity_threshold_case3": 0.3,
    "max_tfidf_features": 1500,
    "n_svd_factors": 85,
    "svd_reg": 0.01,
    "svd_lr": 0.01,
    "svd_epochs": 40,
    "min_product_ratings": 2,
    "confidence_divisor": 5,
    "user_product_weights": (0.65, 0.35),
    "use_quality_adjustment": True,
    "quality_confidence_threshold": 0.7,
}

TIER_SVD = "ENHANCED_SVD"
TIER_STATS = "HYBRID_STATISTICS"
TIER_CONTENT = "CONTENT_SIMILARITY"


# ----------------------------------------------------------------------------- #
# Text cleaning — shared by training and inference
# ----------------------------------------------------------------------------- #
_EDITION_PATTERNS = [
    r"\(special edition\)", r"\(director\'s cut\)", r"\(extended edition\)",
    r"\(widescreen edition\)", r"\(fullscreen edition\)", r"\(remastered\)",
    r"\(unrated\)", r"\(soundtrack\)", r"\(dvd\)", r"\(blu ray\)",
    r"\(collectors edition\)", r"\(anniversary edition\)", r"\(ultimate edition\)",
    r"\(original motion picture soundtrack\)", r"\(deluxe edition\)",
]


def clean_product_name(name) -> str:
    """Enhanced cleaning that preserves pure-number names (matches the notebook)."""
    if name is None or (isinstance(name, float) and np.isnan(name)) or name == "":
        return ""
    name = str(name).lower()
    if re.match(r"^[\d\-\/\s]+$", name):          # purely numbers/dates → keep
        return name.strip()
    for pattern in _EDITION_PATTERNS:
        name = re.sub(pattern, "", name)
    name = re.sub(r"\b(19|20)\d{2}\b", "", name)  # drop years, keep sequel numbers
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def calculate_original_confidence(count, confidence_divisor) -> float:
    """Confidence grows with how many ratings back a statistic, capped at 1.0."""
    return min(1.0, count / confidence_divisor)


@dataclass
class Prediction:
    """Explainable prediction result."""
    rating: float
    tier: str
    reason: str
    similar_items: list = field(default_factory=list)

    def as_dict(self):
        return {
            "rating": round(float(self.rating), 4),
            "tier": self.tier,
            "reason": self.reason,
            "similar_items": self.similar_items,
        }


class HybridPredictor:
    """Holds all trained components and serves explainable predictions."""

    def __init__(self, hyperparams=None):
        self.hp = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        # populated by fit()
        self.global_mean = None
        self.product_name_stats = {}
        self.user_stats = {}
        self.train_product_names = set()
        self.svd_model = None
        self.user_encoder = {}
        self.name_encoder = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.unique_names_list = []
        self.product_quality_scores = {}

    # ------------------------------------------------------------------ fit --- #
    def fit(self, df_train, verbose=True):
        """Train all three tiers on a ratings dataframe."""
        # Lazy import so the serving container can skip surprise if desired.
        from surprise import Dataset, Reader, SVD

        def log(msg):
            if verbose:
                print(msg, flush=True)

        hp = self.hp
        df = df_train.copy()
        df["clean_product_name"] = df["product_name"].apply(clean_product_name)
        df = df[df["clean_product_name"] != ""].reset_index(drop=True)
        log(f"After cleaning: {len(df):,} ratings")

        # --- statistics (user + product bias, confidence) ---
        self.global_mean = float(df["rating"].mean())
        self.product_name_stats = self._agg_stats(df, "clean_product_name", hp["confidence_divisor"])
        self.user_stats = self._agg_stats(df, "user_id", hp["confidence_divisor"])
        self.train_product_names = set(df["clean_product_name"].unique())
        log(f"Global mean: {self.global_mean:.3f} | "
            f"{len(self.product_name_stats):,} products | {len(self.user_stats):,} users")

        # --- selective quality scores (helpfulness signal) ---
        if hp["use_quality_adjustment"] and {"votes", "helpful_votes"}.issubset(df.columns):
            self.product_quality_scores = self._quality_scores(df, hp["quality_confidence_threshold"])
            log(f"Quality scores for {len(self.product_quality_scores):,} high-confidence products")

        # --- Tier 1: enhanced SVD on products with enough ratings ---
        name_counts = df["clean_product_name"].value_counts()
        popular = name_counts[name_counts >= hp["min_product_ratings"]].index.tolist()
        log(f"Products with >= {hp['min_product_ratings']} ratings: {len(popular):,}")
        svd_data = df[df["clean_product_name"].isin(popular)].copy()
        self.user_encoder = {u: i for i, u in enumerate(svd_data["user_id"].unique())}
        self.name_encoder = {n: i for i, n in enumerate(svd_data["clean_product_name"].unique())}
        svd_data["user_idx"] = svd_data["user_id"].map(self.user_encoder)
        svd_data["name_idx"] = svd_data["clean_product_name"].map(self.name_encoder)
        log(f"Training SVD on {len(svd_data):,} ratings "
            f"({len(self.user_encoder):,} users x {len(self.name_encoder):,} products)...")
        reader = Reader(rating_scale=(1, 5))
        trainset = Dataset.load_from_df(
            svd_data[["user_idx", "name_idx", "rating"]], reader
        ).build_full_trainset()
        self.svd_model = SVD(
            n_factors=hp["n_svd_factors"], reg_all=hp["svd_reg"],
            lr_all=hp["svd_lr"], n_epochs=hp["svd_epochs"],
            random_state=42, verbose=False,
        )
        t0 = time.time()
        self.svd_model.fit(trainset)
        log(f"SVD trained in {time.time() - t0:.1f}s")

        # --- Tier 3: TF-IDF over unique item names ---
        unique_names = df["clean_product_name"].unique()
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 4), min_df=2, max_df=0.85,
            max_features=hp["max_tfidf_features"], stop_words="english",
            sublinear_tf=True, smooth_idf=True, norm="l2",
        )
        t0 = time.time()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(unique_names)
        self.unique_names_list = list(unique_names)
        log(f"TF-IDF matrix {self.tfidf_matrix.shape} in {time.time() - t0:.1f}s")
        return self

    @staticmethod
    def _agg_stats(df, key, confidence_divisor):
        g = df.groupby(key)["rating"].agg(["mean", "std", "count"])
        g["confidence"] = g["count"].apply(lambda c: calculate_original_confidence(c, confidence_divisor))
        return {
            k: {"mean": float(r["mean"]), "std": float(r["std"]) if r["std"] == r["std"] else 0.0,
                "count": int(r["count"]), "confidence": float(r["confidence"])}
            for k, r in g.iterrows()
        }

    @staticmethod
    def _quality_scores(df, threshold):
        q = df.groupby("clean_product_name").agg(
            votes_sum=("votes", "sum"), helpful_sum=("helpful_votes", "sum"),
            rating_count=("rating", "count"), rating_std=("rating", "std"),
        ).fillna(0)
        scores = {}
        for name, row in q.iterrows():
            if row["rating_count"] >= 10 and row["votes_sum"] >= 5:
                helpfulness = (row["helpful_sum"] + 1) / (row["votes_sum"] + 2)
                count_conf = min(1.0, row["rating_count"] / 10)
                consistency = max(0, 1 - row["rating_std"]) if row["rating_std"] > 0 else 0.5
                overall = 0.7 * count_conf + 0.3 * consistency
                if overall >= threshold:
                    scores[name] = {"helpfulness_ratio": float(helpfulness), "confidence": float(overall)}
        return scores

    # -------------------------------------------------------------- predict --- #
    def predict(self, user_id, product_name, explain=False):
        """Route a (user, item) pair to the right tier. Returns Prediction if explain else float."""
        clean = clean_product_name(product_name)
        if not clean:
            return self._wrap(self.global_mean, TIER_STATS, "Empty item name → global mean fallback.", explain)

        # Tier 1: SVD
        if clean in self.train_product_names and clean in self.name_encoder and self.svd_model is not None:
            val = self._predict_svd(user_id, clean)
            if val is not None:
                return self._wrap(val, TIER_SVD,
                                  "Item has enough ratings → collaborative-filtering (SVD) prediction, "
                                  "confidence-blended with the global mean.", explain)

        # Tier 2: seen but thin
        if clean in self.train_product_names:
            val = self._predict_stats(user_id, clean)
            return self._wrap(val, TIER_STATS,
                              "Item seen but sparsely rated → confidence-weighted blend of user bias, "
                              "product bias, and similar-item bias.", explain)

        # Tier 3: cold-start
        val, similars = self._predict_content(user_id, clean)
        if similars:
            reason = "Item never seen in training (cold-start) → borrowed ratings from the most similar " \
                     "known items via TF-IDF on the name (NLP)."
        else:
            reason = "Cold-start with no similar items found → fell back to user/product statistics."
        return self._wrap(val, TIER_CONTENT, reason, explain, similars)

    def _wrap(self, rating, tier, reason, explain, similars=None):
        rating = float(np.clip(rating, 1.0, 5.0))
        if not explain:
            return rating
        return Prediction(rating, tier, reason, similars or [])

    # -------- tier implementations (faithful to the notebook) --------------- #
    def _predict_svd(self, user_id, clean):
        try:
            u = self.user_encoder.get(user_id)
            n = self.name_encoder.get(clean)
            if u is None or n is None:
                return None
            pred = self.svd_model.predict(u, n).est
            if clean in self.product_quality_scores:
                helpfulness = self.product_quality_scores[clean]["helpfulness_ratio"]
                pred *= 1.0 + 0.05 * (helpfulness - 0.5)
            if clean in self.product_name_stats:
                c = self.product_name_stats[clean]["confidence"]
                pred = c * pred + (1 - c) * self.global_mean
            else:
                pred = 0.9 * pred + 0.1 * self.global_mean
            return float(np.clip(pred, 1.0, 5.0))
        except Exception:
            return None

    def _predict_stats(self, user_id, clean):
        gm = self.global_mean
        pred = gm
        user_bias = 0.0
        if user_id in self.user_stats:
            s = self.user_stats[user_id]
            user_bias = (s["mean"] - gm) * s["confidence"]
        product_bias = 0.0
        if clean in self.product_name_stats:
            s = self.product_name_stats[clean]
            product_bias = (s["mean"] - gm) * s["confidence"]
        similar_bias = self._similar_bias(clean, self.hp["similarity_threshold_case2"], k=3)
        uw, pw = self.hp["user_product_weights"]
        pred += uw * user_bias + pw * product_bias + 0.1 * similar_bias
        return float(np.clip(pred, 1.0, 5.0))

    def _predict_content(self, user_id, clean):
        similars = self._find_similar(clean, k=10, threshold=self.hp["similarity_threshold_case3"])
        filtered = [(n, s) for n, s in similars
                    if n in self.product_name_stats and self.product_name_stats[n]["confidence"] >= 0.4]
        if not filtered:
            return self._predict_stats(user_id, clean), []
        ratings = np.array([self.product_name_stats[n]["mean"] for n, _ in filtered])
        weights = np.array([s * self.product_name_stats[n]["confidence"] for n, s in filtered])
        content = float(np.sum(ratings * weights) / np.sum(weights))
        if user_id in self.user_stats:
            cnt = self.user_stats[user_id]["count"]
            bw = 0.85 if cnt >= 15 else 0.75 if cnt >= 5 else 0.65
            pred = bw * content + (1 - bw) * self.user_stats[user_id]["mean"]
        else:
            pred = 0.8 * content + 0.2 * self.global_mean
        top = [{"name": n, "similarity": round(float(s), 3),
                "mean_rating": round(self.product_name_stats[n]["mean"], 2)} for n, s in filtered[:5]]
        return float(np.clip(pred, 1.0, 5.0)), top

    def _find_similar(self, product_name, k=5, threshold=0.2):
        if self.tfidf_matrix is None:
            return []
        try:
            vec = self.tfidf_vectorizer.transform([product_name])
            sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
            valid = np.where(sims >= threshold)[0]
            if len(valid) == 0:
                return []
            order = valid[sims[valid].argsort()[::-1]]
            return [(self.unique_names_list[i], sims[i]) for i in order[:k]]
        except Exception:
            return []

    def _similar_bias(self, clean, threshold, k=3):
        sims = self._find_similar(clean, k=k, threshold=threshold)
        biases, weights = [], []
        for name, sim in sims:
            if name in self.product_name_stats and self.product_name_stats[name]["confidence"] >= 0.3:
                biases.append(self.product_name_stats[name]["mean"] - self.global_mean)
                weights.append(sim * self.product_name_stats[name]["confidence"])
        if not biases:
            return 0.0
        biases, weights = np.array(biases), np.array(weights)
        return float(np.sum(biases * weights) / np.sum(weights))

    # -------------------------------------------------------------- tier of -- #
    def tier_of(self, user_id, product_name):
        """Which tier WOULD serve this pair (for monitoring tier-mix), without full compute."""
        clean = clean_product_name(product_name)
        if clean in self.train_product_names and clean in self.name_encoder:
            return TIER_SVD
        if clean in self.train_product_names:
            return TIER_STATS
        return TIER_CONTENT

    # ------------------------------------------------------------ persistence - #
    def save(self, path):
        import joblib
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)
