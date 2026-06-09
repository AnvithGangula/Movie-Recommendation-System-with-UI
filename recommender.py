"""
Movie Recommendation System
Uses the Movies.csv dataset (10,000 movies) for content-based filtering
via TF-IDF on movie overviews + popularity/vote weighting.
Also includes a popularity-based collaborative-style ranker and hybrid blend.

References:
  - scikit-learn TF-IDF / cosine_similarity documentation
  - Surprise library concepts for collaborative filtering
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ── Load dataset ──────────────────────────────────────────────────────────────
_BASE = os.path.dirname(__file__)
_CSV  = os.path.join(_BASE, "Movies.csv")

MOVIES = pd.read_csv(_CSV)
MOVIES = MOVIES.dropna(subset=["title", "overview"]).reset_index(drop=True)
MOVIES["overview"] = MOVIES["overview"].fillna("")
MOVIES["vote_average"] = pd.to_numeric(MOVIES["vote_average"], errors="coerce").fillna(0)
MOVIES["vote_count"]   = pd.to_numeric(MOVIES["vote_count"],   errors="coerce").fillna(0)
MOVIES["popularity"]   = pd.to_numeric(MOVIES["popularity"],   errors="coerce").fillna(0)
MOVIES["year"] = pd.to_datetime(MOVIES["release_date"], errors="coerce").dt.year.fillna(0).astype(int)

ALL_TITLES = MOVIES["title"].tolist()
ALL_USER_IDS = list(range(1, 6))   # demo user IDs (app still shows collaborative tab)

# ── Weighted rating (IMDB-style) for popularity-aware ranking ─────────────────
_C = MOVIES["vote_average"].mean()
_M = MOVIES["vote_count"].quantile(0.70)

def _weighted_rating(row):
    v, R = row["vote_count"], row["vote_average"]
    return (v / (v + _M)) * R + (_M / (v + _M)) * _C

MOVIES["weighted_score"] = MOVIES.apply(_weighted_rating, axis=1)


# ── Content-Based Recommender ─────────────────────────────────────────────────
class ContentBasedRecommender:
    """TF-IDF on movie overviews + cosine similarity."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=15000)
        tfidf_matrix = self.tfidf.fit_transform(MOVIES["overview"])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(MOVIES.index, index=MOVIES["title"]).drop_duplicates()

    def recommend(self, title: str, n: int = 5):
        if title not in self.indices:
            return []
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: n + 1]
        movie_indices = [s[0] for s in sim_scores]
        results = MOVIES.iloc[movie_indices][["title", "year", "vote_average", "popularity"]].copy()
        results["score"] = [round(s[1], 3) for s in sim_scores]
        results = results.rename(columns={"vote_average": "rating"})
        return results.to_dict("records")


# ── Popularity-Based Recommender (acts as "collaborative" for demo) ───────────
class PopularityRecommender:
    """Returns top-rated movies weighted by vote count and popularity."""

    def recommend(self, user_id: int = 1, n: int = 5):
        top = (
            MOVIES.sort_values("weighted_score", ascending=False)
            .head(50)
            .sample(n)          # add variety per call
        )
        results = []
        for _, row in top.iterrows():
            results.append({
                "title":  row["title"],
                "year":   int(row["year"]),
                "rating": round(row["vote_average"], 2),
                "score":  round(row["weighted_score"], 3),
            })
        return sorted(results, key=lambda x: x["score"], reverse=True)


# ── Hybrid Recommender ────────────────────────────────────────────────────────
class HybridRecommender:
    """Blends content-based cosine score with normalised weighted_score (60/40)."""

    def __init__(self):
        self.cb  = ContentBasedRecommender()
        self.pop = PopularityRecommender()

    def recommend(self, user_id: int, liked_title: str, n: int = 5):
        cb_recs  = self.cb.recommend(liked_title, n=20)
        pop_recs = self.pop.recommend(user_id, n=20)

        # Normalise popularity scores to [0,1]
        max_pop = max((r["score"] for r in pop_recs), default=1) or 1
        pop_map = {r["title"]: r["score"] / max_pop for r in pop_recs}

        blended = []
        for r in cb_recs:
            cb_s  = r["score"]
            pop_s = pop_map.get(r["title"], 0)
            hybrid = round(0.6 * cb_s + 0.4 * pop_s, 4)
            blended.append({
                "title":  r["title"],
                "year":   r["year"],
                "rating": r["rating"],
                "score":  hybrid,
            })
        blended.sort(key=lambda x: x["score"], reverse=True)
        return blended[:n]


# ── Singletons ────────────────────────────────────────────────────────────────
content_rec = ContentBasedRecommender()
collab_rec  = PopularityRecommender()
hybrid_rec  = HybridRecommender()
