import pandas as pd
import pickle
import logging
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeFilteringModel

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Weighted hybrid: final_score = alpha * collab + (1 - alpha) * content"""

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        collab_model: CollaborativeFilteringModel,
        products_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        alpha: float = 0.6,
    ):
        self.content_model = content_model
        self.collab_model = collab_model
        self.products_df = products_df
        self.ratings_df = ratings_df
        self.alpha = alpha

    def recommend_for_user(self, user_id: str, top_n: int = 5) -> dict:
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]

        if user_ratings.empty:
            logger.info(f"Cold-start user '{user_id}'  falling back to popular products.")
            return self._cold_start_recommendations(user_id, top_n)

        rated_ids = user_ratings["product_id"].tolist()
        collab_recs = self.collab_model.recommend(user_id, rated_ids, top_n=top_n * 3)

        if not collab_recs:
            return self._cold_start_recommendations(user_id, top_n)

        best_product = user_ratings.sort_values("rating", ascending=False).iloc[0]["product_id"]
        content_recs = self.content_model.recommend(best_product, exclude_ids=rated_ids)
        content_score_map = {r["product_id"]: r["content_score"] for r in content_recs}

        collab_ratings = [r["predicted_rating"] for r in collab_recs]
        min_r, max_r = min(collab_ratings), max(collab_ratings)
        range_r = max_r - min_r if max_r != min_r else 1.0

        blended = []
        for rec in collab_recs:
            pid = rec["product_id"]
            norm_collab = (rec["predicted_rating"] - min_r) / range_r
            content_score = content_score_map.get(pid, 0.0)
            final_score = self.alpha * norm_collab + (1 - self.alpha) * content_score

            if pid in self.products_df.index:
                row = self.products_df.loc[pid]
                blended.append({
                    "product_id": pid,
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "predicted_rating": rec["predicted_rating"],
                    "hybrid_score": round(final_score, 4),
                    "confidence": self._score_to_confidence(final_score),
                    "explanation": self._explain_user_rec(norm_collab, content_score),
                })

        blended.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return {
            "user_id": user_id,
            "mode": "hybrid",
            "anchor_product": best_product,
            "recommendations": blended[:top_n],
        }

    def recommend_for_product(self, product_id: str, top_n: int = 5) -> dict:
        if product_id not in self.products_df.index:
            raise ValueError(f"Product '{product_id}' not in catalog.")
        recs = self.content_model.recommend(product_id, top_n=top_n)
        return {
            "product_id": product_id,
            "product_name": self.products_df.loc[product_id, "product_name"],
            "mode": "content_based",
            "recommendations": recs,
        }

    def _cold_start_recommendations(self, user_id: str, top_n: int) -> dict:
        top_products = (
            self.ratings_df.groupby("product_id")["rating"]
            .agg(["mean", "count"])
            .query("count >= 2")
            .sort_values("mean", ascending=False)
            .head(top_n)
        )
        recs = []
        for pid, row in top_products.iterrows():
            if pid in self.products_df.index:
                p = self.products_df.loc[pid]
                recs.append({
                    "product_id": pid,
                    "product_name": p["product_name"],
                    "category": p["category"],
                    "avg_rating": round(row["mean"], 2),
                    "hybrid_score": round(row["mean"] / 5.0, 4),
                    "confidence": "Medium",
                    "explanation": f"Top-rated product in '{p['category']}' (avg rating: {row['mean']:.2f})",
                })
        return {"user_id": user_id, "mode": "cold_start", "recommendations": recs}

    @staticmethod
    def _score_to_confidence(score: float) -> str:
        if score >= 0.75:
            return "High"
        elif score >= 0.45:
            return "Medium"
        return "Low"

    @staticmethod
    def _explain_user_rec(collab: float, content: float) -> str:
        parts = []
        if collab > 0.6:
            parts.append("users with similar taste highly rated this")
        if content > 0.3:
            parts.append("it matches your recently liked products")
        if not parts:
            parts.append("it aligns with your browsing patterns")
        return f"Recommended because {' and '.join(parts)}."

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Hybrid model saved to {path}")

    @staticmethod
    def load(path: str) -> "HybridRecommender":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Hybrid model loaded from {path}")
        return model
