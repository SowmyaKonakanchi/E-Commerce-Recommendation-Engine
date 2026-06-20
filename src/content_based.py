import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContentBasedRecommender:

    def __init__(self, top_n: int = 5):
        self.top_n = top_n
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        self.tfidf_matrix = None
        self.products_df: pd.DataFrame = None
        self.product_ids: list = []
        self._is_fitted = False

    def fit(self, products_df: pd.DataFrame):
        self.products_df = products_df.copy()
        self.product_ids = list(products_df.index)
        corpus = products_df["content_features"].fillna("").tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self._is_fitted = True
        logger.info(f"Content-based model fitted on {len(self.product_ids)} products. "
                    f"TF-IDF shape: {self.tfidf_matrix.shape}")
        return self

    def _get_similarity_scores(self, product_id: str) -> np.ndarray:
        if product_id not in self.product_ids:
            raise ValueError(f"Product '{product_id}' not found in training data.")
        idx = self.product_ids.index(product_id)
        return cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

    def recommend(self, product_id: str, exclude_ids: list = None, top_n: int = None) -> list[dict]:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        scores = self._get_similarity_scores(product_id)
        exclude = set(exclude_ids or []) | {product_id}
        n = top_n or self.top_n

        candidates = [
            (self.product_ids[i], scores[i])
            for i in np.argsort(scores)[::-1]
            if self.product_ids[i] not in exclude
        ]

        recommendations = []
        for pid, score in candidates[:n]:
            row = self.products_df.loc[pid]
            recommendations.append({
                "product_id": pid,
                "product_name": row["product_name"],
                "category": row["category"],
                "content_score": round(float(score), 4),
                "explanation": self._explain(product_id, pid, score),
            })

        return recommendations

    def _explain(self, source_id: str, target_id: str, score: float) -> str:
        source_cat = self.products_df.loc[source_id, "category"]
        target_cat = self.products_df.loc[target_id, "category"]
        target_name = self.products_df.loc[target_id, "product_name"]

        if source_cat == target_cat:
            return f"Similar to your product in the '{source_cat}' category (similarity: {score:.2f})"
        return (f"'{target_name}' shares content features with your product "
                f"across categories (similarity: {score:.2f})")

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Content-based model saved to {path}")

    @staticmethod
    def load(path: str) -> "ContentBasedRecommender":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Content-based model loaded from {path}")
        return model
