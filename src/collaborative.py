import pandas as pd
import pickle
import logging
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CollaborativeFilteringModel:

    def __init__(self, n_factors: int = 50, n_epochs: int = 20, random_state: int = 42):
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
        self.trainset = None
        self.all_product_ids: list = []
        self._is_fitted = False
        self.metrics: dict = {}

    def fit(self, ratings_df: pd.DataFrame, evaluate: bool = True):
        self.all_product_ids = ratings_df["product_id"].unique().tolist()

        reader = Reader(rating_scale=(1.0, 5.0))
        dataset = Dataset.load_from_df(ratings_df[["user_id", "product_id", "rating"]], reader)

        if evaluate:
            trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
            self.model.fit(trainset)
            predictions = self.model.test(testset)
            self.metrics = {
                "RMSE": round(accuracy.rmse(predictions, verbose=False), 4),
                "MAE": round(accuracy.mae(predictions, verbose=False), 4),
            }
            logger.info(f"Evaluation  RMSE: {self.metrics['RMSE']} | MAE: {self.metrics['MAE']}")

        full_trainset = dataset.build_full_trainset()
        self.model.fit(full_trainset)
        self.trainset = full_trainset
        self._is_fitted = True
        logger.info("Collaborative filtering model trained on full dataset.")
        return self

    def predict_rating(self, user_id: str, product_id: str) -> float:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return round(self.model.predict(user_id, product_id).est, 4)

    def recommend(self, user_id: str, rated_product_ids: list, top_n: int = 5) -> list[dict]:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        seen = set(rated_product_ids)
        candidates = [pid for pid in self.all_product_ids if pid not in seen]

        if not candidates:
            logger.warning(f"No unseen products for user {user_id}.")
            return []

        scored = sorted(
            [(pid, self.predict_rating(user_id, pid)) for pid in candidates],
            key=lambda x: x[1], reverse=True
        )

        return [
            {"product_id": pid, "predicted_rating": rating, "confidence": self._confidence(rating)}
            for pid, rating in scored[:top_n]
        ]

    @staticmethod
    def _confidence(predicted_rating: float) -> str:
        if predicted_rating >= 4.5:
            return "High"
        elif predicted_rating >= 3.5:
            return "Medium"
        return "Low"

    def get_metrics(self) -> dict:
        return self.metrics

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Collaborative model saved to {path}")

    @staticmethod
    def load(path: str) -> "CollaborativeFilteringModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Collaborative model loaded from {path}")
        return model
