import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataPreprocessor:

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df: pd.DataFrame = None
        self.ratings_df: pd.DataFrame = None
        self.products_df: pd.DataFrame = None

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records from {self.data_path}")
        return self.df

    def clean_data(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Call load_data() first.")

        initial_count = len(self.df)
        self.df.dropna(how="all", inplace=True)
        self.df.drop_duplicates(subset=["user_id", "product_id"], keep="last", inplace=True)

        self.df["description"] = self.df["description"].fillna("")
        self.df["category"] = self.df["category"].fillna("Unknown")
        self.df["product_name"] = self.df["product_name"].fillna("Unknown Product")

        self.df["rating"] = pd.to_numeric(self.df["rating"], errors="coerce")
        self.df.dropna(subset=["rating"], inplace=True)
        self.df["rating"] = self.df["rating"].clip(1.0, 5.0)

        logger.info(f"Cleaned data: {initial_count} -> {len(self.df)} records")
        return self.df

    def engineer_features(self) -> pd.DataFrame:
        self.df["content_features"] = (
            self.df["product_name"].str.lower()
            + " "
            + self.df["category"].str.lower()
            + " "
            + self.df["description"].str.lower()
        )
        return self.df

    def build_sub_dataframes(self):
        self.products_df = (
            self.df.drop_duplicates(subset=["product_id"], keep="last")
            .set_index("product_id")[["product_name", "category", "description", "content_features"]]
        )
        self.ratings_df = self.df[["user_id", "product_id", "rating"]].copy()
        logger.info(f"Products: {len(self.products_df)} | Ratings: {len(self.ratings_df)}")
        return self.products_df, self.ratings_df

    def run(self):
        self.load_data()
        self.clean_data()
        self.engineer_features()
        self.build_sub_dataframes()
        return self.df, self.products_df, self.ratings_df

    def get_summary(self) -> dict:
        if self.df is None:
            return {}
        return {
            "total_records": len(self.df),
            "unique_users": self.df["user_id"].nunique(),
            "unique_products": self.df["product_id"].nunique(),
            "categories": self.df["category"].unique().tolist(),
            "rating_mean": round(self.df["rating"].mean(), 2),
            "rating_std": round(self.df["rating"].std(), 2),
        }
