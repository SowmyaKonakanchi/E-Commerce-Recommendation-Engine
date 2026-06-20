import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from src.preprocessing import DataPreprocessor
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeFilteringModel
from src.hybrid import HybridRecommender
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("models", exist_ok=True)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

print("\n[1] Loading and preprocessing data...")
preprocessor = DataPreprocessor("sample_dataset.csv")
df, products_df, ratings_df = preprocessor.run()
print(f"   Dataset summary: {preprocessor.get_summary()}")

print("\n[2] Training content-based model...")
content_model = ContentBasedRecommender(top_n=10)
content_model.fit(products_df)
content_model.save("models/content_model.pkl")

sample_product = products_df.index[0]
for r in content_model.recommend(sample_product)[:3]:
    print(f"   -> {r['product_id']} | {r['product_name']} | score={r['content_score']}")

print("\n[3] Training collaborative filtering model...")
collab_model = CollaborativeFilteringModel(n_factors=50, n_epochs=20)
collab_model.fit(ratings_df, evaluate=True)
collab_model.save("models/collab_model.pkl")
metrics = collab_model.get_metrics()
print(f"   RMSE: {metrics.get('RMSE')}  |  MAE: {metrics.get('MAE')}")

print("\n[4] Building hybrid recommender...")
hybrid = HybridRecommender(content_model, collab_model, products_df, ratings_df)
hybrid.save("models/hybrid_model.pkl")

sample_user = ratings_df["user_id"].iloc[0]
user_recs = hybrid.recommend_for_user(sample_user, top_n=5)
print(f"\n   Hybrid recs for user '{sample_user}':")
for r in user_recs["recommendations"]:
    print(f"   -> {r['product_id']} | {r['product_name']} | score={r['hybrid_score']} | {r['confidence']}")

print("\n[5] Generating visualizations...")
plt.style.use("seaborn-v0_8-whitegrid")
pid_to_name = products_df["product_name"].to_dict()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(ratings_df["rating"], bins=9, color=COLORS[0], edgecolor="white", linewidth=0.8)
axes[0].set_title("Rating Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Count")
cat_counts = df.groupby("category")["product_id"].nunique().sort_values(ascending=True)
axes[1].barh(cat_counts.index, cat_counts.values, color=COLORS[1], edgecolor="white")
axes[1].set_title("Products per Category", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Unique Products")
plt.tight_layout()
plt.savefig("models/fig1_rating_distribution.png", dpi=120)
plt.close()
print("   Saved: models/fig1_rating_distribution.png")

top_products = ratings_df.groupby("product_id")["rating"].mean().sort_values(ascending=False).head(10)
labels = [pid_to_name.get(p, p)[:25] for p in top_products.index]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(labels, top_products.values, color=COLORS[2], edgecolor="white")
ax.set_xlim(0, 5.5)
ax.set_title("Top 10 Products by Average Rating", fontsize=13, fontweight="bold")
ax.set_xlabel("Average Rating")
for bar, val in zip(bars, top_products.values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("models/fig2_top_products.png", dpi=120)
plt.close()
print("   Saved: models/fig2_top_products.png")

sample_ids = list(products_df.index[:8])
sample_matrix = content_model.tfidf_matrix[
    [content_model.product_ids.index(pid) for pid in sample_ids]
]
sim_matrix = cosine_similarity(sample_matrix)
short_labels = [pid_to_name.get(p, p)[:18] for p in sample_ids]
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(sim_matrix, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(short_labels)))
ax.set_yticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(short_labels, fontsize=8)
plt.colorbar(im, ax=ax, label="Cosine Similarity")
ax.set_title("Content-Based Similarity Heatmap", fontsize=13, fontweight="bold")
for i in range(len(sample_ids)):
    for j in range(len(sample_ids)):
        ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if sim_matrix[i, j] > 0.5 else "black")
plt.tight_layout()
plt.savefig("models/fig3_similarity_heatmap.png", dpi=120)
plt.close()
print("   Saved: models/fig3_similarity_heatmap.png")

fig, ax = plt.subplots(figsize=(5, 4))
metric_names = list(metrics.keys())
metric_vals = list(metrics.values())
bars = ax.bar(metric_names, metric_vals, color=[COLORS[3], COLORS[4]], edgecolor="white", width=0.4)
ax.set_ylim(0, max(metric_vals) * 1.4)
ax.set_title("Collaborative Filtering Evaluation", fontsize=13, fontweight="bold")
ax.set_ylabel("Error")
for bar, val in zip(bars, metric_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.4f}",
            ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig("models/fig4_evaluation_metrics.png", dpi=120)
plt.close()
print("   Saved: models/fig4_evaluation_metrics.png")

user_cat = df.pivot_table(index="user_id", columns="category", values="rating",
                           aggfunc="mean").fillna(0)
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(user_cat.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(user_cat.columns)))
ax.set_yticks(range(len(user_cat.index)))
ax.set_xticklabels(user_cat.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(user_cat.index, fontsize=9)
plt.colorbar(im, ax=ax, label="Avg Rating")
ax.set_title("User-Category Rating Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("models/fig5_user_category_heatmap.png", dpi=120)
plt.close()
print("   Saved: models/fig5_user_category_heatmap.png")

print(f"\nDone! RMSE: {metrics.get('RMSE')} | MAE: {metrics.get('MAE')}")
print("Start the API with:  python src/app.py")
