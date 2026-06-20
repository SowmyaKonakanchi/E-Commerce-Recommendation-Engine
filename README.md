# E-Commerce Recommendation Engine

A production-ready **Hybrid Recommendation System** combining Content-Based Filtering and Collaborative Filtering, served via a Flask REST API.

---

## Architecture

```
sample_dataset.csv
        │
        ▼
┌─────────────────────┐
│   DataPreprocessor  │  ← Handle missing values, duplicates, feature engineering
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────────┐  ┌────────────────────┐
│ TF-IDF   │  │  SVD (Surprise)    │
│ Cosine   │  │  Collaborative     │
│ Content  │  │  Filtering         │
│ Based    │  │  RMSE / MAE eval   │
└────┬─────┘  └────────┬───────────┘
     │                 │
     └────────┬────────┘
              ▼
     ┌─────────────────┐
     │ HybridRecommender│  ← Weighted blend + cold-start fallback
     └────────┬─────────┘
              ▼
     ┌─────────────────┐
     │   Flask API     │  ← /recommend/user/<id>  /recommend/product/<id>
     └─────────────────┘
```

---

## Project Structure

```
ecommerce_recommendation_engine/
├── data/                          # (reserved for additional datasets)
├── models/                        # Pickled models + visualizations (auto-generated)
├── notebooks/
│   └── exploration.ipynb          # Interactive demo notebook
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Data loading, cleaning, feature engineering
│   ├── content_based.py           # TF-IDF + Cosine Similarity recommender
│   ├── collaborative.py           # SVD collaborative filtering (Surprise)
│   ├── hybrid.py                  # Weighted hybrid + cold-start support
│   └── app.py                     # Flask REST API
├── train_and_visualize.py         # One-shot training + 5 visualizations
├── sample_dataset.csv             # 50-row e-commerce dataset
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation

### Local Setup

```bash
# 1. Clone / navigate to project
cd "E-Commerce Recommendation Engine"

# 2. Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models and generate visualizations
python train_and_visualize.py

# 5. Start the Flask API
python src/app.py
```

### Docker Setup

```bash
docker build -t ecommerce-recommender .
docker run -p 5000:5000 ecommerce-recommender
```

---

## API Reference

### Health Check

```http
GET /health
```

Response:
```json
{ "status": "ok", "model_loaded": true }
```

---

### Evaluation Metrics

```http
GET /metrics
```

Response:
```json
{
  "dataset_summary": {
    "total_records": 50,
    "unique_users": 10,
    "unique_products": 20,
    "rating_mean": 4.33
  },
  "collaborative_filtering_metrics": {
    "RMSE": 0.3812,
    "MAE": 0.2947
  }
}
```

---

### User Recommendations (Hybrid)

```http
GET /recommend/user/<user_id>?top_n=5
```

Example:
```bash
curl http://localhost:5000/recommend/user/U001?top_n=5
```

Response:
```json
{
  "success": true,
  "data": {
    "user_id": "U001",
    "mode": "hybrid",
    "anchor_product": "P003",
    "recommendations": [
      {
        "product_id": "P017",
        "product_name": "Machine Learning Book",
        "category": "Books",
        "predicted_rating": 4.72,
        "hybrid_score": 0.8431,
        "confidence": "High",
        "explanation": "Recommended because users with similar taste highly rated this and it matches your recently liked products."
      }
    ]
  }
}
```

Cold-start users (no prior ratings) automatically receive popularity-based recommendations.

---

### Product Recommendations (Content-Based)

```http
GET /recommend/product/<product_id>?top_n=5
```

Example:
```bash
curl http://localhost:5000/recommend/product/P001?top_n=5
```

Response:
```json
{
  "success": true,
  "data": {
    "product_id": "P001",
    "product_name": "Wireless Bluetooth Headphones",
    "mode": "content_based",
    "recommendations": [
      {
        "product_id": "P008",
        "product_name": "Smart Watch Fitness Tracker",
        "category": "Electronics",
        "content_score": 0.2134,
        "explanation": "Similar to your product in the 'Electronics' category (similarity: 0.21)"
      }
    ]
  }
}
```

---

## Key Features

| Feature | Description |
|---|---|
| Content-Based Filtering | TF-IDF (bigrams, 5000 features) + Cosine Similarity on product name, category, and description |
| Collaborative Filtering | SVD (50 latent factors) from Surprise library with RMSE/MAE evaluation |
| Hybrid Blending | Weighted score: `0.6 × collab + 0.4 × content` |
| Cold-Start Support | Falls back to popularity-based recommendations for new users |
| Confidence Scores | High / Medium / Low based on hybrid score threshold |
| Explanations | Human-readable reason for each recommendation |
| Model Persistence | All models saved with Pickle for fast API startup |
| Visualizations | 5 Matplotlib charts saved to `models/` |

---

## Generated Visualizations

After running `train_and_visualize.py`:

| File | Description |
|---|---|
| `models/fig1_rating_distribution.png` | Histogram of ratings + products per category |
| `models/fig2_top_products.png` | Top 10 products by average rating |
| `models/fig3_similarity_heatmap.png` | Cosine similarity heatmap between sample products |
| `models/fig4_evaluation_metrics.png` | RMSE and MAE bar chart |
| `models/fig5_user_category_heatmap.png` | User × Category average rating heatmap |

---

## Technologies Used

- Python 3.11
- Flask 3.0
- scikit-learn (TF-IDF, Cosine Similarity)
- scikit-surprise (SVD Collaborative Filtering)
- Pandas / NumPy
- Matplotlib
- Docker

---

## License

MIT License
