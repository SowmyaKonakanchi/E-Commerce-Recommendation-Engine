# Project Description: E-Commerce Recommendation Engine

## Overview

A production-ready Hybrid Recommendation System built with Python that combines Content-Based Filtering and Collaborative Filtering to deliver personalized product recommendations. The system is served via a Flask REST API and is fully containerized with Docker.

---

## Problem Statement

E-commerce platforms struggle to surface relevant products for each user from thousands of listings. Generic product displays lead to poor user experience and lower conversion rates. This project solves that by building a personalized recommendation engine that learns from user behavior (ratings) and product content (descriptions, categories) to suggest the most relevant products.

---

## Solution

A hybrid recommendation system that:
- Analyzes product content using NLP (TF-IDF) to find similar products
- Learns user preferences from past ratings using SVD matrix factorization
- Blends both signals with a weighted scoring formula
- Handles brand-new users with a cold-start fallback (popularity-based)
- Exposes all functionality through a clean REST API

---

## Domains Used

| Domain | Technology |
|---|---|
| Machine Learning | SVD, Cosine Similarity, Hybrid Blending |
| Natural Language Processing | TF-IDF, Bigrams, Stop Word Removal |
| Data Engineering | Pandas, NumPy, Feature Engineering |
| Backend Development | Flask REST API |
| Data Visualization | Matplotlib |
| DevOps | Docker, Git, GitHub |
| Software Engineering | OOP, Modular Architecture, Logging |

---

## Dataset

- 50 user-product interaction records
- 10 unique users (U001–U010)
- 20 unique products across 5 categories: Electronics, Books, Sports, Kitchen, Footwear
- Features: user_id, product_id, product_name, category, description, rating (1–5)
- Average rating: 4.41 | Std: 0.34

---

## System Architecture

```
sample_dataset.csv
        │
        ▼
┌─────────────────────┐
│   DataPreprocessor  │  ← Clean, deduplicate, feature engineer
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
     │ HybridRecommender│  ← 0.6 * collab + 0.4 * content
     └────────┬─────────┘
              ▼
     ┌─────────────────┐
     │   Flask API     │  ← /recommend/user/<id>
     └─────────────────┘    /recommend/product/<id>
```

---

## Module Breakdown

### `src/preprocessing.py` — DataPreprocessor
- Loads CSV dataset
- Drops nulls and duplicates
- Clips ratings to [1.0, 5.0]
- Engineers `content_features` column by combining product name, category, and description
- Splits data into products and ratings DataFrames

### `src/content_based.py` — ContentBasedRecommender
- Fits TF-IDF vectorizer (bigrams, 5000 max features) on product content
- Computes pairwise Cosine Similarity matrix
- Returns top-N similar products with similarity score and explanation
- Supports per-call `top_n` override

### `src/collaborative.py` — CollaborativeFilteringModel
- Trains SVD (50 latent factors, 20 epochs) using the Surprise library
- Evaluates on 80/20 train-test split — reports RMSE and MAE
- Re-trains on full data after evaluation
- Predicts ratings for any user-product pair
- Returns top-N unseen products with predicted rating and confidence label

### `src/hybrid.py` — HybridRecommender
- Blends collaborative and content scores: `0.6 × collab + 0.4 × content`
- Uses user's highest-rated product as content anchor
- Falls back to popularity-based recommendations for cold-start users
- Attaches confidence (High / Medium / Low) and human-readable explanation to every recommendation

### `src/app.py` — Flask REST API
- `GET /health` — API health check
- `GET /metrics` — Dataset summary + RMSE/MAE
- `GET /recommend/user/<user_id>?top_n=5` — Hybrid user recommendations
- `GET /recommend/product/<product_id>?top_n=5` — Content-based product recommendations
- Loads pre-trained pickle models on startup for fast response

### `train_and_visualize.py`
- End-to-end training pipeline
- Saves all models as `.pkl` files to `models/`
- Generates 5 Matplotlib visualizations

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Returns API status |
| GET | `/metrics` | Dataset stats + model evaluation scores |
| GET | `/recommend/user/<user_id>` | Hybrid recommendations for a user |
| GET | `/recommend/product/<product_id>` | Content-based similar products |

---

## Model Evaluation

| Metric | Value |
|---|---|
| RMSE | 0.4146 |
| MAE | 0.3668 |

Low RMSE and MAE indicate the SVD model predicts user ratings accurately, within ~0.4 stars on a 1–5 scale.

---

## Visualizations Generated

| File | Description |
|---|---|
| `fig1_rating_distribution.png` | Rating histogram + products per category |
| `fig2_top_products.png` | Top 10 products by average rating |
| `fig3_similarity_heatmap.png` | Cosine similarity heatmap (sample 8 products) |
| `fig4_evaluation_metrics.png` | RMSE and MAE bar chart |
| `fig5_user_category_heatmap.png` | User × Category average rating heatmap |

---

## Key Features

- Hybrid scoring with configurable alpha weight
- Cold-start support for new users with no history
- Confidence labels (High / Medium / Low) on every recommendation
- Human-readable explanation for every recommendation
- Model persistence with Pickle for fast API startup
- Fully containerized with Docker
- Modular OOP codebase — each component is independently testable

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML / NLP | scikit-learn, scikit-surprise |
| Data | Pandas, NumPy |
| API | Flask 3.1 |
| Visualization | Matplotlib |
| Containerization | Docker |
| Version Control | Git, GitHub |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train models and generate visualizations
python train_and_visualize.py

# Start the API
python src/app.py
```

API will be available at `http://localhost:5000`

---

## Repository

GitHub: https://github.com/SowmyaKonakanchi/E-Commerce-Recommendation-Engine
