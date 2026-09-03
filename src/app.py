import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, request, abort
from src.preprocessing import DataPreprocessor
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeFilteringModel
from src.hybrid import HybridRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

recommender: HybridRecommender = None
eval_metrics: dict = {}
data_summary: dict = {}

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATHS = {
    "content": os.path.join(BASE_DIR, "models/content_model.pkl"),
    "collab":  os.path.join(BASE_DIR, "models/collab_model.pkl"),
}
DATASET_PATH = os.path.join(BASE_DIR, "sample_dataset.csv")


def _load_or_train():
    global recommender, eval_metrics, data_summary

    preprocessor = DataPreprocessor(DATASET_PATH)
    _, products_df, ratings_df = preprocessor.run()
    data_summary = preprocessor.get_summary()

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    if os.path.exists(MODEL_PATHS["content"]):
        content_model = ContentBasedRecommender.load(MODEL_PATHS["content"])
    else:
        content_model = ContentBasedRecommender(top_n=10)
        content_model.fit(products_df)
        content_model.save(MODEL_PATHS["content"])

    if os.path.exists(MODEL_PATHS["collab"]):
        collab_model = CollaborativeFilteringModel.load(MODEL_PATHS["collab"])
    else:
        collab_model = CollaborativeFilteringModel()
        collab_model.fit(ratings_df, evaluate=True)
        collab_model.save(MODEL_PATHS["collab"])

    eval_metrics = collab_model.get_metrics()
    recommender = HybridRecommender(content_model, collab_model, products_df, ratings_df)
    logger.info("All models ready.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": recommender is not None})


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "dataset_summary": data_summary,
        "collaborative_filtering_metrics": eval_metrics,
    })


@app.route("/recommend/user/<user_id>", methods=["GET"])
def recommend_user(user_id: str):
    if recommender is None:
        abort(503, description="Models not loaded yet.")
    top_n = max(1, min(request.args.get("top_n", default=5, type=int), 20))
    try:
        return jsonify({"success": True, "data": recommender.recommend_for_user(user_id, top_n=top_n)})
    except Exception as e:
        logger.error(f"Error recommending for user {user_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/recommend/product/<product_id>", methods=["GET"])
def recommend_product(product_id: str):
    if recommender is None:
        abort(503, description="Models not loaded yet.")
    top_n = max(1, min(request.args.get("top_n", default=5, type=int), 20))
    try:
        return jsonify({"success": True, "data": recommender.recommend_for_product(product_id, top_n=top_n)})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error recommending for product {product_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(503)
def service_unavailable(e):
    return jsonify({"success": False, "error": str(e)}), 503


if __name__ == "__main__":
    _load_or_train()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
