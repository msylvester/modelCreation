from flask import Flask, request, jsonify
from main_model_two import load_resources
from main_model_two import predict_sets_for_parts as predict_sets
#from main_model_two import load_resources, predict_sets

# Load resources once during server startup
df, scaler, pca, knn_model, y = load_resources()

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict LEGO set IDs based on a parts list.
    Expects a JSON payload with parts and quantities.
    Example:
    {
        "parts_list": {
            "6141": 2,
            "3001": 5
        }
    }
    """
    data = request.json
    if not data or "parts_list" not in data:
        return jsonify({"error": "Missing parts_list in request"}), 400

    parts_list = data["parts_list"]

    try:
        # Delegate prediction logic to main_model.py
        closest_set_ids = predict_sets(parts_list, df, scaler, pca, knn_model, y)
        return jsonify({"closest_set_ids": closest_set_ids.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return jsonify({"message": "LEGO Set Prediction API is running!"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

~                                                         