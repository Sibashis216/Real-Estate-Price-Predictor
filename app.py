"""
Flask API - Bangalore House Price Predictor
"""
from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

# ─── Load Model Artifacts ────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

with open(os.path.join(MODEL_DIR, "house_price_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    feature_columns = json.load(f)

with open(os.path.join(MODEL_DIR, "locations.json")) as f:
    locations = json.load(f)

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

print(f"✓ Model loaded: {metadata['model_name']} (R²={metadata['r2_score']})")

# ─── Helper ──────────────────────────────────────────────
def predict_price(location, sqft, bath, bhk, balcony):
    """Build feature vector and return predicted price in Lakhs."""
    # Init all features to 0
    x = np.zeros(len(feature_columns))

    # Numeric features
    col_map = {
        "total_sqft_num": sqft,
        "bath": bath,
        "bhk": bhk,
        "balcony": balcony,
    }
    for col, val in col_map.items():
        if col in feature_columns:
            x[feature_columns.index(col)] = val

    # Location dummy
    loc_col = f"loc_{location}"
    if loc_col in feature_columns:
        x[feature_columns.index(loc_col)] = 1
    else:
        # Use 'other' if location not in training data
        other_col = "loc_other"
        if other_col in feature_columns:
            x[feature_columns.index(other_col)] = 1

    price = model.predict([x])[0]
    return round(float(price), 2)


# ─── Routes ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", locations=locations, metadata=metadata)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        location = data.get("location", "other")
        sqft = float(data.get("sqft", 1000))
        bath = int(data.get("bath", 2))
        bhk = int(data.get("bhk", 2))
        balcony = int(data.get("balcony", 1))

        # Validation
        if sqft < 300 or sqft > 30000:
            return jsonify({"error": "Square footage must be between 300 and 30,000"}), 400
        if bath < 1 or bath > 10:
            return jsonify({"error": "Bathrooms must be between 1 and 10"}), 400
        if bhk < 1 or bhk > 10:
            return jsonify({"error": "BHK must be between 1 and 10"}), 400

        price = predict_price(location, sqft, bath, bhk, balcony)
        price_per_sqft = round((price * 100000) / sqft)

        return jsonify({
            "price_lakhs": price,
            "price_crores": round(price / 100, 3),
            "price_per_sqft": price_per_sqft,
            "location": location,
            "sqft": sqft,
            "bhk": bhk,
            "bath": bath,
            "balcony": balcony,
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/locations")
def get_locations():
    return jsonify({"locations": locations})


@app.route("/api/metadata")
def get_metadata():
    return jsonify(metadata)


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": metadata["model_name"]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
