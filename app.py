"""
OptiCrop - Flask Web Application
Loads the trained Random Forest model and serves crop recommendations.
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
CROP_STATS_PATH = os.path.join(BASE_DIR, "model", "crop_stats.json")
CLUSTER_CSV_PATH = os.path.join(BASE_DIR, "model", "crop_clusters.csv")
CLUSTER_PROFILE_PATH = os.path.join(BASE_DIR, "model", "cluster_profile.csv")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(CROP_STATS_PATH, "r") as f:
    CROP_STATS = json.load(f)

ALL_CROPS = sorted(CROP_STATS.keys())

cluster_df = pd.read_csv(CLUSTER_CSV_PATH)
cluster_profile_df = pd.read_csv(CLUSTER_PROFILE_PATH)

# Reasonable real-world bounds for basic input validation
FIELD_BOUNDS = {
    "N": (0, 150),
    "P": (0, 150),
    "K": (0, 250),
    "temperature": (-10, 60),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 500),
}


def validate_inputs(form):
    """Returns (values_dict, errors_list)."""
    values = {}
    errors = []
    for field, (lo, hi) in FIELD_BOUNDS.items():
        raw = form.get(field, "").strip()
        if raw == "":
            errors.append(f"{field} is required.")
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{field} must be a number.")
            continue
        if not (lo <= val <= hi):
            errors.append(f"{field} should be between {lo} and {hi}.")
            continue
        values[field] = val
    return values, errors


def compute_suitability(crop, values):
    """
    Compares user's input values against the typical (mean +/- std) range
    for the chosen crop, feature by feature. Returns a per-feature verdict
    and an overall suitability score (0-100).
    """
    feature_order = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    crop_data = CROP_STATS[crop]
    breakdown = []
    score_sum = 0

    for feat in feature_order:
        user_val = values[feat]
        mean = crop_data[feat]["mean"]
        std = crop_data[feat]["std"] or 1e-6  # avoid divide-by-zero
        lo = crop_data[feat]["min"]
        hi = crop_data[feat]["max"]

        # Distance from mean, in standard deviations
        z = abs(user_val - mean) / std

        if z <= 1:
            status, feat_score = "Ideal", 100
        elif z <= 2:
            status, feat_score = "Acceptable", 70
        elif z <= 3:
            status, feat_score = "Marginal", 40
        else:
            status, feat_score = "Unsuitable", 10

        score_sum += feat_score
        breakdown.append({
            "feature": feat,
            "user_value": user_val,
            "typical_range": f"{lo} – {hi}",
            "typical_mean": mean,
            "status": status,
        })

    overall_score = round(score_sum / len(feature_order), 1)

    if overall_score >= 85:
        verdict = "Highly Suitable"
    elif overall_score >= 65:
        verdict = "Suitable"
    elif overall_score >= 40:
        verdict = "Marginally Suitable"
    else:
        verdict = "Not Suitable"

    return overall_score, verdict, breakdown


@app.route("/suitability", methods=["GET", "POST"])
def suitability():
    if request.method == "GET":
        return render_template("suitability_form.html", crops=ALL_CROPS)

    crop = request.form.get("crop", "")
    if crop not in CROP_STATS:
        return render_template(
            "suitability_form.html", crops=ALL_CROPS,
            errors=["Please select a valid crop."], form_data=request.form
        )

    values, errors = validate_inputs(request.form)
    if errors:
        return render_template(
            "suitability_form.html", crops=ALL_CROPS,
            errors=errors, form_data=request.form
        )

    score, verdict, breakdown = compute_suitability(crop, values)

    return render_template(
        "suitability_result.html",
        crop=crop.capitalize(),
        score=score,
        verdict=verdict,
        breakdown=breakdown,
    )


@app.route("/research")
def research():
    clusters = cluster_df.groupby("cluster")["crop"].apply(list).to_dict()
    profile_records = cluster_profile_df.to_dict(orient="records")
    return render_template(
        "research.html",
        clusters=clusters,
        profile=profile_records,
        total_crops=len(ALL_CROPS),
        total_samples=2200,
    )





@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    values, errors = validate_inputs(request.form)

    if errors:
        return render_template("index.html", errors=errors, form_data=request.form)

    feature_order = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    features = pd.DataFrame([[values[f] for f in feature_order]], columns=feature_order)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    # Top-3 probabilities for extra insight, if the model supports it
    top_predictions = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_scaled)[0]
        classes = model.classes_
        top_idx = np.argsort(probs)[::-1][:3]
        top_predictions = [(classes[i], round(probs[i] * 100, 2)) for i in top_idx]

    return render_template(
        "result.html",
        crop=prediction.capitalize(),
        top_predictions=top_predictions,
        inputs=values,
    )


if __name__ == "__main__":
    app.run(debug=True)
