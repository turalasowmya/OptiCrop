# OptiCrop — Smart Agricultural Production Optimization Engine

A machine learning–powered crop recommendation system. Enter soil nutrients
(N, P, K) and environmental conditions (temperature, humidity, pH, rainfall)
to get an instant crop recommendation, backed by a Random Forest classifier
trained on 2,200 samples across 22 crop types.

## Project Structure

```
opticrop/
├── data/
│   └── Crop_recommendation.csv      # training dataset
├── model/
│   ├── crop_model.pkl               # trained Random Forest model
│   ├── scaler.pkl                   # StandardScaler used for preprocessing
│   ├── correlation_heatmap.png      # EDA output
│   ├── feature_boxplots.png         # EDA output
│   ├── confusion_matrix.png         # model evaluation output
│   └── model_comparison.csv         # accuracy comparison across models
├── notebooks/
│   └── train_model.py               # data prep + model training + comparison
├── templates/
│   ├── index.html                   # input form
│   └── result.html                  # recommendation result page
├── static/
│   └── css/style.css
├── app.py                           # Flask application
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Retrain the model (optional — a trained model is already included)

```bash
python notebooks/train_model.py
```

This re-runs EDA, trains KNN, Logistic Regression, Decision Tree, and
Random Forest, compares them, and overwrites `model/crop_model.pkl` and
`model/scaler.pkl` with the best performer.

## Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Model Performance

| Model               | Test Accuracy | CV Accuracy |
|----------------------|---------------|-------------|
| KNN                  | 97.95%        | 96.53%      |
| Logistic Regression  | 97.27%        | 96.76%      |
| Decision Tree        | 97.95%        | 98.47%      |
| **Random Forest**    | **99.55%**    | **99.43%**  |

Random Forest was selected as the production model and is what `app.py` loads.

## How It Works

1. The user submits N, P, K, temperature, humidity, pH, and rainfall via the
   web form.
2. `app.py` validates the inputs are present, numeric, and within realistic
   agricultural ranges.
3. Inputs are scaled with the same `StandardScaler` used during training.
4. The Random Forest model predicts the most suitable crop, along with the
   top-3 most likely crops and their probabilities.
5. The result is rendered on `result.html`.

## Next Steps / Possible Extensions

- Persist user submissions to a database for trend analysis over time.
- Deploy to a cloud host (Render, Railway, PythonAnywhere) once local testing
  is complete.
- Add user authentication for the research dashboard if it should be
  restricted to researchers/policymakers only.

## All Three Scenarios — Now Implemented

**Scenario 1 — Smart Crop Recommendation** (`/`)
Farmer enters N, P, K, temperature, humidity, pH, rainfall and gets the
single best-matching crop plus top-3 alternative matches with probabilities.

**Scenario 2 — Crop Suitability & Environmental Assessment** (`/suitability`)
User selects a specific crop and enters their conditions. The app compares
each input against that crop's typical range (mean ± std, computed from the
training data) and returns a per-parameter status (Ideal / Acceptable /
Marginal / Unsuitable) plus an overall 0–100 suitability score and verdict.

**Scenario 3 — Agricultural Research & Policy Planning** (`/research`)
A dashboard showing a feature correlation heatmap and K-Means clustering of
all 22 crops by their average environmental needs (visualized via PCA),
along with the average N/P/K/climate profile of each cluster. Useful for
spotting which crops have similar resource requirements.

To regenerate the suitability stats and clustering analysis after retraining:
```bash
python notebooks/cluster_analysis.py
```
This writes `model/crop_stats.json`, `model/crop_clusters.csv`,
`model/cluster_profile.csv`, and `model/crop_clusters_pca.png`. If you
re-run it, also copy the new PNGs into `static/images/` so the research page
picks up the updated chart:
```bash
cp model/crop_clusters_pca.png model/correlation_heatmap.png static/images/
```
