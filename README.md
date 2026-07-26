# Crop Yield Prediction — Django Edition

A full-stack machine learning web app that predicts crop production across Indian states using a Random Forest Regressor trained on 6,900+ historical agricultural records.

🔗 **Live demo:** https://crop-yield-django.onrender.com
🔗 **Lightweight client-side version (no backend):** https://crop-yieldprediction.netlify.app
📓 **Training notebook:** [Crop_Yield_Prediction_Clean.ipynb](./Crop_Yield_Prediction_Clean.ipynb)

> Note: the live demo runs on a free Render instance and sleeps after ~15 minutes of inactivity — the first request after that may take 30–50 seconds to respond while it wakes up.

---

## Overview

Crop production estimation in India traditionally relies on manual, survey-based reporting that is slow and doesn't adapt well across the country's diversity of states, crops, and seasons. This project addresses that gap with a data-driven system that predicts production from historical patterns — validated properly, explained clearly, and deployed as something anyone can actually use.

## Features

- **Predict** — enter state, crop, season, year, and cultivated area to get a live production estimate from the real trained model (not a simplified copy)
- **Analytics** — production trends by year, top crops, top states, and season distribution, computed live via pandas
- **Model** — real evaluation metrics (R², MAE, RMSE) and a predicted-vs-actual scatter plot, recomputed against the actual saved model on every page load

## Results

| Metric | Value |
|---|---|
| R² (log-space, test set) | ≈ 0.91 |
| Records used | 6,918 (cleaned from 7,383 raw) |
| States / Crops / Seasons | 25 / 97 / 6 |
| Year range | 1997–2014 |

**Data quality note:** the raw dataset had a column-shift corruption affecting ~5% of rows (missing `State_Name` caused every subsequent field to shift one column left). These rows were detected and removed rather than silently trained on.

**Why log-space R²:** Production spans single digits to hundreds of millions in this dataset. The target is log-transformed (`log1p`) before training to prevent a handful of extreme values from dominating the loss — R² is reported in that same space, where it reflects fit quality across the full range of crops rather than just the largest few.

## Tech stack

| Layer | Tools |
|---|---|
| Data cleaning & training | Python, Pandas, NumPy, scikit-learn (in Google Colab) |
| Backend | Django, Gunicorn |
| Model serving | joblib (loads the real trained `RandomForestRegressor` directly, no reimplementation) |
| Frontend | Django templates, Chart.js |
| Static files | WhiteNoise |
| Deployment | Render (backend) |
| Version control | Git / GitHub |

## Project structure

```
crop-yield-django/
├── cropsite/              # Django project (settings, URLs, WSGI)
├── predictor/              # Django app
│   ├── views.py            # Predict / Analytics / Model view logic
│   ├── urls.py              # App-level routing
│   ├── ml_model/            # Trained model, encoders, cleaned dataset
│   ├── templates/predictor/ # HTML templates
│   └── static/predictor/    # CSS
├── manage.py
├── requirements.txt
└── Procfile                 # Tells Render how to start the app
```

## Running it locally

```bash
git clone https://github.com/SweathaAngappan/crop-yield-django.git
cd crop-yield-django
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python manage.py runserver
```

Then open `http://127.0.0.1:8000/` in your browser.

## Data source & known limitations

- Dataset: Indian crop production records (1997–2014), 25 states, 97 crops.
- The model uses state, crop, season, year, and cultivated area only — it does not use soil or weather data.
- `Crop_Year` contributes only ~1% to the model's predictions, so predictions for years beyond the training range (post-2014) extrapolate rather than reflect a learned future trend.
- The model was trained with scikit-learn 1.6.1; `requirements.txt` pins this exact version to avoid version-mismatch warnings when unpickling the saved model.