import os
import numpy as np
import joblib
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'crop_yield_rf_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'ml_model', 'crop_yield_encoders.pkl')

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

state_map = encoders['state_map']
season_map = encoders['season_map']
crop_map = encoders['crop_map']


def predict_view(request):
    prediction = None

    if request.method == 'POST':
        state = request.POST.get('state')
        crop = request.POST.get('crop')
        season = request.POST.get('season')
        year = int(request.POST.get('year'))
        area = float(request.POST.get('area'))

        state_code = state_map[state]
        season_code = season_map[season]
        crop_code = crop_map[crop]
        log_area = np.log1p(area)

        features = [[state_code, season_code, crop_code, year, log_area]]
        log_pred = model.predict(features)[0]
        prediction = round(float(np.expm1(log_pred)), 2)

    context = {
        'states': sorted(state_map.keys()),
        'crops': sorted(crop_map.keys()),
        'seasons': sorted(season_map.keys()),
        'prediction': prediction,
        'active_tab': 'predict',
    }
    return render(request, 'predictor/predict.html', context)



import json
import pandas as pd

CSV_PATH = os.path.join(BASE_DIR, 'ml_model', 'Crop_clean.csv')
df = pd.read_csv(CSV_PATH)


def analytics_view(request):
    yearly = df.groupby('Crop_Year')['Production'].sum().reset_index()
    top_crops = df.groupby('Crop')['Production'].sum().sort_values(ascending=False).head(10).reset_index()
    top_states = df.groupby('State_Name')['Production'].sum().sort_values(ascending=False).head(10).reset_index()
    season_counts = df['Season'].value_counts().reset_index()
    season_counts.columns = ['Season', 'count']

    context = {
        'active_tab': 'analytics',
        'yearly_labels': json.dumps(yearly['Crop_Year'].tolist()),
        'yearly_values': json.dumps(yearly['Production'].tolist()),
        'crop_labels': json.dumps(top_crops['Crop'].tolist()),
        'crop_values': json.dumps(top_crops['Production'].tolist()),
        'state_labels': json.dumps(top_states['State_Name'].tolist()),
        'state_values': json.dumps(top_states['Production'].tolist()),
        'season_labels': json.dumps(season_counts['Season'].tolist()),
        'season_values': json.dumps(season_counts['count'].tolist()),
    }
    return render(request, 'predictor/analytics.html', context)


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Reproduce the same train/test split used in the notebook, so metrics
# reported here match your actual trained model's real performance.
df['state_code'] = df['State_Name'].map(state_map)
df['season_code'] = df['Season'].map(season_map)
df['crop_code'] = df['Crop'].map(crop_map)
df['log_area'] = np.log1p(df['Area'])
df['log_production'] = np.log1p(df['Production'])

_features = ['state_code', 'season_code', 'crop_code', 'Crop_Year', 'log_area']
_X = df[_features]
_y = df['log_production']
_X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=0.2, random_state=42)

_pred_log = model.predict(_X_test)
_pred = np.expm1(_pred_log)
_actual = np.expm1(_y_test)

_r2_log = round(r2_score(_y_test, _pred_log), 4)
_mae = round(mean_absolute_error(_actual, _pred), 2)
_rmse = round(np.sqrt(mean_squared_error(_actual, _pred)), 2)


def model_view(request):
    sample_idx = np.random.RandomState(0).choice(len(_actual), size=100, replace=False)
    scatter = [
        {'x': round(float(np.log1p(_actual.values[i])), 3),
         'y': round(float(np.log1p(_pred[i])), 3)}
        for i in sample_idx
    ]
    context = {
        'active_tab': 'model',
        'r2_log': _r2_log,
        'mae': _mae,
        'rmse': _rmse,
        'n_train': len(_X_train),
        'n_test': len(_X_test),
        'scatter_json': json.dumps(scatter),
    }
    return render(request, 'predictor/model.html', context)