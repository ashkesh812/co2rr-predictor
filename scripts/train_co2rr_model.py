"""
Physics-informed XGBoost CO2RR predictor training.
Saves models + encoder + metrics summary.
"""

import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

# CONFIG
DATA_FILE = "data/raw/co2rr_dataset.xlsx"
MODEL_DIR = "models/"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# LOAD DATA
df = pd.read_excel(DATA_FILE)
print("Dataset columns:", df.columns.tolist())

# V_thermo map
V_thermo_map = {'CO': -0.11, 'HCOOH': -0.25, 'CH4': -0.24, 'C2H5OH': -0.33, 'H2': 0.00}
df['V_thermo'] = df['Product'].map(V_thermo_map)
df['Overpotential'] = df['Applied Potential (V)'] - df['V_thermo']
df['exp_overpotential'] = np.exp(df['Overpotential'])
df['CatalystClass'] = df['Catalyst'].map({
    'Cu': 'Cu', 'Ag': 'Noble', 'Au': 'Noble', 'Sn': 'PostTransition'
})

# FEATURES
input_columns = ['Catalyst', 'Product', 'CatalystClass', 'Applied Potential (V)', 'Overpotential', 'exp_overpotential']
output_columns = ['Faradaic Efficiency (%)', 'Current Density (mAcm2)', 'Selectivity', 'Cost ($tCO2)']

# ENCODING
encoder = OneHotEncoder()
X_cat = encoder.fit_transform(df[['Catalyst', 'Product', 'CatalystClass']]).toarray()
X_num = df[['Applied Potential (V)', 'Overpotential', 'exp_overpotential']].values
X = np.hstack([X_cat, X_num])
joblib.dump(encoder, f"{MODEL_DIR}/onehot_encoder.pkl")

# TRAINING
metrics_summary = {}
log_targets = ['Cost ($tCO2)']

for target in output_columns:
    y = df[target]
    if y.isna().sum() > 0:
        print(f"⚠️ Skipping {target}: missing values present.")
        continue

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = pd.DataFrame(X[train_idx]), pd.DataFrame(X[test_idx])
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        if target in log_targets:
            y_train = np.log1p(y_train)
            y_test = np.log1p(y_test)

        model = XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=6,
                             learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if target in log_targets:
            y_pred = np.expm1(y_pred)
            y_test = np.expm1(y_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    metrics_summary[target] = {
        'MAE': np.mean(maes),
        'RMSE': np.mean(rmses),
        'R2': np.mean(r2s)
    }

    model_final = XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=6,
                               learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model_final.fit(pd.DataFrame(X), np.log1p(y) if target in log_targets else y)
    joblib.dump(model_final, f"{MODEL_DIR}/{target}_xgb_model.pkl")

# Save metrics
with open(f"{MODEL_DIR}/metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

# SUMMARY
print("✅ All models trained and saved. Metrics summary:")
for k, v in metrics_summary.items():
    print(f"{k}: MAE={v['MAE']:.3f}, RMSE={v['RMSE']:.3f}, R²={v['R2']:.3f}")
