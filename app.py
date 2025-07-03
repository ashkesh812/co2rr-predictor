import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load models + metrics
MODEL_DIR = "models/"
output_columns = ['Faradaic Efficiency (%)', 'Current Density (mAcm2)', 'Selectivity', 'Cost ($tCO2)']
models = {t: joblib.load(f"{MODEL_DIR}/{t}_xgb_model.pkl") for t in output_columns}
encoder = joblib.load(f"{MODEL_DIR}/onehot_encoder.pkl")
with open(f"{MODEL_DIR}/metrics_summary.json") as f:
    metrics_summary = json.load(f)

# V_thermo
V_thermo_map = {'CO': -0.11, 'HCOOH': -0.25, 'CH4': -0.24, 'C2H5OH': -0.33, 'H2': 0.00}

# UI
st.title("CO2RR Predictor")
catalyst = st.selectbox("Catalyst", ['Cu', 'Ag', 'Au', 'Sn'])
product = st.selectbox("Product", ['CO', 'HCOOH', 'CH4', 'C2H5OH', 'H2'])
potential = st.number_input("Applied Potential (V)", -2.0, 0.0, -1.0, 0.05)

V_thermo = V_thermo_map.get(product, 0.0)
overpotential = potential - V_thermo
exp_overpotential = np.exp(overpotential)

# Build input
input_df = pd.DataFrame({
    'Catalyst': [catalyst],
    'Product': [product],
    'CatalystClass': ['Cu' if catalyst == 'Cu' else ('Noble' if catalyst in ['Ag', 'Au'] else 'PostTransition')]
})
X_cat = encoder.transform(input_df).toarray()
X_num = np.array([[potential, overpotential, exp_overpotential]])
X_input = pd.DataFrame(np.hstack([X_cat, X_num]))

# Predict
results = {}
for target in output_columns:
    pred = models[target].predict(X_input)[0]
    if target == 'Cost ($tCO2)':
        pred = np.expm1(pred)
    results[target] = pred

# Derived production rate
prod_rate = results['Current Density (mAcm2)'] * results['Faradaic Efficiency (%)'] * 0.0001

# Show results
st.subheader("Predictions")
for target in output_columns:
    st.write(f"**{target}:** {results[target]:.2f}")
st.write(f"**Derived Production Rate (m³/hr):** {prod_rate:.4f}")

# Show model performance
st.subheader("Model performance (10-fold CV)")
for target in output_columns:
    st.write(f"**{target}** - R²: {metrics_summary[target]['R2']:.3f}, RMSE: {metrics_summary[target]['RMSE']:.3f}, MAE: {metrics_summary[target]['MAE']:.3f}")
