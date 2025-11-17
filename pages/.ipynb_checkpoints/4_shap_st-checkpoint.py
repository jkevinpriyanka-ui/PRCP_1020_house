import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("🔍 SHAP Model Explainability")

csv_path = "./house_data_with_predictions.csv"
pipeline_path = "house_pred_files/best_pipeline.joblib"

df = pd.read_csv(csv_path)
best_enet = joblib.load(pipeline_path)

num_cols = best_enet.named_steps['preprocessor'].transformers_[0][2]
cat_cols = best_enet.named_steps['preprocessor'].transformers_[1][2]

preprocessor = best_enet.named_steps["preprocessor"]
model = best_enet.named_steps["model"]

# SHAP Global summary plot
st.subheader("📌 SHAP Summary Plot (Global Importance)")

X = df.drop(["SalePrice", "PredictedPrice"], axis=1)
X_sample = X.sample(200, random_state=42)
X_sample_transformed = preprocessor.transform(X_sample)

# Convert sparse to dense
if hasattr(X_sample_transformed, "toarray"):
    X_sample_transformed = X_sample_transformed.toarray()

# SHAP Explainer
explainer = shap.Explainer(model, X_sample_transformed)
shap_values = explainer(X_sample_transformed)

# Feature names
ohe = preprocessor.named_transformers_['cat']['encoder']
feature_names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))

# Summary plot
fig_summary = plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values.values,
    X_sample_transformed,
    feature_names=feature_names,
    show=False
)
st.pyplot(fig_summary)

# SHAP Waterfall plot
st.subheader("📌 SHAP Waterfall Plot (Single Prediction Explanation)")

selected_idx = st.number_input(
    "Select a row index to explain:",
    min_value=0,
    max_value=len(X) - 1,
    value=10
)

single_row = X.iloc[[selected_idx]]
single_transformed = preprocessor.transform(single_row)

# Convert sparse → dense
if hasattr(single_transformed, "toarray"):
    single_transformed = single_transformed.toarray()

shap_single = explainer(single_transformed)
shap_single.feature_names = feature_names  # attach names manually

# Waterfall plot
fig_waterfall = plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_single[0], show=False)
st.pyplot(fig_waterfall)
