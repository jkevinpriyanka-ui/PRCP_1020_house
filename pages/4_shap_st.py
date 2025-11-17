import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="SHAP Explainability", layout="wide")
st.title("🔍 SHAP Model Explainability")

# --- File paths ---
csv_path = "house_app_files/house_data_with_predictions.csv"
pipeline_path = "house_app_files/best_pipeline.pkl"


# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

df = load_data()
st.success("Data Loaded Successfully!")

# --- Load trained pipeline ---
with open(pipeline_path, "rb") as f:
    best_enet = cloudpickle.load(f)

preprocessor = best_enet.named_steps["preprocessor"]
model = best_enet.named_steps["model"]

# --- Define numeric and categorical features ---
num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]

# --- Prepare data for SHAP ---
X = df.drop(["SalePrice", "PredictedPrice"], axis=1)
X_sample = X.sample(200, random_state=42)
X_sample_transformed = preprocessor.transform(X_sample)

# Convert sparse to dense if needed
if hasattr(X_sample_transformed, "toarray"):
    X_sample_transformed = X_sample_transformed.toarray()

# --- SHAP Explainer ---
explainer = shap.Explainer(model, X_sample_transformed)
shap_values = explainer(X_sample_transformed)

# --- Feature names ---
try:
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehotencoder']
except AttributeError:
    ohe = preprocessor.named_transformers_['cat']  # fallback if not Pipeline

feature_names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))

# --- SHAP Summary Plot ---
st.subheader("📌 SHAP Summary Plot (Global Feature Importance)")
fig_summary = plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values.values,
    X_sample_transformed,
    feature_names=feature_names,
    show=False
)
st.pyplot(fig_summary)

# --- SHAP Waterfall Plot for Single Prediction ---
st.subheader("📌 SHAP Waterfall Plot (Single Prediction Explanation)")
selected_idx = st.number_input(
    "Select a row index to explain:",
    min_value=0,
    max_value=len(X) - 1,
    value=10
)

single_row = X.iloc[[selected_idx]]
single_transformed = preprocessor.transform(single_row)
if hasattr(single_transformed, "toarray"):
    single_transformed = single_transformed.toarray()

shap_single = explainer(single_transformed)
shap_single.feature_names = feature_names  # attach names

fig_waterfall = plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_single[0], show=False)
st.pyplot(fig_waterfall)
