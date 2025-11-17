import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="SHAP Explainability", layout="wide")
st.title("🔍 SHAP Model Explainability")

csv_path = "house_app_files/house_data_with_predictions.csv"
pipeline_path = "house_app_files/best_pipeline.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

df = load_data()
st.success("Data Loaded Successfully!")

with open(pipeline_path, "rb") as f:
    model = cloudpickle.load(f)

preprocessor = model.named_steps['preprocessor']
regressor = model.named_steps['regressor']

num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]

X = df[num_cols.tolist() + cat_cols.tolist()]

X_sample = X.sample(200, random_state=42)
X_sample_transformed = preprocessor.transform(X_sample)

if hasattr(X_sample_transformed, "toarray"):
    X_sample_transformed = X_sample_transformed.toarray()

#  Feature names
ohe = preprocessor.named_transformers_['cat']  # OneHotEncoder object
feature_names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))

# SHAP Explainer
explainer = shap.LinearExplainer(regressor, X_sample_transformed, feature_dependence="independent")
shap_values = explainer(X_sample_transformed)

# SHAP Summary Plot
st.subheader("📌 SHAP Summary Plot (Global Feature Importance)")
fig_summary = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample_transformed, feature_names=feature_names, show=False)
st.pyplot(fig_summary)

# SHAP Waterfall Plot (Single Prediction) 
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

fig_waterfall = plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_single[0], show=False)
st.pyplot(fig_waterfall)

#  display original row and prediction
st.subheader("🔹 Original Data and Predicted Price")
st.write(single_row)
pred_log = regressor.predict(single_transformed)[0]
pred_price = np.expm1(pred_log)
st.success(f"Predicted Sale Price: ${pred_price:,.0f}")
