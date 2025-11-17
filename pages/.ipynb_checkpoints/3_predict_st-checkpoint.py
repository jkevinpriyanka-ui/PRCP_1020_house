import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

st.set_page_config(page_title="Predict", layout="wide")
st.title("🔮 Predict House Price")

# --- Load CSV data ---
csv_path = "house_app_files/house_data_with_predictions.csv"
df = pd.read_csv(csv_path)

pipeline_path = "house_app_files/pipeline.pkl"
with open(pipeline_path, "rb") as f:
    model = cloudpickle.load(f)

#  Features
numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
categorical_features = ['Neighborhood']

#  Sidebar
st.sidebar.header("Set Feature Values")
input_data = {}

# Numeric sliders
for f in numeric_features:
    min_val, max_val = int(df[f].min()), int(df[f].max())
    default_val = int(df[f].median())
    input_data[f] = st.sidebar.slider(f, min_val, max_val, default_val, step=1)

# Categorical selectbox
input_data['Neighborhood'] = st.sidebar.selectbox(
    "Neighborhood", sorted(df['Neighborhood'].unique())
)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

#  Predict 
pred_price = np.expm1(model.predict(input_df)[0])

# Display 
st.subheader("Predicted House Price")
st.success(f"${pred_price:,.0f}")
