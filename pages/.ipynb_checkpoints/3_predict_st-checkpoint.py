import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet

st.set_page_config(page_title="Predict House Price", layout="wide")
st.title("🔮 Predict House Price")

# --- Load data ---
csv_path = "house_app_files/house_data_with_predictions.csv"
df = pd.read_csv(csv_path)

# --- Features ---
numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
categorical_features = ['Neighborhood']

# --- Load or train pipeline ---
pipeline_path = "house_app_files/best_pipeline.pkl"
if os.path.exists(pipeline_path):
    with open(pipeline_path, "rb") as f:
        model = cloudpickle.load(f)
    st.info("✅ Loaded saved pipeline")
else:
    st.warning("Pipeline not found, training a new one...")
    X_train = df[numeric_features + categorical_features]
    y_train = np.log1p(df['SalePrice'])

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5))
    ])

    model.fit(X_train, y_train)
    os.makedirs("house_app_files", exist_ok=True)
    with open(pipeline_path, "wb") as f:
        cloudpickle.dump(model, f)
    st.success("✅ Pipeline trained and saved")

# --- Sidebar: User input ---
st.sidebar.header("Set Feature Values")
input_data = {}
for f in numeric_features:
    min_val, max_val = int(df[f].min()), int(df[f].max())
    default_val = int(df[f].median())
    step = 1 if max_val < 50 else 50 if max_val > 1000 else 10
    input_data[f] = st.sidebar.slider(f, min_val, max_val, default_val, step=step)

input_data['Neighborhood'] = st.sidebar.selectbox(
    "Neighborhood", sorted(df['Neighborhood'].unique())
)

input_df = pd.DataFrame([input_data])

# --- Predict ---
pred_price = np.expm1(model.predict(input_df)[0])

# --- Display ---
st.subheader("Predicted House Price")
st.success(f"${pred_price:,.0f}")
st.write("### Input Features")
st.table(input_df)
