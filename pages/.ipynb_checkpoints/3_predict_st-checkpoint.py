import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet

st.set_page_config(page_title="Predict", layout="wide")
st.title("🔮 Predict House Price")

# Load data
df =pd.read_csv("house_data_with_predictions.csv")

# Features
numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
categorical_features = ['Neighborhood']

#  pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
model = Pipeline([('preprocessor', preprocessor), ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5))])


X = df[numeric_features + categorical_features]
y = np.log1p(df['SalePrice'])
model.fit(X, y)

# User inputs
st.sidebar.header("Set Feature Values")
input_data = {}
for f in numeric_features:
    min_val, max_val = int(df[f].min()), int(df[f].max())
    default_val = int(df[f].median())
    input_data[f] = st.sidebar.slider(f, min_val, max_val, default_val)

input_data['Neighborhood'] = st.sidebar.selectbox("Neighborhood", sorted(df['Neighborhood'].unique()))

input_df = pd.DataFrame([input_data])
pred_price = np.expm1(model.predict(input_df)[0])
st.subheader("Predicted House Price")
st.success(f"${pred_price:,.0f}")
