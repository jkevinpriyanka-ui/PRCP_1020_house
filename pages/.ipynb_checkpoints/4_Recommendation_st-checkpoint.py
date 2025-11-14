import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
import plotly.express as px

st.title("🏠 House Recommendations")

# Load CSV
df = pd.read_csv("house_data_with_predictions.csv")

# pipeline
numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
categorical_features = ['Neighborhood']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
model = Pipeline([('preprocessor', preprocessor), ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5))])

X = df[numeric_features + categorical_features]
y = np.log1p(df['SalePrice'])
model.fit(X, y)

# Sidebar filters
st.sidebar.header("Filter Houses")
max_price = st.sidebar.slider("Maximum Sale Price", int(df['SalePrice'].min()), int(df['SalePrice'].max()), 300000, step=5000)
min_quality = st.sidebar.slider("Minimum Overall Quality", 1, 10, 5)
neighborhood_options = ['All'] + sorted(df['Neighborhood'].unique())
neighborhood = st.sidebar.selectbox("Select Neighborhood", neighborhood_options)

filtered_df = df.copy()
if neighborhood != 'All':
    filtered_df = filtered_df[filtered_df['Neighborhood'] == neighborhood]
filtered_df = filtered_df[(filtered_df['SalePrice'] <= max_price) & (filtered_df['OverallQual'] >= min_quality)]

# Predict filtered houses
X_filtered = filtered_df[numeric_features + categorical_features]
filtered_df['PredictedPrice'] = np.expm1(model.predict(X_filtered))
filtered_df['DiffPercent'] = ((filtered_df['PredictedPrice'] - filtered_df['SalePrice']) / filtered_df['SalePrice']) * 100

# Top 5 deals
top_deals = filtered_df.sort_values('DiffPercent', ascending=False).head(5)
st.subheader("Top 5 Recommended Houses")
st.dataframe(top_deals[['Neighborhood','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','YearBuilt','SalePrice','PredictedPrice','DiffPercent']])

# Filtered Actual vs Predicted
st.subheader("Predicted vs Sale Price - Filtered Houses")
fig = px.scatter(filtered_df, x='SalePrice', y='PredictedPrice', color='Neighborhood',
                 size='OverallQual', hover_data=numeric_features + ['YearBuilt'])
fig.add_shape(type="line", line=dict(color='red', dash='dash'),
              x0=filtered_df['SalePrice'].min(), x1=filtered_df['SalePrice'].max(),
              y0=filtered_df['SalePrice'].min(), y1=filtered_df['SalePrice'].max())
st.plotly_chart(fig, use_container_width=True)
