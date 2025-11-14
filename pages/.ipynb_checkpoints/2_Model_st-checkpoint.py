import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet

st.title("📊 Model Insights")

df = pd.read_csv("house_data_with_predictions.csv")

numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
categorical_features = ['Neighborhood']

# pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5))
])

X = df[numeric_features + categorical_features]
y = np.log1p(df['SalePrice'])  # log-transform
model.fit(X, y)

# Predictions
df['PredictedPrice'] = np.expm1(model.predict(X))

# Actual vs Predicted
st.subheader("Actual vs Predicted Prices")
fig = px.scatter(df, x='SalePrice', y='PredictedPrice', hover_data=['Neighborhood', 'GrLivArea'])
fig.add_shape(type="line", line=dict(dash="dash", color="red"),
              x0=df['SalePrice'].min(), x1=df['SalePrice'].max(),
              y0=df['SalePrice'].min(), y1=df['SalePrice'].max())
st.plotly_chart(fig, use_container_width=True)

# Top 10 influential features
st.subheader("Top 10 Influential Features")
feature_names = numeric_features + list(model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
coefs = model.named_steps['regressor'].coef_
feat_imp = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs, 'AbsCoeff': np.abs(coefs)})
top10 = feat_imp.sort_values('AbsCoeff', ascending=False).head(10)
fig2 = px.bar(top10, x='AbsCoeff', y='Feature', orientation='h', color='AbsCoeff', title="Top 10 Features")
fig2.update_yaxes(categoryorder='total ascending')
st.plotly_chart(fig2, use_container_width=True)
