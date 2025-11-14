import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="EDA",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis")

csv_path = "./house_data_with_predictions.csv"  # Relative path
df = pd.read_csv(csv_path)

# --- SalePrice Distribution ---
st.subheader("SalePrice Distribution")
fig = px.histogram(df, x='SalePrice', nbins=50, title="SalePrice Distribution")
st.plotly_chart(fig, use_container_width=True)

# --- Log-transformed SalePrice ---
st.subheader("Log(SalePrice + 1) Distribution")
df['LogSalePrice'] = np.log1p(df['SalePrice'])
fig_log = px.histogram(df, x='LogSalePrice', nbins=50, title="Log-transformed SalePrice")
st.plotly_chart(fig_log, use_container_width=True)

# --- Top 10 Numeric Correlations ---
st.subheader("Top 10 Numeric Features Correlated with SalePrice")
numeric_df = df.select_dtypes(include=['float64','int64'])
if 'SalePrice' in numeric_df.columns:
    corr = numeric_df.corr()['SalePrice'].sort_values(ascending=False)[1:11]
    corr_df = corr.reset_index().rename(columns={'index':'Feature','SalePrice':'Correlation'})
    fig_corr = px.bar(corr_df, x='Feature', y='Correlation', color='Correlation', title="Top 10 Correlated Features")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.write("No numeric columns found.")

# --- Categorical Feature Analysis ---
st.subheader("Categorical Feature Analysis")
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    selected_cat = st.selectbox("Select Category", cat_cols)
    fig_box = px.box(df, x=selected_cat, y='LogSalePrice', title=f"Log(SalePrice) vs {selected_cat}")
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.write("No categorical columns found.")
