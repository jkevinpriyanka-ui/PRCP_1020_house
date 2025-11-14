import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="EDA", page_icon="📊")
st.title("Exploratory Data Analysis")

# Load data
df = pd.read_csv("house_data_with_predictions.csv")

# SalePrice distribution
st.subheader("SalePrice Distribution")
fig = px.histogram(df, x='SalePrice', nbins=50, title="SalePrice Distribution")
st.plotly_chart(fig, use_container_width=True)

# Buttons for correlation
st.subheader("Top Features Correlated with SalePrice")
col1, col2 = st.columns(2)

with col1:
    if st.button("Original SalePrice"):
        corr = df.corr()['SalePrice'].sort_values(ascending=False)[1:11]
        corr_df = corr.reset_index().rename(columns={'index':'Feature', 'SalePrice':'Correlation'})
        fig_corr = px.bar(corr_df, x='Feature', y='Correlation', color='Correlation', title="Top 10 Correlated Features")
        st.plotly_chart(fig_corr, use_container_width=True)

with col2:
    if st.button("Log-Transformed SalePrice"):
        # Log-transform SalePrice
        df_log = df.copy()
        df_log['LogSalePrice'] = np.log1p(df_log['SalePrice'])
        
        # Log-transform numeric features if skewed
        numeric_cols = df_log.select_dtypes(include=['int64','float64']).columns
        skew_thresh = 1
        skewed_features = df_log[numeric_cols].skew().abs() > skew_thresh
        for col in numeric_cols:
            if skewed_features[col]:
                df_log[col] = np.log1p(df_log[col])
        
        corr_log = df_log.corr()['LogSalePrice'].sort_values(ascending=False)[1:11]
        corr_log_df = corr_log.reset_index().rename(columns={'index':'Feature', 'LogSalePrice':'Correlation'})
        fig_log = px.bar(corr_log_df, x='Feature', y='Correlation', color='Correlation', title="Top 10 Correlated Features (Log-Transformed)")
        st.plotly_chart(fig_log, use_container_width=True)
