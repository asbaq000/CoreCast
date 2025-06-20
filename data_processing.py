import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_and_preprocess_data(uploaded_file):
   
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(uploaded_file)


        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            st.error("Error: No 'Date' column found in the uploaded CSV. Please ensure your date column is named 'Date' or contains 'date'.")
            return pd.DataFrame()

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.dropna(subset=[date_col], inplace=True)
        df.set_index(date_col, inplace=True)

        sales_col = None
        if 'Sales' in df.columns:
            sales_col = 'Sales'
        else:
            for col in df.columns:
                if any(name in col.lower() for name in ['sales', 'quantity', 'amount', 'value']):
                    df.rename(columns={col: 'Sales'}, inplace=True)
                    sales_col = 'Sales'
                    break
        
        if sales_col is None:
            st.error("Error: No 'Sales' column found. Please name your target column 'Sales' or a similar term.")
            return pd.DataFrame()

        df_resampled = df[[sales_col]].resample('D').sum()

        df_resampled[sales_col].replace(0, np.nan, inplace=True)
        df_resampled.fillna(method='ffill', inplace=True)
        df_resampled.fillna(method='bfill', inplace=True)
        
        if df_resampled.isnull().values.any():
             st.warning("Could not fill all missing values. Consider cleaning the data source.")
             df_resampled.fillna(0, inplace=True)


        return df_resampled

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return pd.DataFrame()

def create_features(df):
    df_featured = df.copy()
    df_featured['dayofweek'] = df_featured.index.dayofweek
    df_featured['quarter'] = df_featured.index.quarter
    df_featured['month'] = df_featured.index.month
    df_featured['year'] = df_featured.index.year
    df_featured['dayofyear'] = df_featured.index.dayofyear
    
    df_featured['lag1'] = df_featured['Sales'].shift(1)
    df_featured['lag7'] = df_featured['Sales'].shift(7)
    
    df_featured['rolling_mean_7'] = df_featured['Sales'].rolling(window=7).mean()

    df_featured.dropna(inplace=True)
    
    return df_featured
