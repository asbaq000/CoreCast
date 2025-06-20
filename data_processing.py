import pandas as pd
from typing import Tuple

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads sales data from a CSV, sets a daily frequency index, and fills missing values.
    """
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'])
        data.set_index('Date', inplace=True)
        # Resample to daily frequency, filling missing days with 0 sales
        data = data.asfreq('D').fillna(0)
        return data
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return pd.DataFrame()

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-series features from the date index.
    """
    df_featured = df.copy()
    
    # --- THIS IS THE CORRECTED LINE ---
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