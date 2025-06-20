import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple

def train_evaluate_arima(train_series: pd.Series, test_series: pd.Series) -> Tuple[any, pd.Series]:

    model = ARIMA(train_series, order=(5, 1, 0)) 
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=len(test_series))
    return fitted_model, pd.Series(forecast, index=test_series.index)

def train_evaluate_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[RandomForestRegressor, pd.Series]:
   
    features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'lag1', 'lag7', 'rolling_mean_7']
    target = 'Sales'

    X_train, y_train = train_df[features], train_df[target]
    X_test = test_df[features]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    prediction_series = pd.Series(predictions, index=X_test.index)
    
    return model, prediction_series
