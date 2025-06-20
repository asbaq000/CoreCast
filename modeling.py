import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


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


def train_evaluate_lstm(train_series: pd.Series, test_series: pd.Series, look_back: int = 14) -> pd.Series:
   
    scaler = MinMaxScaler(feature_range=(0, 1))

    full_series = pd.concat([train_series, test_series])
    full_series_scaled = scaler.fit_transform(full_series.values.reshape(-1, 1))
    
    train_scaled = full_series_scaled[:len(train_series)]
 
    def create_dataset(dataset, look_back=look_back):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_scaled, look_back)
  
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)

    test_predictions_scaled = []
   
    current_batch = train_scaled[-look_back:].reshape(1, look_back, 1)

    for i in range(len(test_series)):
        pred = model.predict(current_batch, verbose=0)[0]
        test_predictions_scaled.append(pred)
       
        next_val_scaled = full_series_scaled[len(train_series) + i]
        current_batch = np.append(current_batch[:, 1:, :], [[next_val_scaled]], axis=1)

    forecast = scaler.inverse_transform(test_predictions_scaled)
    
    return pd.Series(forecast.flatten(), index=test_series.index)
