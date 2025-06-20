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
    """
    Trains an ARIMA model and forecasts future values.
    """
    # Note: ARIMA model from statsmodels might not have a specific 'fit' return type annotation
    # Using 'any' for the fitted_model is a practical choice here.
    model = ARIMA(train_series, order=(5, 1, 0)) # Note: Order can be tuned
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=len(test_series))
    return fitted_model, forecast

def train_evaluate_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame, use_tuning: bool = False) -> pd.Series:
    """
    Trains a RandomForestRegressor model, with optional hyperparameter tuning.
    """
    features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'lag1', 'lag7', 'rolling_mean_7']
    target = 'Sales'

    X_train, y_train = train_df[features], train_df[target]
    X_test = test_df[features]

    if use_tuning:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_leaf': [1, 2, 4]
        }
        tscv = TimeSeriesSplit(n_splits=3)
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = RandomForestRegressor(n_estimators=100, random_state=42)
        best_model.fit(X_train, y_train)

    forecast = best_model.predict(X_test)
    return pd.Series(forecast, index=X_test.index)


def train_evaluate_lstm(train_series: pd.Series, test_series: pd.Series, look_back: int = 7) -> pd.Series:
    """
    Trains a Long Short-Term Memory (LSTM) network and forecasts future values.
    """
    # 1. Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
    
    # 2. Create sequences for LSTM
    def create_dataset(dataset, look_back=look_back):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_scaled, look_back)
    
    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 3. Build and train the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0) 

    # 4. Make predictions on the test set
    # We need to use the training data to scale the test data correctly
    full_series_values = pd.concat([train_series, test_series]).values
    inputs = full_series_values[len(full_series_values) - len(test_series) - look_back:]
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(len(test_series)):
        a = inputs[i:(i + look_back), 0]
        X_test.append(a)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    forecast_scaled = model.predict(X_test)
    forecast = scaler.inverse_transform(forecast_scaled)
    
    return pd.Series(forecast.flatten(), index=test_series.index)