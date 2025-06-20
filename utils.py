import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict

def calculate_evaluation_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Calculate MAPE, handling potential division by zero
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }
    return metrics
