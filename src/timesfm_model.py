import torch
import numpy as np
import timesfm
from abc import ABC, abstractmethod

class PredictionModel(ABC):
    """Base class for time series prediction models."""
    
    @abstractmethod
    def predict(self, window: np.ndarray) -> dict:
        """
        Make a prediction on a time series window.
        
        Args:
            window: numpy array of shape (n, 5) with columns [Date, Open, High, Low, Close, Volume]
                   or (n, 5) with columns [Open, High, Low, Close, Volume] if Date is already removed
        
        Returns:
            dict with keys 'high' and 'low' containing predicted values
        """
        pass


class TimesFmModel(PredictionModel):
    """TimesFM 2.5 200M model implementation."""
    
    def __init__(self):
        torch.set_float32_matmul_precision("high")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
    
    def predict(self, window: np.ndarray) -> dict:
        """
        Make a prediction using TimesFM model.
        
        Args:
            window: numpy array of shape (n, 5) with columns [Open, High, Low, Close, Volume]
        
        Returns:
            dict with 'high' and 'low' keys
        """
        # Ensure window is 2D
        if len(window.shape) == 1:
            window = window.reshape(-1, 1)
        
        # Extract OHLCV series
        series_list = [window[:, i].astype(np.float32) for i in range(window.shape[1])]
        
        # Forecast each feature
        point_forecast, _ = self.model.forecast(
            horizon=1,
            inputs=series_list,
        )
        
        # point_forecast shape (num_features, 1)
        # Column order: Open (0), High (1), Low (2), Close (3), Volume (4)
        return {
            'high': float(point_forecast[1][0]),
            'low': float(point_forecast[2][0]),
            'open': float(point_forecast[0][0]),
            'close': float(point_forecast[3][0]),
        }


if __name__ == '__main__':
    # Example usage for testing
    model = TimesFmModel()
    
    in1 = np.linspace(0, 1, 100)
    in2 = np.sin(np.linspace(0, 20, 67))
    
    point_forecast, quantile_forecast = model.model.forecast(
        horizon=12,
        inputs=[in1, in2],
    )
    print("forecast results:")
    print(point_forecast.shape)  # (2, 12)
    print(quantile_forecast.shape)  # (2, 12, 10)