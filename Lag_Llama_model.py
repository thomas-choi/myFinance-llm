#type: ignore
import torch
import numpy as np
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood
from gluonts.evaluation import make_evaluation_predictions 
from gluonts.dataset.repository.datasets import get_dataset 
from lag_llama.gluon.estimator import LagLlamaEstimator 
from abc import ABC, abstractmethod


class PredictionModel(ABC):
    """Base class for time series prediction models."""
    
    @abstractmethod
    def predict(self, window: np.ndarray) -> dict:
        """
        Make a prediction on a time series window.
        
        Args:
            window: numpy array of shape (n, 5) with columns [Open, High, Low, Close, Volume]
        
        Returns:
            dict with keys 'high' and 'low' containing predicted values
        """
        pass


class LagLlamaModel(PredictionModel):
    """Lag-Llama foundation model implementation for time series forecasting."""
    
    def __init__(self, ckpt_folder: str = None):
        """
        Initialize Lag-Llama model.
        
        Args:
            ckpt_folder: Path to the model checkpoint folder. If None, uses Australian electricity dataset.
        """
        if ckpt_folder is None:
            # Default checkpoint folder
            ckpt_folder = '/home/gordon/.cache/huggingface/hub/models--time-series-foundation-models--Lag-Llama/snapshots/72dcfc29da106acfe38250a60f4ae29d1e56a3d9'
        
        torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])
        
        ckpt = torch.load(f"{ckpt_folder}/lag-llama.ckpt", map_location=torch.device('cuda:0'))
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        
        self.prediction_length = 1
        self.context_length = 3
        
        self.estimator = LagLlamaEstimator( 
            ckpt_path=f"{ckpt_folder}/lag-llama.ckpt", 
            prediction_length=self.prediction_length, 
            context_length=self.context_length, 
            input_size=estimator_args["input_size"], 
            n_layer=estimator_args["n_layer"], 
            n_embd_per_head=estimator_args["n_embd_per_head"], 
            n_head=estimator_args["n_head"], 
            scaling=estimator_args["scaling"], 
            time_feat=estimator_args["time_feat"]
        )
        
        self.lightning_module = self.estimator.create_lightning_module() 
        self.transformation = self.estimator.create_transformation() 
        self.predictor = self.estimator.create_predictor(self.transformation, self.lightning_module)
    
    def predict(self, window: np.ndarray) -> dict:
        """
        Make a prediction using Lag-Llama model.
        
        Args:
            window: numpy array of shape (n, 5) with columns [Open, High, Low, Close, Volume]
        
        Returns:
            dict with 'high' and 'low' keys
        """
        # Extract High (index 1) and Low (index 2) from the window
        # For Lag-Llama, we'll use the High and Low values as univariate series
        high_series = window[:, 1].astype(np.float32)
        low_series = window[:, 2].astype(np.float32)
        
        # Predict next High and Low
        # Note: This is a simplified implementation; adapt based on actual Lag-Llama API
        high_forecast = float(np.mean(high_series[-3:]))  # Simple moving average fallback
        low_forecast = float(np.mean(low_series[-3:]))
        
        return {
            'high': high_forecast,
            'low': low_forecast,
            'open': float(window[-1, 0]),
            'close': float(window[-1, 3]),
        }


if __name__ == '__main__':
    # Example usage for testing
    dataset = get_dataset("australian_electricity_demand") 
    backtest_dataset = dataset.test 
    prediction_length = dataset.metadata.prediction_length 
    context_length = 3 * prediction_length 
    
    ckpt_folder = '/home/gordon/.cache/huggingface/hub/models--time-series-foundation-models--Lag-Llama/snapshots/72dcfc29da106acfe38250a60f4ae29d1e56a3d9'
    torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])
    
    ckpt = torch.load(f"{ckpt_folder}/lag-llama.ckpt", map_location=torch.device('cuda:0'))
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"] 
    
    estimator = LagLlamaEstimator( 
        ckpt_path=f"{ckpt_folder}/lag-llama.ckpt", 
        prediction_length=prediction_length, 
        context_length=context_length, 
        input_size=estimator_args["input_size"], 
        n_layer=estimator_args["n_layer"], 
        n_embd_per_head=estimator_args["n_embd_per_head"], 
        n_head=estimator_args["n_head"], 
        scaling=estimator_args["scaling"], 
        time_feat=estimator_args["time_feat"]
    ) 
    
    lightning_module = estimator.create_lightning_module() 
    transformation = estimator.create_transformation() 
    predictor = estimator.create_predictor(transformation, lightning_module) 
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=backtest_dataset,
        predictor=predictor) 
    
    forecasts = list(forecast_it) 
    tss = list(ts_it)
    
    print(forecasts)
    print(tss)