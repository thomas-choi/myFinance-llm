"""
Model registry for managing different prediction models.
This provides an extensible way to register and instantiate models.
"""

from typing import Dict, Type, Optional
from timesfm_model import TimesFmModel, PredictionModel
import sys
import os

# Import Lag-Llama model from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from Lag_Llama_model import LagLlamaModel


class ModelRegistry:
    """Registry for time series prediction models."""
    
    _models: Dict[str, Type[PredictionModel]] = {}
    _instances: Dict[str, PredictionModel] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[PredictionModel]) -> None:
        """
        Register a new model class.
        
        Args:
            name: Model identifier (e.g., 'timesfm', 'lag_llama')
            model_class: Model class that implements PredictionModel interface
        """
        cls._models[name.lower()] = model_class
    
    @classmethod
    def get_model(cls, name: str, **kwargs) -> PredictionModel:
        """
        Get or instantiate a model by name.
        
        Args:
            name: Model identifier
            **kwargs: Additional arguments to pass to the model constructor
        
        Returns:
            Instance of the requested model
        
        Raises:
            ValueError: If model is not registered
        """
        name = name.lower()
        
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available}")
        
        # Return cached instance if available (models are stateful)
        if name not in cls._instances:
            cls._instances[name] = cls._models[name](**kwargs)
        
        return cls._instances[name]
    
    @classmethod
    def list_models(cls) -> list:
        """Return list of registered model names."""
        return list(cls._models.keys())


# Register default models
ModelRegistry.register('timesfm', TimesFmModel)
# ModelRegistry.register('lag_llama', LagLlamaModel)
