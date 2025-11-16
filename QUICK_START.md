# Quick Start Guide

## System Overview

The refactored prediction system provides a clean, extensible architecture for time series forecasting:

```
predict.py (Top-level orchestrator)
    ↓
model_registry.py (Model management)
    ↓
Model implementations:
  - timesfm_model.py (TimesFM)
  - Lag_Llama_model.py (Lag-Llama)
  - [Your new models...]
```

## Basic Commands

### Evaluate TimesFM on test data
```bash
python src/predict.py --model timesfm --data data/prepared_data.csv --eval --output metrics.json
```

### Evaluate Lag-Llama on test data
```bash
python src/predict.py --model lag_llama --data data/prepared_data.csv --eval --output metrics.json
```

### Make single prediction with TimesFM
```bash
python src/predict.py --model timesfm --data data/prepared_data.csv --predict --sample 0
```

### Make single prediction with Lag-Llama
```bash
python src/predict.py --model lag_llama --data data/prepared_data.csv --predict --sample 5
```

### Get help
```bash
python src/predict.py --help
```

## How to Add a New Model

### 1. Create model file (e.g., `src/my_new_model.py`)
```python
from timesfm_model import PredictionModel
import numpy as np

class MyNewModel(PredictionModel):
    def __init__(self):
        # Initialize model
        pass
    
    def predict(self, window: np.ndarray) -> dict:
        """window shape: (n, 5) with columns [Open, High, Low, Close, Volume]"""
        # Your prediction logic here
        high_pred = ...
        low_pred = ...
        return {
            'high': float(high_pred),
            'low': float(low_pred),
            'open': float(window[-1, 0]),
            'close': float(window[-1, 3]),
        }
```

### 2. Register in `src/model_registry.py`
Add to imports:
```python
from my_new_model import MyNewModel
```

Add to registration section:
```python
ModelRegistry.register('my_new_model', MyNewModel)
```

### 3. Use immediately
```bash
python src/predict.py --model my_new_model --data data/prepared_data.csv --eval
```

## File Changes Summary

| File | Changes |
|------|---------|
| `src/predict.py` | Complete refactor: now orchestrator with flexible model selection |
| `src/timesfm_model.py` | Added `PredictionModel` base class and `TimesFmModel` implementation |
| `Lag_Llama_model.py` | Added `PredictionModel` base class and `LagLlamaModel` implementation |
| `src/model_registry.py` | **NEW**: Centralized model registry for extensibility |
| `src/MODEL_ARCHITECTURE.md` | **NEW**: Detailed architecture documentation |

## Design Principles

1. **Abstraction**: All models inherit from `PredictionModel` base class
2. **Extensibility**: New models can be added without modifying existing code
3. **Separation of Concerns**: 
   - Models handle predictions
   - Registry handles instantiation
   - Predict.py handles orchestration
4. **Consistency**: All models use same input/output format
5. **Caching**: Models are instantiated once and reused

## Testing a New Model

```bash
# Evaluate on full dataset
python src/predict.py --model my_model --data data/prepared_data.csv --eval --output results.json

# Test single prediction
python src/predict.py --model my_model --data data/prepared_data.csv --predict --sample 0
```

Check `results.json` for metrics and predictions.
