# Extensible Time Series Prediction System

This document describes the refactored prediction system with support for multiple models.

## Architecture

The system is built on an extensible architecture with the following components:

### 1. **Base Interface** (`timesfm_model.py` and `Lag_Llama_model.py`)
- `PredictionModel`: Abstract base class defining the interface for all prediction models
- All models must implement `predict(window: np.ndarray) -> dict` method
- Returns a dictionary with keys: `'high'`, `'low'`, `'open'`, `'close'`

### 2. **Model Implementations**

#### TimesFM Model (`src/timesfm_model.py`)
- Implements Google's TimesFM 2.5 200M model
- Features: Handles multiple univariate time series in parallel
- Configuration: Normalized inputs, continuous quantile head, flip invariance

#### Lag-Llama Model (`Lag_Llama_model.py`)
- Implements Lag-Llama foundation model
- Features: Supports multivariate time series forecasting
- Configuration: Checkpoint-based initialization

### 3. **Model Registry** (`src/model_registry.py`)
- Central registry for model management
- Handles model instantiation and caching
- Extensible design for adding new models

### 4. **Prediction Interface** (`src/predict.py`)
- Top-level orchestrator for predictions
- Supports both evaluation and single sample prediction
- Metrics calculation (accuracy, F1)
- Data loading and preprocessing

## Usage

### Evaluate a Model
```bash
python src/predict.py --model timesfm --data data/prepared_data.csv --eval --output metrics.json
```

### Make a Single Prediction
```bash
python src/predict.py --model timesfm --data data/prepared_data.csv --predict --sample 0
```

### Available Models
- `timesfm`: Google TimesFM 2.5 200M model
- `lag_llama`: Lag-Llama foundation model

## Adding a New Model

### Step 1: Create Model Implementation
Create a new file (e.g., `src/my_model.py`):

```python
from timesfm_model import PredictionModel
import numpy as np

class MyModel(PredictionModel):
    def __init__(self):
        # Initialize your model
        pass
    
    def predict(self, window: np.ndarray) -> dict:
        """
        Args:
            window: numpy array of shape (n, 5) with columns [Open, High, Low, Close, Volume]
        
        Returns:
            dict with 'high', 'low', 'open', 'close' keys
        """
        # Implement prediction logic
        return {
            'high': predicted_high,
            'low': predicted_low,
            'open': predicted_open,
            'close': predicted_close,
        }
```

### Step 2: Register the Model
In `src/model_registry.py`, add:

```python
from my_model import MyModel

ModelRegistry.register('my_model', MyModel)
```

### Step 3: Use the Model
```bash
python src/predict.py --model my_model --data data/prepared_data.csv --eval
```

## API Reference

### predict.py Functions

#### `load_data(data_path: str) -> pd.DataFrame`
Loads and parses the prepared CSV data with string-formatted windows.

#### `prepare_window(window: list) -> np.ndarray`
Converts a window list to a numpy array with shape (n, 5) containing OHLCV data.

#### `predict_single(model, window: list) -> tuple`
Makes a single prediction and returns (pred_high, pred_low, last_high, last_low).

#### `evaluate(model_name: str, data_path: str, output_path: str = None) -> dict`
Evaluates a model on the entire dataset and returns metrics.

#### `predict(model_name: str, data_path: str, sample_index: int = 0) -> dict`
Makes a single prediction on a sample and returns detailed results.

### model_registry.py API

#### `ModelRegistry.register(name: str, model_class: Type[PredictionModel])`
Registers a new model class with the registry.

#### `ModelRegistry.get_model(name: str, **kwargs) -> PredictionModel`
Retrieves or instantiates a model by name. Models are cached after first instantiation.

#### `ModelRegistry.list_models() -> list`
Returns a list of all registered model names.

## Data Format

Expected CSV format for `--data` argument:
```
window,label
"[[date1, o1, h1, l1, c1, v1], [date2, o2, h2, l2, c2, v2], ...]",UP
```

Each window contains daily OHLCV data in the format: `[Date, Open, High, Low, Close, Volume]`

The system automatically extracts the OHLCV portion (removing the Date column) for predictions.

## Output Formats

### Evaluation Output
```json
{
  "model": "timesfm",
  "accuracy": 0.6234,
  "f1": 0.5892,
  "num_samples": 1500
}
```

### Prediction Output
```json
{
  "model": "timesfm",
  "sample_index": 0,
  "predicted_high": 125.43,
  "predicted_low": 120.15,
  "last_high": 124.50,
  "last_low": 119.80,
  "predicted_label": "UP",
  "true_label": "UP"
}
```
