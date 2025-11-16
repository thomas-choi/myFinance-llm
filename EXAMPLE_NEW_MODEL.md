# Example: Adding a Simple Linear Regression Model

This example demonstrates how to extend the system with a new model.

## Step 1: Create the Model Implementation

Create `src/linear_regression_model.py`:

```python
from timesfm_model import PredictionModel
import numpy as np
from sklearn.linear_model import LinearRegression


class LinearRegressionModel(PredictionModel):
    """Simple linear regression baseline model for comparison."""
    
    def __init__(self):
        """Initialize the linear regression models for High and Low."""
        self.model_high = LinearRegression()
        self.model_low = LinearRegression()
        self._is_trained = False
    
    def fit(self, windows: list, labels: list = None) -> None:
        """
        Train the models on historical data.
        
        Args:
            windows: List of numpy arrays with shape (n, 5) containing OHLCV data
            labels: Optional labels (not used for regression, but kept for compatibility)
        """
        X = []  # Features: flatten each window
        y_high = []
        y_low = []
        
        for window in windows:
            # Flatten window to 1D feature vector
            X.append(window.flatten())
            
            # Last day's High and Low as targets
            y_high.append(window[-1, 1])  # High at index 1
            y_low.append(window[-1, 2])   # Low at index 2
        
        X = np.array(X)
        
        self.model_high.fit(X, y_high)
        self.model_low.fit(X, y_low)
        self._is_trained = True
    
    def predict(self, window: np.ndarray) -> dict:
        """
        Make a prediction using linear regression.
        
        Args:
            window: numpy array of shape (n, 5) with columns [Open, High, Low, Close, Volume]
        
        Returns:
            dict with 'high' and 'low' keys
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction. Call fit() first.")
        
        X = window.flatten().reshape(1, -1)
        
        pred_high = float(self.model_high.predict(X)[0])
        pred_low = float(self.model_low.predict(X)[0])
        
        return {
            'high': pred_high,
            'low': pred_low,
            'open': float(window[-1, 0]),
            'close': float(window[-1, 3]),
        }
```

## Step 2: Register the Model

In `src/model_registry.py`, add:

```python
# At the top with other imports
from linear_regression_model import LinearRegressionModel

# In the registration section
ModelRegistry.register('linear_regression', LinearRegressionModel)
```

## Step 3: Use the Model

```bash
# First, train the model on data
python src/train_linear_model.py --data data/prepared_data.csv

# Then evaluate
python src/predict.py --model linear_regression --data data/prepared_data.csv --eval --output metrics.json
```

## Optional: Create a Training Utility

Create `src/train_linear_model.py` for training-capable models:

```python
import argparse
import pandas as pd
import numpy as np
from model_registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, default='linear_regression')
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    df['window'] = df['window'].apply(eval)
    
    windows = []
    for window in df['window']:
        window_np = np.array(window)[:, 1:].astype(np.float32)
        windows.append(window_np)
    
    # Get model and train if it has fit method
    model = ModelRegistry.get_model(args.model)
    
    if hasattr(model, 'fit'):
        print(f"Training {args.model} on {len(windows)} samples...")
        model.fit(windows, df.get('label'))
        print("Training complete!")
    else:
        print(f"Model {args.model} does not support training")


if __name__ == '__main__':
    main()
```

## Benefits of This Approach

1. **Isolation**: Each model is independent and self-contained
2. **Reusability**: Can test, train, and evaluate any model
3. **Consistency**: All models follow the same interface
4. **Extensibility**: Easy to add features without breaking existing code
5. **Flexibility**: Models can be stateful (like those with training) or stateless

## Comparison Workflow

Now you can easily compare multiple models:

```bash
# Evaluate TimesFM
python src/predict.py --model timesfm --data data/prepared_data.csv --eval --output results_timesfm.json

# Evaluate Lag-Llama
python src/predict.py --model lag_llama --data data/prepared_data.csv --eval --output results_lag_llama.json

# Evaluate Linear Regression baseline
python src/predict.py --model linear_regression --data data/prepared_data.csv --eval --output results_linear.json

# Compare results
cat results_*.json | jq '.accuracy'
```
