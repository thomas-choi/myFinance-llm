import argparse
import ast
import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score
from utils import derive_label
from model_registry import ModelRegistry
import re

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load and parse the prepared CSV data.
    
    Args:
        data_path: Path to the CSV file
    
    Returns:
        DataFrame with parsed windows
    """
    df = pd.read_csv(data_path)
    
    # Convert string representation of nested lists back to actual lists
    def parse_window(window_str):
        if not isinstance(window_str, str):
            return window_str
        
        window_str = window_str.strip()
        
        if not window_str or window_str == '[]':
            return []
        
        # 1. Try ast.literal_eval first (works for normal Python str(list))
        try:
            return ast.literal_eval(window_str)
        except (ValueError, SyntaxError):
            pass
        
        # 2. Try json.loads (works if the data was saved with json.dumps)
        try:
            return json.loads(window_str)
        except json.JSONDecodeError:
            pass
        
        # 3. Handle common data issues: nan, inf, -inf
        fixed_str = re.sub(r'\bnan\b', 'float("nan")', window_str, flags=re.IGNORECASE)
        fixed_str = re.sub(r'\binf\b', 'float("inf")', fixed_str, flags=re.IGNORECASE)
        fixed_str = re.sub(r'-inf\b', 'float("-inf")', fixed_str, flags=re.IGNORECASE)
        
        # 4. Safe eval with minimal namespace for Timestamp and float
        safe_globals = {
            "__builtins__": {"float": float},
            "Timestamp": pd.Timestamp
        }
        try:
            return eval(fixed_str, safe_globals)
        except Exception as e:
            raise ValueError(f"Could not parse window: {window_str[:100]}\nError: {e}")
    
    df['window'] = df['window'].apply(parse_window)
    return df

# def load_data(data_path: str) -> pd.DataFrame:
#     """
#     Load and parse the prepared CSV data.
    
#     Args:
#         data_path: Path to the CSV file
    
#     Returns:
#         DataFrame with parsed windows
#     """
#     df = pd.read_csv(data_path)
    
#     # Convert string representation of nested lists back to actual lists
#     def parse_window(window_str):
#         try:
#             # Try ast.literal_eval first
#             return ast.literal_eval(window_str)
#         except (ValueError, SyntaxError):
#             # If that fails, use json.loads as fallback
#             try:
#                 return json.loads(window_str)
#             except (json.JSONDecodeError, ValueError):
#                 # Last resort: use eval with controlled namespace (only safe built-ins)
#                 # This handles cases where the string contains list/dict literals
#                 try:
#                     return eval(window_str, {"__builtins__": {}})
#                 except:
#                     raise ValueError(f"Could not parse window: {window_str[:100]}")
    
#     df['window'] = df['window'].apply(parse_window)
#     return df


def prepare_window(window: list) -> np.ndarray:
    """
    Prepare a window for prediction.
    
    Args:
        window: List of daily data, each item is [Date, Open, High, Low, Close, Volume]
    
    Returns:
        numpy array of shape (n, 5) with columns [Open, High, Low, Close, Volume]
    """
    window_np = np.array(window)
    # Remove Date column (index 0) to get OHLCV
    return window_np[:, 1:].astype(np.float32)


def predict_single(model, window: list) -> tuple:
    """
    Make a single prediction.
    
    Args:
        model: Prediction model instance
        window: List of daily data
    
    Returns:
        Tuple of (predicted_high, predicted_low, last_high, last_low)
    """
    window_np = prepare_window(window)
    
    # Get prediction from model
    prediction = model.predict(window_np)
    pred_high = prediction['high']
    pred_low = prediction['low']
    
    # Get last day's High and Low for comparison
    last_day = window_np[-1]
    last_high = last_day[1]  # High is at index 1 in OHLCV
    last_low = last_day[2]   # Low is at index 2 in OHLCV
    
    return pred_high, pred_low, last_high, last_low


def evaluate(model_name: str, data_path: str, output_path: str = None) -> dict:
    """
    Evaluate a model on the test set.
    
    Args:
        model_name: Name of the model to use
        data_path: Path to the prepared CSV file
        output_path: Optional path to save metrics as JSON
    
    Returns:
        Dictionary with accuracy and F1 metrics
    """
    # Load model
    print(f"Loading model: {model_name}")
    model = ModelRegistry.get_model(model_name)
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    
    # Make predictions
    print(f"Making predictions on {len(df)} samples...")
    preds = []
    true = df['label'].tolist()
    
    for idx, window in enumerate(df['window']):
        if (idx + 1) % max(1, len(df) // 10) == 0:
            print(f"  Progress: {idx + 1}/{len(df)}")
        
        pred_high, pred_low, last_high, last_low = predict_single(model, window)
        pred_label = derive_label(pred_high, pred_low, last_high, last_low)
        preds.append(pred_label)
    
    # Calculate metrics
    metrics = {
        'model': model_name,
        'accuracy': float(accuracy_score(true, preds)),
        'f1': float(f1_score(true, preds, average='macro')),
        'num_samples': len(df),
    }
    
    print("\nMetrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Save metrics if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {output_path}")
    
    return metrics


def predict(model_name: str, data_path: str, sample_index: int = 0) -> dict:
    """
    Make a single prediction on a sample.
    
    Args:
        model_name: Name of the model to use
        data_path: Path to the prepared CSV file
        sample_index: Index of the sample to predict
    
    Returns:
        Dictionary with prediction details
    """
    # Load model
    print(f"Loading model: {model_name}")
    model = ModelRegistry.get_model(model_name)
    
    # Load data
    df = load_data(data_path)
    
    if sample_index >= len(df):
        raise ValueError(f"Sample index {sample_index} out of range (max: {len(df) - 1})")
    
    window = df['window'].iloc[sample_index]
    pred_high, pred_low, last_high, last_low = predict_single(model, window)
    pred_label = derive_label(pred_high, pred_low, last_high, last_low)
    
    result = {
        'model': model_name,
        'sample_index': sample_index,
        'predicted_high': float(pred_high),
        'predicted_low': float(pred_low),
        'last_high': float(last_high),
        'last_low': float(last_low),
        'predicted_label': pred_label,
        'true_label': df['label'].iloc[sample_index] if 'label' in df.columns else None,
    }
    
    print("\nPrediction Result:")
    print(f"  Model: {result['model']}")
    print(f"  Predicted High: {result['predicted_high']:.4f}")
    print(f"  Predicted Low: {result['predicted_low']:.4f}")
    print(f"  Last High: {result['last_high']:.4f}")
    print(f"  Last Low: {result['last_low']:.4f}")
    print(f"  Predicted Label: {result['predicted_label']}")
    if result['true_label']:
        print(f"  True Label: {result['true_label']}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Top-level prediction interface for time series forecasting models'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help=f'Model name. Available: {", ".join(ModelRegistry.list_models())}'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to prepared CSV file (windows as list of lists)'
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Evaluate model on test set'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Make a single prediction'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=0,
        help='Sample index for single prediction (default: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for metrics or predictions'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.eval and not args.predict:
        parser.print_help()
        print("\nError: Must specify either --eval or --predict")
        return
    
    try:
        if args.eval:
            evaluate(args.model, args.data, args.output)
        
        if args.predict:
            result = predict(args.model, args.data, args.sample)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Prediction saved to: {args.output}")
    
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())