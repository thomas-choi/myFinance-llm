import argparse
import pandas as pd
import yfinance as yf
import numpy as np
import json
import os
from datetime import datetime

def generate_filename(tickers, window_size, num_samples, output_dir='data'):
    """
    Generate a meaningful filename based on parameters.
    Format: data_{tickers}_{window_size}w_{samples}s_{timestamp}.csv
    Example: data_AAPL_GOOG_60w_500s_20250118_143022.csv
    """
    tickers_str = '_'.join(sorted(tickers))
    samples_str = f'{num_samples}s' if num_samples else 'full'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'data_{tickers_str}_{window_size}w_{samples_str}_{timestamp}.csv'
    return os.path.join(output_dir, filename)

def save_metadata(filename, tickers, window_size, num_samples, num_records):
    """
    Save metadata about the dataset to a JSON file for easy reference.
    This helps track what parameters were used to generate each dataset.
    """
    metadata = {
        'filename': os.path.basename(filename),
        'tickers': tickers,
        'window_size': window_size,
        'samples': num_samples if num_samples else 'all available',
        'total_records': num_records,
        'created_at': datetime.now().isoformat(),
        'description': f'Data prepared from {len(tickers)} ticker(s) with {window_size}-day windows'
    }
    
    metadata_filename = filename.replace('.csv', '_metadata.json')
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_filename

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', type=str, required=True, help='Comma-separated tickers, e.g., AAPL,GOOG')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path (if not specified, auto-generated based on parameters)')
    parser.add_argument('--n', type=int, default=60, help='Window size')
    parser.add_argument('-s', '--samples', type=int, default=None, help='Number of data samples from today counting backward (e.g., 500)')
    args = parser.parse_args()

    tickers = args.tickers.split(',')
    
    # Generate automatic filename if not provided
    if args.output is None:
        output_dir = 'data'
        os.makedirs(output_dir, exist_ok=True)
        args.output = generate_filename(tickers, args.n, args.samples, output_dir)
        print(f'Auto-generated output file: {args.output}')
    else:
        # Ensure directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    data = pd.concat([yf.download(t, period='max', auto_adjust=True, multi_level_index=False)[['Open', 'High', 'Low', 'Close', 'Volume']] for t in tickers], axis=0)
    data = data.reset_index()

    # Slice data to specified number of samples if provided
    if args.samples is not None:
        data = data.tail(args.samples)
        data = data.reset_index(drop=True)

    # Create windows
    windows = []
    labels = []
    for i in range(len(data) - args.n - 1):
        window = data.iloc[i:i+args.n]
        next_day = data.iloc[i+args.n]
        last_day = data.iloc[i+args.n-1]
        label = derive_label(next_day['High'], next_day['Low'], last_day['High'], last_day['Low'])
        windows.append(window.values.tolist())  # List of lists for HF dataset
        labels.append(label)

    # Save as CSV
    df = pd.DataFrame({'window': windows, 'label': labels})
    df.to_csv(args.output, index=False)
    
    # Save metadata file alongside the data
    metadata_file = save_metadata(args.output, tickers, args.n, args.samples, len(df))
    
    print(f'Data prepared: {len(df)} samples')
    print(f'Saved to: {args.output}')
    print(f'Metadata saved to: {metadata_file}')

from utils import derive_label
if __name__ == '__main__':
    main()
