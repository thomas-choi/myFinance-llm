import argparse
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', type=str, required=True, help='Comma-separated tickers, e.g., AAPL,GOOG')
    parser.add_argument('--output', type=str, default='data/prepared_data.csv', help='Output CSV')
    parser.add_argument('--n', type=int, default=60, help='Window size')
    args = parser.parse_args()

    tickers = args.tickers.split(',')
    data = pd.concat([yf.download(t, period='max', auto_adjust=True, multi_level_index=False)[['Open', 'High', 'Low', 'Close', 'Volume']] for t in tickers], axis=0)
    data = data.reset_index()

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

    # Save as CSV (or use HF datasets)
    df = pd.DataFrame({'window': windows, 'label': labels})
    df.to_csv(args.output, index=False)
    print(f'Data prepared: {len(df)} samples')

from utils import derive_label
if __name__ == '__main__':
    main()
