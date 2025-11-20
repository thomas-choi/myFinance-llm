#!/usr/bin/env python3
"""
Data Manifest Viewer and Manager

This script helps you view metadata about all prepared datasets,
making it easy to identify which data files correspond to which parameters.

Usage:
    python src/data_manifest.py list              # List all datasets
    python src/data_manifest.py show <filename>   # Show details of a specific dataset
    python src/data_manifest.py clean             # Remove old datasets (keeps 3 most recent)
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from tabulate import tabulate


def find_metadata_files(data_dir='data'):
    """Find all metadata files in the data directory."""
    if not os.path.exists(data_dir):
        return []
    
    metadata_files = sorted(
        Path(data_dir).glob('*_metadata.json'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return metadata_files


def load_metadata(metadata_file):
    """Load metadata from a JSON file."""
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {metadata_file}: {e}")
        return None


def list_datasets(data_dir='data'):
    """List all available datasets with their metadata."""
    metadata_files = find_metadata_files(data_dir)
    
    if not metadata_files:
        print(f"No datasets found in {data_dir}/")
        return
    
    print(f"\n{'='*100}")
    print(f"Available Datasets ({len(metadata_files)} total)")
    print(f"{'='*100}\n")
    
    table_data = []
    for metadata_file in metadata_files:
        metadata = load_metadata(metadata_file)
        if metadata:
            csv_file = metadata_file.parent / metadata['filename']
            file_size = os.path.getsize(csv_file) if os.path.exists(csv_file) else 0
            size_mb = file_size / (1024 * 1024)
            
            created = datetime.fromisoformat(metadata['created_at'])
            created_str = created.strftime('%Y-%m-%d %H:%M:%S')
            
            table_data.append([
                metadata['filename'],
                ','.join(metadata['tickers']),
                metadata['window_size'],
                metadata['samples'],
                metadata['total_records'],
                f"{size_mb:.2f} MB",
                created_str
            ])
    
    headers = ['Filename', 'Tickers', 'Window', 'Samples', 'Records', 'Size', 'Created']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print()


def show_dataset(filename, data_dir='data'):
    """Show detailed information about a specific dataset."""
    metadata_file = Path(data_dir) / f"{filename.replace('.csv', '')}_metadata.json"
    
    if not metadata_file.exists():
        print(f"Metadata not found for {filename}")
        return
    
    metadata = load_metadata(metadata_file)
    if not metadata:
        return
    
    csv_file = Path(data_dir) / filename
    file_size = os.path.getsize(csv_file) if csv_file.exists() else 0
    
    print(f"\n{'='*60}")
    print(f"Dataset Details: {filename}")
    print(f"{'='*60}\n")
    
    print(f"Filename:        {metadata['filename']}")
    print(f"Tickers:         {', '.join(metadata['tickers'])}")
    print(f"Window Size:     {metadata['window_size']} days")
    print(f"Samples:         {metadata['samples']}")
    print(f"Total Records:   {metadata['total_records']}")
    print(f"File Size:       {file_size / (1024 * 1024):.2f} MB")
    print(f"Created:         {metadata['created_at']}")
    print(f"Description:     {metadata['description']}")
    print()


def clean_old_datasets(keep=3, data_dir='data'):
    """Keep only the most recent N datasets, remove older ones."""
    metadata_files = find_metadata_files(data_dir)
    
    if len(metadata_files) <= keep:
        print(f"Only {len(metadata_files)} dataset(s) found. Keeping all.")
        return
    
    print(f"\nFound {len(metadata_files)} datasets. Keeping {keep} most recent...\n")
    
    to_remove = metadata_files[keep:]
    for metadata_file in to_remove:
        metadata = load_metadata(metadata_file)
        if metadata:
            csv_file = Path(data_dir) / metadata['filename']
            print(f"Removing: {metadata['filename']}")
            
            if csv_file.exists():
                csv_file.unlink()
            metadata_file.unlink()
    
    print(f"\nRemoved {len(to_remove)} old dataset(s).\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1]
    
    if command == 'list':
        list_datasets()
    elif command == 'show' and len(sys.argv) > 2:
        show_dataset(sys.argv[2])
    elif command == 'clean':
        keep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        clean_old_datasets(keep)
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
