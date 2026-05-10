"""
Convert Wikipedia CDN TSV trace to pickle format matching synthetic traces.
"""
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

def convert_tsv_to_pkl():
    """Convert cache-t-00.tsv to Wikipedia_CDN.pkl"""
    
    # Read TSV
    tsv_path = Path('data/wikipedia_traces/cache-t-00.tsv')
    print(f'Reading TSV: {tsv_path}')
    df = pd.read_csv(tsv_path, sep='\t')
    
    print(f'TSV loaded: {len(df)} rows')
    print(f'Columns: {list(df.columns)}')
    
    # Extract request sequence from hashed_host_path_query column
    trace_raw = df['hashed_host_path_query'].values.astype(np.int64)  # Use int64 for mapping
    
    print(f'Raw trace - Min: {trace_raw.min()}, Max: {trace_raw.max()}')
    print(f'Unique items in raw trace: {len(np.unique(trace_raw))}')
    
    # Normalize to non-negative range [0, num_unique_items - 1]
    unique_items = np.unique(trace_raw)
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    trace = np.array([item_to_idx[item] for item in trace_raw], dtype=np.int32)
    
    print(f'\nNormalized trace:')
    print(f'Trace shape: {trace.shape}, dtype: {trace.dtype}')
    print(f'Unique items: {len(np.unique(trace))}')
    print(f'Min item ID: {trace.min()}, Max item ID: {trace.max()}')
    print(f'Sample requests (first 10): {trace[:10]}')
    
    # Prepare metadata matching synthetic trace format
    num_items = len(np.unique(trace))
    num_requests = len(trace)
    
    data_to_save = {
        'trace': trace,
        'num_items': num_items,
        'num_requests': num_requests,
        'seed': 0,
        'source': 'Wikipedia CDN trace (cache-t-00.tsv, normalized to [0, num_items-1])',
    }
    
    # Save as pickle with fixed naming
    output_path = Path('data/wikipedia_traces/Wikipedia_CDN_fixed.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f'\n✓ Converted and saved to {output_path}')
    print(f'\nMetadata:')
    for k, v in data_to_save.items():
        if k != 'trace':
            print(f'  {k}: {v}')

if __name__ == '__main__':
    convert_tsv_to_pkl()
