import os
import sys
from pathlib import Path

def setup_paths():
    """Setup project directory structure."""
    paths = {
        'data': Path('data'),
        'data_synthetic': Path('data/synthetic_traces'),
        'data_wikipedia': Path('data/wikipedia_traces'),
        'results': Path('results'),
        'results_plots': Path('results/plots'),
        'src': Path('src'),
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths