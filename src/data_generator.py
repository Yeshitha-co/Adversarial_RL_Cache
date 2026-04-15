import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple

class TraceGenerator:
    """Generate synthetic request traces with Zipfian distribution."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize trace generator with reproducible randomness.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
    
    def zipfian_trace(
        self, 
        num_requests: int,
        num_items: int,
        alpha: float = 1.0,
        save_path: str = None
    ) -> np.ndarray:
        """
        Generate Zipfian-distributed request trace.
        
        Mathematical Model:
            P(rank k) = (1/k^α) / H(N, α)
            where H(N, α) = sum(1/i^α for i in 1..N) [Harmonic number]
        
        Args:
            num_requests: Total requests in trace (e.g., 100,000)
            num_items: Number of unique items in catalog (e.g., 10,000)
            alpha: Zipfian exponent (1.0 = realistic web, 2.0 = very skewed)
            save_path: Optional path to save trace as pickle
            
        Returns:
            np.ndarray: Request sequence where each element is an item ID (0 to num_items-1)
        """
        # Generate Zipfian probabilities
        ranks = np.arange(1, num_items + 1)
        probabilities = (1.0 / (ranks ** alpha)) / np.sum(1.0 / (ranks ** alpha))
        
        # Sample requests according to Zipfian distribution
        trace = np.random.choice(
            num_items, 
            size=num_requests, 
            p=probabilities,
            replace=True
        )
        
        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'trace': trace,
                    'num_items': num_items,
                    'alpha': alpha,
                    'num_requests': num_requests,
                    'seed': self.seed
                }, f)
            print(f"[OK] Trace saved to {save_path}")
        
        return trace
    
    def working_set_trace(
        self,
        num_requests: int,
        num_items: int,
        working_set_size: int,
        locality: float = 0.8,
        save_path: str = None
    ) -> np.ndarray:
        """
        Generate trace with temporal locality (working set model).
        
        Concept: Real traffic exhibits locality - some "working set" of items
        are frequently accessed, while occasionally new items enter the set.
        
        Args:
            num_requests: Total requests
            num_items: Total catalog size
            working_set_size: Size of hot items (e.g., 100 out of 10,000)
            locality: Probability of staying in current working set (0.8 = 80%)
            save_path: Optional save path
            
        Returns:
            np.ndarray: Trace with temporal locality
        """
        trace = []
        current_working_set = np.random.choice(num_items, working_set_size, replace=False)
        
        for _ in range(num_requests):
            if np.random.random() < locality:
                # Stay in working set (Uniform for now, could be Zipfian)
                item = np.random.choice(current_working_set)
            else:
                # Jump to new working set
                current_working_set = np.random.choice(num_items, working_set_size, replace=False)
                item = np.random.choice(current_working_set)
            
            trace.append(item)
        
        trace = np.array(trace)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'trace': trace,
                    'num_items': num_items,
                    'working_set_size': working_set_size,
                    'locality': locality,
                    'num_requests': num_requests,
                    'seed': self.seed
                }, f)
            print(f"[OK] Trace saved to {save_path}")
        
        return trace
    
    @staticmethod
    def load_trace(trace_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load a previously saved trace.
        
        Args:
            trace_path: Path to pickled trace file
            
        Returns:
            Tuple of (trace, metadata)
        """
        with open(trace_path, 'rb') as f:
            data = pickle.load(f)
        return data['trace'], {k: v for k, v in data.items() if k != 'trace'}


def analyze_trace(trace: np.ndarray, top_k: int = 10) -> dict:
    """
    Analyze trace properties (for understanding dataset characteristics).
    
    Args:
        trace: Request sequence (np.ndarray)
        top_k: Show top-k items by frequency
        
    Returns:
        dict: Analysis results
    """
    unique_items = len(np.unique(trace))
    item_counts = np.bincount(trace)
    top_items = np.argsort(item_counts)[-top_k:][::-1]
    
    top_k_requests = item_counts[top_items].sum()
    top_k_percentage = (top_k_requests / len(trace)) * 100
    
    analysis = {
        'total_requests': len(trace),
        'unique_items': unique_items,
        'coverage': unique_items / len(trace),  # How many unique items as fraction
        'top_k_items': top_k,
        'top_k_percentage': top_k_percentage,
        'top_k_items_list': top_items.tolist(),
        'top_k_counts': item_counts[top_items].tolist(),
        'mean_requests_per_item': np.mean(item_counts),
        'std_requests_per_item': np.std(item_counts),
        'max_requests': np.max(item_counts),
        'min_requests': np.min(item_counts),
    }
    
    return analysis
