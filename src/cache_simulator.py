import numpy as np
from typing import Callable, Dict, List, Tuple
from abc import ABC, abstractmethod

class CachePolicy(ABC):
    """Abstract base class for all cache eviction policies."""
    
    @abstractmethod
    def on_hit(self, item_id: int, timestamp: int) -> None:
        """Update policy state on cache hit."""
        pass
    
    @abstractmethod
    def on_miss(self, item_id: int, timestamp: int) -> int:
        """
        Decide which item to evict on cache miss.
        
        Returns:
            item_id: ID of item to evict (or -1 if cache not full)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset policy to initial state."""
        pass


class CacheSimulator:
    """
    Simulates cache behavior given a request trace and eviction policy.
    
    Attributes:
        cache_size: Maximum items in cache
        trace: Request sequence (np.ndarray of item IDs)
        policy: CachePolicy instance
    """
    
    def __init__(self, cache_size: int, policy: CachePolicy):
        """
        Initialize cache simulator.
        
        Args:
            cache_size: Maximum number of items in cache
            policy: CachePolicy instance (LRU, LFU, etc.)
        """
        self.cache_size = cache_size
        self.policy = policy
        self.cache = set()  # Currently cached items
        self.hits = 0
        self.misses = 0
        self.timestamp = 0
    
    def reset(self):
        """Reset simulator to initial state."""
        self.cache.clear()
        self.policy.reset()
        self.hits = 0
        self.misses = 0
        self.timestamp = 0
    
    def run(self, trace: np.ndarray) -> Dict:
        """
        Simulate cache behavior on given trace.
        
        Args:
            trace: Request sequence (np.ndarray)
            
        Returns:
            dict: Simulation results {
                'hit_rate': float,
                'hits': int,
                'misses': int,
                'total_requests': int,
                'evictions': int,
            }
        """
        self.reset()
        evictions = 0
        
        for item_id in trace:
            self.timestamp += 1
            
            if item_id in self.cache:
                # Cache HIT
                self.hits += 1
                self.policy.on_hit(item_id, self.timestamp)
            else:
                # Cache MISS
                self.misses += 1
                
                if len(self.cache) < self.cache_size:
                    # Cache not full, just add
                    self.cache.add(item_id)
                else:
                    # Cache full, evict an item
                    evicted_item = self.policy.on_miss(item_id, self.timestamp)
                    if evicted_item != -1:
                        self.cache.remove(evicted_item)
                        evictions += 1
                    self.cache.add(item_id)
                
                self.policy.on_miss(item_id, self.timestamp)
        
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_rate': hit_rate,
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'evictions': evictions,
        }
    
    def run_streaming(self, trace: np.ndarray) -> Tuple[List[float], Dict]:
        """
        Run simulation and track hit rate over time (for plotting).
        
        Args:
            trace: Request sequence
            
        Returns:
            Tuple of (hit_rate_over_time, final_stats)
        """
        self.reset()
        hit_rates = []
        
        for item_id in trace:
            self.timestamp += 1
            
            if item_id in self.cache:
                self.hits += 1
                self.policy.on_hit(item_id, self.timestamp)
            else:
                self.misses += 1
                if len(self.cache) < self.cache_size:
                    self.cache.add(item_id)
                else:
                    evicted_item = self.policy.on_miss(item_id, self.timestamp)
                    if evicted_item != -1:
                        self.cache.remove(evicted_item)
                    self.cache.add(item_id)
                self.policy.on_miss(item_id, self.timestamp)
            
            total = self.hits + self.misses
            current_hit_rate = self.hits / total if total > 0 else 0.0
            hit_rates.append(current_hit_rate)
        
        final_stats = {
            'hit_rate': hit_rates[-1],
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': len(trace),
        }
        
        return hit_rates, final_stats
