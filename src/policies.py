from collections import defaultdict, deque
from cache_simulator import CachePolicy
import heapq

class LRUPolicy(CachePolicy):
    """
    Least Recently Used (LRU) eviction policy.
    
    Evicts the item that was accessed longest ago.
    
    Implementation: OrderedDict for O(1) ordering, or timestamp-based.
    """
    
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.last_access_time = {}  # item_id -> timestamp
        self.reset()
    
    def on_hit(self, item_id: int, timestamp: int) -> None:
        """Record access time on cache hit."""
        self.last_access_time[item_id] = timestamp
    
    def on_miss(self, item_id: int, timestamp: int) -> int:
        """
        Find least recently used item and evict it.
        
        Returns:
            Item ID to evict (or -1 if cache not full)
        """
        # Find item with minimum last access time
        if self.last_access_time:
            lru_item = min(self.last_access_time, key=self.last_access_time.get)
            del self.last_access_time[lru_item]
            self.last_access_time[item_id] = timestamp
            return lru_item
        return -1
    
    def reset(self) -> None:
        """Reset LRU state."""
        self.last_access_time.clear()


class LFUPolicy(CachePolicy):
    """
    Least Frequently Used (LFU) eviction policy.
    
    Evicts the item with the lowest access frequency.
    
    Implementation: Frequency map + double-linked list for O(1) operations.
    Simplified version: For Week 1, we use direct frequency tracking.
    """
    
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.frequency = defaultdict(int)  # item_id -> access count
        self.min_frequency = 0
        self.reset()
    
    def on_hit(self, item_id: int, timestamp: int) -> None:
        """Increment frequency on cache hit."""
        self.frequency[item_id] += 1
    
    def on_miss(self, item_id: int, timestamp: int) -> int:
        """
        Find least frequently used item and evict it.
        
        Returns:
            Item ID to evict (or -1 if cache not full)
        """
        if self.frequency:
            # Find item with minimum frequency
            # Tie-breaker: evict the one added earliest (not tracked, so arbitrary)
            lfu_item = min(self.frequency, key=self.frequency.get)
            del self.frequency[lfu_item]
            self.frequency[item_id] = 1
            return lfu_item
        return -1
    
    def reset(self) -> None:
        """Reset LFU state."""
        self.frequency.clear()
        self.min_frequency = 0


class OptimalPolicy(CachePolicy):
    """
    Optimal (Belady's) policy - knows future requests.
    
    For comparison only. Evicts the item whose next access is furthest in future.
    Provides upper-bound on hit rate.
    """
    
    def __init__(self, cache_size: int, trace: list = None):
        """
        Args:
            cache_size: Cache size
            trace: Full request trace (needed to look ahead)
        """
        self.cache_size = cache_size
        self.trace = trace
        self.next_access = {}  # item_id -> next timestamp it will be accessed
        self.current_index = 0
        self.reset()
    
    def _build_next_access_map(self) -> None:
        """Pre-compute next access time for each item in trace."""
        self.next_access = defaultdict(lambda: float('inf'))
        
        for i in range(self.current_index + 1, len(self.trace)):
            item = self.trace[i]
            if item not in self.next_access:
                self.next_access[item] = i
    
    def on_hit(self, item_id: int, timestamp: int) -> None:
        """Update on hit."""
        self.current_index = timestamp
        self._build_next_access_map()
    
    def on_miss(self, item_id: int, timestamp: int) -> int:
        """Evict item whose next access is furthest."""
        self.current_index = timestamp
        self._build_next_access_map()
        
        if self.next_access:
            optimal_item = max(self.next_access, key=self.next_access.get)
            del self.next_access[optimal_item]
            return optimal_item
        return -1
    
    def reset(self) -> None:
        """Reset optimal policy."""
        self.next_access.clear()
        self.current_index = 0
