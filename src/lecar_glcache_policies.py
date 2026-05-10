"""
lecar_glcache_policies.py
LeCaR and GL-Cache eviction policies compatible with CacheSimulator.
Optimized for large traces — frequency discount batched every 1000 misses.
"""

import math
import numpy as np
from collections import OrderedDict
from cache_simulator import CachePolicy


class LeCaRPolicy(CachePolicy):
    """
    LeCaR: Learning Cache Replacement via online expert blending.
    Two experts (LRU + LFU) with Hedge multiplicative weight updates.
    """

    def __init__(self, cache_size: int, eta: float = 0.005):
        self.cache_size = cache_size
        self.eta        = eta
        self.dr         = 0.005 * math.exp(-1.0 / cache_size)
        self.reset()

    def on_hit(self, item_id: int, timestamp: int) -> None:
        if item_id in self.lru_order:
            self.lru_order.move_to_end(item_id)
        self.frequency[item_id] = self.frequency.get(item_id, 0) + 1
        # Regret update: penalise the expert that evicted this item
        if item_id in self.eviction_history:
            bad_expert  = self.eviction_history.pop(item_id)
            idx         = 0 if bad_expert == 'lru' else 1
            self.w[idx] *= math.exp(-self.eta)
            self.w       /= self.w.sum()

    def on_miss(self, item_id: int, timestamp: int) -> int:
        if not self.lru_order:
            self._add(item_id)
            return -1

        lru_candidate = next(iter(self.lru_order))
        lfu_candidate = min(self.frequency, key=self.frequency.get)
        use_lru       = np.random.rand() < (self.w[0] / self.w.sum())
        evict_item    = lru_candidate if use_lru else lfu_candidate

        self.eviction_history[evict_item] = 'lru' if use_lru else 'lfu'
        self._remove(evict_item)
        self._add(item_id)

        # Batch discount every 1000 misses for speed
        self.miss_count += 1
        if self.miss_count % 1000 == 0:
            discount = (1.0 - self.dr) ** 1000
            for k in self.frequency:
                self.frequency[k] *= discount

        return evict_item

    def reset(self) -> None:
        self.w                = np.array([0.5, 0.5], dtype=np.float64)
        self.lru_order        = OrderedDict()
        self.frequency        = {}
        self.eviction_history = {}
        self.miss_count       = 0

    def _add(self, item_id):
        self.lru_order[item_id] = True
        self.lru_order.move_to_end(item_id)
        self.frequency[item_id] = self.frequency.get(item_id, 0) + 1

    def _remove(self, item_id):
        self.lru_order.pop(item_id, None)
        self.frequency.pop(item_id, None)


class GLCachePolicy(CachePolicy):
    """
    GL-Cache: Group-Level Learned Cache Replacement (simplified).
    Items bucketed by item_id % num_groups. Group utility scores
    updated online; lowest-score group evicted on miss.
    """

    def __init__(self, cache_size: int, num_groups: int = 64, lr: float = 0.01):
        self.cache_size = cache_size
        self.num_groups = num_groups
        self.lr         = lr
        self.reset()

    def on_hit(self, item_id: int, timestamp: int) -> None:
        g = item_id % self.num_groups
        if item_id in self.group_items[g]:
            self.group_items[g].move_to_end(item_id)
        self.group_scores[g] += self.lr * (1.0 - self.group_scores[g])

    def on_miss(self, item_id: int, timestamp: int) -> int:
        g_evict = self._lowest_group()
        if g_evict is None:
            self._insert(item_id)
            return -1
        evict_item, _ = next(iter(self.group_items[g_evict].items()))
        del self.group_items[g_evict][evict_item]
        self.group_scores[g_evict] = max(
            0.0, self.group_scores[g_evict] * (1.0 - self.lr)
        )
        self._insert(item_id)
        return evict_item

    def reset(self) -> None:
        self.group_scores = np.ones(self.num_groups, dtype=np.float64)
        self.group_items  = [OrderedDict() for _ in range(self.num_groups)]

    def _insert(self, item_id):
        g = item_id % self.num_groups
        self.group_items[g][item_id] = True
        self.group_items[g].move_to_end(item_id)

    def _lowest_group(self):
        for g in np.argsort(self.group_scores):
            if self.group_items[g]:
                return int(g)
        return None