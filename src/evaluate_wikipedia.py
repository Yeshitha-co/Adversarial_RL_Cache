"""
Simple comparison of LRU, LFU, RARL Protagonist, LeCaR, GL-Cache
on Wikipedia trace at cache size 500.
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent  # Adversarial_RL_Cache-main/
sys.path.insert(0, str(ROOT / 'src'))

from cache_simulator import CacheSimulator
from policies import LRUPolicy, LFUPolicy
from dqn_agent import DQNAgent
from lecar_glcache_policies import LeCaRPolicy, GLCachePolicy


def load_wikipedia_trace(path: Path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['trace']


def evaluate_policy(policy_name: str, policy, trace: np.ndarray, cache_size: int):
    simulator = CacheSimulator(cache_size, policy)
    stats = simulator.run(trace)
    return {
        'policy': policy_name,
        'cache_size': cache_size,
        'hit_rate': stats['hit_rate'],
        'hits': stats['hits'],
        'misses': stats['misses'],
        'total_requests': stats['total_requests'],
    }


def plot_hit_rate_comparison(df: pd.DataFrame, output_dir: Path, title: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['policy'], df['hit_rate'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Policy', fontsize=12)
    ax.set_ylabel('Hit Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    for bar, value in zip(bars, df['hit_rate']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{value:.4f}", ha='center', va='bottom', fontsize=10)

    output_path = output_dir / 'wikipedia_cache500_hit_rate_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


class RLAgentPolicy:
    """Wrapper to use DQNAgent as a cache policy."""

    def __init__(self, agent: DQNAgent, num_items: int, cache_size: int):
        self.agent = agent
        self.num_items = num_items
        self.cache_size = cache_size
        self.env = DQNCacheEnvironment(cache_size, num_items)

    def on_hit(self, item_id: int, timestamp: int) -> None:
        pass

    def on_miss(self, item_id: int, timestamp: int) -> int:
        state = self.env._get_state_vector()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            q_values = self.agent.q_network(state_tensor)
            action = q_values.max(1)[1].item()
        action = min(action, len(self.env.cache) - 1)
        cached_items = sorted(list(self.env.cache))
        if cached_items and action < len(cached_items):
            evict_item = cached_items[action]
            self.env.cache.discard(evict_item)
            return evict_item
        return -1

    def reset(self) -> None:
        self.env = DQNCacheEnvironment(self.cache_size, self.num_items)


class DQNCacheEnvironment:
    """Simplified cache environment for DQN agent evaluation."""

    def __init__(self, cache_size: int, num_items: int):
        self.cache_size = cache_size
        self.num_items = num_items
        self.cache = set()
        self.recency = {}
        self.frequency = {}
        self.total_steps = 0

    def _get_state_vector(self) -> np.ndarray:
        state = []
        cached_items = sorted(list(self.cache))
        max_slots = 64
        for i in range(max_slots):
            if i < len(cached_items):
                it = cached_items[i]
                rec = max(0, 100 - (self.total_steps - self.recency.get(it, 0)))
                freq = min(100, self.frequency.get(it, 0))
            else:
                rec = 0
                freq = 0
            state.append(rec / 100.0)
            state.append(freq / 100.0)
        return np.array(state, dtype=np.float32)

    def process_request(self, item: int) -> bool:
        is_hit = item in self.cache
        if is_hit:
            self.recency[item] = self.total_steps
            self.frequency[item] = self.frequency.get(item, 0) + 1
        else:
            if len(self.cache) < self.cache_size:
                self.cache.add(item)
                self.recency[item] = self.total_steps
                self.frequency[item] = 1
        self.total_steps += 1
        return is_hit

    def evict(self, item_idx: int) -> None:
        cached_items = sorted(list(self.cache))
        if 0 <= item_idx < len(cached_items):
            item_to_evict = cached_items[item_idx]
            self.cache.discard(item_to_evict)


def load_protagonist_agent(cache_size: int, num_items: int) -> DQNAgent:
    state_size = 128
    action_size = cache_size
    agent = DQNAgent(state_size, action_size, device='cpu')
    checkpoint_path = ROOT / 'results' / 'week3_adversarial' / 'protagonist_agent.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        agent.q_network.load_state_dict(checkpoint)
        agent.target_network.load_state_dict(checkpoint)
        agent.epsilon = 0.0
        print(f"Loaded protagonist agent from {checkpoint_path}")
    else:
        print(f"Protagonist agent not found at {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return agent


def main():
    cache_size = 500
    wiki_path = ROOT / 'data' / 'wikipedia_traces' / 'Wikipedia_CDN_fixed.pkl'
    trace = load_wikipedia_trace(wiki_path)
    #trace = trace[::10]  # every 10th request 
    trace = trace[:1_000_000]   # subsample for speed

    print(f'Using cache_size={cache_size}')
    print(f'Wikipedia trace requests: {len(trace):,}')

    # Load protagonist agent
    protagonist = load_protagonist_agent(cache_size, len(np.unique(trace)))
    rl_policy = RLAgentPolicy(protagonist, len(np.unique(trace)), cache_size)

    results = [
        evaluate_policy('LRU',              LRUPolicy(cache_size),                            trace, cache_size),
        evaluate_policy('LFU',              LFUPolicy(cache_size),                            trace, cache_size),
        evaluate_policy('RARL Protagonist', rl_policy,                                        trace, cache_size),
        evaluate_policy('LeCaR',            LeCaRPolicy(cache_size),                          trace, cache_size),
        evaluate_policy('GL-Cache',         GLCachePolicy(cache_size),                        trace, cache_size),
    ]

    df = pd.DataFrame(results)
    print('\nResults on Wikipedia trace:')
    print(df[['policy', 'hit_rate', 'hits', 'misses']])

    out_path = ROOT / 'results' / 'wikipedia_comparison_5policies.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'\nSaved results to {out_path}')

    plot_path = plot_hit_rate_comparison(
        df, out_path.parent,
        'Wikipedia CDN — Policy Comparison (cache_size=500)'
    )
    print(f'Saved plot to {plot_path}')


if __name__ == '__main__':
    main()
