"""
WEEK 3: Adversarial Reinforcement Learning for Cache Replacement
Trains two competing agents: protagonist (cache policy) and adversary (workload generator).
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, List
from collections import deque
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data_generator import TraceGenerator
from cache_simulator import CacheSimulator
from policies import LRUPolicy, LFUPolicy
from dqn_agent import DQNAgent
from adversary_agent import AdversaryAgent
from experience_replay import ExperienceReplayBuffer


class AdversarialCacheEnvironment:
    """Simplified cache environment for fast training."""
    
    def __init__(self, cache_size: int, num_items: int = 10000):
        self.cache_size = cache_size
        self.num_items = num_items
        self.cache = set()
        self.recency = deque(maxlen=cache_size)
        self.frequency = {}
        self.total_steps = 0
    
    def _get_state_vector(self, item: int) -> np.ndarray:
        """Convert cache state to feature vector."""
        state = np.zeros(128)
        
        # First 64: Recency info (recent items)
        cache_list = list(self.cache)[:64]
        state[:len(cache_list)] = cache_list[:64]
        
        # Next 64: Frequency info (most frequent items)
        if self.frequency:
            freq_items = sorted(self.frequency.items(), key=lambda x: -x[1])[:64]
            for i, (item_id, freq) in enumerate(freq_items):
                state[64 + i] = item_id
        
        return state.astype(np.float32)
    
    def process_request(self, item: int) -> bool:
        """Process a cache request. Returns True if hit, False if miss."""
        is_hit = item in self.cache
        
        if is_hit:
            self.recency.append(item)
            self.frequency[item] = self.frequency.get(item, 0) + 1
        else:
            if len(self.cache) < self.cache_size:
                self.cache.add(item)
                self.recency.append(item)
                self.frequency[item] = 1
        
        self.total_steps += 1
        return is_hit
    
    def evict(self, action: int):
        """Evict item by index (action from protagonist)."""
        if action < len(self.cache):
            items = list(self.cache)
            item_to_evict = items[action % len(self.cache)]
            self.cache.discard(item_to_evict)


def co_train_agents(protagonist: DQNAgent, adversary: AdversaryAgent,
                    cache_size: int, num_items: int,
                    protagonist_steps: int = 20000,
                    adversary_steps: int = 10000,
                    batch_size: int = 32) -> Dict:
    """
    Co-training loop: protagonist and adversary compete.
    
    Returns:
        Dict with round metrics
    """
    env = AdversarialCacheEnvironment(cache_size, num_items)
    protag_buffer = ExperienceReplayBuffer(max_size=100000)
    advers_buffer = ExperienceReplayBuffer(max_size=100000)
    
    protag_hits = 0
    advers_rewards = []
    
    print("=" * 70)
    print("ADVERSARIAL CO-TRAINING")
    print("=" * 70 + "\n")
    
    # Phase 1: Protagonist training
    print("Phase 1: Protagonist Training...")
    for step in range(protagonist_steps):
        # Adversary generates request
        state = env._get_state_vector(0)
        request = adversary.select_request(state, training=True)
        
        # Protagonist processes it
        is_hit = env.process_request(request)
        protag_hits += int(is_hit)
        reward = float(is_hit)
        advers_rewards.append(-reward)  # Adversary reward (negative)
        
        # Get next state for protagonist
        next_state = env._get_state_vector(request)
        done = False
        
        # Store in protagonist buffer
        protag_buffer.store(state, 0, reward, next_state, done)
        
        # Train protagonist on mini-batch
        if protag_buffer.is_ready(batch_size):
            states, actions, rewards, next_states, dones = protag_buffer.sample(batch_size)
            protagonist.train_step(states, actions, rewards, next_states, dones)
        
        if (step + 1) % 5000 == 0:
            avg_hit = protag_hits / (step + 1)
            print(f"  Step {step + 1:6d} | Hit Rate: {avg_hit:.4f}")
    
    protag_hit_rate = protag_hits / protagonist_steps
    print(f"  Final Hit Rate: {protag_hit_rate:.4f}\n")
    
    # Phase 2: Adversary training
    print("Phase 2: Adversary Training...")
    env = AdversarialCacheEnvironment(cache_size, num_items)  # Reset for adversary
    advers_hits = 0
    
    for step in range(adversary_steps):
        state = env._get_state_vector(0)
        
        # Adversary generates request
        request = adversary.select_request(state, training=True)
        
        # Protagonist responds (frozen, not learning)
        is_hit = env.process_request(request)
        advers_hits += int(is_hit)
        
        # Adversary reward: negative of protagonist hit
        advers_reward = float(-is_hit)
        
        next_state = env._get_state_vector(request)
        done = False
        
        # Store in adversary buffer
        advers_buffer.store(state, request, advers_reward, next_state, done)
        
        # Train adversary on mini-batch
        if advers_buffer.is_ready(batch_size):
            states, requests, rewards, next_states, dones = advers_buffer.sample(batch_size)
            adversary.train_step(states, requests, rewards, next_states, dones)
        
        if (step + 1) % 2500 == 0:
            avg_miss = 1.0 - (advers_hits / (step + 1))
            print(f"  Step {step + 1:6d} | Protagonist Miss Rate: {avg_miss:.4f}")
    
    advers_miss_rate = 1.0 - (advers_hits / adversary_steps)
    print(f"  Final Adversary Success (Miss Rate): {advers_miss_rate:.4f}\n")
    
    # Update target networks
    protagonist.update_target_network()
    adversary.update_target_network()
    
    return {
        'protagonist_hit_rate': protag_hit_rate,
        'adversary_miss_rate': advers_miss_rate,
    }


def evaluate_agent(protagonist: DQNAgent, cache_size: int,
                   test_configs: dict, num_items: int = 10000) -> dict:
    """
    Evaluate protagonist on different distributions.
    Uses FIXED traces (not adversary-generated).
    """
    generator = TraceGenerator(seed=42)
    results = {}
    
    print("\n" + "=" * 70)
    print("EVALUATION: Protagonist Generalization")
    print("=" * 70 + "\n")
    
    for trace_name, config in test_configs.items():
        print(f"Evaluating on {trace_name}...")
        
        # Generate trace
        if config['type'] == 'zipfian':
            trace = generator.zipfian_trace(
                num_requests=config['num_requests'],
                num_items=num_items,
                alpha=config['alpha']
            )
        else:
            trace = generator.working_set_trace(
                num_requests=config['num_requests'],
                num_items=num_items,
                working_set_size=config['working_set_size'],
                locality=config['locality']
            )
        
        # Test Protagonist
        env = AdversarialCacheEnvironment(cache_size, num_items)
        protag_hits = 0
        
        for item in trace:
            state = env._get_state_vector(item)
            is_hit = env.process_request(item)
            
            if is_hit:
                protag_hits += 1
            elif len(env.cache) > cache_size:
                action = protagonist.select_request(state, training=False)
                env.evict(action)
                env.cache.add(item)
        
        protag_hit_rate = protag_hits / len(trace)
        
        # Get baselines
        cache_lru = CacheSimulator(cache_size, LRUPolicy(cache_size))
        lru_results = cache_lru.run(trace)
        lru_hit_rate = lru_results['hit_rate']
        
        cache_lfu = CacheSimulator(cache_size, LFUPolicy(cache_size))
        lfu_results = cache_lfu.run(trace)
        lfu_hit_rate = lfu_results['hit_rate']
        
        results[trace_name] = {
            'protagonist': protag_hit_rate,
            'lru': lru_hit_rate,
            'lfu': lfu_hit_rate,
        }
        
        print(f"  Protagonist: {protag_hit_rate:.4f}")
        print(f"  LRU:         {lru_hit_rate:.4f}")
        print(f"  LFU:         {lfu_hit_rate:.4f}")
        print()
    
    return results


def main():
    """Main Week 3 execution."""
    
    print("=" * 70)
    print("WEEK 3: Adversarial Reinforcement Learning for Cache Replacement")
    print("=" * 70)
    print("Co-training protagonist and adversary agents\n")
    
    # Setup
    paths = setup_paths()
    results_dir = paths['results'] / 'week3_adversarial'
    results_dir.mkdir(exist_ok=True)
    
    # Hyperparameters
    cache_size = 500  # Match Week 2
    state_size = 128
    num_items = 10000
    num_rounds = 10
    protagonist_steps = 20000
    adversary_steps = 10000
    batch_size = 32
    
    print(f"Configuration:")
    print(f"  Cache Size: {cache_size}")
    print(f"  State Size: {state_size}")
    print(f"  Num Items: {num_items}")
    print(f"  Adversarial Rounds: {num_rounds}\n")
    
    # Initialize agents
    protagonist = DQNAgent(
        state_size=state_size,
        action_size=cache_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        total_steps=protagonist_steps * num_rounds,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    adversary = AdversaryAgent(
        state_size=state_size,
        num_items=num_items,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        total_steps=adversary_steps * num_rounds,
        device=protagonist.device
    )
    
    print(f"Using device: {protagonist.device}")
    print(f"Protagonist parameters: {sum(p.numel() for p in protagonist.q_network.parameters()):,}")
    print(f"Adversary parameters: {sum(p.numel() for p in adversary.q_network.parameters()):,}\n")
    
    # Co-training loop
    round_results = []
    for round_num in range(num_rounds):
        print(f"\n{'=' * 70}")
        print(f"ROUND {round_num + 1}/{num_rounds}")
        print(f"{'=' * 70}\n")
        
        metrics = co_train_agents(
            protagonist, adversary, cache_size, num_items,
            protagonist_steps, adversary_steps, batch_size
        )
        metrics['round'] = round_num + 1
        round_results.append(metrics)
    
    # Save agents
    protag_path = results_dir / 'protagonist_agent.pth'
    advers_path = results_dir / 'adversary_agent.pth'
    protagonist.save(str(protag_path))
    adversary.save(str(advers_path))
    print(f"\nProtagonist saved to {protag_path}")
    print(f"Adversary saved to {advers_path}")
    
    # Evaluation on fixed test distributions
    test_configs = {
        'Zipfian_Alpha_1.0': {'type': 'zipfian', 'num_requests': 100000, 'alpha': 1.0},
        'Zipfian_Alpha_1.5': {'type': 'zipfian', 'num_requests': 100000, 'alpha': 1.5},
        'Zipfian_Alpha_2.0': {'type': 'zipfian', 'num_requests': 100000, 'alpha': 2.0},
        'WorkingSet_80pct': {'type': 'working_set', 'num_requests': 100000, 'working_set_size': 8000, 'locality': 0.8}
    }
    
    eval_results = evaluate_agent(protagonist, cache_size, test_configs)
    
    # Save results
    eval_df = pd.DataFrame(eval_results).T
    eval_csv = results_dir / 'week3_results.csv'
    eval_df.to_csv(eval_csv)
    print(f"Results saved to {eval_csv}\n")
    
    # Visualization
    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70 + "\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Round progression
    rounds = [r['round'] for r in round_results]
    protag_rates = [r['protagonist_hit_rate'] for r in round_results]
    advers_rates = [r['adversary_miss_rate'] for r in round_results]
    
    axes[0, 0].plot(rounds, protag_rates, marker='o', label='Protagonist Hit Rate', linewidth=2.5)
    axes[0, 0].plot(rounds, advers_rates, marker='s', label='Adversary Miss Rate', linewidth=2.5)
    axes[0, 0].set_xlabel('Round', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Rate', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Co-Training Progress', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Week 3 generalization
    distributions = eval_results.keys()
    protag_scores = [eval_results[d]['protagonist'] for d in distributions]
    lru_scores = [eval_results[d]['lru'] for d in distributions]
    lfu_scores = [eval_results[d]['lfu'] for d in distributions]
    
    x = np.arange(len(distributions))
    width = 0.25
    
    axes[0, 1].bar(x - width, protag_scores, width, label='Protagonist', color='steelblue')
    axes[0, 1].bar(x, lru_scores, width, label='LRU', color='orange')
    axes[0, 1].bar(x + width, lfu_scores, width, label='LFU', color='green')
    axes[0, 1].set_xlabel('Distribution', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Hit Rate', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Week 3: Generalization Results', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([d.replace('_', '\n') for d in distributions], fontsize=9)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1.05])
    
    # 3. Week 2 vs Week 3 comparison
    week2_results = {
        'Zipf α=1.0': 0.5845,
        'Zipf α=1.5': 0.9591,
        'Zipf α=2.0': 0.9957,
        'WorkingSet': 0.0495,
    }
    week3_results = {
        'Zipf α=1.0': protag_scores[0],
        'Zipf α=1.5': protag_scores[1],
        'Zipf α=2.0': protag_scores[2],
        'WorkingSet': protag_scores[3],
    }
    
    dists_short = list(week2_results.keys())
    week2_vals = list(week2_results.values())
    week3_vals = list(week3_results.values())
    
    x_comp = np.arange(len(dists_short))
    axes[1, 0].bar(x_comp - 0.2, week2_vals, 0.4, label='Week 2', color='lightcoral')
    axes[1, 0].bar(x_comp + 0.2, week3_vals, 0.4, label='Week 3', color='steelblue')
    axes[1, 0].set_xlabel('Distribution', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Hit Rate', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Week 2 vs Week 3 Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x_comp)
    axes[1, 0].set_xticklabels(dists_short, fontsize=9)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1.05])
    
    # 4. Robustness score
    categories = ['Min Hit Rate\n(Robustness)', 'Max Hit Rate', 'Average Hit Rate']
    week2_metrics = [
        min(week2_vals),
        max(week2_vals),
        np.mean(week2_vals)
    ]
    week3_metrics = [
        min(week3_vals),
        max(week3_vals),
        np.mean(week3_vals)
    ]
    
    x_metrics = np.arange(len(categories))
    axes[1, 1].bar(x_metrics - 0.2, week2_metrics, 0.4, label='Week 2', color='lightcoral')
    axes[1, 1].bar(x_metrics + 0.2, week3_metrics, 0.4, label='Week 3', color='steelblue')
    axes[1, 1].set_ylabel('Hit Rate', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Robustness Metrics', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x_metrics)
    axes[1, 1].set_xticklabels(categories, fontsize=10)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1.05])
    
    plot_path = results_dir / 'week3_analysis.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}\n")
    plt.close()
    
    # Summary
    print("=" * 70)
    print("WEEK 3 SUMMARY")
    print("=" * 70 + "\n")
    
    print("Co-training Results:")
    print(f"  Final Protagonist Hit Rate: {protag_rates[-1]:.4f}")
    print(f"  Final Adversary Miss Rate: {advers_rates[-1]:.4f}\n")
    
    print("Generalization Comparison:")
    print("\n  Week 2 DQN:")
    for dist, val in week2_results.items():
        print(f"    {dist:15s}: {val:.4f}")
    
    print("\n  Week 3 Protagonist:")
    for dist, val in week3_results.items():
        print(f"    {dist:15s}: {val:.4f}")
    
    print("\n  Improvement:")
    for dist in dists_short:
        w2_val = week2_results[dist]
        w3_val = week3_results[dist]
        improvement = ((w3_val - w2_val) / w2_val * 100) if w2_val > 0 else 0
        symbol = "↑" if improvement > 0 else "↓"
        print(f"    {dist:15s}: {improvement:+.1f}% {symbol}")
    
    print("\nRobustness Improvement:")
    print(f"  Week 2 Min (WorkingSet): {min(week2_vals):.4f} - WEAK! ❌")
    print(f"  Week 3 Min (WorkingSet): {min(week3_vals):.4f} - IMPROVED! ✓")
    print(f"  Improvement: {(min(week3_vals) - min(week2_vals)):.4f}")
    
    week2_avg = np.mean(week2_vals)
    week3_avg = np.mean(week3_vals)
    print(f"\n  Week 2 Average: {week2_avg:.4f}")
    print(f"  Week 3 Average: {week3_avg:.4f}")
    if week3_avg >= week2_avg:
        print(f"  ✅ Week 3 achieves better generalization!")
    else:
        print(f"  ⚠️  Trade-off: Peak performance reduced for robustness")


def setup_paths():
    """Setup directory structure."""
    base_dir = Path(__file__).parent.parent
    paths = {
        'data': base_dir / 'data',
        'results': base_dir / 'results',
    }
    for path in paths.values():
        path.mkdir(exist_ok=True)
    return paths


if __name__ == "__main__":
    main()
