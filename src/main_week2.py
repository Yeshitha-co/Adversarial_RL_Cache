"""
WEEK 2: Single-Agent DQN Training
Trains a DQN agent on Zipfian trace and tests generalization to other distributions.
"""

import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

from data_generator import TraceGenerator
from cache_simulator import CacheSimulator
from policies import LRUPolicy, LFUPolicy
from dqn_agent import DQNAgent
from experience_replay import ExperienceReplayBuffer
from utils import setup_paths

import torch


class DQNCacheEnvironment:
    """Simplified cache environment for DQN agent training."""
    
    def __init__(self, cache_size: int):
        """Initialize cache environment."""
        self.cache_size = cache_size
        self.cache = set()
        self.recency = {}
        self.frequency = {}
        self.total_steps = 0
    
    def _get_state_vector(self, item: int) -> np.ndarray:
        """Get simplified state: recency and frequency of items in cache."""
        state = []
        cached_items = sorted(list(self.cache))
        
        # Use fixed-size state vector
        max_slots = 64  # Reduced from cache_size
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
        """Process request, return whether it's a hit."""
        self.total_steps += 1
        
        if item in self.cache:
            # Hit
            self.recency[item] = self.total_steps
            self.frequency[item] = self.frequency.get(item, 0) + 1
            return True
        else:
            # Miss
            if len(self.cache) < self.cache_size:
                # Just add it
                self.cache.add(item)
            # Else: wait for agent action
            
            self.recency[item] = self.total_steps
            self.frequency[item] = 1
            return False
    
    def evict(self, action: int):
        """Execute eviction action."""
        cached_items = sorted(list(self.cache))
        if action < len(cached_items):
            self.cache.remove(cached_items[action])


def train_dqn_fast(agent: DQNAgent, trace: list, cache_size: int,
                   replay_buffer: ExperienceReplayBuffer,
                   num_steps: int = 50000,
                   batch_size: int = 32) -> list:
    """Faster training using direct trace simulation."""
    hit_rates = []
    recent_rewards = deque(maxlen=1000)
    
    print("Starting DQN training with ε-greedy exploration\n")
    
    env = DQNCacheEnvironment(cache_size)
    trace_idx = 0
    step = 0
    
    while step < num_steps:
        if trace_idx >= len(trace):
            trace_idx = 0
            env.cache.clear()
            env.recency.clear()
            env.frequency.clear()
            env.total_steps = 0
        
        item = trace[trace_idx]
        state = env._get_state_vector(item)
        
        # Process request
        hit = env.process_request(item)
        reward = float(hit)
        recent_rewards.append(reward)
        
        # If cache full, need to evict
        if not hit and len(env.cache) > cache_size:
            action = agent.select_action(state, training=True)
            env.evict(action)
            env.cache.add(item)
        
        # Get next state
        trace_idx += 1
        if trace_idx < len(trace):
            next_item = trace[trace_idx]
            next_state = env._get_state_vector(next_item)
        else:
            next_state = state
        
        done = (trace_idx >= len(trace))
        
        # Store in replay buffer
        replay_buffer.store(state, 0, reward, next_state, done)
        
        # Train on mini-batch
        if replay_buffer.is_ready(batch_size):
            states, actions, rewards_batch, next_states, dones = replay_buffer.sample(batch_size)
            agent.train_step(states, actions, rewards_batch, next_states, dones)
        
        step += 1
        
        # Update target network and log
        if step % 1000 == 0:
            agent.update_target_network()
            avg_reward = np.mean(list(recent_rewards)) if recent_rewards else 0.0
            print(f"Step {step:6d} | Hit Rate: {avg_reward:.4f} | ε: {agent.epsilon:.4f}")
            hit_rates.append(avg_reward)
    
    return hit_rates


def evaluate_agent(agent: DQNAgent, cache_size: int, 
                   trace_configs: dict, num_items: int = 10000) -> dict:
    """
    Evaluate trained agent on different distributions.
    
    Args:
        agent: Trained DQN agent
        cache_size: Cache size for evaluation
        trace_configs: Dict of trace configurations
        num_items: Total number of items
        
    Returns:
        Dict of results
    """
    generator = TraceGenerator(seed=42)
    results = {}
    
    print("\n" + "-"*70)
    print("EVALUATION: Testing Generalization")
    print("-"*70 + "\n")
    
    for trace_name, config in trace_configs.items():
        print(f"Evaluating on {trace_name}")
        
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
        
        # Test DQN Agent
        env = DQNCacheEnvironment(cache_size)
        dqn_hits = 0
        
        for item in trace:
            # Get state and check if hit
            state = env._get_state_vector(item)
            is_hit = env.process_request(item)
            
            if is_hit:
                dqn_hits += 1
            elif len(env.cache) > cache_size:
                # Cache full, need eviction
                action = agent.select_action(state, training=False)
                env.evict(action)
                env.cache.add(item)
        
        dqn_hit_rate = dqn_hits / len(trace)
        
        # Compare with LRU and LFU baselines
        cache_lru = CacheSimulator(cache_size, LRUPolicy(cache_size))
        lru_results = cache_lru.run(trace)
        lru_hit_rate = lru_results['hit_rate']
        
        cache_lfu = CacheSimulator(cache_size, LFUPolicy(cache_size))
        lfu_results = cache_lfu.run(trace)
        lfu_hit_rate = lfu_results['hit_rate']
        
        results[trace_name] = {
            'dqn': dqn_hit_rate,
            'lru': lru_hit_rate,
            'lfu': lfu_hit_rate,
        }
        
        print(f"  DQN: {dqn_hit_rate:.4f}")
        print(f"  LRU: {lru_hit_rate:.4f}")
        print(f"  LFU: {lfu_hit_rate:.4f}")
        print()
    
    return results


def main():
    """Main Week 2 execution."""
    
    print("-"*70)
    print("Training DQN with ε-greedy exploration on Zipfian alpha=1.0\n")
    print("-"*70)
    
    # Setup
    paths = setup_paths()
    results_dir = paths['results'] / 'week2_training'
    results_dir.mkdir(exist_ok=True)
    
    # Hyperparameters (optimized for CPU training)
    cache_size = 500  # Match Week 1 baseline testing
    state_size = 128  # Reduced for faster computation
    action_size = cache_size
    num_training_steps = 50000  # Reduced steps for demonstration
    batch_size = 32
    
    print(f"Network Architecture:")
    print(f"  Input: {state_size} (reduced state vector for speed)")
    print(f"  Dense layers: 128 → 128 → 64 → 64 (denser network)")
    print(f"  Output: {action_size} (Q-values for each action)")
    print(f"  Exploration: ε-greedy (start=1.0, end=0.05, linear decay)\n")
    
    # Generate training trace (Zipfian α=1.0)
    print("Generating training trace (Zipfian alpha=1.0)...")
    generator = TraceGenerator(seed=42)
    train_trace = generator.zipfian_trace(
        num_requests=100000,
        num_items=10000,
        alpha=1.0
    )
    print(f"Training trace: {len(train_trace)} requests\n")
    
    # Initialize agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        total_steps=num_training_steps,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Using device: {agent.device}")
    print(f"Total parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}\n")
    
    # Create environment and replay buffer
    env = DQNCacheEnvironment(cache_size)
    replay_buffer = ExperienceReplayBuffer(max_size=100000)
    
    # Train agent
    print("-"*70)
    print("Training")
    print("-"*70 + "\n")
    
    hit_rates = train_dqn_fast(
        agent, train_trace, cache_size, replay_buffer,
        num_steps=num_training_steps,
        batch_size=batch_size
    )
    
    # Save trained agent
    agent_path = results_dir / 'dqn_agent.pth'
    agent.save(str(agent_path))
    print(f"\nAgent saved to {agent_path}\n")
    
    # Evaluate generalization
    test_configs = {
        'Zipfian_Alpha_1.0': {
            'type': 'zipfian',
            'num_requests': 100000,
            'num_items': 10000,
            'alpha': 1.0,
        },
        'Zipfian_Alpha_1.5': {
            'type': 'zipfian',
            'num_requests': 100000,
            'num_items': 10000,
            'alpha': 1.5,
        },
        'Zipfian_Alpha_2.0': {
            'type': 'zipfian',
            'num_requests': 100000,
            'num_items': 10000,
            'alpha': 2.0,
        },
        'WorkingSet_80pct': {
            'type': 'working_set',
            'num_requests': 100000,
            'num_items': 10000,
            'working_set_size': 100,
            'locality': 0.8,
        },
    }
    
    eval_results = evaluate_agent(agent, cache_size, test_configs)
    
    # Create results dataframe
    df_results = []
    for dist, metrics in eval_results.items():
        for policy, hit_rate in metrics.items():
            df_results.append({
                'distribution': dist,
                'policy': policy,
                'hit_rate': hit_rate,
            })
    
    results_df = pd.DataFrame(df_results)
    results_csv = results_dir / 'week2_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")
    
    # Plot learning curve
    print("\n" + "-"*70)
    print("Visualization")
    print("-"*70 + "\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Learning curve
    steps = [i * 1000 for i in range(len(hit_rates))]
    
    # Get actual baseline values for training distribution (cache_size=256)
    lfu_baseline = eval_results['Zipfian_Alpha_1.0']['lfu']
    lru_baseline = eval_results['Zipfian_Alpha_1.0']['lru']
    
    axes[0].plot(steps, hit_rates, marker='o', linewidth=2.5, markersize=6, label='DQN')
    axes[0].axhline(y=lfu_baseline, color='green', linestyle='--', linewidth=2, label=f'LFU Baseline ({lfu_baseline:.2%})')
    axes[0].axhline(y=lru_baseline, color='orange', linestyle='--', linewidth=2, label=f'LRU Baseline ({lru_baseline:.2%})')
    axes[0].set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Hit Rate', fontsize=12, fontweight='bold')
    axes[0].set_title('DQN Learning Curve (Training on Zipfian α=1.0, cache_size=256)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Generalization comparison
    distributions = eval_results.keys()
    dqn_scores = [eval_results[d]['dqn'] for d in distributions]
    lru_scores = [eval_results[d]['lru'] for d in distributions]
    lfu_scores = [eval_results[d]['lfu'] for d in distributions]
    
    x = np.arange(len(distributions))
    width = 0.25
    
    axes[1].bar(x - width, dqn_scores, width, label='DQN', color='steelblue')
    axes[1].bar(x, lru_scores, width, label='LRU', color='orange')
    axes[1].bar(x + width, lfu_scores, width, label='LFU', color='green')
    
    axes[1].set_xlabel('Distribution', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Hit Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('Generalization: DQN vs Classical Baselines', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([d.replace('_', '\n') for d in distributions], fontsize=10)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.05])
    
    plot_path = results_dir / 'week2_analysis.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()
    
    # Print summary
    print("\n" + "-"*70)
    print("Week 2 summary:")
    print("-"*70 + "\n")
    
    print("Training on Zipfian alpha=1.0:")
    print(f"  Final Hit Rate: {hit_rates[-1]:.4f}")
    print(f"  Improvement over LRU: {(hit_rates[-1] - eval_results['Zipfian_Alpha_1.0']['lru']):.4f}")
    print(f"  Improvement over LFU: {(hit_rates[-1] - eval_results['Zipfian_Alpha_1.0']['lfu']):.4f}\n")
    
    print("Generalization Test Results:")
    avg_gap = 0
    for dist in distributions:
        dqn = eval_results[dist]['dqn']
        gap = hit_rates[-1] - dqn
        avg_gap += gap
        print(f"  {dist:20s}: {dqn:.4f} (gap: {gap:+.4f})")
    
    avg_gap /= len(distributions) - 1  # Exclude training distribution
    print(f"\nAverage Generalization Gap: {avg_gap:.4f}")

if __name__ == '__main__':
    main()
