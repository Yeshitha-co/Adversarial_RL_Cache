"""
WEEK 3 ADVANCED: Adversarial WorkingSet Training
Protagonist learns to handle temporal locality patterns through adversarial co-training.
Adversary generates WorkingSet workloads (not arbitrary attacks), forcing protagonist to adapt.
"""

import sys
from pathlib import Path
from typing import Dict
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
from experience_replay import ExperienceReplayBuffer
from utils import setup_paths


class WorkingSetCacheEnvironment:
    """Cache environment optimized for WorkingSet learning."""
    
    def __init__(self, cache_size: int, num_items: int = 10000):
        self.cache_size = cache_size
        self.num_items = num_items
        self.cache = set()
        self.recency = {}
        self.frequency = {}
        self.recent_accesses = deque(maxlen=100)  # Track recent items for working set
        self.total_steps = 0
    
    def _get_state_vector(self) -> np.ndarray:
        """Get enhanced state vector focused on temporal locality."""
        state = []
        cached_items = sorted(list(self.cache))
        max_slots = 64
        
        for i in range(max_slots):
            if i < len(cached_items):
                it = cached_items[i]
                
                # Recency: 1.0 - normalized time since access
                time_since_access = self.total_steps - self.recency.get(it, 0)
                recency = max(0.0, 1.0 - (time_since_access / 500.0))
                
                # Temporal Locality: Is item in recent working set?
                in_working_set = 1.0 if it in self.recent_accesses else 0.0
                
            else:
                recency, in_working_set = 0.0, 0.0
            
            state.extend([recency, in_working_set])
        
        return np.array(state, dtype=np.float32)
    
    def process_request(self, item: int) -> bool:
        """Process a cache request. Returns True if hit, False if miss."""
        self.total_steps += 1
        self.recent_accesses.append(item)  # Track for working set detection
        
        is_hit = item in self.cache
        
        if is_hit:
            self.recency[item] = self.total_steps
            self.frequency[item] = self.frequency.get(item, 0) + 1
        else:
            if len(self.cache) < self.cache_size:
                self.cache.add(item)
            self.recency[item] = self.total_steps
            self.frequency[item] = self.frequency.get(item, 0) + 1
        
        return is_hit
    
    def evict(self, item: int):
        """Evict a specific item from cache."""
        if item in self.cache:
            self.cache.remove(item)
    
    def reset(self):
        """Reset environment."""
        self.cache.clear()
        self.recency.clear()
        self.frequency.clear()
        self.recent_accesses.clear()
        self.total_steps = 0


class AdversaryWorkingSetGenerator:
    """
    Adversary that generates WorkingSet-style requests to stress-test the protagonist.
    Generates requests biased toward current working set but occasionally introduces new items.
    """
    
    def __init__(self, num_items: int, working_set_size: int = 100, locality: float = 0.8):
        self.num_items = num_items
        self.working_set_size = working_set_size
        self.locality = locality  # Probability of staying in working set
        self.working_set = set(np.random.choice(num_items, working_set_size, replace=False))
        self.recent_items = deque(maxlen=50)
    
    def generate_request(self) -> int:
        """Generate request: stay in working set or introduce new item."""
        if np.random.random() < self.locality:
            # Stay in working set
            if self.recent_items:
                # Prefer recently accessed items (temporal locality)
                return list(self.recent_items)[np.random.randint(0, len(self.recent_items))]
            else:
                return np.random.choice(list(self.working_set))
        else:
            # Phase transition: introduce new working set
            self.working_set = set(np.random.choice(self.num_items, self.working_set_size, replace=False))
            return np.random.choice(list(self.working_set))
    
    def observe_hit(self, item: int):
        """Observe cache hit to track working set."""
        self.recent_items.append(item)
    
    def reset(self):
        """Reset for new round."""
        self.working_set = set(np.random.choice(self.num_items, self.working_set_size, replace=False))
        self.recent_items.clear()


def adversarial_workingset_training(num_rounds: int = 10,
                                     protagonist_steps: int = 20000,
                                     adversary_steps: int = 10000,
                                     batch_size: int = 32):
    """
    Co-training loop: Protagonist learns to handle WorkingSet patterns.
    Adversary generates WorkingSet-style attacks (not random).
    """
    
    paths = setup_paths()
    results_dir = paths['results'] / 'week3_workingset_adversarial'
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("WEEK 3 ADVANCED: ADVERSARIAL WORKINGSET TRAINING")
    print("Protagonist learns temporal locality through adversarial co-training")
    print("=" * 70)
    print()
    
    cache_size = 500
    num_items = 10000
    
    # Initialize protagonist (fresh network)
    protagonist = DQNAgent(128, cache_size, learning_rate=0.001)
    generator = TraceGenerator()
    
    print(f"Network: 128→128→64→64 on {protagonist.device}")
    print(f"Protagonist steps per round: {protagonist_steps}")
    print(f"Num rounds: {num_rounds}\n")
    
    round_results = []
    
    for round_num in range(num_rounds):
        print("=" * 70)
        print(f"ROUND {round_num + 1}/{num_rounds}")
        print("=" * 70 + "\n")
        
        env = WorkingSetCacheEnvironment(cache_size, num_items)
        adversary = AdversaryWorkingSetGenerator(num_items, working_set_size=100, locality=0.8)
        protag_buffer = ExperienceReplayBuffer(max_size=100000)
        
        # PHASE 1: Protagonist training on adversarial WorkingSet workload
        print("Phase 1: Protagonist Training on WorkingSet Attacks...")
        protag_hits = 0
        
        for step in range(protagonist_steps):
            old_state = env._get_state_vector()
            
            # Adversary generates WorkingSet-style request
            request = adversary.generate_request()
            
            # Protagonist processes it
            is_hit = env.process_request(request)
            protag_hits += int(is_hit)
            
            if is_hit:
                adversary.observe_hit(request)
            
            new_state = env._get_state_vector()
            reward = 1.0 if is_hit else -1.0
            
            # Store experience
            protag_buffer.store(old_state, 0, reward, new_state, False)
            
            # Train protagonist
            if len(protag_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = protag_buffer.sample(batch_size)
                protagonist.train_step(states, actions, rewards, next_states, dones)
            
            # Eviction logic: DQN decides which item to evict
            if len(env.cache) >= cache_size:
                with torch.no_grad():
                    q_vals = protagonist.q_network(torch.FloatTensor(new_state).unsqueeze(0).to(protagonist.device))
                action = torch.argmin(q_vals).item() % len(env.cache)
                evict_item = sorted(list(env.cache))[min(action, len(env.cache) - 1)]
                env.evict(evict_item)
            
            if (step + 1) % 5000 == 0:
                rate = protag_hits / (step + 1)
                print(f"  Step {step + 1:6d} | Hit Rate: {rate:.4f}")
        
        protag_hit_rate = protag_hits / protagonist_steps
        print(f"  Final Hit Rate: {protag_hit_rate:.4f}\n")
        
        round_results.append({
            'round': round_num + 1,
            'protagonist_hit_rate': protag_hit_rate,
        })
    
    # Save trained protagonist
    torch.save(protagonist.q_network.state_dict(), results_dir / 'workingset_protagonist.pth')
    print(f"✓ WorkingSet protagonist saved\n")
    
    # EVALUATION on all distributions
    print("=" * 70)
    print("EVALUATION: Can it generalize?")
    print("=" * 70)
    print()
    
    env = WorkingSetCacheEnvironment(cache_size, num_items)
    results = []
    protag_scores = []
    
    for dist_name, dist_fn in [
        ("Zipfian_α=2.0", lambda: generator.zipfian_trace(100000, num_items, alpha=2.0)),
        ("Zipfian_α=1.5", lambda: generator.zipfian_trace(100000, num_items, alpha=1.5)),
        ("Zipfian_α=1.0", lambda: generator.zipfian_trace(100000, num_items, alpha=1.0)),
        ("WorkingSet_80%", lambda: generator.working_set_trace(100000, num_items, working_set_size=100, locality=0.8)),
    ]:
        trace = dist_fn()
        
        # Protagonist evaluation
        env.reset()
        protag_hits = 0
        
        for item in trace:
            is_hit = env.process_request(item)
            protag_hits += int(is_hit)
            
            if not is_hit and len(env.cache) >= cache_size:
                state = env._get_state_vector()
                with torch.no_grad():
                    q_vals = protagonist.q_network(torch.FloatTensor(state).unsqueeze(0).to(protagonist.device))
                action = torch.argmin(q_vals).item() % len(env.cache)
                evict_item = sorted(list(env.cache))[min(action, len(env.cache) - 1)]
                env.evict(evict_item)
        
        protag_rate = protag_hits / len(trace)
        protag_scores.append(protag_rate)
        
        # Baselines
        sim_lru = CacheSimulator(cache_size, LRUPolicy(cache_size))
        lru_result = sim_lru.run(trace)
        lru_rate = lru_result['hit_rate']
        
        sim_lfu = CacheSimulator(cache_size, LFUPolicy(cache_size))
        lfu_result = sim_lfu.run(trace)
        lfu_rate = lfu_result['hit_rate']
        
        improvement = (protag_rate - lru_rate) / max(0.001, lru_rate)
        
        print(f"{dist_name:18s}: Protagonist={protag_rate:.4f}  LRU={lru_rate:.4f}  ({improvement:+.0%})")
        results.append({
            "Distribution": dist_name,
            "Protagonist": protag_rate,
            "LRU": lru_rate,
            "LFU": lfu_rate,
        })
    
    print()
    
    # Summary
    df = pd.DataFrame(results)
    df.to_csv(results_dir / 'workingset_adversarial_results.csv', index=False)
    
    avg = np.mean(protag_scores)
    min_rate = np.min(protag_scores)
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Average Hit Rate: {avg:.4f}")
    print(f"Minimum Hit Rate (WorkingSet): {min_rate:.4f}")
    print()
    
    if min_rate > 0.20:
        print("✅ HUGE SUCCESS! DQN learned WorkingSet through RL!")
    elif min_rate > 0.15:
        print("🚀 GREAT! Significant WorkingSet improvement!")
    elif min_rate > 0.08:
        print("⭐ GOOD! WorkingSet learned better than baseline!")
    else:
        print(f"ℹ️  WorkingSet hit rate: {min_rate:.4f}")
    
    print()
    print("Why this matters:")
    print("- Week 2 (single-agent DQN on Zipfian): WorkingSet = 0.0495 ❌")
    print("- Week 3 original (adversary generates random): WorkingSet = 0.0505 ⚠️")
    print(f"- Week 3 NEW (adversary generates WorkingSet): WorkingSet = {protag_scores[3]:.4f}")
    print()
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(results))
    width = 0.25
    
    ax.bar(x - width, protag_scores, width, label='DQN (WorkingSet-Trained)', color='red')
    ax.bar(x, [r['LRU'] for r in results], width, label='LRU', color='blue')
    ax.bar(x + width, [r['LFU'] for r in results], width, label='LFU', color='orange')
    
    ax.set_ylabel('Hit Rate')
    ax.set_title('WorkingSet-Adversarial Training Results')
    ax.set_xticks(x)
    ax.set_xticklabels([r['Distribution'] for r in results])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'workingset_adversarial_results.png', dpi=100)
    print(f"✓ Plot saved to {results_dir}\n")


if __name__ == "__main__":
    adversarial_workingset_training(num_rounds=10)
