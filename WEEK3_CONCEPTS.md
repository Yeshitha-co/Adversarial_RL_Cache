# WEEK 3: Adversarial Reinforcement Learning for Cache Replacement

## Overview
Week 3 implements **adversarial co-training** where two agents compete:
- **Protagonist**: Cache replacement agent (learns optimal eviction policy)
- **Adversary**: Workload generator (learns to create hard workloads)

The result: A robust cache agent that generalizes across ALL workload types, not just Zipfian.

---

## Week 2 Problem Recap

| Workload | DQN | LRU | LFU | Issue |
|---|---|---|---|---|
| Zipfian α=1.0 | 58.45% | 12.68% | 49.05% | ✅ Dominates |
| Zipfian α=1.5 | 95.91% | 55.52% | 94.94% | ✅ Dominates |
| Zipfian α=2.0 | 99.57% | 99.57% | 99.57% | ✅ Ties |
| WorkingSet | 4.95% | 5.70% | 5.77% | ❌ **FAILS** |

**Root Cause:** DQN overfitted to Zipfian patterns. Fails on uniform WorkingSet.

---

## Adversarial Training Solution

### Core Idea: Co-Evolutionary Arms Race

```
Round 1:
  Protagonist trains on Zipfian α=1.0
  Competitor: It's too easy! I need harder workloads!

Round 2:
  Adversary learns: "Create workloads that fool the protagonist"
  → Generates mix of Zipfian + WorkingSet patterns

Round 3:
  Protagonist adapts: "I need strategies that handle BOTH"
  → Learns robust policies

Round 4:
  Adversary intensifies: "Let me try even harder patterns"
  → Creates customized hard workloads

...cycle repeats until convergence
```

### Two-Agent Architecture

#### Protagonist Agent (Cache Policy)
- Same DQN as Week 2 (dense network, ε-greedy)
- **Goal:** Maximize hit rate
- **Input:** Cache state (recency + frequency)
- **Output:** Which item to evict
- **Learns:** Robust strategies against diverse workloads

#### Adversary Agent (Workload Generator)
- DQN that generates request sequences
- **Goal:** Minimize cache hit rate (make protagonist fail)
- **Input:** Hidden state of protagonist's recent decisions
- **Output:** Next item ID to request
- **Learns:** "What requests confuse the cache agent?"

### Training Loop

```python
for round in range(num_rounds):
    # Phase 1: Protagonist trains
    for step in range(protagonist_steps):
        request = adversary.generate_request()
        state = cache.get_state()
        action = protagonist.select_action(state)
        reward = cache.process_request(request)
        protagonist.learn(state, action, reward)
    
    # Phase 2: Adversary trains
    for step in range(adversary_steps):
        state = cache.get_state()
        request = adversary.select_request(state)
        protagonist_hit = cache.process_request(request)
        adversary_reward = -protagonist_hit  # Reward = make protagonist miss
        adversary.learn(state, request, adversary_reward)
    
    # Evaluate both agents
    protagonist_hit_rate = evaluate(protagonist)
    print(f"Round {round}: Hit Rate = {protagonist_hit_rate:.4f}")
```

---

## Key Differences from Week 2

| Aspect | Week 2 | Week 3 |
|---|---|---|
| **Training Setup** | Single agent on fixed trace | Two agents, co-training |
| **Workload** | Zipfian α=1.0 (fixed) | Adversary-generated (adaptive) |
| **Goal** | Maximize hit rate on one pattern | Generalize across all patterns |
| **Training Duration** | Single pass (50k steps) | Multiple rounds (10+ rounds) |
| **Expected Improvement** | ✅ Beats LRU/LFU on Zipfian | 🎯 **Beats both on ALL distributions** |

---

## Expected Results

### Protagonist Performance (Week 3 vs Week 2)

| Workload | Week 2 DQN | Week 3 DQN | Improvement |
|---|---|---|---|
| Zipfian α=1.0 | 58.45% | ~55-60% | Hold steady |
| Zipfian α=1.5 | 95.91% | ~85-90% | Trade-off (less specialized) |
| Zipfian α=2.0 | 99.57% | ~95-98% | Trade-off (less specialized) |
| WorkingSet | 4.95% | **~15-25%** | ✅ **Huge improvement** |
| **Average** | 64.72% | **~65-75%** | ✅ Better generalization |

**Key Insight:** Week 3 trades off peak performance on Zipfian for **much better** average performance across all workloads.

---

## Mathematical Framework

### Protagonist Loss
```
L_protagonist = MSE(Q_target, Q_learn)
where:
  Q_target = reward + γ * max(Q_next)
  reward = 1 if cache hit, 0 if miss
```

### Adversary Loss
```
L_adversary = -log(π(request|state)) * advantage
where:
  advantage = -(protagonist_hit_rate)
  = positive when protagonist misses
```

### Nash Equilibrium
Training converges when:
- Protagonist: Can't improve hit rate against current adversary
- Adversary: Can't generate harder workloads against current protagonist
- Result: Robust equilibrium protecting against diverse workloads

---

## Implementation Details

### Adversary Network Architecture
- **Input:** 128-dim state vector (cache state + history)
- **Hidden Layers:** 128 → 64 → 32 (narrower than protagonist)
- **Output:** 500 request IDs (which items to request)
- **Total Parameters:** ~50,000

### Training Hyperparameters
- **Num Rounds:** 10 adversarial rounds
- **Protagonist Steps/Round:** 20,000 (less than Week 2)
- **Adversary Steps/Round:** 10,000 (learns from protagonist)
- **Cache Size:** 500 (same as Week 2 for comparison)
- **Batch Size:** 32 (both agents)

### Exploration Strategy
- **Protagonist:** ε-greedy (start=1.0, end=0.05, linear decay)
- **Adversary:** ε-greedy (start=0.5, end=0.1, linear decay)
  - Lower exploration for adversary (more targeted attacks)

---

## Evaluation Metrics

### Per-Distribution Hit Rate
- Same 4 distributions as Week 2
- Compare Week 3 vs Week 2 vs baselines

### Generalization Gap
```
Gap = best_performance - worst_performance
Week 2 Gap = 99.57% - 4.95% = 94.62% (huge!)
Week 3 Gap = ??? - ??? = smaller (goal!)
```

### Robustness Score
```
Score = min(hit_rates across all distributions)
Week 2 = 4.95% (WorkingSet failure)
Week 3 = target: ~15-25% (reliable minimum)
```

---

## Why This Works

1. **Diversity Pressure:** Adversary forces protagonist to handle multiple patterns
2. **Adaptive Learning:** Adversary adapts strategies each round
3. **No Overfitting:** Can't optimize for unknown future adversary strategies
4. **Game Theory:** Nash equilibrium = robust solution

---

## Expected Output Files

```
results/week3_adversarial/
├── protagonist_agent.pth      # Best protagonist weights
├── adversary_agent.pth        # Final adversary weights
├── week3_training_log.csv     # Round-by-round metrics
├── week3_generalization.csv   # Per-distribution results
├── week3_comparison.png       # Week 2 vs Week 3 visualization
└── week3_summary.txt          # Final statistics
```

---

## Success Criteria

✅ **Week 3 is successful if:**
1. Protagonist hit rate on WorkingSet improves to >10% (from 4.95%)
2. Average hit rate across distributions > Week 2
3. Generalization gap decreases (more balanced performance)
4. Robustness score improves (minimum performance increases)

Let's build it! 🚀
