# Week 2: Single-Agent DQN Training

## Learning Phase: DQN for Cache Replacement

### What is DQN (Deep Q-Network)?

DQN is a **reinforcement learning** algorithm that learns decision-making by trial and error.

**Basic Idea:**
```
Observation (State) → Agent Brain (Neural Network) → Action Decision → Feedback (Reward)
                                                            ↓
                                                    Learn from feedback
```

### Cache Replacement as an RL Problem

**State:** Current snapshot of cache
- Which items are in cache?
- How recently was each accessed?
- How frequently has each been accessed?
- What did we just serve?

Example state vector (for a small cache):
```
State = [
  recency_A,      # How long ago item A was accessed (-1 if not cached)
  frequency_A,    # Total times item A accessed
  recency_B,
  frequency_B,
  recency_C,
  frequency_C,
  current_request # The item just requested (miss)
]
```

**Action:** Which item to evict (if new request misses)
```
Action space = [0, 1, ..., cache_size-1]
Action 0 = Evict item at cache position 0
Action 1 = Evict item at cache position 1
...
```

**Reward:** Immediate feedback on decision quality
```
Reward = +1 if next request hits cache (good decision)
Reward = 0 if next request misses cache (bad decision)

Long-term goal: Maximize cumulative rewards (hit rate)
```

**Transition:** State → Action → Reward → Next State
```
Old State: Cache=[A,B,C], Request=D
Action: Evict B
New State: Cache=[A,D,C], D is now cached

Next Request: D
Reward: +1 (HIT! Our eviction choice was good)
```

### Why DQN vs Classical Policies?

**Classical policies (LRU, LFU):** Hard-coded heuristics
```
LRU: Always evict least recently used (fixed rule)
LFU: Always evict least frequently used (fixed rule)

Problem: Same rule everywhere; can't adapt to distribution changes
```

**DQN:** Learns flexible policy
```
DQN learns: "On this distribution, favor recency. On that distribution, favor frequency"
Can adapt to any pattern in training data
```

### The Neural Network Brain

DQN uses a neural network to predict "Q-values" (goodness of actions):

```
Input: State
  ↓
Hidden Layer 1 (64 neurons, ReLU activation)
  ↓
Hidden Layer 2 (64 neurons, ReLU activation)
  ↓
Output: Q-values for each action [Q(evict item 0), Q(evict item 1), ...]

Decision: Pick action with highest Q-value
```

**What Q-value means:**
```
Q(state, action) = Expected future reward if I take this action now

High Q → Action expected to lead to hits (good)
Low Q → Action expected to lead to misses (bad)

Example:
Q(current_state, evict_LRU_item) = 0.92 (high: LRU usually works)
Q(current_state, evict_LFU_item) = 0.45 (low: LFU bad for this distribution)

Decision: Choose action 1 (evict LRU item)
```

### Learning Process: Q-Learning

**Goal:** Teach network to predict accurate Q-values

**Algorithm (simplified):**

```
1. Take random action in cache
2. Observe reward and next state
3. Compute Q-value target:
   target = reward + γ * max(Q_next_state)
   
   Where:
   - reward = +1 (hit) or 0 (miss)
   - γ (gamma) = 0.99 (discount factor: future rewards matter less)
   - max(Q_next_state) = Best possible Q-value in next state

4. Update network: Minimize (predicted_Q - target)^2
5. Repeat 1000s of times until network converges
```

**Intuition:**
```
If action leads to hit, boost its Q-value
If action leads to miss, lower its Q-value
Over time, network learns accurate predictions
```

### Exploration vs Exploitation

**Problem:** DQN must balance two goals:
- **Exploitation:** Use best-learned policy (act greedily)
- **Exploration:** Try new actions to discover better policies

**Solution: ε-greedy exploration**
```
With probability ε (e.g., 10%): Pick random action (explore)
With probability 1-ε (e.g., 90%): Pick best-learned action (exploit)

Early training: Higher ε (more exploration)
Late training: Lower ε (more exploitation)
```

**Why this matters for cache:**
```
Early: "Try evicting different items, see what works"
Late: "I've learned good eviction policy, follow it"

Without exploration: Agent gets stuck in local optima
```

---

## Week 2 Implementation Plan

### File Structure

```
Week 2 files to create:
├── src/dqn_agent.py          # DQN agent implementation
├── src/experience_replay.py   # Memory buffer for training
├── src/main_week2.py          # Training orchestration
└── results/week2_training/    # Training plots and results
```

### State Representation

For each cached item, we track:
```python
state = {
    "recencies": [...],        # -1 if not cached, else steps ago
    "frequencies": [...],       # Total accesses
    "cache_positions": [...],   # Where each item is in cache
    "current_request": item_id
}
```

### Action Space

```python
num_actions = cache_size
action = which_item_to_evict (0 to cache_size-1)
```

### Reward Signal

```python
reward = 1.0 if next_request_hits else 0.0
```

### Network Architecture

```
Input: 256-dim state vector
  ↓
Dense(128, ReLU)
  ↓
Dense(64, ReLU)
  ↓
Dense(cache_size) → Q-values for each action
```

---

## Training Setup

### Hyperparameters

```python
learning_rate = 0.001           # How much to update weights
gamma = 0.99                    # Discount future rewards
epsilon_start = 1.0             # Full exploration initially
epsilon_end = 0.05              # Minimal exploration after
epsilon_decay = 0.995           # Decay exploration over time
batch_size = 32                 # Mini-batch for training
replay_buffer_size = 100000     # Store last 100k transitions
target_update_freq = 1000       # Sync target network every 1000 steps
```

### Training Strategy

**Phase 1: Training on Zipfian α=1.0 (100k steps)**
- Agent plays cache game against Zipfian α=1.0 trace
- Collect experience, update network
- Track hit rate every 1000 steps
- Expected: Converges to 95%+ hit rate

**Phase 2: Testing generalization (no training)**
- Freeze network weights
- Test on Zipfian α=1.0 (training dist): Should see 95%+
- Test on Zipfian α=1.5: Expect 70-80%?
- Test on Zipfian α=2.0: Expect 60-70%?
- Test on WorkingSet: Expect 30-40% (if recency learned)

**Measurement: Generalization Gap**
```
Gap = Performance(training_dist) - Performance(other_dist)
    = 95% - (avg of other distributions)
    
Expected gap: 40-60 percentage points
→ Shows why adversarial training needed (Week 3)
```

---

## Expected Results

### Learning Curve (Training on Zipfian α=1.0)

```
Hit Rate %
   100 |                           ----
    90 |                  /--------/
    80 |          /------/
    70 |        /
    60 |      /
    50 |    /
    40 |  /
    30 |/
    20 |___________________
     0 |
       0       25k     50k     75k    100k Steps
       
Expected: Starts 30% (random), ends 95%+
```

### Generalization Test Results

```
Distribution    Training    Validation    Gap
Zipfian α=1.0   95%+        95%+ ✓        ~0% (same dist)
Zipfian α=1.5   95%+        75%  ✗        20%
Zipfian α=2.0   95%+        65%  ✗        30%
WorkingSet      95%+        35%  ✗        60%

Average gap: ~35 percentage points
```

**Key insight:** Agent learns frequency-based strategy that:
- ✓ Crushes Zipfian α=1.0 (right distribution)
- ✗ Fails on WorkingSet (different distribution)
- → Week 3: Fix with adversarial training!

---

## Why This Matters

**Week 1 Result:** LFU gets 71.92% on Zipfian α=1.0

**Week 2 Expected:** DQN gets 95%+ on Zipfian α=1.0

**Question:** Why so different?
- LFU: Holds fixed frequency counts (loses recency info)
- DQN: Learns richer features (frequency + recency + position)

**Week 3 Challenge:**
- DQN overfits to Zipfian distribution
- Adversary will generate WorkingSet-like traces
- Protagonist forced to learn more robust strategy

