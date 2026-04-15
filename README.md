# RARL-Cache: Adversarial Reinforcement Learning for Robust Web Cache Replacement

## Abstract

Cache replacement decides which item to evict when the cache is full, directly determining hit rate in web systems. Classical policies like LRU and LFU use fixed rules; recent ML agents like LeCaR and GL-Cache learn from traffic traces and do better but only on patterns they have seen. 

We propose **RARL-Cache**, which co-trains two DQN agents:
- **Protagonist:** Learns the optimal cache eviction policy
- **Adversary:** Acts as a training partner by continuously generating request sequences the protagonist handles poorly

This forces the protagonist to generalize beyond any fixed trace. The adversary is discarded at deployment.

**Expected Result:** RARL-Cache outperforms plain DQN and classical baselines (LRU, LFU) when traffic patterns shift outside the training distribution.

---

## Project Structure

```
RL_Final_Project/
├── data/
│   ├── synthetic_traces/          # Generated Zipfian and working-set traces
│   └── wikipedia_traces/          # Real Wikipedia CDN traces (Week 4)
├── src/
│   ├── data_generator.py          # Trace generation (Zipfian, working-set)
│   ├── cache_simulator.py         # Generic cache simulator
│   ├── policies.py                # LRU, LFU, Optimal baseline policies
│   ├── benchmark.py               # Benchmarking and comparison
│   ├── main_week1.py              # Week 1 main script
│   └── utils.py                   # Utility functions
├── results/
│   ├── week1_baseline_results.csv  # W1 numeric results
│   ├── week1_baseline_comparison.png # W1 comparison plots
│   └── plots/                      # Additional visualizations
├── requirements.txt
└── README.md                       # THIS FILE
```

---

## Project Phases

### **Week 1: Dataset Exploration & Classical Baselines** ✅ [COMPLETE]
**Goal:** Establish baseline performance metrics using classical policies.

**Deliverables:**
- Synthetic Zipfian request traces with tunable skewness (α)
- Working-set model traces (temporal locality)
- LRU and LFU baseline implementations
- Benchmark comparison across cache sizes and workloads
- Hit rate plots and CSV results

**Key Insights:**
- LRU performs well on temporal patterns
- LFU performs well on power-law (popularity) distributions
- Both are fixed policies with limited adaptability

---

### **Week 2: Single-Agent DQN Training**
**Goal:** Train a DQN agent on fixed traces; measure overfitting.

**Deliverables:**
- DQN agent implementation (PyTorch)
- Training on fixed Zipfian trace
- Evaluation on training vs. unseen data
- Generalization gap plots

**Expected Result:** DQN overfits to training pattern; poor performance when traffic shifts.

---

### **Week 3: Adversarial Co-Training (RARL)**
**Goal:** Implement adversarial training; demonstrate robustness.

**Deliverables:**
- Protagonist DQN (eviction policy learner)
- Adversary DQN (request sequence generator)
- Co-training loop with alternating updates
- Training curves showing protagonist recovery

**Expected Result:** Protagonist learns robust policy that handles distribution shifts.

---

### **Week 4: Comprehensive Evaluation**
**Goal:** Compare RARL-Cache against all baselines on real Wikipedia traces.

**Deliverables:**
- Real Wikipedia CDN trace evaluation
- Comparison: LRU, LFU, DQN, LeCaR, GL-Cache, RARL-Cache
- Robustness evaluation (distribution shift scenarios)
- Ablation studies
- Publication-ready figures

**Expected Result:** RARL-Cache has highest hit rate under distribution shifts.

---

## Quick Start & Reproduction

### **Prerequisites**

- **Python 3.8+**
- **pip** or **conda**

### **1. Environment Setup**

```bash
cd c:\Users\yeshi\Downloads\RL_Final_Project
pip install -r requirements.txt
```

### **2. Run Week 1 Experiments**

```bash
python src/main_week1.py
```

**What this does:**
1. Generates synthetic Zipfian traces (α=1.0, 1.5, 2.0)
2. Generates working-set traces (temporal locality)
3. Initializes LRU and LFU policies
4. Runs benchmarks across cache sizes: [100, 500, 1000, 2000]
5. Produces comparison plots and CSV results

**Output:**
- `results/week1_baseline_results.csv` - Numeric results
- `results/week1_baseline_comparison.png` - Plots

**Runtime:** ~2-5 minutes on standard CPU

### **3. View Results**

```bash
# View CSV results
cat results/week1_baseline_results.csv

# View plot (Windows)
start results/week1_baseline_comparison.png
```

---

## Week 1 Results Summary

### **Key Findings**

| Trace Type | LRU Hit Rate | LFU Hit Rate | Winner |
|-----------|--------------|--------------|--------|
| Zipfian α=1.0 | ~48-52% | ~50-55% | LFU |
| Zipfian α=1.5 | ~50-55% | ~52-58% | LFU |
| Zipfian α=2.0 | ~52-58% | ~54-62% | LFU |
| Working-Set 80% | ~55-65% | ~50-60% | LRU |

### **Interpretation**

1. **LFU dominates on Zipfian traces** - Makes sense; popularity (frequency) matters more than recency on power-law distributions
2. **LRU dominates on working-set model** - Temporal locality aligns with LRU's design
3. **Both leave room for improvement** - Classical fixed policies plateau quickly; motivates Week 2-3

---

## Technical Deep-Dive

### **Cache Replacement Problem**

```
When cache is FULL and new request arrives:
├─ Option A: Keep everything (impossible, cache full)
├─ Option B: Drop request (unacceptable, lose content)
└─ Option C: Evict item, add new one ← We're here

Which item to evict? That's the policy decision.
Goal: Maximize hit rate = (hits / total_requests)
```

### **LRU Policy**

**Intuition:** Items accessed recently are likely reused soon.

**Algorithm:**
```
On HIT:    Update last_access_time[item] = now
On MISS:   Evict item with minimum last_access_time
```

**Trade-offs:**
- Simple, O(1) operations
- Good for temporal locality
- Ignores frequency (can't distinguish hot vs cold)
- Vulnerable to scans (sequential access evicts hot items)

### **LFU Policy**

**Intuition:** Items accessed many times are important.

**Algorithm:**
```
On HIT:    frequency[item] += 1
On MISS:   Evict item with minimum frequency
```

**Trade-offs:**
- Captures popularity
- Good for Zipfian distributions
- Frequency stales over time (can't forget old patterns)
- Higher memory overhead

---

## API Reference

### **Data Generation**

```python
from src.data_generator import TraceGenerator, analyze_trace

gen = TraceGenerator(seed=42)

# Zipfian trace
trace = gen.zipfian_trace(
    num_requests=100000,
    num_items=10000,
    alpha=1.0,  # 1.0=realistic, 2.0=highly skewed
    save_path='data/synthetic_traces/zipfian.pkl'
)

# Working-set trace
trace = gen.working_set_trace(
    num_requests=100000,
    num_items=10000,
    working_set_size=100,
    locality=0.8  # 80% stay in working set
)

# Analyze
analysis = analyze_trace(trace, top_k=10)
print(f"Top-10 items: {analysis['top_k_percentage']:.2f}% of traffic")
```

### **Cache Simulation**

```python
from src.cache_simulator import CacheSimulator
from src.policies import LRUPolicy

policy = LRUPolicy(cache_size=1000)
sim = CacheSimulator(cache_size=1000, policy=policy)

results = sim.run(trace)
print(f"Hit Rate: {results['hit_rate']:.4f}")
print(f"Hits: {results['hits']}, Misses: {results['misses']}")
```

### **Benchmarking**

```python
from src.benchmark import Benchmark
from src.policies import LRUPolicy, LFUPolicy

bench = Benchmark(results_dir='results')

policies = {
    'LRU': LRUPolicy(cache_size=1000),
    'LFU': LFUPolicy(cache_size=1000),
}

results_df = bench.run_comparison(
    cache_sizes=[100, 500, 1000],
    trace_configs=[...],
    policies=policies
)

bench.print_summary()
bench.plot_comparison()
```

---

## Reproducing Exact Results

To replicate our Week 1 results exactly:

```bash
# 1. Clone/download project
cd c:\Users\yeshi\Downloads\RL_Final_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run experiments
python src/main_week1.py

# 4. Check outputs
ls results/
# week1_baseline_results.csv
# week1_baseline_comparison.png
```

**System Requirements:**
- Python 3.8+
- RAM: ≥2GB
- Storage: ~500MB
- Runtime: 2-5 minutes

---

## References

1. **Pinto, L., et al.** (2017). "Robust Adversarial Reinforcement Learning." *arXiv:1703.02702*
2. **Yang, J., et al.** (2023). "GL-Cache: Group-level Learning for Efficient and High-Performance Caching." *USENIX FAST*
3. **Vietri, G., et al.** (2018). "Driving Cache Replacement with ML-based LeCaR." *USENIX HotStorage*
4. **Kim, H., et al.** (2024). "T-CacheNet: Transformer-based Deep RL for NGI Content Caching." *NCCI*

---

## Team

- **Yeshitha Bhuvanesh** (yb2649)
- **Keerthilakshmi Sivakumar** (ks4420)

---

## License & Citation

```bibtex
@article{rarl-cache-2024,
  title={Adversarial Reinforcement Learning for Robust Web Cache Replacement},
  author={Bhuvanesh, Yeshitha and Sivakumar, Keerthilakshmi},
  year={2024}
}
```