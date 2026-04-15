"""
WEEK 1: Dataset Exploration & Classical Baselines
Main entry point for Week 1 experiments.

This script:
1. Generates synthetic Zipfian and working-set traces
2. Implements LRU and LFU baselines
3. Runs comprehensive benchmarks
4. Generates comparison plots and results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_generator import TraceGenerator, analyze_trace
from cache_simulator import CacheSimulator
from policies import LRUPolicy, LFUPolicy
from benchmark import Benchmark
from utils import setup_paths


def main():
    """Main Week 1 execution."""
    # Setup directories
    paths = setup_paths()
    
    # STAGE 1: Generate Traces
    print("-"*70)
    print("Stage 1: Generating Synthetic Request Traces")
    print("-"*70 + "\n")
    
    generator = TraceGenerator(seed=42)
    
    # Trace configurations to test
    trace_configs = [
        # Zipfian with different skewness levels
        {
            'name': 'Zipfian_Alpha_1.0',
            'type': 'zipfian',
            'num_requests': 100000,
            'num_items': 10000,
            'alpha': 1.0,
        },
        {
            'name': 'Zipfian_Alpha_1.5',
            'type': 'zipfian',
            'num_requests': 100000,
            'num_items': 10000,
            'alpha': 1.5,
        },
        {
            'name': 'Zipfian_Alpha_2.0',
            'type': 'zipfian',
            'num_requests': 100000,
            'num_items': 10000,
            'alpha': 2.0,
        },
        # Working set model (temporal locality)
        {
            'name': 'WorkingSet_80pct',
            'type': 'working_set',
            'num_requests': 100000,
            'num_items': 10000,
            'working_set_size': 100,
            'locality': 0.8,
        },
    ]
    
    traces = {}
    for idx, config in enumerate(trace_configs, 1):
        print(f"Trace {idx}/4: {config['name']}")
        
        if config['type'] == 'zipfian':
            trace = generator.zipfian_trace(
                num_requests=config['num_requests'],
                num_items=config['num_items'],
                alpha=config['alpha'],
                save_path=str(paths['data_synthetic'] / f"{config['name']}.pkl")
            )
        elif config['type'] == 'working_set':
            trace = generator.working_set_trace(
                num_requests=config['num_requests'],
                num_items=config['num_items'],
                working_set_size=config['working_set_size'],
                locality=config['locality'],
                save_path=str(paths['data_synthetic'] / f"{config['name']}.pkl")
            )
        
        traces[config['name']] = trace
        
        # Analyze
        analysis = analyze_trace(trace, top_k=10)
        print(f"  Total requests: {analysis['total_requests']:,}")
        print(f"  Unique items accessed: {analysis['unique_items']:,}")
        print(f"  Top-10 concentration: {analysis['top_k_percentage']:.1f}% of traffic")
        print(f"  Coverage: {analysis['coverage']:.2%}\n")
    
    # STAGE 2: Run Benchmarks
    print("-"*70)
    print("Stage 2: Running Cache Policy Benchmarks")
    print("-"*70)
    print("Testing LRU and LFU on all traces\n")
    
    benchmark = Benchmark(results_dir=str(paths['results']))
    
    # Cache sizes to test
    cache_sizes = [100, 500, 1000, 2000]
    
    # Prepare trace configs for benchmark
    benchmark_configs = [
        {
            'type': 'zipfian',
            'num_requests': 100000,
            'num_items': 10000,
            'alpha': 1.0,
        },
        {
            'type': 'zipfian',
            'num_requests': 100000,
            'num_items': 10000,
            'alpha': 1.5,
        },
        {
            'type': 'zipfian',
            'num_requests': 100000,
            'num_items': 10000,
            'alpha': 2.0,
        },
        {
            'type': 'working_set',
            'num_requests': 100000,
            'num_items': 10000,
            'working_set_size': 100,
            'locality': 0.8,
        },
    ]
    
    # Policies to compare
    policies = {
        'LRU': LRUPolicy(cache_size=2000),
        'LFU': LFUPolicy(cache_size=2000),
    }
    
    # Run comprehensive benchmark
    results_df = benchmark.run_comparison(
        cache_sizes=cache_sizes,
        trace_configs=benchmark_configs,
        policies=policies
    )

    # STAGE 3: Analysis & Visualization
    print("-"*70)
    print("Stage 3: Analyzing Results & Creating Visualizations")
    print("-"*70 + "\n")
    
    benchmark.print_summary()
    benchmark.plot_comparison(metric='hit_rate')

    print(f"\nResults saved to: {paths['results']}")
    print(f"CSV data: {paths['results'] / 'week1_baseline_results.csv'}")
    print(f"Plots: {paths['results'] / 'week1_baseline_comparison.png'}\n")


if __name__ == '__main__':
    main()
