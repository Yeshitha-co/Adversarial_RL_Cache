import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import time

from data_generator import TraceGenerator, analyze_trace
from cache_simulator import CacheSimulator
from policies import LRUPolicy, LFUPolicy


class Benchmark:
    """Run and compare cache policies."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_policy(
        self,
        policy_name: str,
        policy: 'CachePolicy',
        trace: np.ndarray,
        cache_size: int,
        verbose: bool = True
    ) -> Dict:
        """
        Run a single policy on a trace.
        
        Args:
            policy_name: Name of policy (for logging)
            policy: CachePolicy instance
            trace: Request trace
            cache_size: Cache size
            verbose: Print progress
            
        Returns:
            dict: Results including hit_rate, time_ms, etc.
        """
        simulator = CacheSimulator(cache_size, policy)
        
        start_time = time.time()
        results = simulator.run(trace)
        elapsed_ms = (time.time() - start_time) * 1000
        
        results['policy'] = policy_name
        results['cache_size'] = cache_size
        results['trace_length'] = len(trace)
        results['time_ms'] = elapsed_ms
        
        if verbose:
            hit_rate_pct = results['hit_rate'] * 100
            print(f"    {policy_name:8s}: {hit_rate_pct:6.2f}% hit rate ({results['hits']:7,} hits)")
        
        return results
    
    def run_comparison(
        self,
        cache_sizes: List[int],
        trace_configs: List[Dict],
        policies: Dict[str, 'CachePolicy']
    ) -> pd.DataFrame:
        """
        Comprehensive comparison across multiple configurations.
        
        Args:
            cache_sizes: List of cache sizes to test (e.g., [100, 500, 1000])
            trace_configs: List of trace generation configs
            policies: Dict mapping policy_name -> policy_instance
            
        Returns:
            pd.DataFrame: Results table
        """
        generator = TraceGenerator(seed=42)
        results_list = []
        
        for config_idx, trace_config in enumerate(trace_configs):
            print(f"\n[*] Trace {config_idx + 1}/{len(trace_configs)}: {trace_config.get('type').upper()}", end="")
            if 'alpha' in trace_config:
                print(f" (α={trace_config['alpha']})")
            else:
                print()
            
            # Generate trace
            if trace_config['type'] == 'zipfian':
                trace = generator.zipfian_trace(
                    num_requests=trace_config['num_requests'],
                    num_items=trace_config['num_items'],
                    alpha=trace_config.get('alpha', 1.0)
                )
            elif trace_config['type'] == 'working_set':
                trace = generator.working_set_trace(
                    num_requests=trace_config['num_requests'],
                    num_items=trace_config['num_items'],
                    working_set_size=trace_config.get('working_set_size', 100),
                    locality=trace_config.get('locality', 0.8)
                )
            else:
                raise ValueError(f"Unknown trace type: {trace_config['type']}")
            
            # Analyze trace
            trace_analysis = analyze_trace(trace)
            print(f"    Requests: {trace_analysis['total_requests']:,} | " +
                  f"Unique items: {trace_analysis['unique_items']:,} | " +
                  f"Top-10: {trace_analysis['top_k_percentage']:.1f}%")
            
            # Run each cache size
            for cache_size in cache_sizes:
                print(f"    Cache size: {cache_size:4d} items")
                
                # Run each policy
                for policy_name, policy in policies.items():
                    result = self.run_policy(
                        policy_name,
                        policy,
                        trace,
                        cache_size,
                        verbose=True
                    )
                    
                    # Add trace config info
                    result.update({
                        'trace_type': trace_config['type'],
                        'num_items': trace_config['num_items'],
                        'trace_length': len(trace),
                    })
                    if 'alpha' in trace_config:
                        result['alpha'] = trace_config['alpha']
                    if 'working_set_size' in trace_config:
                        result['working_set_size'] = trace_config['working_set_size']
                    
                    results_list.append(result)
        
        self.results_df = pd.DataFrame(results_list)
        
        # Save results
        csv_path = self.results_dir / "week1_baseline_results.csv"
        self.results_df.to_csv(csv_path, index=False)
        
        return self.results_df
    
    def plot_comparison(self, metric: str = 'hit_rate'):
        """
        Plot comparison across policies and configurations.
        Separate graphs for each alpha value, showing both LRU and LFU.
        
        Args:
            metric: Metric to plot ('hit_rate', 'time_ms', etc.)
        """
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results to plot. Run run_comparison first.")
            return
        
        df = self.results_df
        
        # Separate Zipfian traces by alpha
        zipfian_df = df[df['trace_type'] == 'zipfian']
        working_set_df = df[df['trace_type'] == 'working_set']
        
        # Get unique alpha values
        alphas = sorted(zipfian_df['alpha'].unique())
        
        # Create subplots: one for each alpha + one for working-set
        num_plots = len(alphas) + 1
        fig, axes = plt.subplots(
            num_plots,
            1,
            figsize=(10, 5 * num_plots)
        )
        
        if num_plots == 1:
            axes = [axes]
        
        # Plot each Zipfian alpha
        for idx, alpha in enumerate(alphas):
            ax = axes[idx]
            subset = zipfian_df[zipfian_df['alpha'] == alpha]
            
            for policy in sorted(subset['policy'].unique()):
                policy_data = subset[subset['policy'] == policy].sort_values('cache_size')
                ax.plot(
                    policy_data['cache_size'],
                    policy_data['hit_rate'],
                    marker='o',
                    label=policy,
                    linewidth=2.5,
                    markersize=8
                )
            
            ax.set_xlabel('Cache Size', fontsize=12, fontweight='bold')
            ax.set_ylabel('Hit Rate', fontsize=12, fontweight='bold')
            ax.set_title(f'Zipfian α={alpha}: Hit Rate vs Cache Size', fontsize=13, fontweight='bold')
            ax.legend(fontsize=11, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        # Plot Working-Set model
        ax = axes[len(alphas)]
        subset = working_set_df
        
        for policy in sorted(subset['policy'].unique()):
            policy_data = subset[subset['policy'] == policy].sort_values('cache_size')
            ax.plot(
                policy_data['cache_size'],
                policy_data['hit_rate'],
                marker='s',
                label=policy,
                linewidth=2.5,
                markersize=8
            )
        
        ax.set_xlabel('Cache Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hit Rate', fontsize=12, fontweight='bold')
        ax.set_title('Working-Set Model (80% Locality): Hit Rate vs Cache Size', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plot_path = self.results_dir / "week1_baseline_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Plot saved to {plot_path}")
        plt.close()
    
    def print_summary(self):
        """Print summary statistics of results."""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("[!] No results to summarize.")
            return
        
        df = self.results_df

        print("Baseline Results Summary")
        
        # Summary by policy
        print("\nBy Policy:")
        policy_summary = df.groupby('policy')['hit_rate'].agg(['mean', 'std', 'min', 'max'])
        for policy, row in policy_summary.iterrows():
            print(f"  {policy:8s}  Mean: {row['mean']:.4f}  Std: {row['std']:.4f}  " +
                  f"Min: {row['min']:.4f}  Max: {row['max']:.4f}")
        
        # Summary by trace type
        print("\nBy Trace Type:")
        trace_summary = df.groupby('trace_type')['hit_rate'].agg(['mean', 'std', 'min', 'max'])
        for trace_type, row in trace_summary.iterrows():
            print(f"  {trace_type:15s}  Mean: {row['mean']:.4f}  Std: {row['std']:.4f}  " +
                  f"Min: {row['min']:.4f}  Max: {row['max']:.4f}")
        
        # Best performance per configuration
        print("\nBest Performers:")
        for trace_type in df['trace_type'].unique():
            subset = df[df['trace_type'] == trace_type]
            best_idx = subset['hit_rate'].idxmax()
            best_row = subset.loc[best_idx]
            print(f"  {trace_type:15s}: {best_row['policy']:4s} " +
                  f"hit_rate={best_row['hit_rate']:.4f}  cache_size={best_row['cache_size']}")
