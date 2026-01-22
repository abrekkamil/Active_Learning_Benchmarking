#!/usr/bin/env python3
"""
Benchmark multiple active learning strategies.
"""

import itertools
from pathlib import Path
import json
import wandb
import time

from experiments.run_experiment import main as run_experiment
from config.config import ActiveLearningConfig
from src.active_learning import ActiveLearningSystem

def benchmark_cold_start_strategies():
    """Benchmark different cold start strategies."""
    
    cold_start_strategies = [
        'random',
        'simple_diversity',
        'diversity',
        'entropy_based_uncertainty',
        'weak_supervision',
        'self_supervised'
    ]
    
    results = {}
    timing = {}
    
    for strategy in cold_start_strategies:
        print(f"\n{'='*60}")
        print(f"Testing cold start strategy: {strategy}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Update config
        config = ActiveLearningConfig()
        config.cold_start_strategy = strategy
        config.dataset_name = f"cold_start_{strategy}"
        
        # Initialize WandB
        wandb.init(
            project="AL_benchmark_cold_start",
            name=strategy,
            config=config.to_dict()
        )
        
        # Run experiment
        al_system = ActiveLearningSystem(config)
        metrics = al_system.run()
        
        # Record results
        results[strategy] = {
            'final_ap': al_system.best_ap,
            'labeled_counts': al_system.history['labeled_count'],
            'val_aps': al_system.history['val_ap']
        }
        
        # Record timing
        end_time = time.time()
        timing[strategy] = end_time - start_time
        
        # Save individual results
        al_system.save_results()
        al_system.plot_results()
        
        wandb.finish()
    
    # Save benchmark results
    save_benchmark_results(results, timing, "cold_start_benchmark.json")
    
    return results, timing

def benchmark_query_strategies():
    """Benchmark different query strategies."""
    
    query_strategies = [
        'uncertainty',
        'diversity',
        'hybrid',
        'k_center',
        'feature'
    ]
    
    results = {}
    
    for strategy in query_strategies:
        print(f"\n{'='*60}")
        print(f"Testing query strategy: {strategy}")
        print(f"{'='*60}")
        
        # Update config
        config = ActiveLearningConfig()
        config.query_strategy = strategy
        config.dataset_name = f"query_{strategy}"
        
        # Run experiment
        wandb.init(
            project="AL_benchmark_query",
            name=strategy,
            config=config.to_dict()
        )
        
        al_system = ActiveLearningSystem(config)
        metrics = al_system.run()
        
        results[strategy] = {
            'final_ap': al_system.best_ap,
            'labeled_counts': al_system.history['labeled_count'],
            'val_aps': al_system.history['val_ap']
        }
        
        al_system.save_results()
        al_system.plot_results()
        
        wandb.finish()
    
    save_benchmark_results(results, {}, "query_strategy_benchmark.json")
    
    return results

def save_benchmark_results(results, timing, filename):
    """Save benchmark results to JSON file."""
    output = {
        'results': results,
        'timing': timing
    }
    
    with open(Path("results") / filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Benchmark results saved to results/{filename}")

if __name__ == "__main__":
    print("Benchmarking cold start strategies...")
    cold_start_results, cold_start_timing = benchmark_cold_start_strategies()
    
    print("\n\nBenchmarking query strategies...")
    query_results = benchmark_query_strategies()
    
    print("\nBenchmark completed!")