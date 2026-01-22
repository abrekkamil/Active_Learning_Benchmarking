#!/usr/bin/env python3
"""
Run a single active learning experiment with specified configuration.
"""

import argparse
import yaml
import wandb
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.active_learning import ActiveLearningSystem
from config.config import ActiveLearningConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Run active learning experiment")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--cold-start", type=str,
                       help="Override cold start strategy")
    parser.add_argument("--query-strategy", type=str,
                       help="Override query strategy")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable WandB logging")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = ActiveLearningConfig.from_yaml(args.config)
    
    # Override config if specified
    if args.cold_start:
        config.cold_start_strategy = args.cold_start
    if args.query_strategy:
        config.query_strategy = args.query_strategy
    if args.no_wandb:
        config.use_wandb = False
    
    # Initialize WandB
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=f"{config.cold_start_strategy}_{config.query_strategy}",
            config=config.to_dict()
        )
    
    # Create results directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    print(f"Starting experiment with configuration:")
    print(f"  Cold Start: {config.cold_start_strategy}")
    print(f"  Query Strategy: {config.query_strategy}")
    print(f"  Initial labeled: {config.initial_labeled}")
    print(f"  Query size: {config.query_size}")
    print(f"  AL cycles: {config.al_cycles}")
    
    al_system = ActiveLearningSystem(config)
    results = al_system.run()
    
    # Save results
    al_system.save_results()
    al_system.plot_results()
    
    # Cleanup
    if config.use_wandb:
        wandb.finish()
    
    print(f"Experiment completed!")

if __name__ == "__main__":
    main()