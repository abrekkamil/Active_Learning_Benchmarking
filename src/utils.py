"""
Utility functions for active learning experiments.
"""

import os
import json
import logging
import time
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import wandb

def setup_logging(experiment_name: str, log_dir: str = "results/logs") -> logging.Logger:
    """
    Set up logging for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    log_file = Path(log_dir) / f"{experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_checkpoint(
    model,
    cycle: int,
    epoch: int,
    ap: float,
    config,
    is_best: bool = False,
    additional_info: Optional[Dict] = None
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        cycle: Current active learning cycle
        epoch: Current epoch
        ap: Current average precision
        config: Configuration object
        is_best: Whether this is the best model so far
        additional_info: Additional information to save
        
    Returns:
        Path to saved checkpoint
    """
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir) / config.dataset_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename
    if is_best:
        filename = f"{config.dataset_name}_best.pth"
    else:
        filename = f"{config.dataset_name}_cycle{cycle}_epoch{epoch}.pth"
    
    checkpoint_path = checkpoint_dir / filename
    
    # Prepare checkpoint data
    checkpoint = {
        'cycle': cycle,
        'epoch': epoch,
        'ap': ap,
        'model_state_dict': model.model.state_dict() if hasattr(model, 'model') else model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict() if hasattr(model, 'optimizer') else None,
        'config': config.__dict__ if hasattr(config, '__dict__') else config,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    if not is_best:
        latest_path = checkpoint_dir / f"{config.dataset_name}_latest.pth"
        torch.save(checkpoint, latest_path)
    
    logging.info(f"Checkpoint saved to {checkpoint_path}")
    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model,
    device: torch.device,
    load_optimizer: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: PyTorch device
        load_optimizer: Whether to load optimizer state
        
    Returns:
        Checkpoint data
    """
    if not os.path.exists(checkpoint_path):
        # Try to find latest checkpoint
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
        
        if not checkpoints:
            logging.warning(f"No checkpoints found in {checkpoint_dir}")
            return {}
        
        # Sort by modification time
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        checkpoint_path = checkpoints[0]
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        
        # Handle loading into wrapped model
        if hasattr(model, 'model'):
            model.model.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
    
    # Load optimizer state
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logging.info(f"Loaded checkpoint: cycle={checkpoint.get('cycle', 'N/A')}, "
                 f"epoch={checkpoint.get('epoch', 'N/A')}, "
                 f"AP={checkpoint.get('ap', 'N/A'):.4f}")
    
    return checkpoint


def calculate_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_thresholds: List[float] = [0.5, 0.75]
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for object detection.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_thresholds: IoU thresholds for AP calculation
        
    Returns:
        Dictionary of metrics
    """
    # Simplified metric calculation
    # In practice, you'd use COCO evaluation metrics
    metrics = {}
    
    # Calculate basic metrics
    total_predictions = 0
    total_ground_truth = 0
    true_positives = 0
    
    for pred, gt in zip(predictions, ground_truth):
        total_predictions += len(pred.get('boxes', []))
        total_ground_truth += len(gt.get('boxes', []))
        
        # Simple matching (for illustration)
        # In practice, use proper IoU-based matching
        if len(pred.get('boxes', [])) > 0 and len(gt.get('boxes', [])) > 0:
            true_positives += min(len(pred['boxes']), len(gt['boxes']))
    
    # Calculate precision and recall
    if total_predictions > 0:
        precision = true_positives / total_predictions
    else:
        precision = 0.0
    
    if total_ground_truth > 0:
        recall = true_positives / total_ground_truth
    else:
        recall = 0.0
    
    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_predictions': total_predictions,
        'num_ground_truth': total_ground_truth,
        'true_positives': true_positives
    })
    
    return metrics


def plot_learning_curves(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot learning curves from training history.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the figure (optional)
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: AP over epochs
    if 'val_ap' in history:
        ax = axes[0, 0]
        epochs = range(1, len(history['val_ap']) + 1)
        ax.plot(epochs, history['val_ap'], 'b-', label='Validation AP')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AP')
        ax.set_title('Average Precision over Time')
        ax.grid(True)
        ax.legend()
    
    # Plot 2: Labeled samples vs AP
    if 'labeled_count' in history and 'val_ap' in history:
        ax = axes[0, 1]
        
        # Group by labeled count
        grouped = defaultdict(list)
        for count, ap in zip(history['labeled_count'], history['val_ap']):
            grouped[count].append(ap)
        
        # Calculate average AP for each labeled count
        avg_counts = []
        avg_aps = []
        for count in sorted(grouped.keys()):
            avg_counts.append(count)
            avg_aps.append(np.mean(grouped[count]))
        
        ax.plot(avg_counts, avg_aps, 'r-o', label='Average AP')
        ax.set_xlabel('Number of Labeled Samples')
        ax.set_ylabel('Average AP')
        ax.set_title('Performance vs Labeled Data')
        ax.grid(True)
        ax.legend()
    
    # Plot 3: Loss curves (if available)
    if 'train_loss' in history:
        ax = axes[1, 0]
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'g-', label='Training Loss')
        
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.grid(True)
        ax.legend()
    
    # Plot 4: Query efficiency
    if 'query_efficiency' in history:
        ax = axes[1, 1]
        cycles = range(1, len(history['query_efficiency']) + 1)
        ax.plot(cycles, history['query_efficiency'], 'm-s', label='Query Efficiency')
        ax.set_xlabel('AL Cycle')
        ax.set_ylabel('AP Gain per Sample')
        ax.set_title('Query Efficiency over AL Cycles')
        ax.grid(True)
        ax.legend()
    else:
        # Plot labeled count over cycles
        if 'labeled_count' in history:
            ax = axes[1, 1]
            unique_counts = sorted(set(history['labeled_count']))
            cycles_at_counts = [history['labeled_count'].index(c) for c in unique_counts]
            
            ax.plot(cycles_at_counts, unique_counts, 'c-^', label='Labeled Samples')
            ax.set_xlabel('AL Cycle')
            ax.set_ylabel('Number of Labeled Samples')
            ax.set_title('Labeled Data Growth')
            ax.grid(True)
            ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Figure saved to {save_path}")
    
    # Show figure
    if show:
        plt.show()
    
    return fig


def compare_strategies(
    strategy_results: Dict[str, Dict],
    metric: str = 'AP@[IoU=0.50:0.95]',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare performance of different strategies.
    
    Args:
        strategy_results: Dictionary mapping strategy names to results
        metric: Metric to compare
        save_path: Path to save the figure
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    strategies = list(strategy_results.keys())
    final_scores = []
    learning_curves = {}
    
    for strategy, results in strategy_results.items():
        if 'val_ap' in results:
            final_scores.append(results['val_ap'][-1] if results['val_ap'] else 0)
            learning_curves[strategy] = results['val_ap']
        elif 'final_performance' in results:
            # Handle nested structure
            perf = results['final_performance']
            if isinstance(perf, (list, tuple)) and len(perf) > 0:
                if isinstance(perf[-1], dict) and metric in perf[-1]:
                    final_scores.append(perf[-1][metric])
                elif isinstance(perf[-1], (int, float)):
                    final_scores.append(perf[-1])
            else:
                final_scores.append(0)
    
    # Plot 1: Bar chart of final scores
    ax = axes[0]
    bars = ax.bar(range(len(strategies)), final_scores, color='skyblue')
    ax.set_xlabel('Strategy')
    ax.set_ylabel(f'Final {metric}')
    ax.set_title(f'Strategy Comparison - Final {metric}')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, final_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Plot 2: Learning curves
    ax = axes[1]
    for strategy, curve in learning_curves.items():
        if curve:  # Only plot if we have data
            epochs = range(1, len(curve) + 1)
            ax.plot(epochs, curve, '-o', label=strategy, markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.set_title('Learning Curves Comparison')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Comparison figure saved to {save_path}")
    
    # Show figure
    if show:
        plt.show()
    
    return fig


def log_to_wandb(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True
):
    """
    Log metrics to Weights & Biases.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Step number (e.g., epoch, cycle)
        commit: Whether to commit the log
    """
    if wandb.run is not None:
        if step is not None:
            wandb.log(metrics, step=step, commit=commit)
        else:
            wandb.log(metrics, commit=commit)


def save_results_json(
    results: Dict[str, Any],
    experiment_name: str,
    results_dir: str = "results"
) -> str:
    """
    Save experiment results to JSON file.
    
    Args:
        results: Results dictionary
        experiment_name: Name of the experiment
        results_dir: Directory to save results
        
    Returns:
        Path to saved file
    """
    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Create filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = Path(results_dir) / filename
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Results saved to {filepath}")
    return str(filepath)


def load_results_json(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded results dictionary
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def compute_query_efficiency(
    history: Dict[str, List],
    query_size: int
) -> List[float]:
    """
    Compute query efficiency (AP gain per queried sample).
    
    Args:
        history: Training history
        query_size: Number of samples queried per cycle
        
    Returns:
        List of query efficiency values
    """
    if 'val_ap' not in history or len(history['val_ap']) < 2:
        return []
    
    efficiencies = []
    ap_values = history['val_ap']
    
    # Group by AL cycle (assuming each cycle adds query_size samples)
    for i in range(1, len(ap_values)):
        ap_gain = ap_values[i] - ap_values[i-1]
        efficiency = ap_gain / query_size if query_size > 0 else 0
        efficiencies.append(efficiency)
    
    return efficiencies


def create_experiment_summary(
    config: Any,
    results: Dict[str, Any],
    timing: Optional[Dict[str, float]] = None
) -> str:
    """
    Create a text summary of experiment results.
    
    Args:
        config: Experiment configuration
        results: Experiment results
        timing: Timing information
        
    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("=" * 60)
    summary.append("EXPERIMENT SUMMARY")
    summary.append("=" * 60)
    summary.append("")
    
    # Configuration
    summary.append("CONFIGURATION:")
    summary.append("-" * 40)
    
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config
    
    for key, value in config_dict.items():
        if not key.startswith('_'):
            summary.append(f"  {key}: {value}")
    
    # Results
    summary.append("")
    summary.append("RESULTS:")
    summary.append("-" * 40)
    
    if 'final_ap' in results:
        summary.append(f"  Final AP: {results['final_ap']:.4f}")
    
    if 'best_ap' in results:
        summary.append(f"  Best AP: {results['best_ap']:.4f}")
    
    if 'final_labeled_count' in results:
        summary.append(f"  Final labeled samples: {results['final_labeled_count']}")
    
    # Timing
    if timing:
        summary.append("")
        summary.append("TIMING:")
        summary.append("-" * 40)
        
        for key, value in timing.items():
            summary.append(f"  {key}: {value:.2f} seconds")
    
    summary.append("")
    summary.append("=" * 60)
    
    return "\n".join(summary)


def setup_experiment_directory(
    experiment_name: str,
    base_dir: str = "experiments"
) -> Path:
    """
    Create directory structure for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for experiments
        
    Returns:
        Path to experiment directory
    """
    exp_dir = Path(base_dir) / experiment_name
    subdirs = [
        "configs",
        "logs",
        "checkpoints",
        "results",
        "figures"
    ]
    
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return exp_dir