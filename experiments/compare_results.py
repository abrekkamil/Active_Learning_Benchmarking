#!/usr/bin/env python3
"""
Compare results from multiple active learning experiments.
"""

import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import wandb

def load_all_results(results_dir: str = "results") -> Dict[str, Dict]:
    """
    Load all experiment results from JSON files.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Dictionary of all experiment results
    """
    results_dir = Path(results_dir)
    all_results = {}
    
    # Find all JSON result files
    result_files = list(results_dir.glob("*.json"))
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            # Extract experiment name from filename
            exp_name = file_path.stem
            all_results[exp_name] = result
            
            print(f"Loaded results from: {exp_name}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_results

def compare_cold_start_strategies(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare performance of different cold start strategies.
    
    Args:
        results: Dictionary of experiment results
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for exp_name, result in results.items():
        # Skip if not a cold start experiment
        if "cold_start" not in exp_name.lower():
            continue
        
        # Extract strategy name
        strategy = None
        for s in ["random", "diversity", "simple_diversity", "entropy", 
                  "weak_supervision", "self_supervised", "uncertainty_weak"]:
            if s in exp_name.lower():
                strategy = s
                break
        
        if strategy is None:
            strategy = "unknown"
        
        # Extract performance metrics
        if 'final_ap' in result:
            final_ap = result['final_ap']
        elif 'history' in result and 'val_ap' in result['history']:
            val_ap = result['history']['val_ap']
            final_ap = val_ap[-1] if val_ap else 0
        else:
            final_ap = 0
        
        # Extract training time if available
        training_time = result.get('timing', {}).get('total_time', 0)
        
        # Extract other metrics
        best_ap = result.get('best_ap', 0)
        final_labeled = result.get('final_labeled_count', 0)
        
        comparison_data.append({
            'strategy': strategy,
            'experiment': exp_name,
            'final_ap': final_ap,
            'best_ap': best_ap,
            'training_time_min': training_time / 60 if training_time else 0,
            'final_labeled': final_labeled,
            'ap_per_sample': final_ap / final_labeled if final_labeled > 0 else 0
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Sort by final AP
    df = df.sort_values('final_ap', ascending=False)
    
    return df

def compare_query_strategies(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare performance of different query strategies.
    
    Args:
        results: Dictionary of experiment results
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for exp_name, result in results.items():
        # Skip if not a query strategy experiment
        if "query" not in exp_name.lower():
            continue
        
        # Extract strategy name
        strategy = None
        for s in ["uncertainty", "diversity", "hybrid", "k_center", "feature"]:
            if s in exp_name.lower():
                strategy = s
                break
        
        if strategy is None:
            strategy = "unknown"
        
        # Extract performance metrics
        if 'final_ap' in result:
            final_ap = result['final_ap']
        elif 'history' in result and 'val_ap' in result['history']:
            val_ap = result['history']['val_ap']
            final_ap = val_ap[-1] if val_ap else 0
        else:
            final_ap = 0
        
        # Extract query efficiency if available
        query_efficiency = 0
        if 'history' in result and 'val_ap' in result['history']:
            val_ap = result['history']['val_ap']
            if len(val_ap) > 1:
                ap_gain = val_ap[-1] - val_ap[0]
                # Assuming 5 queries per cycle, 5 cycles = 25 samples
                query_efficiency = ap_gain / 25 if 25 > 0 else 0
        
        comparison_data.append({
            'strategy': strategy,
            'experiment': exp_name,
            'final_ap': final_ap,
            'best_ap': result.get('best_ap', 0),
            'query_efficiency': query_efficiency,
            'final_labeled': result.get('final_labeled_count', 0)
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Sort by final AP
    df = df.sort_values('final_ap', ascending=False)
    
    return df

def plot_strategy_comparison(
    df: pd.DataFrame,
    metric: str = 'final_ap',
    title: str = 'Strategy Comparison',
    save_path: Optional[str] = None
):
    """
    Create bar plot comparing strategies.
    
    Args:
        df: DataFrame with strategy comparison data
        metric: Metric to compare
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    # Group by strategy and calculate mean
    grouped = df.groupby('strategy')[metric].agg(['mean', 'std', 'count'])
    grouped = grouped.sort_values('mean', ascending=False)
    
    # Create bar plot
    bars = plt.bar(range(len(grouped)), grouped['mean'], 
                   yerr=grouped['std'], capsize=5, 
                   color=sns.color_palette("viridis", len(grouped)))
    
    plt.xlabel('Strategy')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.xticks(range(len(grouped)), grouped.index, rotation=45, ha='right')
    
    # Add value labels
    for i, (idx, row) in enumerate(grouped.iterrows()):
        plt.text(i, row['mean'] + 0.01, f"{row['mean']:.3f}", 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_learning_curve_comparison(
    results: Dict[str, Dict],
    strategies_to_plot: List[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot learning curves for multiple strategies.
    
    Args:
        results: Dictionary of experiment results
        strategies_to_plot: List of strategies to plot
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    
    for i, (exp_name, result) in enumerate(results.items()):
        # Check if we should plot this strategy
        should_plot = False
        strategy_name = exp_name
        
        if strategies_to_plot:
            for strategy in strategies_to_plot:
                if strategy in exp_name.lower():
                    should_plot = True
                    strategy_name = strategy
                    break
        else:
            should_plot = True
        
        if not should_plot:
            continue
        
        # Extract learning curve
        if 'history' in result and 'val_ap' in result['history']:
            val_ap = result['history']['val_ap']
            epochs = range(1, len(val_ap) + 1)
            
            plt.plot(epochs, val_ap, '-o', 
                    label=strategy_name, 
                    color=colors[i % len(colors)],
                    markersize=4,
                    linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('AP@[IoU=0.50:0.95]')
    plt.title('Learning Curves Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {save_path}")
    
    plt.show()

def create_comparison_table(df: pd.DataFrame, metrics: List[str]) -> str:
    """
    Create a formatted comparison table.
    
    Args:
        df: DataFrame with comparison data
        metrics: List of metrics to include
        
    Returns:
        Formatted table string
    """
    # Group by strategy
    grouped = df.groupby('strategy')[metrics].agg(['mean', 'std'])
    
    # Format table
    table_data = []
    for strategy in grouped.index:
        row = [strategy]
        for metric in metrics:
            mean_val = grouped.loc[strategy, (metric, 'mean')]
            std_val = grouped.loc[strategy, (metric, 'std')]
            row.append(f"{mean_val:.4f} ± {std_val:.4f}")
        table_data.append(row)
    
    # Create headers
    headers = ['Strategy'] + [m.replace('_', ' ').title() for m in metrics]
    
    # Format as markdown
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "|" + "|".join(["---"] * len(headers)) + "|\n"
    
    for row in table_data:
        markdown += "| " + " | ".join(row) + " |\n"
    
    return markdown

def analyze_query_efficiency(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Analyze query efficiency for different strategies.
    
    Args:
        results: Dictionary of experiment results
        
    Returns:
        DataFrame with query efficiency analysis
    """
    efficiency_data = []
    
    for exp_name, result in results.items():
        # Extract strategy name
        strategy = exp_name.split('_')[0] if '_' in exp_name else exp_name
        
        # Calculate query efficiency
        if 'history' in result and 'val_ap' in result['history']:
            val_ap = result['history']['val_ap']
            if len(val_ap) > 1:
                initial_ap = val_ap[0]
                final_ap = val_ap[-1]
                ap_gain = final_ap - initial_ap
                
                # Get number of queries (assuming from experiment name or config)
                # This is a simplified calculation
                queries_made = result.get('final_labeled_count', 0) - result.get('initial_labeled_count', 0)
                queries_made = max(queries_made, 1)  # Avoid division by zero
                
                query_efficiency = ap_gain / queries_made
                
                efficiency_data.append({
                    'strategy': strategy,
                    'experiment': exp_name,
                    'initial_ap': initial_ap,
                    'final_ap': final_ap,
                    'ap_gain': ap_gain,
                    'queries_made': queries_made,
                    'query_efficiency': query_efficiency,
                    'efficiency_per_query': query_efficiency
                })
    
    df = pd.DataFrame(efficiency_data)
    
    if len(df) > 0:
        df = df.sort_values('query_efficiency', ascending=False)
    
    return df

def generate_report(results: Dict[str, Dict], output_dir: str = "results"):
    """
    Generate a comprehensive comparison report.
    
    Args:
        results: Dictionary of experiment results
        output_dir: Directory to save report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("# Active Learning Benchmark Report")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Compare cold start strategies
    report_lines.append("## Cold Start Strategy Comparison")
    cold_start_df = compare_cold_start_strategies(results)
    
    if len(cold_start_df) > 0:
        report_lines.append("### Performance Metrics")
        metrics = ['final_ap', 'best_ap', 'training_time_min', 'ap_per_sample']
        report_lines.append(create_comparison_table(cold_start_df, metrics))
        report_lines.append("")
        
        # Save cold start comparison
        cold_start_df.to_csv(output_dir / "cold_start_comparison.csv", index=False)
        
        # Plot cold start comparison
        plot_strategy_comparison(
            cold_start_df,
            metric='final_ap',
            title='Cold Start Strategy Comparison',
            save_path=str(output_dir / "cold_start_comparison.png")
        )
    
    # Compare query strategies
    report_lines.append("## Query Strategy Comparison")
    query_df = compare_query_strategies(results)
    
    if len(query_df) > 0:
        report_lines.append("### Performance Metrics")
        metrics = ['final_ap', 'query_efficiency', 'best_ap']
        report_lines.append(create_comparison_table(query_df, metrics))
        report_lines.append("")
        
        # Save query strategy comparison
        query_df.to_csv(output_dir / "query_strategy_comparison.csv", index=False)
        
        # Plot query strategy comparison
        plot_strategy_comparison(
            query_df,
            metric='final_ap',
            title='Query Strategy Comparison',
            save_path=str(output_dir / "query_strategy_comparison.png")
        )
    
    # Query efficiency analysis
    report_lines.append("## Query Efficiency Analysis")
    efficiency_df = analyze_query_efficiency(results)
    
    if len(efficiency_df) > 0:
        report_lines.append("### Efficiency Metrics")
        efficiency_metrics = ['ap_gain', 'queries_made', 'query_efficiency']
        report_lines.append(create_comparison_table(efficiency_df, efficiency_metrics))
        report_lines.append("")
        
        # Save efficiency analysis
        efficiency_df.to_csv(output_dir / "query_efficiency.csv", index=False)
    
    # Learning curve comparison
    report_lines.append("## Learning Curve Comparison")
    report_lines.append("![Learning Curves](learning_curves_comparison.png)")
    report_lines.append("")
    
    # Plot learning curves
    plot_learning_curve_comparison(
        results,
        save_path=str(output_dir / "learning_curves_comparison.png")
    )
    
    # Best performing strategy
    report_lines.append("## Best Performing Strategies")
    report_lines.append("")
    
    if len(cold_start_df) > 0:
        best_cold_start = cold_start_df.iloc[0]
        report_lines.append(f"### Best Cold Start Strategy: {best_cold_start['strategy']}")
        report_lines.append(f"- Final AP: {best_cold_start['final_ap']:.4f}")
        report_lines.append(f"- Training Time: {best_cold_start['training_time_min']:.1f} minutes")
        report_lines.append(f"- AP per Sample: {best_cold_start['ap_per_sample']:.4f}")
        report_lines.append("")
    
    if len(query_df) > 0:
        best_query = query_df.iloc[0]
        report_lines.append(f"### Best Query Strategy: {best_query['strategy']}")
        report_lines.append(f"- Final AP: {best_query['final_ap']:.4f}")
        report_lines.append(f"- Query Efficiency: {best_query['query_efficiency']:.4f}")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")
    report_lines.append("Based on the benchmark results, we recommend:")
    report_lines.append("")
    
    if len(cold_start_df) > 0 and len(query_df) > 0:
        best_combination = f"{best_cold_start['strategy']} + {best_query['strategy']}"
        report_lines.append(f"1. **Strategy Combination**: {best_combination}")
        report_lines.append("2. **Cold Start**: Use diversity-based methods for better initial performance")
        report_lines.append("3. **Active Learning**: Hybrid strategies balance uncertainty and diversity")
        report_lines.append("4. **Budget Allocation**: Focus on high-uncertainty samples first")
    
    # Save report
    report_path = output_dir / "benchmark_report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"Report saved to {report_path}")
    
    # Also save as HTML for easier viewing
    import markdown
    html_content = markdown.markdown("\n".join(report_lines))
    
    html_path = output_dir / "benchmark_report.html"
    with open(html_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Active Learning Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
                h3 {{ color: #2ecc71; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """)
    
    print(f"HTML report saved to {html_path}")
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Compare active learning experiment results")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory containing result files")
    parser.add_argument("--output-dir", type=str, default="results/comparison",
                       help="Directory to save comparison results")
    parser.add_argument("--wandb-project", type=str,
                       help="WandB project to fetch results from")
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate comprehensive report")
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    results = load_all_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        
        # Try to fetch from WandB
        if args.wandb_project:
            print(f"Fetching results from WandB project: {args.wandb_project}")
            results = fetch_wandb_results(args.wandb_project)
    
    if not results:
        print("No results to compare. Exiting.")
        return
    
    print(f"Loaded {len(results)} experiments")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report if requested
    if args.generate_report:
        print("Generating comprehensive report...")
        report_path = generate_report(results, args.output_dir)
        print(f"Report generated: {report_path}")
    else:
        # Just create basic comparisons
        print("Creating basic comparisons...")
        
        # Cold start comparison
        cold_start_df = compare_cold_start_strategies(results)
        if len(cold_start_df) > 0:
            cold_start_df.to_csv(output_dir / "cold_start_comparison.csv", index=False)
            plot_strategy_comparison(
                cold_start_df,
                save_path=str(output_dir / "cold_start_comparison.png")
            )
            print(f"Cold start comparison saved to {output_dir}")
        
        # Query strategy comparison
        query_df = compare_query_strategies(results)
        if len(query_df) > 0:
            query_df.to_csv(output_dir / "query_strategy_comparison.csv", index=False)
            plot_strategy_comparison(
                query_df,
                save_path=str(output_dir / "query_strategy_comparison.png")
            )
            print(f"Query strategy comparison saved to {output_dir}")
        
        # Learning curve comparison
        plot_learning_curve_comparison(
            results,
            save_path=str(output_dir / "learning_curves_comparison.png")
        )
        print(f"Learning curve comparison saved to {output_dir}")
    
    print("\nComparison completed!")

def fetch_wandb_results(project_name: str) -> Dict[str, Dict]:
    """
    Fetch experiment results from WandB.
    
    Args:
        project_name: WandB project name
        
    Returns:
        Dictionary of experiment results
    """
    import wandb
    
    api = wandb.Api()
    
    # Get all runs from the project
    runs = api.runs(project_name)
    
    results = {}
    
    for run in runs:
        try:
            # Get run configuration
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            
            # Get run summary
            summary = run.summary._json_dict
            
            # Get run history
            history = run.scan_history()
            history_data = list(history)
            
            # Combine into result dictionary
            result = {
                'config': config,
                'summary': summary,
                'history': history_data,
                'name': run.name,
                'id': run.id,
                'url': run.url
            }
            
            results[run.name] = result
            
            print(f"Fetched results for: {run.name}")
            
        except Exception as e:
            print(f"Error fetching run {run.name}: {e}")
    
    return results

if __name__ == "__main__":
    main()