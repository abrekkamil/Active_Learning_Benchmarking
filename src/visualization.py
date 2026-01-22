"""
Visualization utilities for active learning experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.cm as cm

class ActiveLearningVisualizer:
    """
    Visualization class for active learning experiments.
    Provides various plotting functions for analyzing results.
    """
        
    def __init__(self, style: str = "seaborn-v0_8", color_palette: str = "husl"):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style (e.g., "seaborn-v0_8", "seaborn-darkgrid", "ggplot")
            color_palette: Color palette for plots
        """
        try:
            # First check if seaborn-v0_8 is available
            if style == "seaborn-v0_8" and style not in plt.style.available:
                # Try alternative seaborn styles
                if "seaborn-whitegrid" in plt.style.available:
                    style = "seaborn-whitegrid"
                elif "seaborn-darkgrid" in plt.style.available:
                    style = "seaborn-darkgrid"
                elif "seaborn" in plt.style.available:
                    style = "seaborn"
                else:
                    # Fall back to a default matplotlib style
                    style = "default"
                    
            plt.style.use(style)
        except Exception as e:
            print(f"Warning: Could not set style '{style}': {e}")
            plt.style.use("default")  # Fall back to default style
        
        try:
            sns.set_palette(color_palette)
        except Exception as e:
            print(f"Warning: Could not set color palette '{color_palette}': {e}")
            # Set a default palette
            sns.set_palette("viridis")
        
        self.color_palette = color_palette
        
    def plot_learning_curves(
        self,
        history: Dict[str, List],
        title: str = "Learning Curves",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        show: bool = True
    ) -> plt.Figure:
        """
        Plot learning curves from training history.
        
        Args:
            history: Dictionary with training history
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: AP over epochs
        if 'val_ap' in history:
            ax = axes[0, 0]
            epochs = range(1, len(history['val_ap']) + 1)
            ax.plot(epochs, history['val_ap'], 'b-', linewidth=2, marker='o', markersize=4, label='Validation AP')
            
            # Add smoothing if enough points
            if len(history['val_ap']) > 5:
                smoothed = self._smooth_curve(history['val_ap'])
                ax.plot(epochs, smoothed, 'r--', linewidth=1, alpha=0.7, label='Smoothed')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('AP')
            ax.set_title('Average Precision over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot 2: Training loss
        if 'train_loss' in history:
            ax = axes[0, 1]
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], 'g-', linewidth=2, label='Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add exponential moving average
            if len(history['train_loss']) > 10:
                ema = pd.Series(history['train_loss']).ewm(span=5).mean()
                ax.plot(epochs, ema, 'r--', linewidth=1, alpha=0.7, label='EMA')
        
        # Plot 3: Labeled samples vs AP
        if 'labeled_count' in history and 'val_ap' in history:
            ax = axes[1, 0]
            
            # Scatter plot with regression line
            scatter = ax.scatter(history['labeled_count'], history['val_ap'], 
                                c=range(len(history['val_ap'])), cmap='viridis',
                                s=50, alpha=0.7, edgecolors='k')
            
            # Add regression line
            if len(history['val_ap']) > 2:
                z = np.polyfit(history['labeled_count'], history['val_ap'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(history['labeled_count']), max(history['labeled_count']), 100)
                ax.plot(x_range, p(x_range), "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
            
            ax.set_xlabel('Number of Labeled Samples')
            ax.set_ylabel('AP')
            ax.set_title('Performance vs Labeled Data')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Epoch')
        
        # Plot 4: Query efficiency
        if 'val_ap' in history and len(history['val_ap']) > 1:
            ax = axes[1, 1]
            
            # Calculate query efficiency (AP gain per sample)
            ap_gains = np.diff(history['val_ap'])
            cycles = range(1, len(ap_gains) + 1)
            
            bars = ax.bar(cycles, ap_gains, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('AL Cycle')
            ax.set_ylabel('AP Gain')
            ax.set_title('Query Efficiency per AL Cycle')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_strategy_comparison(
        self,
        strategy_results: Dict[str, Dict],
        metric: str = 'final_ap',
        title: str = 'Strategy Comparison',
        plot_type: str = 'bar',  # 'bar', 'radar', or 'scatter'
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
        show: bool = True
    ) -> plt.Figure:
        """
        Compare performance of different strategies.
        
        Args:
            strategy_results: Dictionary mapping strategy names to results
            metric: Metric to compare
            title: Plot title
            plot_type: Type of plot
            save_path: Path to save figure
            figsize: Figure size
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure
        """
        if plot_type == 'radar':
            return self._plot_radar_comparison(strategy_results, metric, title, save_path, figsize, show)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Prepare data
        strategies = list(strategy_results.keys())
        metrics_data = []
        
        for strategy, results in strategy_results.items():
            # Extract the metric value
            if metric in results:
                value = results[metric]
            elif 'history' in results and 'val_ap' in results['history']:
                val_ap = results['history']['val_ap']
                value = val_ap[-1] if val_ap else 0
            else:
                value = 0
            
            metrics_data.append({
                'strategy': strategy,
                'value': value,
                'color': plt.cm.Set2(len(strategies))
            })
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(metrics_data)
        df = df.sort_values('value', ascending=False)
        
        # Plot 1: Bar chart
        ax = axes[0]
        bars = ax.bar(range(len(df)), df['value'], 
                     color=plt.cm.Set3(range(len(df))),
                     edgecolor='black', alpha=0.8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by Strategy')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['strategy'], rotation=45, ha='right')
        
        # Add value labels
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(i, row['value'] + 0.01, f"{row['value']:.3f}", 
                   ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Scatter plot with error bars (if available)
        ax = axes[1]
        
        # Try to get multiple metrics for error bars
        for i, (strategy, results) in enumerate(strategy_results.items()):
            if isinstance(results.get(metric), (list, np.ndarray)):
                values = results[metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                ax.errorbar(i, mean_val, yerr=std_val, 
                           fmt='o', capsize=5, capthick=2,
                           label=strategy, markersize=10)
            else:
                ax.scatter(i, df.loc[df['strategy'] == strategy, 'value'].values[0],
                          s=100, label=strategy, alpha=0.7, edgecolors='k')
        
        ax.set_xlabel('Strategy Index')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_radar_comparison(
        self,
        strategy_results: Dict[str, Dict],
        metric: str,
        title: str,
        save_path: Optional[str],
        figsize: Tuple[int, int],
        show: bool
    ) -> plt.Figure:
        """Create radar chart for strategy comparison."""
        from math import pi
        
        # Extract metrics for each strategy
        metrics = ['final_ap', 'best_ap', 'query_efficiency', 'training_time']
        metrics = [m for m in metrics if any(m in res for res in strategy_results.values())]
        
        if len(metrics) < 3:
            print("Not enough metrics for radar chart. Using bar chart instead.")
            return self.plot_strategy_comparison(
                strategy_results, metric, title, 'bar', save_path, figsize, show
            )
        
        # Normalize metrics
        normalized_data = {}
        for strategy, results in strategy_results.items():
            values = []
            for m in metrics:
                if m in results:
                    val = results[m]
                elif m == 'final_ap' and 'history' in results and 'val_ap' in results['history']:
                    val_ap = results['history']['val_ap']
                    val = val_ap[-1] if val_ap else 0
                else:
                    val = 0
                values.append(val)
            
            # Normalize to [0, 1]
            max_val = max(values) if max(values) > 0 else 1
            normalized = [v / max_val for v in values]
            normalized_data[strategy] = normalized
        
        # Create radar chart
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot each strategy
        colors = plt.cm.Set2(np.linspace(0, 1, len(strategy_results)))
        for (strategy, values), color in zip(normalized_data.items(), colors):
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', 
                   label=strategy, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Draw axis lines
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics])
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=9)
        plt.ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title(title, size=15, fontweight='bold', y=1.1)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_uncertainty_distribution(
        self,
        uncertainties: Dict[str, np.ndarray],
        title: str = "Uncertainty Distribution",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        show: bool = True
    ) -> plt.Figure:
        """
        Plot distribution of uncertainty scores.
        
        Args:
            uncertainties: Dictionary mapping strategy names to uncertainty arrays
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Histogram
        ax = axes[0, 0]
        for strategy, uncerts in uncertainties.items():
            if len(uncerts) > 0:
                ax.hist(uncerts, bins=30, alpha=0.5, label=strategy, density=True)
        
        ax.set_xlabel('Uncertainty Score')
        ax.set_ylabel('Density')
        ax.set_title('Uncertainty Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Box plot
        ax = axes[0, 1]
        data_to_plot = [uncerts for uncerts in uncertainties.values() if len(uncerts) > 0]
        labels = [strategy for strategy, uncerts in uncertainties.items() if len(uncerts) > 0]
        
        if data_to_plot:
            box = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Uncertainty Score')
            ax.set_title('Uncertainty Statistics')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Cumulative distribution
        ax = axes[1, 0]
        for strategy, uncerts in uncertainties.items():
            if len(uncerts) > 0:
                sorted_uncerts = np.sort(uncerts)
                cdf = np.arange(1, len(sorted_uncerts) + 1) / len(sorted_uncerts)
                ax.plot(sorted_uncerts, cdf, label=strategy, linewidth=2)
        
        ax.set_xlabel('Uncertainty Score')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Violin plot
        ax = axes[1, 1]
        if data_to_plot:
            parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
            
            # Color the violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(plt.cm.Set3(i / len(data_to_plot)))
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Uncertainty Score')
            ax.set_title('Uncertainty Violin Plot')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_query_analysis(
        self,
        query_history: List[Dict],
        title: str = "Query Analysis",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        show: bool = True
    ) -> plt.Figure:
        """
        Analyze and visualize query patterns.
        
        Args:
            query_history: List of query information dictionaries
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        if not query_history:
            print("No query history data available")
            return fig
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(query_history)
        
        # Plot 1: Query uncertainty over cycles
        ax = axes[0, 0]
        if 'cycle' in df.columns and 'avg_uncertainty' in df.columns:
            ax.plot(df['cycle'], df['avg_uncertainty'], 'b-o', linewidth=2, markersize=8)
            ax.fill_between(df['cycle'], 
                           df['avg_uncertainty'] - df.get('std_uncertainty', 0),
                           df['avg_uncertainty'] + df.get('std_uncertainty', 0),
                           alpha=0.2, color='b')
            ax.set_xlabel('AL Cycle')
            ax.set_ylabel('Average Uncertainty')
            ax.set_title('Query Uncertainty over Cycles')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Number of queries per cycle
        ax = axes[0, 1]
        if 'cycle' in df.columns and 'num_queries' in df.columns:
            bars = ax.bar(df['cycle'], df['num_queries'], 
                         color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('AL Cycle')
            ax.set_ylabel('Number of Queries')
            ax.set_title('Query Volume per Cycle')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
        
        # Plot 3: Query efficiency (AP gain per query)
        ax = axes[1, 0]
        if 'cycle' in df.columns and 'ap_gain' in df.columns and 'num_queries' in df.columns:
            efficiency = df['ap_gain'] / df['num_queries'].replace(0, 1)
            ax.plot(df['cycle'], efficiency, 'g-s', linewidth=2, markersize=8)
            ax.set_xlabel('AL Cycle')
            ax.set_ylabel('AP Gain per Query')
            ax.set_title('Query Efficiency')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(efficiency) > 2:
                z = np.polyfit(df['cycle'], efficiency, 1)
                p = np.poly1d(z)
                ax.plot(df['cycle'], p(df['cycle']), 'r--', alpha=0.7, 
                       label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
                ax.legend()
        
        # Plot 4: Cumulative queries
        ax = axes[1, 1]
        if 'cycle' in df.columns and 'num_queries' in df.columns:
            cumulative = df['num_queries'].cumsum()
            ax.plot(df['cycle'], cumulative, 'm-^', linewidth=2, markersize=8)
            ax.set_xlabel('AL Cycle')
            ax.set_ylabel('Cumulative Queries')
            ax.set_title('Cumulative Query Volume')
            ax.grid(True, alpha=0.3)
            
            # Fill area under curve
            ax.fill_between(df['cycle'], 0, cumulative, alpha=0.3, color='m')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_interactive_plot(
        self,
        history: Dict[str, List],
        title: str = "Interactive Learning Curves",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> go.Figure:
        """
        Create interactive plot using Plotly.
        
        Args:
            history: Training history
            title: Plot title
            save_path: Path to save HTML file
            show: Whether to display the figure
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Validation AP', 'Training Loss',
                           'Labeled Samples vs AP', 'Query Efficiency'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Validation AP
        if 'val_ap' in history:
            epochs = list(range(1, len(history['val_ap']) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_ap'],
                          mode='lines+markers',
                          name='Validation AP',
                          line=dict(color='blue', width=2),
                          marker=dict(size=6)),
                row=1, col=1
            )
        
        # Plot 2: Training loss
        if 'train_loss' in history:
            epochs = list(range(1, len(history['train_loss']) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=history['train_loss'],
                          mode='lines',
                          name='Training Loss',
                          line=dict(color='green', width=2)),
                row=1, col=2
            )
        
        # Plot 3: Labeled samples vs AP
        if 'labeled_count' in history and 'val_ap' in history:
            fig.add_trace(
                go.Scatter(x=history['labeled_count'], y=history['val_ap'],
                          mode='markers',
                          name='Samples vs AP',
                          marker=dict(
                              size=10,
                              color=list(range(len(history['val_ap']))),
                              colorscale='Viridis',
                              showscale=True,
                              colorbar=dict(title="Epoch")
                          )),
                row=2, col=1
            )
        
        # Plot 4: Query efficiency
        if 'val_ap' in history and len(history['val_ap']) > 1:
            ap_gains = np.diff(history['val_ap']).tolist()
            cycles = list(range(1, len(ap_gains) + 1))
            
            fig.add_trace(
                go.Bar(x=cycles, y=ap_gains,
                      name='AP Gain',
                      marker_color='skyblue'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=title,
            showlegend=True,
            height=800,
            template='plotly_white'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="AP", row=1, col=1)
        
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        
        fig.update_xaxes(title_text="Labeled Samples", row=2, col=1)
        fig.update_yaxes(title_text="AP", row=2, col=1)
        
        fig.update_xaxes(title_text="AL Cycle", row=2, col=2)
        fig.update_yaxes(title_text="AP Gain", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
        
        if show:
            fig.show()
        
        return fig
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        normalize: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        show: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            title: Plot title
            normalize: Whether to normalize the matrix
            save_path: Path to save figure
            figsize: Figure size
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure
        """
        if normalize:
            cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)
            data = cm_norm
        else:
            data = confusion_matrix
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(data, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        ax.set(xticks=np.arange(data.shape[1]),
               yticks=np.arange(data.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = data.max() / 2.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt),
                       ha="center", va="center",
                       color="white" if data[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_feature_space(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        title: str = "Feature Space Visualization",
        method: str = "pca",  # "pca", "tsne", or "umap"
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize feature space using dimensionality reduction.
        
        Args:
            features: Feature vectors
            labels: Corresponding labels
            title: Plot title
            method: Dimensionality reduction method
            save_path: Path to save figure
            figsize: Figure size
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure
        """
        if features.shape[1] <= 2:
            # Features already low-dimensional
            reduced = features
        else:
            # Apply dimensionality reduction
            if method == "pca":
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                reduced = reducer.fit_transform(features)
            elif method == "tsne":
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                reduced = reducer.fit_transform(features)
            elif method == "umap":
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    reduced = reducer.fit_transform(features)
                except ImportError:
                    print("UMAP not installed. Using PCA instead.")
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=2)
                    reduced = reducer.fit_transform(features)
            else:
                raise ValueError(f"Unknown reduction method: {method}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot with different colors for each label
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(reduced[mask, 0], reduced[mask, 1],
                      c=[color], label=f'Class {label}', alpha=0.6, s=50)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'{title} ({method.upper()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature space plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _smooth_curve(self, values: List[float], window_size: int = 5) -> np.ndarray:
        """Apply moving average smoothing to a curve."""
        if len(values) < window_size:
            return np.array(values)
        
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        
        return np.array(smoothed)
    
    def save_all_figures(self, output_dir: str = "results/figures"):
        """
        Save all currently open figures.
        
        Args:
            output_dir: Directory to save figures
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, fig in enumerate(plt.get_fignums()):
            fig_obj = plt.figure(fig)
            save_path = Path(output_dir) / f"figure_{i:03d}.png"
            fig_obj.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure {i} to {save_path}")