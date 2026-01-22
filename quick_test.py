# test_imports.py
import sys
sys.path.append('.')

try:
    from src import (
        ActiveLearningSystem,
        ColdStartStrategies,
        QueryStrategies,
        MaskRCNNModel,
        ActiveLearningVisualizer,
        ActiveLearningConfig
    )
    print("✓ All imports successful!")
    print(f"Version: {ActiveLearningSystem.__module__}")
except ImportError as e:
    print(f"✗ Import error: {e}")


# Example of using the visualization module
from src.visualization import ActiveLearningVisualizer

# Create visualizer
viz = ActiveLearningVisualizer(style="seaborn", color_palette="husl")

# Example history data
history = {
    'val_ap': [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.52, 0.55, 0.57],
    'train_loss': [2.5, 1.8, 1.3, 1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.48],
    'labeled_count': [100, 100, 105, 105, 110, 110, 115, 115, 120, 120]
}

# Plot learning curves
fig = viz.plot_learning_curves(
    history,
    title="Example Learning Curves",
    save_path="results/figures/learning_curves.png",
    show=True
)

# Create interactive plot
interactive_fig = viz.create_interactive_plot(
    history,
    title="Interactive Learning Curves",
    save_path="results/figures/interactive_plot.html",
    show=False
)

print("Visualization complete!")