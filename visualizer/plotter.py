import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_curves(ax, df, metric, dataset=None,dataset_size=None):

    ax.clear()

    metric_map = {
        "f1": "F1_cycles",
        "dice": "dice_cycles",
        "iou": "iou_cycles",
    }

    curve_col = metric_map.get(metric)

    if curve_col not in df.columns:
        return

    # ------------------------------------------------
    # detect full dataset size
    # ------------------------------------------------

    plotted = False
    full_scores = get_full_performance(df, metric)

    for _, row in df.iterrows():
        model = row["model_name"]
        full_value = full_scores.get(model, None)
        y = row[curve_col]
        
        if full_value is not None and full_value > 0:
            y = [v / full_value for v in y]

        labeled = row.get("labeled_curve")

        if y is None or labeled is None:
            continue

        if len(y) != len(labeled):
            continue

        x_percent = [100 * l / dataset_size for l in labeled]

        label = row.get("label", row.get("fname"))

        ax.plot(x_percent, y, label=label)

        plotted = True

    ax.set_xlabel("Labeled data (% of full dataset)")
    ax.set_ylabel(f"{metric.upper()} (% of FULL)")
    ax.grid(True, linestyle="--", alpha=0.5)
    if plotted:
        ax.legend(fontsize=7)
    ax.axhline(
    1.0,
    linestyle=":",
    color="black",
    linewidth=2,
    label="FULL (100%)"
)
    ax.figure.canvas.draw()


def plot_strategy_mean(ax, df, metric, dataset_size=None):

    ax.clear()

    metric_map = {
        "f1": "F1_cycles",
        "dice": "dice_cycles",
        "iou": "iou_cycles",
    }

    curve_col = metric_map[metric]


    for strategy, group in df.groupby("strategy"):

        curves = []
        xs = None

        for _, row in group.iterrows():

            y = row[curve_col]
            labeled = row["labeled_curve"]

            if len(y) != len(labeled):
                continue

            x = [100*l/dataset_size for l in labeled]

            curves.append(y)
            xs = x

        if not curves:
            continue

        curves = np.array(curves)

        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)

        ax.plot(xs, mean_curve, label=strategy)

        ax.fill_between(
            xs,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )

    ax.set_xlabel("Labeled data (% of full dataset)")
    ax.set_ylabel(metric.upper())
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

    ax.figure.canvas.draw()

def plot_strategy_boxplot(ax, df, metric, dataset_size=None):

    ax.clear()

    metric_col = f"{metric}_best"

    # Remove FULL runs from boxplot
    df_plot = df[df["run_type"] != "FULL"]

    data = []
    labels = []
    colors = []

    color_map = {
        "AL": "#4C72B0",
        "RL_AL": "#DD8452",
        "RL_AL_improved": "#55A868",
    }

    for strategy, group in df_plot.groupby("strategy"):

        vals = group[metric_col].dropna().values

        if len(vals) == 0:
            continue

        data.append(vals)
        labels.append(strategy)

        run_type = group["run_type"].iloc[0]
        colors.append(color_map.get(run_type, "gray"))
    strategy_order = (
        df.groupby("strategy")[metric_col]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    strategy_groups = df.groupby("strategy")

    for strategy in strategy_order:
        group = strategy_groups.get_group(strategy)

    ax.boxplot(data, patch_artist=True, showmeans=True)
    box = ax.boxplot(data, patch_artist=True)

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    # ---------------------------------
    # Add FULL reference line
    # ---------------------------------

    full_scores = get_full_performance(df, metric)

    if len(full_scores) > 0:

        full_value = max(full_scores.values())

        ax.axhline(
            full_value,
            linestyle="--",
            color="black",
            linewidth=2,
            label="FULL training"
        )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)

    ax.figure.subplots_adjust(bottom=0.30)

    ax.set_ylabel(metric.upper())

    ax.grid(True, linestyle="--", alpha=0.5)

    ax.legend()

    ax.figure.canvas.draw()




def performance_at_percent(row, percent):

    labeled = row["labeled_curve"]
    f1 = row["F1_cycles"]

    if not labeled or not f1:
        return None

    full = max(labeled)

    target = percent * full

    idx = np.argmin(np.abs(np.array(labeled) - target))

    return f1[idx]



def get_full_performance(df, metric):

    metric_col = f"{metric}_best"

    full_runs = df[df["run_type"] == "FULL"]

    full_scores = {}

    for _, row in full_runs.iterrows():

        model = row["model_name"]

        full_scores[model] = row[metric_col]

    return full_scores

def plot_efficiency(ax, df):

    ax.clear()

    data = []
    labels = []

    for strategy, group in df.groupby("strategy"):

        vals = group["labels_95"].dropna()

        if len(vals) == 0:
            continue

        data.append(vals.mean())
        labels.append(strategy)

    ax.barh(labels, data)

    ax.set_xlabel("Labels needed for 95% performance")
    ax.set_title("Label Efficiency")

    ax.grid(True, linestyle="--", alpha=0.5)

    ax.figure.canvas.draw()