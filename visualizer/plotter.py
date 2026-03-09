import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_curves(ax, df, metric):

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

    full_dataset_size = 0

    for _, r in df.iterrows():
        labeled = r.get("labeled_curve")

        if labeled is not None and len(labeled) > 0:
            full_dataset_size = max(full_dataset_size, max(labeled))
            print("full dataset size:", full_dataset_size)
    if full_dataset_size == 0:
        return

    plotted = False

    for _, row in df.iterrows():

        y = row[curve_col]
        labeled = row.get("labeled_curve")

        if y is None or labeled is None:
            continue

        if len(y) != len(labeled):
            continue

        x_percent = [100 * l / full_dataset_size for l in labeled]

        label = row.get("label", row.get("fname"))

        ax.plot(x_percent, y, label=label)

        plotted = True

    ax.set_xlabel("Labeled data (% of full dataset)")
    ax.set_ylabel(metric.upper())

    if plotted:
        ax.legend(fontsize=7)

    ax.figure.canvas.draw()


def plot_strategy_mean(ax, df, metric):

    ax.clear()

    metric_map = {
        "f1": "F1_cycles",
        "dice": "dice_cycles",
        "iou": "iou_cycles",
    }

    curve_col = metric_map[metric]

    # detect dataset size
    full_dataset_size = 0
    for _, r in df.iterrows():
        labeled = r["labeled_curve"]
        if labeled:
            full_dataset_size = max(full_dataset_size, max(labeled))

    for strategy, group in df.groupby("strategy"):

        curves = []
        xs = None

        for _, row in group.iterrows():

            y = row[curve_col]
            labeled = row["labeled_curve"]

            if len(y) != len(labeled):
                continue

            x = [100*l/full_dataset_size for l in labeled]

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
    ax.legend(fontsize=8)

    ax.figure.canvas.draw()

def plot_strategy_boxplot(ax, df):

    ax.clear()

    data = []
    labels = []

    for strategy, group in df.groupby("strategy"):

        data.append(group["f1_auc"].values)
        labels.append(strategy)

    ax.boxplot(data)

    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("F1 AUC")

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