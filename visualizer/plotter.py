import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import mplcursors

# ── Metric → cycles column ────────────────────────────────────────────────────

METRIC_CYCLES_COL = {
    "f1":      "f1_cycles",
    "dice":    "dice_cycles",
    "iou":     "iou_cycles",
    "mask_AP": "mask_AP_cycles",
    "box_AP":  "box_AP_cycles",
}

# Metrics to save into full_scores per segmentation type
SEMANTIC_METRICS  = ["f1", "dice", "iou"]
INSTANCE_METRICS  = ["mask_AP", "box_AP"]


def _cycles_col(metric: str) -> str:
    return METRIC_CYCLES_COL.get(metric, f"{metric}_cycles")


def _all_metrics_for(metric: str) -> list[str]:
    """Return the full metric family for a given metric string."""
    if metric in SEMANTIC_METRICS:
        return SEMANTIC_METRICS
    return INSTANCE_METRICS


# =============================================================================
# Full-run performance lookup
# =============================================================================

def get_full_performance(df, metric: str, saved_full=None) -> dict:
    """
    Returns {model_name: {metric: value, ...}} for FULL runs.
    Falls back to saved_full (from settings) if no FULL runs in df.
    """
    full_runs = df[df["run_type"] == "FULL"]
    full_scores: dict = {}

    metrics_family = _all_metrics_for(metric)

    for _, row in full_runs.iterrows():
        model = row["model_name"]
        for m in metrics_family:
            col = f"{m}_best"
            if col in row.index and not pd.isna(row[col]):
                full_scores.setdefault(model, {})[m] = row[col]

    if not full_scores and saved_full:
        full_scores = saved_full

    return full_scores


# =============================================================================
# plot_curves  —  raw / best runs
# =============================================================================

def plot_curves(ax, df, metric, saved_full_results=None, model=None, dataset_size=None):

    # clean up previous cursor if any
    if hasattr(ax, "_cursor"):
        try:
            ax._cursor.remove()
        except Exception:
            pass
    ax.clear()

    curve_col = _cycles_col(metric)

    if curve_col not in df.columns:
        ax.set_title(f"Column '{curve_col}' not found in data")
        ax.figure.canvas.draw()
        return

    if dataset_size is None or dataset_size <= 0:
        ax.set_title("Dataset size not set")
        ax.figure.canvas.draw()
        return

    full_scores = get_full_performance(df, metric, saved_full=saved_full_results)

    plotted = False

    for _, row in df.iterrows():
        row_model   = row["model_name"]
        full_values = full_scores.get(row_model)

        y = row.get(curve_col)
        labeled = row.get("labeled_curve")

        if not y or not labeled or len(y) != len(labeled):
            continue

        # normalise to % of full-set performance when available
        if full_values and metric in full_values and full_values[metric] > 0:
            full_val = full_values[metric]
            y = [v / full_val for v in y]

        x_pct = [100.0 * l / dataset_size for l in labeled]
        label = row.get("label") or row.get("fname", "unknown")

        ax.plot(x_pct, y, label=label)
        plotted = True

    ax.axhline(1.0, linestyle=":", color="black", linewidth=2, label="FULL (100%)")
    ax.set_xlabel("Labeled data (% of full dataset)")
    ax.set_ylabel(f"{metric} (ratio of FULL)")
    ax.grid(True, linestyle="--", alpha=0.5)

    if plotted:
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    # interactive cursor
    if ax.lines:
        cursor = mplcursors.cursor(ax.lines, hover=2)

        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            lbl = sel.artist.get_label()
            sel.annotation.set_text(f"{lbl}\n{y:.3f} @ {x:.1f}%")

        ax._cursor = cursor

    ax.figure.canvas.draw()


# =============================================================================
# plot_strategy_mean
# =============================================================================

def plot_strategy_mean(ax, df, metric, dataset=None, dataset_size=None):

    ax.clear()

    curve_col = _cycles_col(metric)

    if curve_col not in df.columns:
        ax.set_title(f"Column '{curve_col}' not found in data")
        ax.figure.canvas.draw()
        return

    if dataset_size is None or dataset_size <= 0:
        ax.set_title("Dataset size not set")
        ax.figure.canvas.draw()
        return

    for strategy, group in df.groupby("strategy"):
        curves = []
        common_x = None

        for _, row in group.iterrows():
            y = row.get(curve_col)
            labeled = row.get("labeled_curve")

            if not y or not labeled or len(y) != len(labeled):
                continue

            x = [100.0 * l / dataset_size for l in labeled]
            curves.append(y)
            common_x = x   # assumes same x grid within a strategy group

        if not curves or common_x is None:
            continue

        arr = np.array(curves)
        mean_curve = arr.mean(axis=0)
        std_curve  = arr.std(axis=0)

        ax.plot(common_x, mean_curve, label=strategy)
        ax.fill_between(common_x,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.2)

    ax.set_xlabel("Labeled data (% of full dataset)")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    ax.figure.canvas.draw()


# =============================================================================
# plot_strategy_boxplot
# =============================================================================
def make_short_label(strategy):
    s = strategy.lower()

    # remove prefixes
    s = s.replace("rl_al_improved_rl_policy_", "")
    s = s.replace("rl_al_rl_policy_", "")
    s = s.replace("al_uncertainty_", "")
    s = s.replace("entropy_based_", "")

    # common replacements
    replacements = {
        "self_supervised": "SS",
        "weak_supervision": "WS",
        "simple_diversity": "SimpDiv",
        "diversity": "Div",
        "random": "Rand",
        "uncertainty": "Unc",
        "dynamic": "Dyn",
    }

    parts = s.split("_")
    short_parts = [replacements.get(p, p.capitalize()) for p in parts if p]

    return "-".join(short_parts)

def plot_strategy_boxplot(ax, df, metric, saved_full_results=None, model=None, dataset_size=None):
    ax.clear()

    metric_col = f"{metric}_best"

    # merge RL_AL and RL_AL_improved into one visual group
    group_color_map = {
        "AL": "#4C72B0",
        "RL": "#55A868",   # one single color for all RL-based methods
    }

    df_plot = df[df["run_type"] != "FULL"].copy()

    # unified group
    df_plot["plot_group"] = df_plot["run_type"].apply(
        lambda x: "AL" if x == "AL" else "RL"
    )

    # order by mean performance
    strategy_order = (
        df_plot.groupby("strategy")[metric_col]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    data, labels, colors, peak_labels_data = [], [], [], []

    for strategy in strategy_order:
        group = df_plot[df_plot["strategy"] == strategy]
        vals = group[metric_col].dropna().values

        if len(vals) == 0:
            continue

        # number of labelled images needed to reach the peak for each run
        peak_imgs = group.apply(
            lambda row: labels_to_peak(row, metric, dataset_size),
            axis=1
        ).dropna().values

        data.append(vals)
        labels.append(make_short_label(strategy))
        peak_labels_data.append(peak_imgs)

        plot_group = group["plot_group"].iloc[0]
        colors.append(group_color_map.get(plot_group, "gray"))

    if not data:
        ax.set_title("No data to plot")
        ax.figure.canvas.draw()
        return

    box = ax.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="^",
            markersize=6,
            markerfacecolor="green",
            markeredgecolor="green"
        ),
        medianprops=dict(color="orange", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
        boxprops=dict(edgecolor="black", linewidth=1),
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    # FULL reference line
    full_scores = get_full_performance(df, metric, saved_full=saved_full_results)
    full_handle = None

    if full_scores:
        ref_model = model if (model and model != "ALL") else next(iter(full_scores))
        full_values = full_scores.get(ref_model)

        if full_values and metric in full_values:
            full_handle = ax.axhline(
                full_values[metric],
                linestyle="--",
                color="black",
                linewidth=2,
                label=f"FULL ({ref_model})"
            )

    # axis formatting
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel(f"Best {metric.upper()} score", fontsize=11)
    ax.set_title(f"{metric.upper()} Peak Performance by Strategy", fontsize=14, pad=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # annotate each box with median number of images needed to reach peak
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    for i, peak_imgs in enumerate(peak_labels_data, start=1):
        if len(peak_imgs) == 0:
            continue

        median_peak_imgs = np.median(peak_imgs)

        if dataset_size and dataset_size > 0:
            median_peak_pct = 100.0 * median_peak_imgs / dataset_size
            txt = f"peak\n{median_peak_imgs:.0f} imgs\n({median_peak_pct:.1f}%)"
        else:
            txt = f"peak\n{median_peak_imgs:.0f} imgs"

        box_top = np.nanmax(data[i - 1])

        ax.text(
            i,
            box_top + 0.03 * y_range,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0
        )

    # give extra space for annotations
    ax.set_ylim(y_min, y_max + 0.15 * y_range)


    # more balanced spacing
    ax.figure.subplots_adjust(bottom=0.22, right=0.84)

    # simpler legend
    group_handles = [
        mpatches.Patch(color=group_color_map["AL"], label="AL"),
        mpatches.Patch(color=group_color_map["RL"], label="RL + AL"),
    ]
    if full_handle is not None:
        group_handles.append(
            Line2D([0], [0], color="black", linestyle="--", linewidth=2, label=f"FULL ({ref_model})")
        )

    legend_groups = ax.legend(
        handles=group_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        title="Groups",
        fontsize=10,
        title_fontsize=11,
        frameon=True
    )
    ax.add_artist(legend_groups)

   # boxplot guide handles
    guide_handles = [
        Line2D([0], [0], color="orange", lw=2, label="Median"),
        Line2D([0], [0], marker="^", color="green", markerfacecolor="green",
            linestyle="None", markersize=7, label="Mean"),
        mpatches.Patch(facecolor="lightgray", edgecolor="black", label="Q1 to Q3"),
        Line2D([0], [0], color="black", lw=1.2, label="Whiskers"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
            linestyle="None", markersize=6, label="Outlier"),

        # --- spacer line ---
        Line2D([], [], linestyle="None", label=""),

        # --- abbreviation title ---
        Line2D([], [], linestyle="None", label="Abbreviations:"),

        # --- abbreviations ---
        Line2D([], [], linestyle="None", label="Dyn = Dynamic pool"),
        Line2D([], [], linestyle="None", label="Rand = Random sampling"),
        Line2D([], [], linestyle="None", label="Unc = Uncertainty-based"),
        Line2D([], [], linestyle="None", label="Div = Diversity-based"),
        Line2D([], [], linestyle="None", label="SimpDiv = Simple diversity"),
        Line2D([], [], linestyle="None", label="SS = Self-supervised"),
        Line2D([], [], linestyle="None", label="WS = Weak supervision"),
    ]

    legend_guide = ax.legend(
        handles=guide_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Boxplot guide",
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        handlelength=1.8,
        handletextpad=0.8
    )

    ax.add_artist(legend_guide)

    ax.figure.canvas.draw()


# =============================================================================
# plot_efficiency
# =============================================================================

def plot_efficiency(ax, df, metric="f1", dataset_size=None):

    ax.clear()

    data, labels = [], []

    for strategy, group in df.groupby("strategy"):
        vals = group["labels_95"].dropna()

        if len(vals) == 0:
            continue

        mean_val = vals.mean()

        # optionally express as % of full dataset
        if dataset_size and dataset_size > 0:
            mean_val = 100.0 * mean_val / dataset_size

        data.append(mean_val)
        labels.append(strategy)

    if not data:
        ax.set_title("No efficiency data available")
        ax.figure.canvas.draw()
        return

    # sort by efficiency (fewer labels = better → ascending x)
    order  = np.argsort(data)
    data   = [data[i]   for i in order]
    labels = [labels[i] for i in order]

    ax.barh(labels, data)

    xlabel = ("Labels needed for 95% performance (% of dataset)"
              if dataset_size else "Labels needed for 95% performance")
    ax.set_xlabel(xlabel)
    ax.set_title("Label Efficiency")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.figure.canvas.draw()


# =============================================================================
# performance_at_percent  (utility, unchanged logic but uses _cycles_col)
# =============================================================================

def performance_at_percent(row, percent, metric="f1"):

    labeled = row.get("labeled_curve")
    curve   = row.get(_cycles_col(metric))

    if not labeled or not curve:
        return None

    full   = max(labeled)
    target = percent * full
    idx    = np.argmin(np.abs(np.array(labeled) - target))

    return curve[idx]


def labels_to_peak(row, metric, dataset_size=None):
    """
    Returns the number of labelled images used when the metric curve reaches its peak.
    """
    curve_col = _cycles_col(metric)

    y = row.get(curve_col)
    labeled = row.get("labeled_curve")

    if not y or not labeled or len(y) != len(labeled):
        return np.nan

    y = np.array(y, dtype=float)
    labeled = np.array(labeled, dtype=float)

    if len(y) == 0:
        return np.nan

    peak_idx = np.nanargmax(y)
    return labeled[peak_idx]