import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================================
# Config fields to extract
# ==========================================================

CONFIG_COLS = [
    "model_name",
    "initial_labeled",
    "query_size",
    "al_cycles",
    "initial_training_epoch",
    "cold_start_strategy",
    "query_strategy",
    "dynamic_query_size",
]


# ==========================================================
# File discovery
# ==========================================================

def discover_run_files(results_dir: Path):

    return sorted(results_dir.rglob("*.json"))


# ==========================================================
# JSON loader
# ==========================================================

def load_json(path: Path):

    with open(path, "r") as f:
        return json.load(f)


def safe_get_config(payload):

    if isinstance(payload, dict) and "config" in payload:
        return payload["config"]

    return {}


def safe_get_history(payload):

    if isinstance(payload, dict) and "history" in payload:
        return payload["history"]

    return {}


# ==========================================================
# Run type inference
# ==========================================================

def infer_run_type(fname: str):

    n = fname.lower()
    if "test" in n:
        return "TEST"
    if "full_set_training" in n:
        return "FULL"

    if "reinforcement_active_learning_improved" in n:
        return "RL_AL_improved"

    if "reinforcement_active_learning" in n:
        return "RL_AL"
    
    if "active_learning" in n:
        return "AL"


    return "OTHER"


# ==========================================================
# Strategy label for plots
# ==========================================================

def infer_strategy(fname: str):

    base = Path(fname).stem

    base = base.replace("Reinforcement_Active_Learning_", "RL_")
    base = base.replace("Active_Learning_", "AL_")
    base = base.replace("Full_set_training_", "FULL_")

    base = base.replace("_uncertainty", "")
    base = base.replace("_rl_policy", "")

    base = re.sub(r"__+", "_", base).strip("_")

    return base


# ==========================================================
# Convert epoch logs → cycle logs
# ==========================================================

def compute_cycle_metrics(history, config, row):

    f1 = history.get("val_F1", [])
    dice = history.get("val_dice", [])
    iou = history.get("val_mean_iou", [])
    labeled = history.get("labeled_count", [])

    epochs_per_cycle = config.get("epochs_per_cycle", 1)
    initial_epochs = config.get("initial_training_epoch", 0)
    al_cycles = config.get("al_cycles", 0)

    F1_cycles = []
    dice_cycles = []
    iou_cycles = []
    labeled_cycles = []

    start = 0

    # initial training
    if initial_epochs > 0:
        if row["run_type"] == "FULL":
             initial_epochs = len(f1)
        F1_cycles.append(max(f1[:initial_epochs]))
        dice_cycles.append(max(dice[:initial_epochs]))
        iou_cycles.append(max(iou[:initial_epochs]))

        labeled_cycles.append(labeled[initial_epochs - 1])

        start = initial_epochs

    # AL cycles
    for i in range(al_cycles):

        s = start + i * epochs_per_cycle
        e = s + epochs_per_cycle

        if e > len(f1):
            break

        F1_cycles.append(max(f1[s:e]))
        dice_cycles.append(max(dice[s:e]))
        iou_cycles.append(max(iou[s:e]))

        labeled_cycles.append(labeled[e - 1])

    return F1_cycles, dice_cycles, iou_cycles, labeled_cycles

# ==========================================================
# Check if run finished
# ==========================================================

def is_run_finished(row):

    if row["run_type"] == "FULL":
        return True

    f1_cycles = row["F1_cycles"]

    if f1_cycles is None:
        return False

    expected_cycles = int(row["al_cycles"])

    return len(f1_cycles) == expected_cycles + 1


# ==========================================================
# Summarize one file
# ==========================================================

def summarize_file(path: Path):

    payload = load_json(path)

    config = safe_get_config(payload)
    history = safe_get_history(payload)

    row = {
        "file": str(path),
        "fname": path.name,
        "run_type": infer_run_type(path.name),
        "label": infer_strategy(path.name),
    }

    # flatten config
    for k in CONFIG_COLS:
        row[k] = config.get(k)


    if row["model_name"] is not None:
        row["model_name"] = row["model_name"].lower()
    else:
        row["model_name"] = "Unet"
    
    # if row["dyamic_query_size"] is None:
    #     row["dynamic_query_size"] = False
    # compute cycle metrics
    F1_cycles, dice_cycles, iou_cycles, labeled_cycles = compute_cycle_metrics(history, config, row)

    row["F1_cycles"] = F1_cycles
    row["dice_cycles"] = dice_cycles
    row["iou_cycles"] = iou_cycles
    row["labeled_curve"] = labeled_cycles

    # summary metrics
    if len(F1_cycles) > 0:
        row["f1_best"] = max(F1_cycles)
    if len(dice_cycles) > 0:
        row["dice_best"] = max(dice_cycles)
    if len(iou_cycles) > 0:
        row["iou_best"] = max(iou_cycles)

    row['f1_auc'] = np.trapz(F1_cycles, labeled_cycles) if len(F1_cycles) > 1 else None
    row['dice_auc'] = np.trapz(dice_cycles, labeled_cycles) if len(dice_cycles) > 1 else None
    row['iou_auc'] = np.trapz(iou_cycles, labeled_cycles) if len(iou_cycles) > 1 else None

    return row


def build_strategy_label(row):

    parts = []

    if row["run_type"]:
        parts.append(row["run_type"])

    if row["query_strategy"]:
        parts.append(row["query_strategy"])

    if row["cold_start_strategy"]:
        parts.append(row["cold_start_strategy"])

    if row["dynamic_query_size"]:
        parts.append("dynamic")

    return "_".join(parts)

def infer_dataset(fname):
    n = fname.lower()

    if "crackseg9k" in n:
        return "CrackSeg9k"

    if "deepcrack" in n:
        return "DeepCrack"

    return "Unknown"
# ==========================================================
# Main loader
# ==========================================================

def load_results(results_dir, metric, dataset=None, dataset_size=None):

    results_dir = Path(results_dir)

    files = discover_run_files(results_dir)

    rows = []

    for f in files:
        if dataset is not None and dataset != infer_dataset(f.name):
            continue
        try:

            row = summarize_file(f)

            rows.append(row)
        except Exception as e:
            pass

    runs_df = pd.DataFrame(rows)

    if len(runs_df) == 0:
        return runs_df

    # remove NaN metrics
    runs_df = runs_df.dropna(subset=["f1_best"])

    # detect finished runs
    runs_df["finished"] = runs_df.apply(is_run_finished, axis=1)
    runs_df["strategy"] = runs_df.apply(build_strategy_label, axis=1)

    runs_df = runs_df[runs_df["finished"]].drop(columns=["finished"])

    runs_df = runs_df[runs_df["run_type"].isin(["FULL", "AL", "RL_AL", "RL_AL_improved"])].copy()
    
    runs_df = runs_df.reset_index(drop=True)
    runs_df["labels_90"] = runs_df.apply(
        lambda r: labels_needed_for_target(runs_df, r, metric, 0.90),
        axis=1
    )

    runs_df["labels_95"] = runs_df.apply(
        lambda r: labels_needed_for_target(runs_df, r, metric, 0.95),
        axis=1
    )
    runs_df["labels_95_pct"] = runs_df["labels_95"] / dataset_size
    runs_df["labels_100"] = runs_df.apply(
        lambda r: labels_needed_for_target(runs_df, r, metric, 1.0),
        axis=1
    )
    print("\nRuns loaded:", len(runs_df))

    return runs_df

def labels_needed_for_target(df, row, metric, target=0.95):

    metric_col = f"{metric}_best"

    # find FULL performance for this model
    full_runs = df[df["run_type"] == "FULL"]

    model = row["model_name"]

    full_row = full_runs[full_runs["model_name"] == model]

    if full_row.empty:
        return None

    full_perf = full_row.iloc[0][metric_col]

    threshold = target * full_perf

    labeled = row["labeled_curve"]
    curve = row[f"{metric.upper()}_cycles"]  # F1_cycles, DICE_cycles etc

    if not labeled or not curve:
        return None

    for l, v in zip(labeled, curve):

        if v >= threshold:
            return l

    return None