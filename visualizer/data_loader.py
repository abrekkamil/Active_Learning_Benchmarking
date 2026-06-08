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
    "task",
]


# ==========================================================
# File discovery
# ==========================================================

def discover_run_files(results_dir: Path):
    print(f"Discovering run files in {results_dir}...")
    print(f"Found {len(list(results_dir.rglob('*.json')))} JSON files.")
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

def infer_strategy(path) -> str:
    """
    Build a short human-readable label from the file path.

    Label format:  <run_prefix>_<MMDD>_<HHMM>
    e.g.  RL_crackseg9k_weak_0314_1217

    The date folder is expected to sit somewhere in the path and match
    the pattern YYYY-MM-DD or YYYYMMDD (common training log conventions).
    The time (HHMM) is taken from the trailing numeric suffix of the filename.
    """
    path = Path(path)
    base = path.stem

    # ── Shorten the filename ──────────────────────────────────────────────
    base = base.replace("Reinforcement_Active_Learning_", "RL_")
    base = base.replace("Active_Learning_", "AL_")
    base = base.replace("Full_set_training_", "FULL_")
    base = base.replace("_uncertainty", "")
    base = base.replace("_rl_policy", "")
    base = re.sub(r"__+", "_", base).strip("_")

    # ── Extract HHMM from trailing numeric suffix of filename ─────────────
    # e.g. "..._1217"  →  "1217"
    time_str = ""
    time_match = re.search(r"_(\d{3,4})$", base)
    if time_match:
        time_str = time_match.group(1).zfill(4)   # ensure 4 digits
        base = base[: time_match.start()]          # strip suffix from label

    # ── Extract MMDD from a date-named parent folder ──────────────────────
    # Supports: 2024-03-14, 20240314, 2024_03_14, 03-14, 0314
    date_str = ""
    for part in path.parts:
        # ISO: 2024-03-14  or  2024_03_14
        m = re.match(r"^\d{4}[-_](\d{2})[-_](\d{2})$", part)
        if m:
            date_str = m.group(1) + m.group(2)   # MMDD
            break
        # compact: 20240314
        m = re.match(r"^\d{4}(\d{2})(\d{2})$", part)
        if m:
            date_str = m.group(1) + m.group(2)
            break
        # short: 03-14 or 0314
        m = re.match(r"^(\d{2})[-_]?(\d{2})$", part)
        if m:
            date_str = m.group(1) + m.group(2)
            break

    # ── Assemble final label ──────────────────────────────────────────────
    suffix_parts = [p for p in [date_str, time_str] if p]
    if suffix_parts:
        return f"{base}_{'_'.join(suffix_parts)}"
    return base


# ==========================================================
# Convert epoch logs → cycle logs
# ==========================================================

def compute_cycle_metrics(history, config, row, task="segmentation"):

    if task == "segmentation":
        m1 = history.get("val_F1", [])
        m2 = history.get("val_dice", [])
        m3 = history.get("val_mean_iou", [])

    elif task == "instance_segmentation":
        m1 = history.get("val_mask_AP", [])
        m2 = history.get("val_bbox_AP", [])
        m3 = None
    else:
        m1 = m2 = m3 = None
        raise ValueError(f"Unsupported task: {task}")

    labeled = history.get("labeled_count", [])

    epochs_per_cycle   = config.get("epochs_per_cycle", 1)
    initial_epochs     = config.get("initial_training_epoch", 0)
    al_cycles          = config.get("al_cycles", 0)

    m1_cycles, m2_cycles, m3_cycles = [], [], []
    labeled_cycles = []

    start = 0

    # initial training
    if row["run_type"] == "FULL":
        # FULL runs have no AL cycles; treat the whole history as one block
        if m1:
            m1_cycles.append(max(m1))
            m2_cycles.append(max(m2))
            if m3 is not None:
                m3_cycles.append(max(m3))
            labeled_cycles.append(labeled[-1] if labeled else None)
    elif initial_epochs > 0:
        if row["run_type"] == "FULL":
            initial_epochs = len(m1)

        m1_cycles.append(max(m1[:initial_epochs]))
        m2_cycles.append(max(m2[:initial_epochs]))

        if m3 is not None:
            m3_cycles.append(max(m3[:initial_epochs]))

        labeled_cycles.append(labeled[initial_epochs - 1])
        start = initial_epochs

    print("Initial epochs finished")

    # AL cycles
    for i in range(al_cycles):
        s = start + i * epochs_per_cycle
        e = s + epochs_per_cycle

        if e > len(m1):
            break

        m1_cycles.append(max(m1[s:e]))
        m2_cycles.append(max(m2[s:e]))

        if m3 is not None:
            m3_cycles.append(max(m3[s:e]))

        labeled_cycles.append(labeled[e - 1])

    print("AL cycles finished")

    if task == "segmentation":
        return m1_cycles, m2_cycles, m3_cycles, labeled_cycles
    else:
        return m1_cycles, m2_cycles, labeled_cycles


# ==========================================================
# Check if run finished
# ==========================================================

def is_run_finished(row):
    if row["run_type"] == "FULL":
        return True
    if pd.isna(row["al_cycles"]) or row["al_cycles"] < 0:
        return False
    
    expected_cycles = int(row["al_cycles"])

    if row["task"] == "segmentation":
        cycles = row["f1_cycles"]   
    elif row["task"] == "instance_segmentation":
        cycles = row["mask_AP_cycles"]
    else:
        return False

    if cycles is None:
        return False

    return len(cycles) == expected_cycles + 1


# ==========================================================
# Summarize one file
# ==========================================================

def summarize_file(path: Path):
    payload = load_json(path)
    config  = safe_get_config(payload)
    history = safe_get_history(payload)

    row = {
        "file":     str(path),
        "fname":    path.name,
        "run_type": infer_run_type(path.name),
        "label":    infer_strategy(path),   # full path so date folder is accessible
    }

    # flatten config
    for k in CONFIG_COLS:
        row[k] = config.get(k)

    if row["model_name"] is not None:
        row["model_name"] = row["model_name"].lower()
    else:
        row["model_name"] = "unet"

    task = row["task"]

    # ── Semantic segmentation ──────────────────────────────────────────────
    if task == "segmentation":
        f1_cycles, dice_cycles, iou_cycles, labeled_cycles = compute_cycle_metrics(
            history, config, row, task="segmentation"
        )

        # consistent snake_case + uppercase metric abbreviation
        row["f1_cycles"]   = f1_cycles
        row["dice_cycles"] = dice_cycles
        row["iou_cycles"]  = iou_cycles
        row["labeled_curve"] = labeled_cycles

        row["f1_best"]   = max(f1_cycles)   if f1_cycles   else None
        row["dice_best"] = max(dice_cycles) if dice_cycles else None
        row["iou_best"]  = max(iou_cycles)  if iou_cycles  else None

        row["f1_auc"]   = np.trapz(f1_cycles,   labeled_cycles) if len(f1_cycles)   > 1 else None
        row["dice_auc"] = np.trapz(dice_cycles, labeled_cycles) if len(dice_cycles) > 1 else None
        row["iou_auc"]  = np.trapz(iou_cycles,  labeled_cycles) if len(iou_cycles)  > 1 else None

    # ── Instance segmentation ──────────────────────────────────────────────
    elif task == "instance_segmentation":
        mask_AP_cycles, box_AP_cycles, labeled_cycles = compute_cycle_metrics(
            history, config, row, task="instance_segmentation"
        )

        # column names match METRIC_SORT_COL / metric_cols_for_dataset in gui.py
        row["mask_AP_cycles"] = mask_AP_cycles
        row["box_AP_cycles"]  = box_AP_cycles
        row["labeled_curve"]  = labeled_cycles

        row["mask_AP_best"] = max(mask_AP_cycles) if mask_AP_cycles else None
        row["box_AP_best"]  = max(box_AP_cycles)  if box_AP_cycles  else None

        row["mask_AP_auc"] = np.trapz(mask_AP_cycles, labeled_cycles) if len(mask_AP_cycles) > 1 else None
        row["box_AP_auc"]  = np.trapz(box_AP_cycles,  labeled_cycles) if len(box_AP_cycles)  > 1 else None

    return row


# ==========================================================
# Strategy / dataset helpers
# ==========================================================

def build_strategy_label(row):
    parts = []
    if row["run_type"]:        parts.append(row["run_type"])
    if row["query_strategy"]:  parts.append(row["query_strategy"])
    if row["cold_start_strategy"]: parts.append(row["cold_start_strategy"])
    if row["dynamic_query_size"]:  parts.append("dynamic")
    return "_".join(parts)


def infer_dataset(fname):
    n = fname.lower()
    if "crackseg9k" in n: return "CrackSeg9k"
    if "deepcrack"  in n: return "DeepCrack"
    if "masonry"    in n: return "Masonry"
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
            print(f"Skipping {f.name} (dataset mismatch)")
            continue
        try:
            rows.append(summarize_file(f))
        except Exception as e:
            print(f"Error processing {f}: {e}")

    runs_df = pd.DataFrame(rows)
    if len(runs_df) == 0:
        return runs_df

    runs_df["finished"] = runs_df.apply(is_run_finished, axis=1)
    runs_df["strategy"] = runs_df.apply(build_strategy_label, axis=1)

    runs_df = runs_df[runs_df["finished"]].drop(columns=["finished"])
    runs_df = runs_df[runs_df["run_type"].isin(["FULL", "AL", "RL_AL", "RL_AL_improved"])].copy()
    runs_df = runs_df.reset_index(drop=True)

    runs_df["labels_90"] = runs_df.apply(
        lambda r: labels_needed_for_target(runs_df, r, dataset, metric, 0.90), axis=1
    )
    runs_df["labels_95"] = runs_df.apply(
        lambda r: labels_needed_for_target(runs_df, r, dataset, metric, 0.95), axis=1
    )
    if dataset_size:
        runs_df["labels_95_pct"] = runs_df["labels_95"] / dataset_size
    runs_df["labels_100"] = runs_df.apply(
        lambda r: labels_needed_for_target(runs_df, r, dataset, metric, 1.0), axis=1
    )

    print(f"\nRuns loaded: {len(runs_df)}")
    return runs_df


# ==========================================================
# Efficiency helper
# ==========================================================

def _metric_cycles_col(metric: str) -> str:
    """Return the cycles column name for a given metric string."""
    mapping = {
        "f1":      "f1_cycles",
        "dice":    "dice_cycles",
        "iou":     "iou_cycles",
        "mask_AP": "mask_AP_cycles",
        "box_AP":  "box_AP_cycles",
    }
    # fallback: try <metric>_cycles directly
    return mapping.get(metric, f"{metric}_cycles")


def labels_needed_for_target(df, row, dataset, metric, target=0.95):
    metric_best_col   = f"{metric}_best"
    metric_cycles_col = _metric_cycles_col(metric)
    model = row["model_name"]

    # ── Find full-set performance for this model ───────────────────────────
    full_runs = df[df["run_type"] == "FULL"]
    if not full_runs.empty:
        full_row = full_runs[full_runs["model_name"] == model]
        if full_row.empty:
            return None
        full_perf = full_row.iloc[0].get(metric_best_col)
    else:
        SETTINGS_FILE = Path("visualizer_settings.json")
        if not SETTINGS_FILE.exists():
            return None
        with open(SETTINGS_FILE) as f:
            settings = json.load(f)
        saved = settings.get("full_results", {})
        try:
            full_perf = saved[dataset][model][metric]
        except KeyError:
            print(f"No saved full result for dataset={dataset}, model={model}, metric={metric}")
            return None

    if full_perf is None:
        return None

    threshold = target * full_perf
    labeled   = row["labeled_curve"]
    curve     = row.get(metric_cycles_col)

    if not labeled or not curve:
        return None

    for l, v in zip(labeled, curve):
        if v >= threshold:
            return l

    return None