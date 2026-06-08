"""
Sewer-ML dataset label distribution report.

Counts how often each of the 17 (or 18 if VA included) defect labels
appears in the train and val splits, as raw counts and as percentage
of the split size. Outputs:

    - dataset_report.json  : machine-readable structured report
    - dataset_report.csv   : spreadsheet-friendly table
    - dataset_report.txt   : human-readable aligned table

Run from the same directory as your AL/supervised scripts:

    python dataset_report.py --dataroot ../../../Datasets/
"""

import os
import sys
import json
import argparse
import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd


# Sewer-ML label order matching MultiLabelDataset.LabelNames
# (verify against your dataloaders/sewerml_dataset.py if in doubt)
SEWER_LABEL_ORDER = [
    "VA", "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN",
    "AF", "BE", "FO", "GR", "PH", "PB", "OS", "OP", "OK",
]

# Class importance weights (from your training code)
CIW = {
    "VA": 0.0310, "RB": 1.0000, "OB": 0.5518, "PF": 0.2896,
    "DE": 0.1622, "FS": 0.6419, "IS": 0.1847, "RO": 0.3559,
    "IN": 0.3131, "AF": 0.0811, "BE": 0.2275, "FO": 0.2477,
    "GR": 0.0901, "PH": 0.4167, "PB": 0.4167, "OS": 0.9009,
    "OP": 0.3829, "OK": 0.4396,
}

# Optional human-readable descriptions
LABEL_DESCRIPTIONS = {
    "VA": "Visible asset (normal)",
    "RB": "Root in pipe",
    "OB": "Obstacle",
    "PF": "Pipe fracture / crack",
    "DE": "Deformation",
    "FS": "Settled deposits",
    "IS": "Intruding sealing material",
    "RO": "Surface damage",
    "IN": "Infiltration",
    "AF": "Attached deposits",
    "BE": "Pipe collapse",
    "FO": "Foreign object",
    "GR": "Sealing ring displacement",
    "PH": "Hole in pipe",
    "PB": "Broken pipe",
    "OS": "Other settled deposits",
    "OP": "Other pipe damage",
    "OK": "Other defects",
}


def load_split(csv_path, label_names):
    """
    Read a Sewer-ML annotation CSV and return:
        df            : pandas DataFrame as loaded
        label_matrix  : np.ndarray [N, num_labels] of 0/1
        present_labels: list of label codes actually found as columns
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError("CSV not found: {}".format(csv_path))

    df = pd.read_csv(csv_path)
    print("Loaded {} ({} rows, {} cols)".format(
        os.path.basename(csv_path), len(df), len(df.columns)
    ))

    # Figure out which label columns are in this CSV
    present = [c for c in label_names if c in df.columns]
    if len(present) == 0:
        # Some Sewer-ML CSVs use different column naming; try lowercase
        lower_map = {c.lower(): c for c in df.columns}
        present = [c for c in label_names if c.lower() in lower_map]
        if present:
            rename = {lower_map[c.lower()]: c for c in present}
            df = df.rename(columns=rename)

    if len(present) == 0:
        raise RuntimeError(
            "No label columns found in {}. "
            "Columns present: {}".format(csv_path, list(df.columns))
        )

    label_matrix = df[present].fillna(0).astype(int).values
    return df, label_matrix, present


def compute_stats(label_matrix, label_names):
    """
    Return a dict of per-label stats:
        count, percent_of_split, ciw, description
    """
    n_rows = len(label_matrix)
    stats = OrderedDict()
    for i, code in enumerate(label_names):
        n_pos = int(label_matrix[:, i].sum())
        pct   = 100.0 * n_pos / max(1, n_rows)
        stats[code] = {
            "count":           n_pos,
            "percent_of_split": round(pct, 4),
            "ciw":             CIW.get(code, None),
            "description":     LABEL_DESCRIPTIONS.get(code, ""),
        }
    return stats


def co_occurrence_summary(label_matrix):
    """High-level co-occurrence and label-cardinality stats."""
    n = len(label_matrix)
    # Number of positive labels per image
    labels_per_img = label_matrix.sum(axis=1)
    n_zero  = int((labels_per_img == 0).sum())
    n_one   = int((labels_per_img == 1).sum())
    n_multi = int((labels_per_img >= 2).sum())
    return {
        "n_images":                       n,
        "n_with_no_labels":               n_zero,
        "n_with_exactly_one_label":       n_one,
        "n_with_two_or_more_labels":      n_multi,
        "pct_no_labels":                  round(100.0 * n_zero  / max(1, n), 4),
        "pct_exactly_one":                round(100.0 * n_one   / max(1, n), 4),
        "pct_two_or_more":                round(100.0 * n_multi / max(1, n), 4),
        "mean_labels_per_image":          round(float(labels_per_img.mean()), 4),
        "max_labels_per_image":           int(labels_per_img.max()),
    }


def format_table(report):
    """Aligned, printable table comparing train and val."""
    train = report["train"]["per_label"]
    val   = report["val"]["per_label"]

    header = "{:<4} {:>6} {:>30} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Code", "CIW", "Description",
        "Train+", "Train%", "Val+", "Val%", "Train/Val"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for code in train:
        t = train[code]
        v = val.get(code, {"count": 0, "percent_of_split": 0.0})
        ratio = (t["count"] / max(1, v["count"]))
        lines.append(
            "{:<4} {:>6} {:>30} {:>10d} {:>9.3f}% {:>10d} {:>9.3f}% {:>10.2f}".format(
                code,
                "{:.3f}".format(t["ciw"]) if t["ciw"] is not None else "-",
                (t["description"] or "")[:30],
                t["count"], t["percent_of_split"],
                v["count"], v["percent_of_split"],
                ratio,
            )
        )

    lines.append(sep)
    train_sum = report["train"]["summary"]
    val_sum   = report["val"]["summary"]
    lines.append("")
    lines.append("Train: {} images | normal={} ({:.2f}%) | "
                 "single-label={} ({:.2f}%) | multi-label={} ({:.2f}%) | "
                 "mean labels/img={}".format(
        train_sum["n_images"],
        train_sum["n_with_no_labels"], train_sum["pct_no_labels"],
        train_sum["n_with_exactly_one_label"], train_sum["pct_exactly_one"],
        train_sum["n_with_two_or_more_labels"], train_sum["pct_two_or_more"],
        train_sum["mean_labels_per_image"],
    ))
    lines.append("Val:   {} images | normal={} ({:.2f}%) | "
                 "single-label={} ({:.2f}%) | multi-label={} ({:.2f}%) | "
                 "mean labels/img={}".format(
        val_sum["n_images"],
        val_sum["n_with_no_labels"], val_sum["pct_no_labels"],
        val_sum["n_with_exactly_one_label"], val_sum["pct_exactly_one"],
        val_sum["n_with_two_or_more_labels"], val_sum["pct_two_or_more"],
        val_sum["mean_labels_per_image"],
    ))
    return "\n".join(lines)


def write_csv(report, csv_path):
    """One row per label, columns for train and val counts/percentages."""
    rows = []
    train = report["train"]["per_label"]
    val   = report["val"]["per_label"]
    for code in train:
        t = train[code]
        v = val.get(code, {"count": 0, "percent_of_split": 0.0})
        rows.append({
            "code":         code,
            "description":  t["description"],
            "ciw":          t["ciw"],
            "train_count":  t["count"],
            "train_pct":    t["percent_of_split"],
            "val_count":    v["count"],
            "val_pct":      v["percent_of_split"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Sewer-ML label distribution report"
    )
    parser.add_argument(
        "--dataroot", type=str, default="../../../Datasets/",
        help="Root containing Sewer_ML/SewerML_train.csv and SewerML_valid.csv"
    )
    parser.add_argument(
        "--include_va", action="store_true",
        help="Include the VA (visible asset / normal) label in the report"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dataset_reports",
        help="Where to write JSON/CSV/TXT outputs"
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Optional tag appended to output filenames"
    )
    args = parser.parse_args()

    # Build label list
    if args.include_va:
        labels = SEWER_LABEL_ORDER
    else:
        labels = [c for c in SEWER_LABEL_ORDER if c != "VA"]
    print("Reporting on {} labels: {}".format(len(labels), labels))

    sewer_root = os.path.join(args.dataroot, "Sewer_ML")
    train_csv  = os.path.join(sewer_root, "SewerML_train.csv")
    val_csv    = os.path.join(sewer_root, "SewerML_valid.csv")

    # Load splits
    _, train_mat, train_present = load_split(train_csv, labels)
    _, val_mat,   val_present   = load_split(val_csv,   labels)
    if train_present != val_present:
        print("WARNING: train and val have different label columns")
        print("  train: {}".format(train_present))
        print("  val:   {}".format(val_present))

    used_labels = train_present  # use the actually-present columns

    # Stats
    train_stats = compute_stats(train_mat, used_labels)
    val_stats   = compute_stats(val_mat,   used_labels)

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataroot":     os.path.abspath(args.dataroot),
        "include_va":   args.include_va,
        "labels":       used_labels,
        "train": {
            "csv":      os.path.abspath(train_csv),
            "summary":  co_occurrence_summary(train_mat),
            "per_label": train_stats,
        },
        "val": {
            "csv":      os.path.abspath(val_csv),
            "summary":  co_occurrence_summary(val_mat),
            "per_label": val_stats,
        },
    }

    # Output paths
    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    tag   = "_{}".format(args.tag) if args.tag else ""
    base  = "dataset_report_{}{}".format(stamp, tag)
    json_path = os.path.join(args.output_dir, base + ".json")
    csv_path  = os.path.join(args.output_dir, base + ".csv")
    txt_path  = os.path.join(args.output_dir, base + ".txt")

    # Write outputs
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    write_csv(report, csv_path)

    table = format_table(report)
    with open(txt_path, "w") as f:
        f.write(table + "\n")

    # Also print to stdout
    print()
    print(table)
    print()
    print("Saved:")
    print("  {}".format(json_path))
    print("  {}".format(csv_path))
    print("  {}".format(txt_path))


if __name__ == "__main__":
    main()
