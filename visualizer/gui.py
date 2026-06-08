from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from datetime import datetime
from data_loader import load_results
from plotter import plot_curves, plot_strategy_mean, plot_strategy_boxplot, plot_efficiency
import pandas as pd
import numpy as np
import json
from pathlib import Path
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import csv
SETTINGS_FILE = Path("visualizer_settings.json")

# ── Per-dataset config ────────────────────────────────────────────────────────
# Add / rename datasets here.  "type" drives which metric columns are shown.
DATASET_CONFIG = {
    "CrackSeg9k": {"type": "semantic"},
    "DeepCrack":  {"type": "semantic"},
    "Masonry":    {"type": "instance"},
    # Example instance-segmentation dataset:
    # "COCO-Crack": {"type": "instance"},
}

SEMANTIC_METRICS  = ["f1", "dice", "iou"]
INSTANCE_METRICS  = ["mask_AP", "box_AP"]

# Columns shown in the table for each segmentation type
SEMANTIC_METRIC_COLS  = ["f1_best", "dice_best", "iou_best",
                          "f1_auc",  "dice_auc",  "iou_auc"]
INSTANCE_METRIC_COLS  = ["mask_AP_best", "box_AP_best",
                          "mask_AP_auc",  "box_AP_auc"]

# Maps the metric combo-box text → the "best" column used for sorting / saving
METRIC_SORT_COL = {
    "f1":      "f1_best",
    "dice":    "dice_best",
    "iou":     "iou_best",
    "mask_AP": "mask_AP_best",
    "box_AP":  "box_AP_best",
}

# Maps metric → column saved into full_results per model
FULL_RESULT_METRICS = {
    "semantic": ["f1", "dice", "iou"],
    "instance": ["mask_AP", "box_AP"],
}


def dataset_type(dataset: str) -> str:
    """Return 'semantic' or 'instance' for a given dataset name."""
    return DATASET_CONFIG.get(dataset, {}).get("type", "semantic")


def metrics_for_dataset(dataset: str) -> list[str]:
    return SEMANTIC_METRICS if dataset_type(dataset) == "semantic" else INSTANCE_METRICS


def metric_cols_for_dataset(dataset: str) -> list[str]:
    return SEMANTIC_METRIC_COLS if dataset_type(dataset) == "semantic" else INSTANCE_METRIC_COLS


# ─────────────────────────────────────────────────────────────────────────────

class ExperimentGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Active Learning Experiment Explorer")
        self.resize(1400, 800)

        self.df = None
        # Loaded once from JSON; keyed by dataset name
        self.saved_full_results: dict = {}

        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout = QVBoxLayout()

        # ── Plot mode ─────────────────────────────────────────────────────────
        self.plot_mode = QComboBox()
        self.plot_mode.addItems([
            "Raw runs",
            "Best runs",
            "Strategy mean",
            "Strategy boxplot",
            "Efficiency analysis",
        ])

        # ── Top controls ──────────────────────────────────────────────────────
        top_bar = QHBoxLayout()

        self.load_btn = QPushButton("Load Results Directory")
        self.load_btn.clicked.connect(self.load_directory)
        top_bar.addWidget(self.load_btn)

        self.dataset_box = QComboBox()
        self.dataset_box.addItems(list(DATASET_CONFIG.keys()))
        self.dataset_box.currentTextChanged.connect(self.dataset_changed)
        top_bar.addWidget(QLabel("Dataset"))
        top_bar.addWidget(self.dataset_box)

        self.dataset_size_box = QLineEdit()
        self.dataset_size_box.setPlaceholderText("Full dataset size")
        top_bar.addWidget(QLabel("Full dataset size"))
        top_bar.addWidget(self.dataset_size_box)

        self.metric_box = QComboBox()
        # Populated properly in load_settings / dataset_changed
        top_bar.addWidget(QLabel("Metric"))
        top_bar.addWidget(self.metric_box)

        self.update_btn = QPushButton("Update Plot")
        self.update_btn.clicked.connect(self.update_plot)
        top_bar.addWidget(self.update_btn)

        main_layout.addLayout(top_bar)

        # ── Filters ───────────────────────────────────────────────────────────
        filter_layout = QHBoxLayout()

        self.run_type_filter  = QComboBox()
        self.model_filter     = QComboBox()
        self.strategy_filter  = QComboBox()
        self.cold_filter      = QComboBox()
        self.initial_filter   = QComboBox()
        self.query_filter     = QComboBox()
        self.dynamic_filter   = QComboBox()
        self.level_filter     = QComboBox()
        self.level_filter.addItems([
            "ALL",
            "Reach 90%",
            "Reach 95%",
            "Reach 100%",
            "Stopped before 20% data",
            "Reach 100% before 20% data",   # fixed: was "50%" in filter_df
            "Reach 100% before 30% data",   # fixed: was "50%" in filter_df
            "Reach 100% before 40% data",   # fixed: was "50%" in filter_df
            "Reach 100% before 50% data",   # fixed: was "50%" in filter_df
            "Reach 100% before 80% data",   # fixed: was "50%" in filter_df
            "Reach 100% before 100% data",   # fixed: was "50%" in filter_df
        ])

        filter_layout.addWidget(QLabel("Plot mode"));   filter_layout.addWidget(self.plot_mode)
        filter_layout.addWidget(QLabel("Run type"));    filter_layout.addWidget(self.run_type_filter)
        filter_layout.addWidget(QLabel("Model"));       filter_layout.addWidget(self.model_filter)
        filter_layout.addWidget(QLabel("Query strategy")); filter_layout.addWidget(self.strategy_filter)
        filter_layout.addWidget(QLabel("Cold start"));  filter_layout.addWidget(self.cold_filter)
        filter_layout.addWidget(QLabel("Initial labeled")); filter_layout.addWidget(self.initial_filter)
        filter_layout.addWidget(QLabel("Query size"));  filter_layout.addWidget(self.query_filter)
        filter_layout.addWidget(QLabel("Dynamic Query")); filter_layout.addWidget(self.dynamic_filter)
        filter_layout.addWidget(QLabel("Performance level")); filter_layout.addWidget(self.level_filter)

        main_layout.addLayout(filter_layout)

        # ── Plot canvas ───────────────────────────────────────────────────────
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        main_layout.addWidget(toolbar)

        # ── Table ─────────────────────────────────────────────────────────────
        self.table = QTableWidget()
        self.table.horizontalHeader().sectionClicked.connect(self._on_header_clicked)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.load_settings()

    # ── Settings ──────────────────────────────────────────────────────────────

    def _read_settings_file(self) -> dict:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        return {}

    def load_settings(self):
        settings = self._read_settings_file()
        dataset = self.dataset_box.currentText()

        # Dataset size
        size = settings.get("dataset_sizes", {}).get(dataset, "")
        self.dataset_size_box.setText(str(size))

        # Saved full-run baselines
        self.saved_full_results = settings.get("full_results", {})

        # Metric selector — restore last used metric for this dataset, else use type default
        self._refresh_metric_box(dataset, settings)

    def save_settings(self):
        settings = self._read_settings_file()
        dataset  = self.dataset_box.currentText()

        # ── Dataset size ──────────────────────────────────────────────────────
        settings.setdefault("dataset_sizes", {})[dataset] = self.dataset_size_box.text()

        # ── Last used metric ──────────────────────────────────────────────────
        settings.setdefault("last_metric", {})[dataset] = self.metric_box.currentText()

        # ── Full-run baselines ────────────────────────────────────────────────
        if self.df is not None:
            full_runs = self.df[self.df["run_type"] == "FULL"]
            full_dict: dict = {}
            dtype = dataset_type(dataset)

            for _, r in full_runs.iterrows():
                model = r["model_name"]
                full_dict.setdefault(model, {})
                for metric in FULL_RESULT_METRICS[dtype]:
                    col = f"{metric}_best"
                    if col in r:
                        full_dict[model][metric] = r[col]

            if full_dict:
                settings.setdefault("full_results", {})[dataset] = full_dict
                self.saved_full_results[dataset] = full_dict   # keep in-memory copy in sync

        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)

    def dataset_changed(self):
        dataset  = self.dataset_box.currentText()
        settings = self._read_settings_file()

        # Restore dataset size
        size = settings.get("dataset_sizes", {}).get(dataset, "")
        self.dataset_size_box.setText(str(size))

        # Refresh metric combo for the new dataset type
        self._refresh_metric_box(dataset, settings)

        # If data is already loaded re-draw with the new dataset context
        if self.df is not None:
            self.update_plot()

    def _refresh_metric_box(self, dataset: str, settings: dict):
        """Populate metric_box for *dataset*, restoring last used metric if available."""
        metrics      = metrics_for_dataset(dataset)
        last_metric  = settings.get("last_metric", {}).get(dataset, metrics[0])

        self.metric_box.blockSignals(True)
        self.metric_box.clear()
        self.metric_box.addItems(metrics)
        # Restore last used metric
        idx = self.metric_box.findText(last_metric)
        if idx >= 0:
            self.metric_box.setCurrentIndex(idx)
        self.metric_box.blockSignals(False)

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_directory(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select results directory",
            r"C:\Users\vzcl68\OneDrive - Durham University\Durham\Active_Learning_Experiments"
        )
        if not folder:
            return

        dataset      = self.dataset_box.currentText()
        size_text    = self.dataset_size_box.text()
        dataset_size = int(size_text) if size_text.isdigit() else None

        self.df = load_results(
            folder,
            self.metric_box.currentText(),
            dataset=dataset,
            dataset_size=dataset_size,
        )

        if self.df is None or len(self.df) == 0:
            QMessageBox.warning(self, "Warning", "No experiment files found")
            return

        self.populate_filters()
        self.populate_table(self.df)
        self.update_plot()

    # ── Filters ───────────────────────────────────────────────────────────────

    def populate_filters(self):
        df = self.df

        def fill(box, column):
            box.clear()
            if column not in df:
                return
            vals = sorted(df[column].dropna().unique())
            box.addItem("ALL")
            for v in vals:
                box.addItem(str(v))

        fill(self.run_type_filter,  "run_type")
        fill(self.model_filter,     "model_name")
        fill(self.strategy_filter,  "query_strategy")
        fill(self.cold_filter,      "cold_start_strategy")
        fill(self.initial_filter,   "initial_labeled")
        fill(self.query_filter,     "query_size")
        fill(self.dynamic_filter,   "dynamic_query_size")

    def filter_df(self):
        if self.df is None:
            return None

        df = self.df

        def apply(box, column):
            val = box.currentText()
            if val != "ALL" and column in df:
                return df[df[column].astype(str) == val]
            return df

        df = apply(self.run_type_filter,  "run_type")
        df = apply(self.model_filter,     "model_name")
        df = apply(self.strategy_filter,  "query_strategy")
        df = apply(self.cold_filter,      "cold_start_strategy")
        df = apply(self.initial_filter,   "initial_labeled")
        df = apply(self.query_filter,     "query_size")
        df = apply(self.dynamic_filter,   "dynamic_query_size")

        level = self.level_filter.currentText()
        if level == "Reach 90%":
            df = df[df["labels_90"].notna()]
        elif level == "Reach 95%":
            df = df[df["labels_95"].notna()]
        elif level == "Reach 100%":
            df = df[df["labels_100"].notna()]

        elif level == "Stopped before 20% data" and self.dataset_size_box.text().isdigit():
            dataset_size = int(self.dataset_size_box.text())
            limit_20 = 0.20 * dataset_size

            label_col = "labeled_curve"

            if label_col not in df.columns:
                print("labeled_curve column not found.")
                print("Available columns:", df.columns.tolist())
                return df

            df = df.copy()

            def max_labeled_used(curve):
                if curve is None:
                    return None
                if not isinstance(curve, (list, tuple, np.ndarray)):
                    return None
                if len(curve) == 0:
                    return None

                values = pd.to_numeric(pd.Series(curve), errors="coerce").dropna()
                if values.empty:
                    return None

                return values.max()

            df["max_labeled_count_for_run"] = df[label_col].apply(max_labeled_used)

            print("Stopped before 20% data filter")
            print("Dataset size:", dataset_size)
            print("20% limit:", limit_20)
            print(
                df[["label", "run_type", label_col, "max_labeled_count_for_run"]]
                .sort_values("max_labeled_count_for_run")
                .tail(20)
            )

            df = df[
                df["max_labeled_count_for_run"].notna() &
                (df["max_labeled_count_for_run"] <= limit_20)
            ]

        elif level == "Reach 100% before 20% data" and self.dataset_size_box.text().isdigit():
            dataset_size = int(self.dataset_size_box.text())
            df = df[
                df["labels_100"].notna() &
                (df["labels_100"] <= 0.20 * dataset_size)
            ]

        elif level == "Reach 100% before 20% data" and self.dataset_size_box.text().isdigit():
            dataset_size = int(self.dataset_size_box.text())
            df = df[
                df["labels_100"].notna() &
                (df["labels_100"] <= 0.20 * dataset_size)
            ]
        elif level == "Reach 100% before 40% data" and self.dataset_size_box.text().isdigit():
            dataset_size = int(self.dataset_size_box.text())
            df = df[
                df["labels_100"].notna() &
                (df["labels_100"] <= 0.40 * dataset_size)
            ]
        elif level == "Reach 100% before 50% data" and self.dataset_size_box.text().isdigit():
            dataset_size = int(self.dataset_size_box.text())
            df = df[
                df["labels_100"].notna() &
                (df["labels_100"] <= 0.50 * dataset_size)
            ]
        elif level == "Reach 100% before 80% data" and self.dataset_size_box.text().isdigit():
            dataset_size = int(self.dataset_size_box.text())
            df = df[
                df["labels_100"].notna() &
                (df["labels_100"] <= 0.80 * dataset_size)
            ]
        elif level == "Reach 100% before 100% data" and self.dataset_size_box.text().isdigit():
            dataset_size = int(self.dataset_size_box.text())
            df = df[
                df["labels_100"].notna() &
                (df["labels_100"] <= dataset_size)
            ]

        return df

    # ── Table ─────────────────────────────────────────────────────────────────

    def populate_table(self, df, sort_col_idx: int = None, sort_ascending: bool = False):
        dataset = self.dataset_box.currentText()
        metric  = self.metric_box.currentText().lower()

        # Default sort column (metric best)
        default_sort = METRIC_SORT_COL.get(metric, "f1_best")

        # Base columns always shown
        base_cols = [
            "run_type", "label", "model_name",
            "initial_labeled", "query_size", "dynamic_query_size",
            "cold_start_strategy", "query_strategy",
        ]
        metric_cols    = metric_cols_for_dataset(dataset)
        milestone_cols = ["labels_90", "labels_95", "labels_100"]

        display_cols = base_cols + metric_cols + milestone_cols
        cols = [c for c in display_cols if c in df.columns]

        # Determine which column to sort by
        if sort_col_idx is not None and sort_col_idx < len(cols):
            sort_col = cols[sort_col_idx]
        else:
            sort_col = default_sort
            sort_ascending = False

        table_df = df[cols].copy()

        # Numeric-aware sort: coerce to float where possible
        if sort_col in table_df.columns:
            try:
                table_df["_sort_key"] = pd.to_numeric(table_df[sort_col], errors="coerce")
                table_df = table_df.sort_values("_sort_key", ascending=sort_ascending, na_position="last")
                table_df = table_df.drop(columns=["_sort_key"])
            except Exception:
                table_df = table_df.sort_values(sort_col, ascending=sort_ascending, na_position="last")

        table_df = table_df.reset_index(drop=True)

        # Store for re-sort on header click
        self._table_df   = table_df
        self._table_cols = cols

        self.table.setSortingEnabled(False)   # we handle sorting manually
        self.table.setRowCount(len(table_df))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)

        for i, (_, row) in enumerate(table_df.iterrows()):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                # right-align numeric cells
                try:
                    float(val)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                except (ValueError, TypeError):
                    pass
                self.table.setItem(i, j, item)

        # Show sort indicator on the header
        header = self.table.horizontalHeader()
        if sort_col in cols:
            header.setSortIndicator(
                cols.index(sort_col),
                Qt.SortOrder.AscendingOrder if sort_ascending else Qt.SortOrder.DescendingOrder
            )
        header.setSortIndicatorShown(True)

    def _on_header_clicked(self, col_idx: int):
        """Toggle sort direction when the same column header is clicked twice."""
        # detect toggle: same column → flip direction
        current_col   = getattr(self, "_sort_col_idx", None)
        current_asc   = getattr(self, "_sort_ascending", False)

        if current_col == col_idx:
            new_asc = not current_asc
        else:
            # numeric cols default descending (higher = better), text cols ascending
            new_asc = False

        self._sort_col_idx  = col_idx
        self._sort_ascending = new_asc

        df = self.filter_df()
        if df is not None:
            self.populate_table(df, sort_col_idx=col_idx, sort_ascending=new_asc)

    # ── Plotting ──────────────────────────────────────────────────────────────

    def update_plot(self):
        self.save_settings()

        mode    = self.plot_mode.currentText()
        df      = self.filter_df()
        metric  = self.metric_box.currentText()
        dataset = self.dataset_box.currentText()

        size_text = self.dataset_size_box.text().strip()
        if not size_text.isdigit():
            QMessageBox.warning(self, "Missing dataset size",
                                "Please enter a valid full dataset size.")
            return

        dataset_size = int(size_text)
        full_results = self.saved_full_results.get(dataset)

        if mode == "Raw runs":
            plot_curves(
                self.ax, df, metric,
                saved_full_results=full_results,
                model=self.model_filter.currentText(),
                dataset_size=dataset_size,
            )

        elif mode == "Best runs":
            df_best = select_best_runs(df, metric)   # fixed: pass metric, use query_strategy
            plot_curves(
                self.ax, df_best, metric,
                saved_full_results=full_results,
                model=self.model_filter.currentText(),
                dataset_size=dataset_size,
            )

        elif mode == "Strategy mean":
            plot_strategy_mean(
                self.ax, df, metric,
                dataset=dataset,
                dataset_size=dataset_size,
            )

        elif mode == "Strategy boxplot":
            plot_strategy_boxplot(
                self.ax, df, metric,
                saved_full_results=full_results,
                model=self.model_filter.currentText(),
                dataset_size=dataset_size,
            )

        elif mode == "Efficiency analysis":
            plot_efficiency(self.ax, df, metric=metric, dataset_size=dataset_size)

        self.populate_table(df)
        self._save_plot_data_csv(df, mode, dataset, metric)


    def _save_plot_data_csv(self, df, mode, dataset, metric):
        """Write a CSV of the rows being plotted, next to wherever plots get saved."""
        if df is None or len(df) == 0:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_mode = mode.replace(" ", "_").lower()
        out = Path(f"plot_data_{dataset}_{safe_mode}_{metric}_{ts}.csv")

        # Pick a small, useful column subset; fall back to whatever exists.
        preferred = [
            "fname", "day_folder", "run_type", "label",
            "model_name", "cold_start_strategy", "query_strategy",
            "initial_labeled", "query_size", "dynamic_query_size",
            "al_cycles", "epochs_per_cycle",
            "initial_training_epoch", "oracle_epochs",
            "f1_best", "dice_best", "iou_best",
            "labeled_initial", "labeled_final", "labeled_chosen", "images_at_best",
            "finished", "bad_params",
        ]
        cols = [c for c in preferred if c in df.columns]
        df.to_csv(out, columns=cols, index=False)
        print(f"[plot data] wrote {len(df)} rows to {out}")
# ── Helpers ───────────────────────────────────────────────────────────────────

def select_best_runs(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Return the single best run per query_strategy, ranked by AUC of *metric*."""
    auc_col = f"{metric}_auc"
    # Graceful fallback if the column doesn't exist
    if auc_col not in df.columns:
        auc_col = [c for c in df.columns if c.endswith("_auc")]
        auc_col = auc_col[0] if auc_col else None

    selected = []
    for _, group in df.groupby("query_strategy"):   # fixed: was "strategy"
        if auc_col and auc_col in group.columns:
            best = group.loc[group[auc_col].idxmax()]
        else:
            best = group.iloc[0]
        selected.append(best)

    return pd.DataFrame(selected)


def select_representative_runs(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_name", "initial_labeled", "query_size", "dynamic_query_size"]
    selected   = []

    for _, group in df.groupby(group_cols):
        if len(group) == 1:
            selected.append(group)
            continue

        top_auc    = group.sort_values("f1_auc", ascending=False).head(2)
        best_final = group.loc[[group["f1_final"].idxmax()]]
        chosen     = pd.concat([top_auc, best_final]).drop_duplicates(subset=["fname"])
        selected.append(chosen)

    return pd.concat(selected)