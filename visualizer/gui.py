from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

from data_loader import load_results
from plotter import plot_curves, plot_strategy_mean, plot_strategy_boxplot, plot_efficiency
import pandas as pd
import json
from pathlib import Path

SETTINGS_FILE = Path("visualizer_settings.json")

class ExperimentGUI(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Active Learning Experiment Explorer")
        self.resize(1400,800)

        self.df = None
        splitter = QSplitter(Qt.Orientation.Vertical)

        main_layout = QVBoxLayout()

        self.plot_mode = QComboBox()
        self.plot_mode.addItems([
            "Raw runs",
            "Best runs",
            "Strategy mean",
            "Strategy boxplot",
            "Efficiency analysis",
        ])

        # -------------------
        # Top controls
        # -------------------

        top_bar = QHBoxLayout()

        self.load_btn = QPushButton("Load Results Directory")
        self.load_btn.clicked.connect(self.load_directory)

        top_bar.addWidget(self.load_btn)

        self.dataset_box = QComboBox()
        self.dataset_box.addItems(["CrackSeg9k","DeepCrack"])

        top_bar.addWidget(QLabel("Dataset"))
        top_bar.addWidget(self.dataset_box)

        self.dataset_size_box = QLineEdit()
        self.dataset_size_box.setPlaceholderText("Full dataset size")
        top_bar.addWidget(QLabel("Full dataset size"))
        top_bar.addWidget(self.dataset_size_box)

        self.metric_box = QComboBox()
        self.metric_box.addItems(["f1","dice","iou"])

        top_bar.addWidget(QLabel("Metric"))
        top_bar.addWidget(self.metric_box)

        self.update_btn = QPushButton("Update Plot")
        self.update_btn.clicked.connect(self.update_plot)

        top_bar.addWidget(self.update_btn)

        main_layout.addLayout(top_bar)

        # -------------------
        # Filters
        # -------------------

        filter_layout = QHBoxLayout()

        self.run_type_filter = QComboBox()
        self.model_filter = QComboBox()
        self.strategy_filter = QComboBox()
        self.cold_filter = QComboBox()
        self.initial_filter = QComboBox()
        self.query_filter = QComboBox()
        self.dynamic_filter = QComboBox()

        filter_layout.addWidget(QLabel("Plot mode"))
        filter_layout.addWidget(self.plot_mode)
        filter_layout.addWidget(QLabel("Run type"))
        filter_layout.addWidget(self.run_type_filter)

        filter_layout.addWidget(QLabel("Model"))
        filter_layout.addWidget(self.model_filter)

        filter_layout.addWidget(QLabel("Query strategy"))
        filter_layout.addWidget(self.strategy_filter)

        filter_layout.addWidget(QLabel("Cold start"))
        filter_layout.addWidget(self.cold_filter)

        filter_layout.addWidget(QLabel("Initial labeled"))
        filter_layout.addWidget(self.initial_filter)

        filter_layout.addWidget(QLabel("Query size"))
        filter_layout.addWidget(self.query_filter)

        filter_layout.addWidget(QLabel("Dynamic Query"))
        filter_layout.addWidget(self.dynamic_filter)

        main_layout.addLayout(filter_layout)

        # -------------------
        # Plot
        # -------------------

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.fig)

        # -------------------
        # Table
        # -------------------

        self.table = QTableWidget()

        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        container = QWidget()
        container.setLayout(main_layout)

        self.setCentralWidget(container)

        self.load_settings()

    # -----------------------------------------------------
    def load_settings(self):

        if SETTINGS_FILE.exists():

            with open(SETTINGS_FILE) as f:
                settings = json.load(f)

            size = settings.get("dataset_size")

            if size:
                self.dataset_size_box.setText(str(size))

    def save_settings(self):

        settings = {
            "dataset_size": self.dataset_size_box.text()
        }

        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)



    def load_directory(self):
        
        folder = QFileDialog.getExistingDirectory(self, "Select results directory")

        if not folder:
            return

        self.df = load_results(folder, self.metric_box.currentText(), dataset=self.dataset_box.currentText(), dataset_size=int(self.dataset_size_box.text()) if self.dataset_size_box.text().isdigit() else None)

        if self.df is None or len(self.df) == 0:
            QMessageBox.warning(self,"Warning","No experiment files found")
            return

        self.populate_filters()
        self.populate_table(self.df)
        self.update_plot()

    # -----------------------------------------------------

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

        fill(self.run_type_filter, "run_type")
        fill(self.model_filter,"model_name")
        fill(self.strategy_filter,"query_strategy")
        fill(self.cold_filter,"cold_start_strategy")
        fill(self.initial_filter,"initial_labeled")
        fill(self.query_filter,"query_size")
        fill(self.dynamic_filter, "dynamic_query_size")

    # -----------------------------------------------------

    def filter_df(self):

        if self.df is None:
            return None

        df = self.df

        def apply(box, column):

            val = box.currentText()

            if val != "ALL" and column in df:
                return df[df[column].astype(str) == val]

            return df

        df = apply(self.run_type_filter, "run_type")
        df = apply(self.model_filter,"model_name")
        df = apply(self.strategy_filter,"query_strategy")
        df = apply(self.cold_filter,"cold_start_strategy")
        df = apply(self.initial_filter,"initial_labeled")
        df = apply(self.query_filter,"query_size")
        df = apply(self.dynamic_filter,"dynamic_query_size")

        return df

    # -----------------------------------------------------

    def populate_table(self, df):

        metric = self.metric_box.currentText().lower()

        metric_map = {
            "f1": "f1_best",
            "dice": "dice_best",
            "iou": "iou_best",
        }

        sort_col = metric_map.get(metric, "f1_best")

        display_cols = [
            "run_type",
            "label",
            "model_name",
            "initial_labeled",
            "query_size",
            "dynamic_query_size",
            "cold_start_strategy",
            "query_strategy",
            "f1_best",
            "dice_best",
            "iou_best",
            "f1_auc",
            "dice_auc",
            "iou_auc",
            "labels_90",
            "labels_95",
            "labels_100",
        ]

        cols = [c for c in display_cols if c in df.columns]

        table_df = df[cols]

        # 🔹 sort by selected metric
        if sort_col in table_df.columns:
            table_df = table_df.sort_values(sort_col, ascending=False)

        table_df = table_df.reset_index(drop=True)

        self.table.setRowCount(len(table_df))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)

        for i, (_, row) in enumerate(table_df.iterrows()):

            for j, val in enumerate(row):

                self.table.setItem(i, j, QTableWidgetItem(str(val)))
    # -----------------------------------------------------

    def update_plot(self):
        self.save_settings()

        mode = self.plot_mode.currentText()
        df = self.filter_df()
        metric = self.metric_box.currentText()
        dataset = self.dataset_box.currentText()

        size_text = self.dataset_size_box.text().strip()

        if not size_text.isdigit():
            QMessageBox.warning(
                self,
                "Missing dataset size",
                "Please enter a valid full dataset size."
            )
            return

        dataset_size = int(size_text)

        if mode == "Raw runs":
            plot_curves(
                self.ax,
                df,
                metric,
                dataset=dataset,
                dataset_size=dataset_size,
            )

        elif mode == "Best runs":
            df_best = select_best_runs(df)
            plot_curves(
                self.ax,
                df_best,
                metric,
                dataset=dataset,
                dataset_size=dataset_size,
            )

        elif mode == "Strategy mean":
            plot_strategy_mean(
                self.ax,
                df,
                metric,
                dataset=dataset,
                dataset_size=dataset_size,
            )

        elif mode == "Strategy boxplot":
            plot_strategy_boxplot(self.ax, df, metric)

        elif mode == "Efficiency analysis":
            plot_efficiency(
                self.ax,
                df,
                metric=metric,
                dataset_size=dataset_size,
            )

        self.populate_table(df)

def select_best_runs(df):

    selected = []

    for strategy, group in df.groupby("strategy"):

        best = group.loc[group["f1_auc"].idxmax()]

        selected.append(best)

    return pd.DataFrame(selected)


def select_representative_runs(df):

    group_cols = [
        "model_name",
        "initial_labeled",
        "query_size",
        "dynamic_query_size",
    ]

    selected = []

    for _, group in df.groupby(group_cols):

        if len(group) == 1:
            selected.append(group)
            continue

        # top by AUC
        top_auc = group.sort_values("f1_auc", ascending=False).head(2)

        # best final
        best_final = group.loc[[group["f1_final"].idxmax()]]

        chosen = pd.concat([top_auc, best_final]).drop_duplicates(subset=["fname"])

        selected.append(chosen)

    return pd.concat(selected)
