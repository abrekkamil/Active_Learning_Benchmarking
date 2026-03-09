from PyQt6.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

from data_loader import load_results
from plotter import plot_curves, plot_strategy_mean, plot_strategy_boxplot
import pandas as pd

class ExperimentGUI(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Active Learning Experiment Explorer")
        self.resize(1400,800)

        self.df = None

        main_layout = QVBoxLayout()

        self.plot_mode = QComboBox()
        self.plot_mode.addItems([
            "Raw runs",
            "Best runs",
            "Strategy mean",
            "Strategy boxplot",
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

        main_layout.addWidget(self.canvas)

        # -------------------
        # Table
        # -------------------

        self.table = QTableWidget()
        main_layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(main_layout)

        self.setCentralWidget(container)

    # -----------------------------------------------------

    def load_directory(self):

        folder = QFileDialog.getExistingDirectory(self, "Select results directory")

        if not folder:
            return

        self.df = load_results(folder, dataset=self.dataset_box.currentText())

        if self.df is None or len(self.df) == 0:
            QMessageBox.warning(self,"Warning","No experiment files found")
            return

        self.populate_filters()
        self.populate_table(self.df)

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
            "f1_final",
            "f1_auc",
        ]

        cols = [c for c in display_cols if c in df.columns]

        table_df = df[cols]

        self.table.setRowCount(len(table_df))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)

        for i, (_, row) in enumerate(table_df.iterrows()):

            for j, val in enumerate(row):

                self.table.setItem(i, j, QTableWidgetItem(str(val)))
    # -----------------------------------------------------

    def update_plot(self):
        mode = self.plot_mode.currentText()
        df = self.filter_df()
        metric = self.metric_box.currentText()
        if mode == "Raw runs":
            plot_curves(self.ax, df, metric)

        elif mode == "Best runs":
            df_best = select_best_runs(df)
            plot_curves(self.ax, df_best, metric)

        elif mode == "Strategy mean":
            plot_strategy_mean(self.ax, df, metric)

        elif mode == "Strategy boxplot":
            plot_strategy_boxplot(self.ax, df)


        if df is None or len(df) == 0:
            return

        

        # update plot
        plot_curves(self.ax, df, metric)

        # update table
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