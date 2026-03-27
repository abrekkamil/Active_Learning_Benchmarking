import sys
from pathlib import Path
import gc
import torch
# Make src importable
PROJECT_ROOT = Path("../../../").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.config import ActiveLearningConfig
from src.utils import set_seed
from src.rl_active_learning import ActiveLearningSystemRL
from src.active_learning import ActiveLearningSystem

import json

def empty():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    

initial_labeled_sizes = [200, 300, 500, 1000]
query_sizes = [50, 100]
cold_start_strategies = ["random",
     "self_supervised",
    "simple_diversity",
    "diversity",
    "entropy_based_uncertainty",
    "weak_supervision",
    ]
config_path = "AL_ins_segmentation_masonry.yaml"
config = ActiveLearningConfig.from_yaml(config_path)
for strategy in cold_start_strategies:
    for initial_labeled_size in initial_labeled_sizes:
        for query_size in query_sizes:
            # Optional overrides for interactive testing
            config.cold_start_strategy = strategy
            config.model_name = 'maskrcnn'
            config.dynamic_query_size = False
            config.use_wandb = False
            config.pool = True
            config.experiment_name = 'Active_Learning_masonry_new_pool'
            config.al_cycles = 10
            config.initial_labeled = initial_labeled_size
            config.query_size = query_size
            config.epochs_per_cycle = 10
            config.initial_training_epoch = 10
            al_system = ActiveLearningSystem(config)
            history_AL = al_system.run()

            print(history_AL)
            del al_system
            empty()