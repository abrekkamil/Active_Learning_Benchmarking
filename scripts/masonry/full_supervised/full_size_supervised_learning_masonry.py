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
    
config_path = "full_ins_segmentation_masonry.yaml"
config = ActiveLearningConfig.from_yaml(config_path)

config.experiment_name = 'Full_set_training_maskrcnn'

al_system_full = ActiveLearningSystem(config, skip_cold_start=True)
al_system_full.train(epochs=100)

full_training_history = al_system_full.history

del al_system_fullw
empty()