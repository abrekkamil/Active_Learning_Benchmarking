import json
import yaml
import random
import matplotlib.pyplot as plt
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): random seed
        deterministic (bool): if True, enforce deterministic CUDA behavior
                              (slower but fully reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_config(path):
    """
    Load a YAML or JSON configuration file.
    """
    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")


def plot_sample(dataset, idx=None):
    """
    Plot a single sample from a dataset.
    Assumes dataset[idx] -> (image, target)
    """
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    image, target = dataset[idx]

    plt.figure(figsize=(4, 4))
    if hasattr(image, "permute"):
        image = image.permute(1, 2, 0)
    plt.imshow(image)
    plt.axis("off")

    if isinstance(target, dict) and "boxes" in target:
        for box in target["boxes"]:
            x1, y1, x2, y2 = box
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              fill=False, edgecolor="red", linewidth=2)
            )
    plt.show()


def load_results(results_dir):
    """
    Load all JSON result files from a directory.
    Returns a list of dicts.
    """
    results = []
    for file in sorted(results_dir.glob("*.json")):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                data["__file__"] = file.name
                results.append(data)
        except Exception as e:
            print(f"Skipping {file.name}: {e}")
    return results
