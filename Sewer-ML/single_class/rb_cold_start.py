"""
Cold start strategies for binary RB classification.

Most strategies are direct adaptations of the multi-label versions.
The main differences:
  - 'stratified' becomes 'balanced' with a target positive ratio, since
    there is only one class
  - 'herding' clusters by positive/negative rather than argmax over CIW
  - No CIW weighting anywhere (only one class matters)

All functions return a list[int] of indices into the passed dataset.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans


# --------------------------------------------------------------
# Feature extractor (reuses ImageNet ResNet-18)
# --------------------------------------------------------------
def _extract_features(dataset, device, batch_size=64, workers=8, cache_path=None):
    if cache_path is not None:
        # Features are class-agnostic — strip the target_class suffix so all binary
        # tasks share the same feature file. Labels stay per-class.
        feat_dir   = os.path.dirname(cache_path)
        feat_file  = os.path.join(feat_dir, "sewerml_features_shared.npy")
        lbl_file   = cache_path + "_binary_labels.npy"
        if os.path.exists(feat_file) and os.path.exists(lbl_file):
            print("ColdStart: loading cached features (shared) + labels (per-class)")
            return (
                np.load(feat_file, mmap_mode="r"),
                np.load(lbl_file,  mmap_mode="r"),
            )

    print("ColdStart: extracting ResNet-18 features")
    extractor    = models.resnet18(pretrained=True)
    extractor.fc = nn.Identity()
    extractor    = extractor.to(device).eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    feats_list, labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            imgs = batch["image"].to(device, non_blocking=True)
            feats_list.append(extractor(imgs).cpu().numpy().astype("float32"))
            labels_list.append(batch["labels"].cpu().numpy().astype("int32"))

    features = np.vstack(feats_list)
    labels   = np.concatenate(labels_list)

    norms    = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features = features / norms

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(feat_file, features)
        np.save(lbl_file,  labels)
        print("ColdStart: features cached to {}".format(feat_file))

    return features, labels


def _get_binary_labels(dataset, batch_size=64, workers=8):
    """Fast path: use cached binary_labels if available; else iterate."""
    cached = getattr(dataset, "binary_labels", None)
    if cached is not None:
        return np.asarray(cached).astype(np.int32)

    # Subset case: walk through underlying dataset
    base = getattr(dataset, "dataset", None)
    if base is not None and hasattr(base, "binary_labels") and base.binary_labels is not None:
        indices = np.asarray(dataset.indices)
        return np.asarray(base.binary_labels)[indices].astype(np.int32)

    print("ColdStart: no cached labels, iterating dataset")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    all_labels = []
    for batch in tqdm(loader, desc="Loading binary labels"):
        all_labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(all_labels).astype(np.int32)


# --------------------------------------------------------------
# 1. Random
# --------------------------------------------------------------
def cold_start_random(dataset, n_initial, seed=42, **kwargs):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=n_initial, replace=False).tolist()
    print("ColdStart [random]: {} samples".format(len(idx)))
    return idx


# --------------------------------------------------------------
# 2. Balanced (the binary equivalent of 'stratified')
# --------------------------------------------------------------
def cold_start_balanced(dataset, n_initial, pos_ratio=0.5, seed=42,
                        batch_size=64, workers=8, **kwargs):
    """
    Sample a target fraction of positives and negatives.
    pos_ratio=0.5 gives a 50/50 split, which is the standard choice for
    rare-class binary training.
    """
    labels = _get_binary_labels(dataset, batch_size, workers)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_pos_target = int(round(n_initial * pos_ratio))
    n_neg_target = n_initial - n_pos_target

    # Clip to what's available
    n_pos = min(n_pos_target, len(pos_idx))
    n_neg = min(n_neg_target, len(neg_idx))

    # If positives are scarce, fill extra from negatives
    if n_pos < n_pos_target:
        n_neg = min(n_initial - n_pos, len(neg_idx))

    rng = np.random.default_rng(seed)
    sel_pos = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    sel_neg = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
    selected = np.concatenate([sel_pos, sel_neg]).tolist()

    print("ColdStart [balanced]: {} positives + {} negatives = {} ({} available pos)".format(
        n_pos, n_neg, len(selected), len(pos_idx)
    ))
    return selected


# --------------------------------------------------------------
# 3. Diversity (KMeans over features)
# --------------------------------------------------------------
def cold_start_diversity(dataset, n_initial, device, cache_path=None,
                         seed=42, batch_size=64, workers=8, **kwargs):
    features, _ = _extract_features(dataset, device, batch_size, workers, cache_path)
    n_samples   = len(features)
    n_clusters  = min(2048, max(16, int(np.sqrt(n_samples))))

    rng         = np.random.default_rng(seed)
    subset_size = min(max(n_clusters * 3, 100_000), n_samples)
    subset_idx  = rng.choice(n_samples, size=subset_size, replace=False)

    km = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=4096,
        random_state=seed, n_init=1,
    )
    km.fit(features[subset_idx])
    cluster_ids = km.predict(features)

    per_cluster = int(np.ceil(n_initial / n_clusters))
    selected = []
    for c in range(n_clusters):
        members = np.where(cluster_ids == c)[0]
        if len(members) == 0:
            continue
        chosen = rng.choice(members, size=min(per_cluster, len(members)), replace=False)
        selected.extend(chosen.tolist())
        if len(selected) >= n_initial:
            break

    selected = selected[:n_initial]
    print("ColdStart [diversity]: {} samples from {} clusters".format(
        len(selected), n_clusters
    ))
    return selected


# --------------------------------------------------------------
# 4. Coreset (greedy k-center on features)
# --------------------------------------------------------------
def cold_start_coreset(dataset, n_initial, device, cache_path=None,
                       seed=42, batch_size=64, workers=8, **kwargs):
    features, _ = _extract_features(dataset, device, batch_size, workers, cache_path)
    n_samples   = len(features)
    rng         = np.random.default_rng(seed)

    feat_t   = torch.tensor(features, dtype=torch.float32)
    selected = [int(rng.integers(n_samples))]
    min_dists = np.full(n_samples, np.inf)

    for _ in tqdm(range(n_initial - 1), desc="coreset"):
        last   = feat_t[selected[-1]].unsqueeze(0)
        dists  = torch.cdist(feat_t, last).squeeze(1).numpy()
        min_dists = np.minimum(min_dists, dists)
        selected.append(int(np.argmax(min_dists)))

    print("ColdStart [coreset]: {} samples".format(len(selected)))
    return selected


# --------------------------------------------------------------
# 5. CLIP-IQA filter + secondary strategy
# --------------------------------------------------------------
def _load_clipiqa_scores(path):
    with open(path, "r") as f:
        return json.load(f)


def cold_start_clipiqa(dataset, n_initial, device,
                       clipiqa_json_path,
                       clipiqa_threshold=0.3,
                       secondary_strategy="balanced",
                       pos_ratio=0.5,
                       cache_path=None, seed=42,
                       batch_size=64, workers=8,
                       save_selection_path=None, **kwargs):
    """
    Same idea as the multi-label version:
      1. Load CLIP-IQA scores
      2. Filter images below threshold
      3. Apply secondary strategy on the filtered pool

    Default secondary is 'balanced' (50/50 pos/neg), which is the binary
    equivalent of 'stratified'.
    """
    scores = _load_clipiqa_scores(clipiqa_json_path)
    print("CLIP-IQA: loaded {} scores".format(len(scores)))

    # Get filenames from the dataset (handles wrapped datasets)
    filenames = None
    for obj in (dataset, getattr(dataset, "base", None), getattr(dataset, "dataset", None)):
        if obj is not None and hasattr(obj, "imgPaths"):
            filenames = obj.imgPaths
            break
    if filenames is None:
        raise RuntimeError("Could not find imgPaths on dataset or base")

    if len(filenames) != len(dataset):
        # If wrapped in a Subset, remap
        if hasattr(dataset, "indices"):
            filenames = [filenames[i] for i in dataset.indices]
        else:
            raise RuntimeError("filenames ({}) != dataset ({})".format(
                len(filenames), len(dataset)
            ))

    score_arr = np.zeros(len(dataset), dtype=np.float32)
    matched = 0
    for i, fn in enumerate(filenames):
        bare = os.path.basename(fn)
        if bare in scores:
            score_arr[i] = float(scores[bare])
            matched += 1
    print("CLIP-IQA: matched {}/{}".format(matched, len(dataset)))

    filtered = np.where(score_arr >= clipiqa_threshold)[0].tolist()
    print("CLIP-IQA: {} pass threshold {:.2f}".format(len(filtered), clipiqa_threshold))

    if len(filtered) == 0:
        raise RuntimeError("No images pass IQA threshold")
    if len(filtered) < n_initial:
        print("WARNING: filtered pool smaller than budget; using all filtered")
        n_initial = len(filtered)

    filtered_ds = Subset(dataset, filtered)

    fn_map = {
        "random":    cold_start_random,
        "balanced":  cold_start_balanced,
        "diversity": cold_start_diversity,
        "coreset":   cold_start_coreset,
    }
    if secondary_strategy not in fn_map:
        raise ValueError("Unknown secondary '{}'".format(secondary_strategy))

    local = fn_map[secondary_strategy](
        dataset=filtered_ds, n_initial=n_initial, device=device,
        cache_path=cache_path, seed=seed, batch_size=batch_size,
        workers=workers, pos_ratio=pos_ratio,
    )
    selected = [filtered[i] for i in local]

    # Selection report
    if save_selection_path is not None:
        labels = _get_binary_labels(dataset, batch_size, workers)
        sel_labels = labels[selected]
        report = {
            "config": {
                "clipiqa_threshold":  clipiqa_threshold,
                "secondary_strategy": secondary_strategy,
                "pos_ratio_target":   pos_ratio,
                "n_initial":          n_initial,
                "seed":               seed,
            },
            "summary": {
                "total":             len(dataset),
                "passed_iqa":        len(filtered),
                "selected":          len(selected),
                "positives_in_sel":  int(sel_labels.sum()),
                "pos_rate_in_sel":   float(sel_labels.mean()),
                "iqa_mean_selected": float(score_arr[selected].mean()),
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(save_selection_path)) or ".", exist_ok=True)
        with open(save_selection_path, "w") as f:
            json.dump(report, f, indent=2)
        print("Report saved to {}".format(save_selection_path))

    print("ColdStart [clipiqa+{}]: {} samples".format(secondary_strategy, len(selected)))
    return selected


# --------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------
COLD_START_STRATEGIES = {
    "random":    cold_start_random,
    "balanced":  cold_start_balanced,
    "diversity": cold_start_diversity,
    "coreset":   cold_start_coreset,
    "clipiqa":   cold_start_clipiqa,
}


def apply_cold_start(strategy_name, dataset, n_initial, device,
                     cache_path=None, seed=42, batch_size=64, workers=8,
                     clipiqa_json_path=None, clipiqa_threshold=0.3,
                     secondary_strategy="balanced", pos_ratio=0.5,
                     save_selection_path=None):
    if strategy_name not in COLD_START_STRATEGIES:
        raise ValueError("Unknown strategy '{}'. Choose from {}".format(
            strategy_name, list(COLD_START_STRATEGIES.keys())
        ))

    fn = COLD_START_STRATEGIES[strategy_name]

    if strategy_name == "clipiqa":
        return fn(
            dataset=dataset, n_initial=n_initial, device=device,
            clipiqa_json_path=clipiqa_json_path,
            clipiqa_threshold=clipiqa_threshold,
            secondary_strategy=secondary_strategy,
            pos_ratio=pos_ratio,
            cache_path=cache_path, seed=seed,
            batch_size=batch_size, workers=workers,
            save_selection_path=save_selection_path,
        )

    return fn(
        dataset=dataset, n_initial=n_initial, device=device,
        cache_path=cache_path, seed=seed,
        batch_size=batch_size, workers=workers,
        pos_ratio=pos_ratio,
    )
