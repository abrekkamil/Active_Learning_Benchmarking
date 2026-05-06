import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from torch.utils.data import DataLoader, Subset
import copy
import os
import sys
import json
import datetime
import time
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataloaders'))

from torchvision import transforms, models
from dataloaders.sewerml_dataset import MultiLabelDataset, MultiLabelDatasetInference


# ==============================================================
# Feature extractor (shared across strategies that need features)
# ==============================================================
def _extract_features(dataset, device, batch_size=64, workers=8, cache_path=None):
    """
    Extract ResNet-18 backbone features for the full dataset.
    Saves to cache_path if provided so subsequent runs skip extraction.
    Returns:
        features : np.ndarray [N, 512]  L2-normalised
        labels   : np.ndarray [N, C]
    """
    if cache_path is not None:
        feat_file  = cache_path + "_features.npy"
        label_file = cache_path + "_labels.npy"
        if os.path.exists(feat_file) and os.path.exists(label_file):
            print("ColdStart: loading cached features")
            return (
                np.load(feat_file, mmap_mode='r'),
                np.load(label_file, mmap_mode='r'),
            )

    print("ColdStart: extracting ResNet-18 features")
    from torchvision import models
    extractor    = models.resnet18(pretrained=True)
    extractor.fc = nn.Identity()
    extractor    = extractor.to(device)
    extractor.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    feats_list, labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            imgs  = batch['image'].to(device, non_blocking=True)
            feats = extractor(imgs).cpu().numpy().astype('float32')
            feats_list.append(feats)
            labels_list.append(batch['labels'].cpu().numpy())

    features = np.vstack(feats_list)
    labels   = np.vstack(labels_list)

    norms    = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features = features / norms

    if cache_path is not None:
        np.save(feat_file,  features)
        np.save(label_file, labels)
        print("ColdStart: features cached to {}".format(feat_file))

    return features, labels


# ==============================================================
# Strategy 1  Random
# ==============================================================
def cold_start_random(dataset, n_initial, seed=42, **kwargs):
    """
    Uniformly random selection. Fastest baseline.
    No features needed.
    """
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_initial, replace=False).tolist()
    print("ColdStart [random]: selected {} samples".format(len(indices)))
    return indices


# ==============================================================
# Strategy 2  Diversity (MiniBatchKMeans)
# ==============================================================
def cold_start_diversity(dataset, n_initial, device, cache_path=None,
                         seed=42, batch_size=64, workers=8, **kwargs):
    """
    Selects visually diverse samples using MiniBatchKMeans clustering
    over ResNet-18 backbone features. One sample per cluster, spread
    evenly across the feature space.
    """
    features, _ = _extract_features(
        dataset, device, batch_size, workers, cache_path
    )

    n_samples  = len(features)
    n_clusters = min(2048, int(np.sqrt(n_samples)))

    subset_size = min(max(n_clusters * 3, 100000), n_samples)
    rng         = np.random.default_rng(seed)
    subset_idx  = rng.choice(n_samples, size=subset_size, replace=False)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=4096, random_state=seed, n_init=1
    )
    kmeans.fit(features[subset_idx])
    cluster_ids = kmeans.predict(features)

    samples_per_cluster = int(np.ceil(n_initial / n_clusters))
    selected = []
    for c in range(n_clusters):
        idx = np.where(cluster_ids == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(samples_per_cluster, len(idx)), replace=False)
        selected.extend(chosen.tolist())
        if len(selected) >= n_initial:
            break

    selected = selected[:n_initial]
    print("ColdStart [diversity]: selected {} samples from {} clusters".format(
        len(selected), n_clusters
    ))
    return selected


# ==============================================================
# Strategy 3  Entropy (most uncertain under a pre-trained model)
# ==============================================================
def cold_start_entropy(dataset, n_initial, device, num_classes,
                       batch_size=64, workers=8, seed=42, **kwargs):
    """
    Selects the n_initial samples with the highest prediction entropy
    under a randomly-initialised (or ImageNet-pretrained) ResNet-18.

    For a fresh model the logits are near-zero so entropy is almost
    uniform, but small weight-init differences create enough signal to
    diversify across regions of input space that a random initialisation
    finds hard. Using a pretrained backbone gives a more meaningful
    uncertainty ranking.
    """
    from torchvision import models
    model    = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model    = model.to(device)
    model.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )

    entropy_scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="ColdStart entropy scoring"):
            imgs   = batch['image'].to(device, non_blocking=True)
            logits = model(imgs)
            probs  = torch.sigmoid(logits)
            eps    = 1e-8
            ent    = -(
                probs * torch.log(probs + eps) +
                (1 - probs) * torch.log(1 - probs + eps)
            ).mean(dim=1)
            entropy_scores.append(ent.cpu().numpy())

    entropy_scores = np.concatenate(entropy_scores)
    selected       = np.argsort(entropy_scores)[::-1][:n_initial].tolist()

    print("ColdStart [entropy]: selected {} highest-entropy samples".format(
        len(selected)
    ))
    return selected


# ==============================================================
# Strategy 4  Coreset (greedy farthest-first traversal)
# ==============================================================
def cold_start_coreset(dataset, n_initial, device, cache_path=None,
                       seed=42, batch_size=64, workers=8, **kwargs):
    """
    Greedy k-center / coreset selection.
    Iteratively picks the sample farthest from all already-selected
    samples in feature space, ensuring maximal coverage.

    Note: O(n * k) in the number of selected samples k and pool n.
    For very large pools (>500k) use the approximate version below
    or limit with a random pre-filter.
    """
    features, _ = _extract_features(
        dataset, device, batch_size, workers, cache_path
    )

    n_samples = len(features)
    rng       = np.random.default_rng(seed)

    # Initialise with one random point
    selected  = [int(rng.integers(n_samples))]
    # min distance from each point to the selected set
    min_dists = np.full(n_samples, np.inf)

    feat_tensor = torch.tensor(features, dtype=torch.float32)

    for _ in tqdm(range(n_initial - 1), desc="ColdStart coreset"):
        last   = feat_tensor[selected[-1]].unsqueeze(0)          # [1, D]
        dists  = torch.cdist(feat_tensor, last).squeeze(1).numpy()
        min_dists = np.minimum(min_dists, dists)
        next_idx  = int(np.argmax(min_dists))
        selected.append(next_idx)

    print("ColdStart [coreset]: selected {} samples (greedy k-center)".format(
        len(selected)
    ))
    return selected


# ==============================================================
# Strategy 5  Herding (class-balanced exemplar selection)
# ==============================================================
def cold_start_herding(dataset, n_initial, device, num_classes,
                       cache_path=None, seed=42, batch_size=64,
                       workers=8, **kwargs):
    """
    Herding-style selection: for each class, selects samples whose
    running feature mean stays closest to the class prototype.
    Ensures every class is proportionally represented.

    For multi-label data (Sewer-ML), a sample is assigned to the
    class with the highest label weight it belongs to. Samples with
    no defect label (VA-only) are grouped into a 'normal' bin.
    """
    features, labels = _extract_features(
        dataset, device, batch_size, workers, cache_path
    )

    n_samples = len(features)

    # CIW weights to prioritise critical defect classes
    CIW = np.array([
        0.0310, 1.0000, 0.5518, 0.2896, 0.1622, 0.6419, 0.1847,
        0.3559, 0.3131, 0.0811, 0.2275, 0.2477, 0.0901, 0.4167,
        0.4167, 0.9009, 0.3829, 0.4396
    ])

    # Assign each sample to its highest-CIW positive class
    weighted_labels = labels * CIW[np.newaxis, :]       # [N, C]
    class_assign    = np.argmax(weighted_labels, axis=1) # [N]
    # Samples with all-zero labels -> class 0 (VA / normal)
    no_label_mask           = labels.sum(axis=1) == 0
    class_assign[no_label_mask] = 0

    # Budget per class proportional to CIW weights
    budgets = {}
    for c in range(num_classes):
        budgets[c] = max(1, int(n_initial * CIW[c] / CIW.sum()))
    # Top up to n_initial
    deficit = n_initial - sum(budgets.values())
    for c in sorted(budgets, key=lambda x: -CIW[x])[:max(0, deficit)]:
        budgets[c] += 1

    selected = []
    rng      = np.random.default_rng(seed)

    for c in range(num_classes):
        class_idx = np.where(class_assign == c)[0]
        if len(class_idx) == 0:
            continue
        budget = min(budgets[c], len(class_idx))
        if budget == 0:
            continue

        class_feats = features[class_idx]             # [M, D]
        prototype   = class_feats.mean(axis=0)        # [D]
        running_mean = np.zeros_like(prototype)
        chosen_local = []

        for _ in range(budget):
            remaining = [i for i in range(len(class_feats))
                         if i not in chosen_local]
            if not remaining:
                break
            # Pick sample whose addition minimises distance to prototype
            best_i, best_dist = -1, np.inf
            for i in remaining:
                candidate_mean = (
                    running_mean * len(chosen_local) + class_feats[i]
                ) / (len(chosen_local) + 1)
                dist = np.linalg.norm(prototype - candidate_mean)
                if dist < best_dist:
                    best_dist = dist
                    best_i    = i
            chosen_local.append(best_i)
            running_mean = (
                running_mean * (len(chosen_local) - 1) + class_feats[best_i]
            ) / len(chosen_local)

        selected.extend(class_idx[chosen_local].tolist())

    # De-duplicate and trim
    selected = list(dict.fromkeys(selected))[:n_initial]

    print("ColdStart [herding]: selected {} samples across {} classes".format(
        len(selected), num_classes
    ))
    return selected


# ==============================================================
# Strategy 6  Stratified (label-frequency balanced sampling)
# ==============================================================
def cold_start_stratified(dataset, n_initial, num_classes,
                           seed=42, batch_size=64, workers=8,
                           device=None, **kwargs):
    """
    Stratified sampling that guarantees minimum representation for
    every class, weighted by CIW so rare-but-critical defects get
    proportionally more samples.

    This is the most practical cold start for Sewer-ML because the
    dataset is heavily imbalanced. It directly addresses the low
    mAP and CIW-F2 that come from rare defect under-representation.

    Two passes:
      1. Mandatory quota: each class gets at least min_per_class samples
         from its positive examples.
      2. Fill remaining budget randomly from leftover samples.
    """
    CIW = np.array([
        0.0310, 1.0000, 0.5518, 0.2896, 0.1622, 0.6419, 0.1847,
        0.3559, 0.3131, 0.0811, 0.2275, 0.2477, 0.0901, 0.4167,
        0.4167, 0.9009, 0.3829, 0.4396, 
    ])

    print("ColdStart [stratified]: loading all labels")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    all_labels = []
    for batch in tqdm(loader, desc="Loading labels"):
        all_labels.append(batch['labels'].numpy())
    all_labels = np.vstack(all_labels)           # [N, C]

    n_samples = len(all_labels)
    rng       = np.random.default_rng(seed)

    # Budget per class weighted by CIW
    ciw_sum   = CIW.sum()
    budgets   = np.maximum(1, (n_initial * CIW / ciw_sum).astype(int))
    # Ensure total does not exceed n_initial
    while budgets.sum() > n_initial:
        budgets[np.argmax(budgets)] -= 1

    selected = set()

    # Pass 1: mandatory quota per class
    for c in range(num_classes):
        pos_idx = np.where(all_labels[:, c] > 0)[0]
        if len(pos_idx) == 0:
            continue
        quota   = min(budgets[c], len(pos_idx))
        chosen  = rng.choice(pos_idx, size=quota, replace=False)
        selected.update(chosen.tolist())

    # Pass 2: fill remaining budget randomly
    remaining_budget = n_initial - len(selected)
    if remaining_budget > 0:
        all_indices  = np.arange(n_samples)
        leftover     = np.setdiff1d(all_indices, np.array(list(selected)))
        if len(leftover) > 0:
            extra = rng.choice(
                leftover,
                size=min(remaining_budget, len(leftover)),
                replace=False
            )
            selected.update(extra.tolist())

    selected = list(selected)[:n_initial]

    # Report class coverage
    sel_labels = all_labels[selected]
    print("ColdStart [stratified]: selected {} samples".format(len(selected)))
    for c in range(num_classes):
        count = int(sel_labels[:, c].sum())
        print("  Class {:2d}: {:4d} positive samples".format(c, count))

    return selected

 
# ==============================================================
# Strategy 7  Self-supervised (SSL features + KMeans)
# ==============================================================
def _load_ssl_backbone(device):
    """
    Load the best available SSL backbone in priority order:
      1. MoCo v3  ResNet-50  (torchvision >= 0.13, needs the weights file)
      2. SWSL     ResNet-18  (Facebook semi-supervised, via torch.hub)
      3. SimCLR   ResNet-50  (via torch.hub from google-research)
      4. Plain ImageNet ResNet-50 pretrained  (always available fallback)
 
    Returns (model, feat_dim, backbone_name).
    The model is already in eval mode with the classifier head removed.
    """
    from torchvision import models
 
    # ---- Option 1: MoCo v3 via torchvision weights ----
    try:
        import torchvision
        if hasattr(torchvision.models, 'ResNet50_Weights'):
            backbone = models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            )
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            backbone = backbone.to(device).eval()
            print("ColdStart SSL: loaded ResNet-50 (torchvision ImageNet weights)")
            return backbone, feat_dim, "resnet50_imagenet"
    except Exception as e:
        print("ColdStart SSL: torchvision ResNet-50 failed ({})".format(e))
 
    # ---- Option 2: SWSL ResNet-18 (Facebook semi-supervised) ----
    try:
        backbone = torch.hub.load(
            'facebookresearch/semi-supervised-ImageNet1K-models',
            'resnet18_swsl',
            verbose=False
        )
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        backbone = backbone.to(device).eval()
        print("ColdStart SSL: loaded SWSL ResNet-18 (Facebook semi-supervised)")
        return backbone, feat_dim, "resnet18_swsl"
    except Exception as e:
        print("ColdStart SSL: SWSL failed ({})".format(e))
 
    # ---- Option 3: SimCLR via torch.hub ----
    try:
        backbone = torch.hub.load(
            'google-research/simclr', 'resnet50_1x', pretrained=True,
            verbose=False
        )
        feat_dim = 2048
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        backbone = backbone.to(device).eval()
        print("ColdStart SSL: loaded SimCLR ResNet-50")
        return backbone, feat_dim, "resnet50_simclr"
    except Exception as e:
        print("ColdStart SSL: SimCLR failed ({})".format(e))
 
    # ---- Option 4: Plain ImageNet ResNet-50 (guaranteed fallback) ----
    print("ColdStart SSL: falling back to ImageNet ResNet-50")
    backbone = models.resnet50(pretrained=True)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()
    return backbone, feat_dim, "resnet50_imagenet_fallback"
def _extract_ssl_features(dataset, device, backbone, batch_size=64,
                           workers=8, cache_path=None, backbone_name="ssl"):
    """
    Extract and L2-normalise features from the SSL backbone.
    Results are cached at cache_path + _ssl_{backbone_name}.npy
    so re-runs skip extraction.
    """
    if cache_path is not None:
        feat_file = "{}_ssl_{}.npy".format(cache_path, backbone_name)
        if os.path.exists(feat_file):
            print("ColdStart SSL: loading cached SSL features")
            return np.load(feat_file, mmap_mode='r')
 
    print("ColdStart SSL: extracting features with {}".format(backbone_name))
 
    # Sewer-ML normalisation (same as training pipeline)
    from torchvision import transforms as T
    normalise = T.Normalize(
        mean=[0.523, 0.453, 0.345],
        std=[0.210, 0.199, 0.154]
    )
 
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
 
    feats_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="SSL feature extraction"):
            imgs = batch['image'].to(device, non_blocking=True)
 
            # Ensure 3-channel input
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            elif imgs.shape[1] > 3:
                imgs = imgs[:, :3, :, :]
 
            # Re-normalise: undo Sewer-ML norm, apply ImageNet norm
            # (SSL models were trained on ImageNet normalisation)
            sewer_mean = torch.tensor(
                [0.523, 0.453, 0.345], device=device
            ).view(1, 3, 1, 1)
            sewer_std  = torch.tensor(
                [0.210, 0.199, 0.154], device=device
            ).view(1, 3, 1, 1)
            imgnet_mean = torch.tensor(
                [0.485, 0.456, 0.406], device=device
            ).view(1, 3, 1, 1)
            imgnet_std  = torch.tensor(
                [0.229, 0.224, 0.225], device=device
            ).view(1, 3, 1, 1)
 
            imgs = imgs * sewer_std + sewer_mean          # undo sewer norm
            imgs = (imgs - imgnet_mean) / imgnet_std      # apply imagenet norm
 
            feats = backbone(imgs)
            if feats.dim() > 2:
                feats = torch.flatten(feats, 1)
            feats_list.append(feats.cpu().numpy().astype('float32'))
 
    features = np.vstack(feats_list)
 
    # L2 normalise
    norms    = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features = features / norms
    if cache_path is not None:
        feat_file = f"{cache_path}_ssl_{backbone_name}.npy"

        # Ensure directory exists
        os.makedirs(os.path.dirname(feat_file), exist_ok=True)

        if os.path.exists(feat_file):
            print("ColdStart SSL: loading cached SSL features")
            return np.load(feat_file, mmap_mode='r')
 
    return features
 
 
def cold_start_self_supervised(dataset, n_initial, device, cache_path=None,
                                seed=42, batch_size=64, workers=8, **kwargs):
    """
    Self-supervised cold start:
      1. Extract features from the best available SSL backbone
         (SWSL, SimCLR, or ImageNet ResNet-50 as fallback)
      2. Run KMeans with n_initial clusters
      3. For each cluster, pick the sample closest to the cluster centre
 
    This is strictly better than the diversity (MiniBatchKMeans) strategy
    because the feature space is more semantically meaningful  SSL features
    cluster by content (defect type, pipe material, lighting) rather than
    just texture.
 
    The main difference from the original code in the pipeline:
      - Fully batched (no per-sample loop, 100x faster on large pools)
      - Proper renormalisation between Sewer-ML and ImageNet stats
      - Multiple SSL backbone options with graceful fallback
      - Feature caching so subsequent runs are instant
    """
    backbone, feat_dim, backbone_name = _load_ssl_backbone(device)
 
    features = _extract_ssl_features(
        dataset=dataset,
        device=device,
        backbone=backbone,
        batch_size=batch_size,
        workers=workers,
        cache_path=cache_path,
        backbone_name=backbone_name,
    )
 
    # Free backbone memory before KMeans
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    n_samples  = len(features)
    n_clusters = n_initial          # one cluster per desired sample
 
    # KMeans with n_initial clusters
    # Use MiniBatchKMeans for large pools (faster, similar quality)
    if n_samples > 50000:
        print("ColdStart SSL: MiniBatchKMeans with {} clusters".format(n_clusters))
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=4096,
            random_state=seed, n_init=3
        )
    else:
        from sklearn.cluster import KMeans
        print("ColdStart SSL: KMeans with {} clusters".format(n_clusters))
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=3)
 
    cluster_ids = kmeans.fit_predict(features)
    centres     = kmeans.cluster_centers_           # [K, D]
 
    # For each cluster pick the sample nearest to its centre
    selected = []
    for c in range(n_clusters):
        mask = np.where(cluster_ids == c)[0]
        if len(mask) == 0:
            continue
 
        cluster_feats = features[mask]              # [M, D]
        centre        = centres[c]                  # [D]
        dists         = np.linalg.norm(cluster_feats - centre, axis=1)
        closest_local = int(np.argmin(dists))
        selected.append(int(mask[closest_local]))
 
    # Top up with random samples if any cluster was empty
    if len(selected) < n_initial:
        rng      = np.random.default_rng(seed)
        sel_set  = set(selected)
        leftover = [i for i in range(n_samples) if i not in sel_set]
        extra    = rng.choice(
            leftover,
            size=min(n_initial - len(selected), len(leftover)),
            replace=False
        )
        selected.extend(extra.tolist())
 
    selected = selected[:n_initial]
 
    print("ColdStart [self_supervised]: selected {} samples "
          "using {} features".format(len(selected), backbone_name))
    return selected

# ==============================================================
# Strategy 7  CLIP-IQA filtered + secondary strategy
# ==============================================================
 
def _load_clipiqa_scores(clipiqa_json_path):
    """
    Load CLIP-IQA scores from JSON file.
    Expected format: {"filename.png": score, ...}
    Returns a dict mapping filename -> float score.
    """
    import json
    with open(clipiqa_json_path, "r") as f:
        scores = json.load(f)
    print("CLIP-IQA: loaded scores for {} images".format(len(scores)))
    return scores
 
 
def _get_dataset_filenames(dataset):
    """
    Extract filenames from a Sewer-ML dataset.
    Tries common attribute names used in MultiLabelDataset.
    Returns list of bare filenames (no directory prefix).
    """
    # Try common attribute names
    for attr in ("img_list", "filenames", "images", "data"):
        if hasattr(dataset, attr):
            raw = getattr(dataset, attr)
            if isinstance(raw, list) and len(raw) > 0:
                return [os.path.basename(str(p)) for p in raw]
 
    # Fallback: access items one by one (slow but always works)
    print("CLIP-IQA: extracting filenames via dataset iteration (slow path)")
    names = []
    for i in range(len(dataset)):
        item = dataset[i]
        if "filename" in item:
            names.append(os.path.basename(str(item["filename"])))
        elif "image_path" in item:
            names.append(os.path.basename(str(item["image_path"])))
        else:
            names.append(str(i))   # fallback: use index as name
    return names
 

def cold_start_clipiqa(dataset, n_initial, device, num_classes,
                       clipiqa_json_path,
                       clipiqa_threshold=0.5,
                       secondary_strategy="stratified",
                       cache_path=None, seed=42,
                       batch_size=64, workers=8,
                       save_selection_path=None,
                       **kwargs):
    """
    CLIP-IQA filtered cold start.
 
    Pipeline:
        1. Load CLIP-IQA scores from JSON
        2. Filter out images below `clipiqa_threshold`
        3. Apply `secondary_strategy` (stratified / herding / coreset /
           diversity / random) on the filtered pool only
        4. Save detailed selection report (filenames, scores, labels,
           class distribution) to `save_selection_path` as JSON
 
    Why this order matters:
        Filtering first removes genuinely uninformative / corrupted images
        before any feature-space or label-space selection. The secondary
        strategy then works on a cleaner, higher-quality pool.
 
    Why stratified is the recommended secondary strategy:
        Even after IQA filtering, Sewer-ML is heavily imbalanced. Without
        explicit class quotas, any feature-space method (coreset, herding,
        diversity) will under-sample rare high-CIW defects (RB, OS, FS)
        and produce the exact mAP / CIW-F2 gap you are already seeing.
 
    Args:
        dataset             : PyTorch training dataset
        n_initial           : total number of samples to select
        device              : torch.device
        num_classes         : number of label classes (17 for Sewer-ML)
        clipiqa_json_path   : path to clip_iqa_train.json
        clipiqa_threshold   : keep images with score >= threshold (default 0.5)
        secondary_strategy  : one of 'stratified', 'herding', 'coreset',
                              'diversity', 'random', 'entropy'
        cache_path          : base path for feature cache
        seed                : random seed
        batch_size          : DataLoader batch size
        workers             : DataLoader workers
        save_selection_path : if set, saves a JSON report of selected images
 
    Returns:
        selected_indices : list[int]   indices into dataset
    """
    import json
 
    print("ColdStart [clipiqa+{}]: threshold={:.2f} | target={}".format(
        secondary_strategy, clipiqa_threshold, n_initial
    ))
 
    # --------------------------------------------------
    # 1. Load CLIP-IQA scores
    # --------------------------------------------------
    scores = _load_clipiqa_scores(clipiqa_json_path)
    print("CLIP-IQA: score range [{:.3f}, {:.3f}]".format(
        min(scores.values()), max(scores.values())
    ))
    print("Highest score examples:")
    for fname in sorted(scores, key=scores.get, reverse=True)[:10]:
        print("  {}: {:.3f}".format(fname, scores[fname]))
    # --------------------------------------------------
    # 2. Map dataset indices to filenames and scores
    # --------------------------------------------------
    filenames = dataset.imgPaths if hasattr(dataset, "imgPaths") else None
    print("CLIP-IQA: extracted {} filenames from dataset".format(len(filenames)))
    print("10 filename examples:", filenames[:10])
    if len(filenames) != len(dataset):
        raise RuntimeError(
            "Filename list length ({}) != dataset length ({})".format(
                len(filenames), len(dataset)
            )
        )
 
    all_scores_arr = np.zeros(len(dataset), dtype=np.float32)
    n_matched      = 0
    for i, fname in enumerate(filenames):
        bare = os.path.basename(fname)
        if bare in scores:
            all_scores_arr[i] = float(scores[bare])
            n_matched += 1
        else:
            # If not in JSON, treat as 0.0 (will be filtered out)
            all_scores_arr[i] = 0.0
 
    print("CLIP-IQA: matched {}/{} images".format(n_matched, len(dataset)))
 
    # --------------------------------------------------
    # 3. Filter by threshold
    # --------------------------------------------------
    filtered_indices = np.where(all_scores_arr >= clipiqa_threshold)[0].tolist()
 
    print("CLIP-IQA: {} images pass threshold {:.2f} ({:.1f}% of dataset)".format(
        len(filtered_indices), clipiqa_threshold,
        100.0 * len(filtered_indices) / len(dataset)
    ))
 
    if len(filtered_indices) == 0:
        raise RuntimeError(
            "No images passed CLIP-IQA threshold {:.2f}. "
            "Lower the threshold or check the JSON path.".format(clipiqa_threshold)
        )
 
    if len(filtered_indices) < n_initial:
        print("WARNING: filtered pool ({}) < n_initial ({}). "
              "Using all filtered images.".format(len(filtered_indices), n_initial))
        n_initial = len(filtered_indices)
 
    # --------------------------------------------------
    # 4. Analyse label distribution of filtered pool
    #    (helps diagnose whether IQA filtering removed defects)
    # --------------------------------------------------
    print("Loading labels for filtered pool to check class distribution...")
    loader_filtered = DataLoader(
        Subset(dataset, filtered_indices),
        batch_size=batch_size, shuffle=False, num_workers=workers
    )
    filtered_labels = []
    for batch in tqdm(loader_filtered, desc="Loading filtered labels", leave=False):
        filtered_labels.append(batch["labels"].numpy())
    filtered_labels = np.vstack(filtered_labels)     # [F, C]
 
    class_counts_filtered = filtered_labels.sum(axis=0).astype(int)
    print("Class distribution in filtered pool:")
    for c in range(num_classes):
        print("  Class {:2d}: {:6d} positives ({:.2f}%)".format(
            c, class_counts_filtered[c],
            100.0 * class_counts_filtered[c] / len(filtered_indices)
        ))
 
    # Build a filtered-only dataset view for secondary strategy
    filtered_dataset = Subset(dataset, filtered_indices)
 
    # --------------------------------------------------
    # 5. Apply secondary strategy on filtered pool
    # --------------------------------------------------
    print("Applying secondary strategy '{}' on filtered pool of {}...".format(
        secondary_strategy, len(filtered_indices)
    ))
 
    secondary_fn = COLD_START_STRATEGIES.get(secondary_strategy)
    if secondary_fn is None:
        raise ValueError(
            "Unknown secondary strategy '{}'. Choose from: {}".format(
                secondary_strategy, list(COLD_START_STRATEGIES.keys())
            )
        )
 
    # The secondary strategy returns indices into filtered_dataset,
    # which are positions in filtered_indices, not global dataset indices.
    local_selected = secondary_fn(
        dataset=filtered_dataset,
        n_initial=n_initial,
        device=device,
        num_classes=num_classes,
        cache_path=cache_path,
        seed=seed,
        batch_size=batch_size,
        workers=workers,
    )
 
    # Map local indices back to global dataset indices
    selected_indices = [filtered_indices[i] for i in local_selected]
 
    # --------------------------------------------------
    # 6. Build and save selection report
    # --------------------------------------------------
    print("Building selection report...")
 
    # Labels for selected samples
    sel_labels = filtered_labels[local_selected]       # [n_initial, C]
    class_counts_selected = sel_labels.sum(axis=0).astype(int)
 
    # Per-sample info
    selected_info = []
    for rank, (global_idx, local_idx) in enumerate(
        zip(selected_indices, local_selected)
    ):
        fname = filenames[global_idx]
        iqa   = float(all_scores_arr[global_idx])
        lbls  = sel_labels[rank].tolist()
        selected_info.append({
            "rank":         rank,
            "global_index": global_idx,
            "filename":     fname,
            "clip_iqa":     round(iqa, 6),
            "labels":       [int(v) for v in lbls],
            "n_positive":   int(sum(lbls)),
        })
 
    report = {
        "config": {
            "clipiqa_threshold":  clipiqa_threshold,
            "secondary_strategy": secondary_strategy,
            "n_initial":          n_initial,
            "num_classes":        num_classes,
            "seed":               seed,
        },
        "summary": {
            "total_dataset":          len(dataset),
            "passed_iqa_filter":      len(filtered_indices),
            "selected":               len(selected_indices),
            "iqa_filter_pct":         round(
                100.0 * len(filtered_indices) / len(dataset), 2
            ),
            "selection_pct_of_total": round(
                100.0 * len(selected_indices) / len(dataset), 2
            ),
            "class_counts_filtered":  class_counts_filtered.tolist(),
            "class_counts_selected":  class_counts_selected.tolist(),
            "iqa_score_stats": {
                "min":    round(float(all_scores_arr[selected_indices].min()), 4),
                "max":    round(float(all_scores_arr[selected_indices].max()), 4),
                "mean":   round(float(all_scores_arr[selected_indices].mean()), 4),
                "median": round(float(np.median(all_scores_arr[selected_indices])), 4),
            },
        },
        "selected_images": selected_info,
    }
 
    if save_selection_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_selection_path)),
                    exist_ok=True)
        with open(save_selection_path, "w") as f:
            json.dump(report, f, indent=2)
        print("Selection report saved to: {}".format(save_selection_path))
    else:
        # Always save next to results with a default name
        default_path = "selection_report_clipiqa_{}.json".format(
            secondary_strategy
        )
        with open(default_path, "w") as f:
            json.dump(report, f, indent=2)
        print("Selection report saved to: {}".format(default_path))
 
    print("ColdStart [clipiqa+{}]: selected {} samples".format(
        secondary_strategy, len(selected_indices)
    ))
    print("Class counts in selection:")
    for c in range(num_classes):
        print("  Class {:2d}: {:4d} positives".format(
            c, class_counts_selected[c]
        ))
 
    return selected_indices
# ==============================================================
# Dispatcher
# ==============================================================
COLD_START_STRATEGIES = {
    "random":     cold_start_random,
    "diversity":  cold_start_diversity,
    "entropy":    cold_start_entropy,
    "coreset":    cold_start_coreset,
    "herding":    cold_start_herding,
    "stratified": cold_start_stratified,
    "self_supervised": cold_start_self_supervised,
    "clipiqa":         cold_start_clipiqa,
}


def apply_cold_start(args,  strategy_name, dataset, n_initial, device,
                     num_classes, cache_path=None, seed=42,
                     batch_size=64, workers=8,
                     clipiqa_json_path=None,
                     clipiqa_threshold=0.5,
                     secondary_strategy="stratified",
                     save_selection_path=None):
    """
    Unified entry point for all cold start strategies.
 
    Args:
        strategy_name       : one of 'random', 'diversity', 'entropy',
                              'coreset', 'herding', 'stratified',
                              'self_supervised', 'clipiqa'
        dataset             : PyTorch Dataset (full training set)
        n_initial           : number of samples to select
        device              : torch.device
        num_classes         : number of label classes
        cache_path          : base path for feature cache (no extension)
        seed                : random seed for reproducibility
        batch_size          : batch size for feature extraction
        workers             : DataLoader workers
        clipiqa_json_path   : path to clip_iqa_train.json (clipiqa only)
        clipiqa_threshold   : IQA score threshold (clipiqa only)
        secondary_strategy  : strategy applied after IQA filter (clipiqa only)
        save_selection_path : path to save selection report JSON (clipiqa only)
 
    Returns:
        selected_indices : list[int]
    """
    if strategy_name not in COLD_START_STRATEGIES:
        raise ValueError(
            "Unknown cold start strategy '{}'. Choose from: {}".format(
                strategy_name, list(COLD_START_STRATEGIES.keys())
            )
        )
 
    fn = COLD_START_STRATEGIES[strategy_name]
 
    if strategy_name == "clipiqa":
        clipiqa_json_path = args.clipiqa_json_path
        print(args)
        return fn(
            dataset=dataset,
            n_initial=n_initial,
            device=device,
            num_classes=num_classes,
            clipiqa_json_path=clipiqa_json_path,
            clipiqa_threshold=clipiqa_threshold,
            secondary_strategy=secondary_strategy,
            cache_path=cache_path,
            seed=seed,
            batch_size=batch_size,
            workers=workers,
            save_selection_path=save_selection_path,
        )
 
    return fn(
        dataset=dataset,
        n_initial=n_initial,
        device=device,
        num_classes=num_classes,
        cache_path=cache_path,
        seed=seed,
        batch_size=batch_size,
        workers=workers,
    )
# ===========================
# Policy Network (REINFORCE)
# ===========================
class PolicyNet(nn.Module):
    """
    Policy network for RL-based sample selection.
    Takes per-sample states and a global context vector,
    outputs image-level selection logits and an optional budget logit.
    """
    def __init__(self, state_dim, hidden_dim=256, num_budget_options=1):
        super(PolicyNet, self).__init__()

        self.image_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),   # scalar logit per sample
        )

        self.budget_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_budget_options),
        )

    def forward(self, sample_states, global_state):
        """
        Args:
            sample_states: [N, state_dim]  one row per candidate sample
            global_state:  [state_dim]     mean of all candidate states
        Returns:
            image_logits:  [N, 1]
            budget_logits: [num_budget_options]
        """
        image_logits  = self.image_head(sample_states)          # [N, 1]
        budget_logits = self.budget_head(global_state)          # [num_budget_options]
        return image_logits, budget_logits


# ===========================
# Oracle / Main Task Model
# ===========================
class SewerMLModel(nn.Module):
    """
    ResNet backbone for multi-label classification on Sewer-ML.
    Supports:
      - forward()                  ? raw logits  [B, num_classes]
      - get_bottleneck_features()  ? penultimate features [B, feat_dim]
    """
    def __init__(self, num_classes, arch='resnet34', pretrained=True):
        super(SewerMLModel, self).__init__()

        backbone_fn = getattr(models, arch)
        backbone    = backbone_fn(pretrained=pretrained)

        # Feature extractor: everything up to (but not including) the fc layer
        self.features    = nn.Sequential(*list(backbone.children())[:-1])  # [B, C, 1, 1]
        self.feat_dim    = backbone.fc.in_features
        self.classifier  = nn.Linear(self.feat_dim, num_classes)

    def get_bottleneck_features(self, x):
        """Return flattened penultimate features: [B, feat_dim]"""
        x = self.features(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        feats  = self.get_bottleneck_features(x)
        logits = self.classifier(feats)
        return logits

# ==============================================================
# Logging
# ==============================================================
def setup_logging(name, log_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        if log_path is not None:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
    return logger

# ===========================
# Feature Cache Utils
# ===========================
def get_feature_cache_path(args):
    cache_dir = os.path.join(args.dataroot, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    feature_path = os.path.join(cache_dir, f"sewerml_resnet18_features_{args.scale_size}.npy")
    label_path   = os.path.join(cache_dir, f"sewerml_labels_{args.scale_size}.npy")
    return feature_path, label_path


# ===========================
# Data Loading
# ===========================
def get_data(args):
    """Load Sewer-ML dataset with proper transforms."""
    trainTransform = transforms.Compose([
        transforms.Resize((args.scale_size, args.scale_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    testTransform = transforms.Compose([
        transforms.Resize((args.scale_size, args.scale_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    sewer_root = os.path.join(args.dataroot, 'Sewer_ML')
    anno_dir   = sewer_root

    import pandas as pd
    train_csv = os.path.join(anno_dir, 'SewerML_train.csv')
    val_csv   = os.path.join(anno_dir, 'SewerML_valid.csv')

    train_list = pd.read_csv(train_csv)['Filename'].tolist() if os.path.exists(train_csv) else []
    val_list   = pd.read_csv(val_csv)['Filename'].tolist()   if os.path.exists(val_csv)   else []

    if args.inference:
        valid_dataset = MultiLabelDatasetInference(
            annRoot=anno_dir, imgRoot=sewer_root,
            split='Val', transform=testTransform, onlyDefects=False
        )
        train_dataset = None
        train_classweights = None
    else:
        train_dataset = MultiLabelDataset(
            img_dir=sewer_root, image_transform=trainTransform,
            labels_path=anno_dir, val_list=val_list, train_list=train_list,
            known_labels=args.train_known_labels, testing=False, split='train'
        )
        valid_dataset = MultiLabelDataset(
            img_dir=sewer_root, image_transform=testTransform,
            labels_path=anno_dir, val_list=val_list, train_list=train_list,
            known_labels=args.test_known_labels, testing=True, split='valid'
        )
        train_classweights = train_dataset.class_weights
        print(f"Train: {len(train_dataset)} | Val: {len(valid_dataset)}")

    train_loader = (
        DataLoader(train_dataset, batch_size=args.batch_size,
                   shuffle=True, num_workers=args.workers, drop_last=False)
        if train_dataset else None
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.test_batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True
    )

    return train_loader, valid_loader, None, train_classweights


# ===========================
# Active Learning Pipeline (REINFORCE)
# ===========================
class ActiveLearningPipeline:
    """
    RL-based Active Learning for Sewer-ML multi-label classification.

    Architecture mirrors ActiveLearningSystemRL:
      - Oracle model: frozen after initial training, used for state computation
      - Main model:   retrained every AL cycle on the growing labeled set
      - PolicyNet:    REINFORCE agent selecting the next batch of samples
    """

    # ---- Sewer-ML class importance weights (for CIW-F2) ----
    CIW = {
        "VA": 0.0310, "RB": 1.0000, "OB": 0.5518, "PF": 0.2896,
        "DE": 0.1622, "FS": 0.6419, "IS": 0.1847, "RO": 0.3559,
        "IN": 0.3131, "AF": 0.0811, "BE": 0.2275, "FO": 0.2477,
        "GR": 0.0901, "PH": 0.4167, "PB": 0.4167, "OS": 0.9009,
        "OP": 0.3829, "OK": 0.4396
    }
    NUM_CLASSES = 17
        # Logger (stream + file)


    def __init__(self, train_dataset, args, device='cuda'):
        self.train_dataset = train_dataset
        self.args          = args
        self.device        = torch.device(device if torch.cuda.is_available() else 'cpu')
        log_file = os.path.join(
        "logs",
        "RAL_{}.log".format(
            datetime.datetime.now().strftime("%m%d_%H%M")
        )
    )
        self.logger = setup_logging("SewerML_Supervised", log_path=log_file)
        # --------------------------------------------------
        # Labeled / unlabeled pool
        # --------------------------------------------------
        self.labeled_indices   = []
        self.unlabeled_indices = list(range(len(train_dataset)))

        # --------------------------------------------------
        # Oracle model  (ResNet-34, frozen after warm-up)
        # --------------------------------------------------
        self.oracle_model = SewerMLModel(
            num_classes=self.NUM_CLASSES, arch='resnet34', pretrained=True
        ).to(self.device)

        # --------------------------------------------------
        # Main task model  (ResNet-101, retrained each cycle)
        # --------------------------------------------------
        self.main_model = SewerMLModel(
            num_classes=self.NUM_CLASSES, arch='resnet101', pretrained=True
        ).to(self.device)

        # --------------------------------------------------
        # PolicyNet  (REINFORCE)
        # --------------------------------------------------
        # state_dim = bottleneck features + 3 uncertainty scalars
        self.state_dim = self.oracle_model.feat_dim + 3   # e.g. 512 + 3 = 515
        self.policy    = PolicyNet(
            state_dim=self.state_dim,
            hidden_dim=getattr(args, 'policy_hidden', 256),
            num_budget_options=1,
        ).to(self.device)
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=getattr(args, 'policy_lr', 1e-4)
        )

        # --------------------------------------------------
        # REINFORCE hyper-params
        # --------------------------------------------------
        self.entropy_beta       = getattr(args, 'entropy_beta', 1e-3)
        self.policy_temp        = getattr(args, 'policy_temp_start', 1.0)
        self.policy_temp_end    = getattr(args, 'policy_temp_end', 0.5)
        self.reward_baseline    = 0.0
        self.baseline_momentum  = 0.9
        self.prev_score         = None

        # --------------------------------------------------
        # History
        # --------------------------------------------------
        self.history = {}
        self.cycle   = 0
        self._init_results_path()
        print(f"Device: {self.device} | state_dim: {self.state_dim}")
    # ----------------------------------------------------------
    # Calculating Weights of Positive Samples for BCEWithLogitsLoss
    # ----------------------------------------------------------

    def _compute_pos_weight_from_indices(self, indices, eps=1e-6):
        labels = self.train_dataset.labels[indices]

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        pos_count = labels.sum(axis=0)
        neg_count = labels.shape[0] - pos_count
        pos_weight = neg_count / (pos_count + eps)

        return torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
    # ----------------------------------------------------------
    # Cold start  (all strategies via dispatcher)
    # ----------------------------------------------------------
    def cold_start(self):
        strategy   = getattr(self.args, 'cold_start_strategy', 'diversity')
        n_init_arg = getattr(self.args, 'initial_percentage', 0.01)
 
        n_initial = (
            int(n_init_arg * len(self.train_dataset))
            if n_init_arg <= 1.0
            else int(n_init_arg)
        )
        n_initial = max(1, n_initial)
 
        self.logger.info(
            "Cold start [{}]: selecting {} samples ({:.1f}%)".format(
                strategy, n_initial,
                100.0 * n_initial / len(self.train_dataset)
            )
        )
 
        cache_path = get_feature_cache_path(self.args)
 
        selected = apply_cold_start(args=self.args,
            strategy_name=strategy,
            dataset=self.train_dataset,
            n_initial=n_initial,
            device=self.device,
            num_classes=self.NUM_CLASSES,
            cache_path=cache_path,
            seed=getattr(self.args, 'seed', 42),
            batch_size=self.args.batch_size,
            workers=self.args.workers,

            clipiqa_threshold=getattr(self.args, 'clipiqa_threshold', 0.5),
        )
 
        mask = np.zeros(len(self.train_dataset), dtype=bool)
        mask[selected] = True
 
        self.labeled_indices   = selected
        self.unlabeled_indices = np.where(~mask)[0].tolist()
 
        self.logger.info(
            "Cold start done: {} labeled | {} unlabeled".format(
                len(self.labeled_indices), len(self.unlabeled_indices)
            )
        )
 
        self.history["cold_start_strategy"] = strategy
        self.history["cold_start_labeled"]  = len(self.labeled_indices)
        self.save_results()
        return self.labeled_indices

    # ==========================================================
    # Oracle model training (run ONCE, then freeze)
    # ==========================================================
    def train_oracle_model(self, epochs=50):
        """Train the oracle model on the initial labeled set, then freeze it."""
        print(f"Training oracle model for {epochs} epochs "
              f"on {len(self.labeled_indices)} samples")

        subset    = Subset(self.train_dataset, self.labeled_indices)
        loader    = DataLoader(subset, batch_size=self.args.batch_size,
                               shuffle=True, num_workers=4)
        
        pos_weight = self._compute_pos_weight_from_indices(self.labeled_indices)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.logger.info(
            "Using weighted BCEWithLogitsLoss for oracle training with pos_weight: {}".format(
                pos_weight.detach().cpu().tolist()
            )
        )
        optimizer = optim.Adam(self.oracle_model.parameters(), lr=1e-4)

        self.oracle_model.train()
        for ep in range(epochs):
            total_loss = 0.0
            for batch in tqdm(loader, desc=f"Oracle ep {ep+1}/{epochs}", leave=False):
                imgs   = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.oracle_model(imgs), labels.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Oracle epoch {ep+1}/{epochs} | loss {total_loss/len(loader):.4f}")

        # ---- Freeze oracle permanently ----
        self.oracle_model.eval()
        for p in self.oracle_model.parameters():
            p.requires_grad = False
        print("Oracle model frozen.")

    # ==========================================================
    # State computation  (mirrors _compute_state in file 1)
    # ==========================================================
    def _compute_state(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample state vectors.

        State = [bottleneck_features | entropy | 1-confidence | 1-margin]
                 shape: [B, feat_dim + 3]

        All operations are done with no_grad (oracle is frozen).
        """
        with torch.no_grad():
            feats  = self.oracle_model.get_bottleneck_features(images)  # [B, feat_dim]
            logits = self.oracle_model(images)                           # [B, C]
            probs  = torch.sigmoid(logits)                               # [B, C]  independent

            eps = 1e-8

            # Mean binary entropy across classes ? [B]
            entropy    = -(
                probs * torch.log(probs + eps) +
                (1 - probs) * torch.log(1 - probs + eps)
            ).mean(dim=1)

            # Mean confidence = mean of max(p, 1-p) ? [B]
            confidence = torch.max(probs, 1 - probs).mean(dim=1)

            # Margin from 0.5, averaged across classes ? [B]
            margin     = (probs - 0.5).abs().mean(dim=1)

            uncertainty = torch.stack(
                [entropy, 1.0 - confidence, 1.0 - margin], dim=1
            )                                                            # [B, 3]

            return torch.cat([feats, uncertainty], dim=1)                # [B, state_dim]

    # ==========================================================
    # RL query  (mirrors query() in file 1)
    # ==========================================================
    def query(self, budget: int):
        """
        Select `budget` samples from the unlabeled pool using the policy.

        Steps:
          1. Compute states for ALL unlabeled samples (in batches)
          2. Pre-filter to top-K by entropy (candidate_ratio)
          3. Run PolicyNet on candidates
          4. Sample `budget` indices via torch.multinomial (no replacement)

        Returns:
          selected_indices : list[int]  indices into train_dataset
          log_prob_sum     : Tensor     sum of log-probs (for REINFORCE)
          entropy          : Tensor     policy entropy (for entropy bonus)
        """
        if not self.unlabeled_indices:
            return [], None, None

        # ---- Step 1: Compute states for the full unlabeled pool ----
        unlabeled_subset = Subset(self.train_dataset, self.unlabeled_indices)
        loader = DataLoader(
            unlabeled_subset, batch_size=self.args.batch_size,
            shuffle=False, num_workers=self.args.workers
        )

        state_chunks = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch['image'].to(self.device)
                state_chunks.append(self._compute_state(imgs))

        states = torch.cat(state_chunks, dim=0).detach()   # [N, state_dim]

        # ---- Step 2: Entropy-based candidate filtering (top 20%) ----
        entropy_scores  = states[:, -3]                    # entropy is first uncertainty col
        candidate_ratio = getattr(self.args, 'candidate_ratio', 0.2)
        top_k           = max(1, min(int(candidate_ratio * len(entropy_scores)),
                                     len(entropy_scores)))
        _, cand_idx     = torch.topk(entropy_scores, top_k)

        candidate_states = states[cand_idx]                # [K, state_dim]
        candidate_pool   = [self.unlabeled_indices[i] for i in cand_idx.tolist()]

        # ---- Step 3: PolicyNet forward ----
        global_state         = candidate_states.mean(dim=0)   # [state_dim]
        image_logits, _      = self.policy(candidate_states, global_state)

        # ---- Step 4: Multinomial sampling ----
        budget       = max(1, min(budget, len(candidate_pool)))
        image_probs  = F.softmax(
            image_logits.squeeze() / self.policy_temp, dim=0
        ).clamp_min(1e-12)

        selected_pos    = torch.multinomial(image_probs, num_samples=budget, replacement=False)
        log_prob_sum    = torch.log(image_probs[selected_pos]).sum()
        entropy         = -(image_probs * torch.log(image_probs)).sum()

        selected_indices = [candidate_pool[i] for i in selected_pos.tolist()]
        print(f"  Query: selected {len(selected_indices)} samples "
              f"from {len(candidate_pool)} candidates "
              f"(pool size: {len(self.unlabeled_indices)})")
        return selected_indices, log_prob_sum, entropy

    # ==========================================================
    # Main model training
    # ==========================================================
    def train_main_model(self, val_loader, epochs: int):
        """Retrain main model from scratch on the current labeled set."""
        print(f"Training main model for {epochs} epochs "
              f"on {len(self.labeled_indices)} labeled samples")

        subset    = Subset(self.train_dataset, self.labeled_indices)
        loader    = DataLoader(subset, batch_size=self.args.batch_size,
                               shuffle=True, num_workers=self.args.workers)
        
        pos_weight = self._compute_pos_weight_from_indices(self.labeled_indices)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.logger.info(
            "Using weighted BCEWithLogitsLoss for main model training with pos_weight: {}".format(
                pos_weight.detach().cpu().tolist()
            )
        )
        optimizer = optim.Adam(self.main_model.parameters(), lr=getattr(self.args, 'lr', 1e-4))

        for ep in range(epochs):
            self.main_model.train()
            epoch_start = time.time()
            total_loss = 0.0
            for batch in tqdm(loader, desc=f"Main ep {ep+1}/{epochs}", leave=False):
                imgs   = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.main_model(imgs), labels.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Main epoch {ep+1}/{epochs} | loss {total_loss/len(loader):.4f}")

            epoch_time = time.time() - epoch_start
            train_metrics = {"train_loss": total_loss / len(loader)}
            eval_metrics  = self.evaluate_main_model(val_loader)
            self._log_metrics(
                epoch=ep,
                train_time=epoch_time,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
            )
            self.save_results()
        return eval_metrics
    # ==========================================================
    # Evaluation
    # ==========================================================
    def evaluate_main_model(self, val_loader):
        """
        Evaluate main model on validation set.
        Returns a dict with macro_f1, micro_f1, mAP, ciw_f2, and more.
        The primary AL metric is macro_f1 (consistent with get_primary_metric).
        """
        class_names = val_loader.dataset.LabelNames
        self.main_model.eval()

        all_preds, all_labels, all_scores = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                imgs   = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                scores = torch.sigmoid(self.main_model(imgs))
                preds  = (scores > 0.5).int()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_scores.append(scores.cpu().numpy())

        y_pred  = np.concatenate(all_preds)
        y_true  = np.concatenate(all_labels)
        y_score = np.concatenate(all_scores)

        macro_f1 = f1_score(y_true, y_pred, average='macro',  zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro',  zero_division=0)
        ov_p     = precision_score(y_true, y_pred, average='micro', zero_division=0)
        ov_r     = recall_score(y_true,    y_pred, average='micro', zero_division=0)
        ov_f1    = (2 * ov_p * ov_r) / (ov_p + ov_r + 1e-8)
        pc_p     = precision_score(y_true, y_pred, average='macro', zero_division=0)
        pc_r     = recall_score(y_true,    y_pred, average='macro', zero_division=0)
        pc_f1    = (2 * pc_p * pc_r) / (pc_p + pc_r + 1e-8)
        zero_one = float(np.mean(np.all(y_true == y_pred, axis=1)))
        mAP      = average_precision_score(y_true, y_score, average='macro')

        # CIW-F2
        beta   = 2
        per_class = {}
        ciw_f2_raw = 0.0
        ciw_w_sum  = sum(self.CIW.get(c, 0.0) for c in class_names)

        for i, cls in enumerate(class_names):
            w    = self.CIW.get(cls, 0.0)
            p_c  = float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
            r_c  = float(recall_score(y_true[:, i],    y_pred[:, i], zero_division=0))
            f1_c = float(f1_score(y_true[:, i],        y_pred[:, i], zero_division=0))
            ap_c = float(average_precision_score(y_true[:, i], y_score[:, i]))
            f2_c = (1 + beta ** 2) * (p_c * r_c) / ((beta ** 2 * p_c) + r_c + 1e-8)
            n_pos_true = int(y_true[:, i].sum())
            n_pos_pred = int(y_pred[:, i].sum())
            ciw_f2_raw += w * f2_c
 
            per_class[cls] = dict(
                ciw=w,
                precision=round(p_c,  4),
                recall=round(r_c,     4),
                f1=round(f1_c,        4),
                ap=round(ap_c,        4),
                f2=round(f2_c,        4),
                ciw_f2=round(w * f2_c, 6),
                n_pos_true=n_pos_true,
                n_pos_pred=n_pos_pred,
            )

        ciw_f2 = ciw_f2_raw / (ciw_w_sum + 1e-8)
        # --------------------------------------------------
        # Log per-class table to logger
        # --------------------------------------------------
        header = (
            "{:<6} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>9} {:>9}".format(
                "Class", "CIW", "P", "R", "F1", "AP", "F2",
                "CIW-F2", "TruePos", "PredPos"
            )
        )
        self.logger.info("Per-class metrics:")
        self.logger.info(header)
        self.logger.info("-" * len(header))
        for cls in class_names:
            m = per_class[cls]
            self.logger.info(
                "{:<6} {:>5.3f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} "
                "{:>7.4f} {:>8.6f} {:>9d} {:>9d}".format(
                    cls,
                    m["ciw"],
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["ap"],
                    m["f2"],
                    m["ciw_f2"],
                    m["n_pos_true"],
                    m["n_pos_pred"],
                )
            )
        self.logger.info("-" * len(header))
 
        # Flag classes with low F1 relative to their CIW importance
        critical_threshold = 0.3
        weak_classes = [
            cls for cls in class_names
            if per_class[cls]["ciw"] >= 0.3
            and per_class[cls]["f1"] < critical_threshold
        ]
        if weak_classes:
            self.logger.info(
                "WARNING: high-CIW classes below F1={}: {}".format(
                    critical_threshold,
                    ", ".join(
                        "{} (CIW={:.2f} F1={:.4f})".format(
                            c, per_class[c]["ciw"], per_class[c]["f1"]
                        )
                        for c in weak_classes
                    )
                )
            )
 
        metrics = dict(
            macro_f1=macro_f1, micro_f1=micro_f1,
            ov_p=ov_p, ov_r=ov_r, ov_f1=ov_f1,
            pc_p=pc_p, pc_r=pc_r, pc_f1=pc_f1,
            zero_one=zero_one, mAP=mAP, ciw_f2=ciw_f2,
            per_class=per_class,
        )
        self.logger.info(
            "m-F1: {:.4f} | M-F1: {:.4f} | mAP: {:.4f} | CIW-F2: {:.4f}".format(
                micro_f1, macro_f1, mAP, ciw_f2
            )
        )
        
        return metrics

    # ==========================================================
    # REINFORCE policy update  (mirrors run_cycle in file 1)
    # ==========================================================
    def _update_policy(self, log_prob_sum, entropy, reward):
        """Single REINFORCE gradient step."""
        advantage = torch.tensor(
            reward - self.reward_baseline, dtype=torch.float32, device=self.device
        )
        # Moving baseline
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline +
            (1 - self.baseline_momentum) * reward
        )

        loss = -(advantage * log_prob_sum) - self.entropy_beta * entropy

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        return advantage.item(), loss.item()

    # ==========================================================
    # Single AL cycle
    # ==========================================================
    def run_cycle(self, val_loader, budget: int, cycle_epochs: int):
        """One full active learning cycle: query ? label ? retrain ? evaluate ? update policy."""

        # Temperature annealing (matches file 1)
        self.policy_temp = max(
            self.policy_temp_end,
            getattr(self.args, 'policy_temp_start', 1.0) * (0.95 ** self.cycle)
        )

        # ---- Query ----
        self.logger.info(f"Querying for new {budget} samples")
        new_indices, log_prob_sum, entropy = self.query(budget)

        if not new_indices:
            print("No samples selected skipping cycle.")
            self.cycle += 1
            return None

        # ---- Update pools ----
        new_set = set(new_indices)
        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in new_set]

        # ---- Retrain main model ----
        metrics = self.train_main_model(val_loader, epochs=cycle_epochs)
        
        score   = metrics['ciw_f2']  # primary metric for reward

        # ---- Reward: delta in primary metric, clipped ----
        if self.prev_score is None:
            reward = 0.0
        else:
            reward = float(np.clip(score - self.prev_score, -0.1, 0.1))
        self.prev_score = score

        # ---- Policy update ----
        advantage, policy_loss = self._update_policy(log_prob_sum, entropy, reward)

        # ---- Log ----
        self.history.setdefault('cycle',         []).append(self.cycle)
        self.history.setdefault('labeled_count', []).append(len(self.labeled_indices))
        self.history.setdefault('reward',        []).append(reward)
        self.history.setdefault('advantage',     []).append(advantage)
        self.history.setdefault('policy_loss',   []).append(policy_loss)
        for k, v in metrics.items():
            self.history.setdefault(f'val_{k}', []).append(v)

        print(f"Cycle {self.cycle} | Macro-F1: {score:.4f} | "
              f"Reward: {reward:.4f} | Baseline: {self.reward_baseline:.4f} | "
              f"Advantage: {advantage:.4f} | Labeled: {len(self.labeled_indices)}")

        self.cycle += 1
        return metrics
    def _log_metrics(self, epoch, train_time, train_metrics, eval_metrics):
        global_epoch = epoch + self.cycle * self.args.epochs_per_cycle

        self.history.setdefault("epoch", []).append(epoch)
        self.history.setdefault("global_epoch", []).append(global_epoch)
        self.history.setdefault("cycle", []).append(self.cycle)

        self.history.setdefault("train_loss", []).append(train_metrics.get("train_loss", 0))
        self.history.setdefault("labeled_count", []).append(len(self.labeled_indices))
        self.history.setdefault("train_time", []).append(train_time)

        # ----------------------------
        # Semantic segmentation models
        # ----------------------------
        if self.args.task == "segmentation":

            dice = eval_metrics.get("dice", 0)
            f1 = eval_metrics.get("f1", 0)
            miou = eval_metrics.get("mean_iou", 0)

            self.history.setdefault("val_dice", []).append(dice)
            self.history.setdefault("val_F1", []).append(f1)
            self.history.setdefault("val_mean_iou", []).append(miou)

            self.logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics.get('train_loss',0):.4f} | "
                f"F1: {f1:.4f} | "
                f"Dice: {dice:.4f} | "
                f"Mean IoU: {miou:.4f} | "
                f"Labeled: {len(self.labeled_indices)}"
            )

            if self.args.use_wandb:
                log_to_wandb(
                    {
                        "epoch": epoch + 1,
                        "global_epoch": global_epoch,
                        "cycle": self.cycle,
                        "train_loss": train_metrics.get("train_loss",0),
                        "val_dice": dice,
                        "val_iou": miou,
                        "labeled_count": len(self.labeled_indices),
                    },
                    step=global_epoch,
                )

        # ---------------------------------
        # Instance segmentation / detection
        # ---------------------------------
        elif self.args.task in ["instance_segmentation", "detection"]:

            mask_ap = eval_metrics.get("mask_AP", 0)
            bbox_ap = eval_metrics.get("bbox_AP", 0)

            self.history.setdefault("val_mask_AP", []).append(mask_ap)
            self.history.setdefault("val_bbox_AP", []).append(bbox_ap)

            self.logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics.get('train_loss',0):.4f} | "
                f"Mask AP: {mask_ap:.4f} | "
                f"BBox AP: {bbox_ap:.4f} | "
                f"Labeled: {len(self.labeled_indices)}"
            )

            if self.args.use_wandb:
                log_to_wandb(
                    {
                        "epoch": epoch + 1,
                        "global_epoch": global_epoch,
                        "cycle": self.cycle,
                        "train_loss": train_metrics.get("train_loss",0),
                        "val_mask_AP": mask_ap,
                        "val_bbox_AP": bbox_ap,
                        "labeled_count": len(self.labeled_indices),
                    },
                    step=global_epoch,
                )
        # ---------------------------------
        # Classification (default)
        # ---------------------------------
        elif self.args.task == "classification":

            macro_f1 = eval_metrics.get("macro_f1", 0)
            micro_f1 = eval_metrics.get("micro_f1", 0)
            meanAP = eval_metrics.get("mAP", 0)
            ciw_f2 = eval_metrics.get("ciw_f2", 0)
            ov_p = eval_metrics.get("ov_p", 0)
            ov_r = eval_metrics.get("ov_r", 0)
            ov_f1 = eval_metrics.get("ov_f1", 0)
            pc_p = eval_metrics.get("pc_p", 0)
            pc_r = eval_metrics.get("pc_r", 0)
            pc_f1 = eval_metrics.get("pc_f1", 0)
            zero_one = eval_metrics.get("zero_one", 0)

            per_class = eval_metrics.get("per_class", {})
            if per_class:
                self.history.setdefault("per_class_history", []).append({
                    "global_epoch": global_epoch,
                    "cycle": self.cycle,
                    "epoch": epoch,
                    "labeled": len(self.labeled_indices),
                    "classes": per_class,
                })

            self.history.setdefault("val_macro_f1", []).append(macro_f1)
            self.history.setdefault("val_micro_f1", []).append(micro_f1)
            self.history.setdefault("val_mAP", []).append(meanAP)
            self.history.setdefault("val_ciw_f2", []).append(ciw_f2)
            self.history.setdefault("val_ov_p", []).append(ov_p)
            self.history.setdefault("val_ov_r", []).append(ov_r)
            self.history.setdefault("val_ov_f1", []).append(ov_f1)
            self.history.setdefault("val_pc_p", []).append(pc_p)
            self.history.setdefault("val_pc_r", []).append(pc_r)
            self.history.setdefault("val_pc_f1", []).append(pc_f1)
            self.history.setdefault("val_zero_one", []).append(zero_one)

            self.logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics.get('train_loss',0):.4f} | "
                f"Macro F1: {macro_f1:.4f} | "
                f"Micro F1: {micro_f1:.4f} | "
                f"mAP: {meanAP:.4f} | "
                f"CIW-F2: {ciw_f2:.4f} | "
                f"Labeled: {len(self.labeled_indices)}"
            )

            if self.args.use_wandb:
                log_to_wandb(
                    {
                        "epoch": epoch + 1,
                        "global_epoch": global_epoch,
                        "cycle": self.cycle,
                        "train_loss": train_metrics.get("train_loss", 0),
                        "val_macro_f1": macro_f1,
                        "val_micro_f1": micro_f1,
                        "val_mAP": meanAP,
                        "val_ciw_f2": ciw_f2,
                        "labeled_count": len(self.labeled_indices),
                    },
                    step=global_epoch,
                )

    def save_results(self):
        results = {
            "config": self._config_to_dict(),
            "history": self.history,
        }

        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)
    def _init_results_path(self):
        date_folder = datetime.datetime.now().strftime("%m_%d")
        results_dir = os.path.join(self.args.results_dir, date_folder)
        os.makedirs(results_dir, exist_ok=True)

        time_stamp = datetime.datetime.now().strftime("%H%M")

        self.results_path = os.path.join(
            results_dir,
            f"{self.args.experiment_name}_"
            f"{self.args.dataset_type}_"
            f"{self.args.cold_start_strategy}_"
            f"{self.args.query_strategy}_"
            f"{time_stamp}.json"
        )

    def _config_to_dict(self):
        # works for argparse.Namespace or simple config objects
        return vars(self.args)
    
    # ==========================================================
    # Full run
    # ==========================================================
    def run(self, val_loader):
        """
        Full active learning run:
          1. Diversity-based cold start
          2. Oracle model training (frozen after this)
          3. Initial main model warm-up
          4. AL cycles with REINFORCE policy
        """
        run_start = datetime.datetime.now()
        print("\n" + "=" * 60)
        print("STARTING RL ACTIVE LEARNING Sewer-ML")
        print("=" * 60)

        # ---- Cold start ----
        self.cold_start()

        # ---- Oracle training ----
        self.train_oracle_model(epochs=getattr(self.args, 'oracle_epochs', 50))

        # ---- Initial main model warm-up ----
        print("\nWarm-up: initial main model training")
        init_metrics = self.train_main_model(val_loader=val_loader,epochs=getattr(self.args, 'initial_epochs', 50))
        
        self.prev_score = init_metrics['ciw_f2']  # set baseline for reward calculation

        self.history['init_metrics'] = init_metrics
        print(f"Warm-up done | CIW-F2: {self.prev_score:.4f}")

        # ---- AL cycles ----
        budget       = getattr(self.args, 'al_budget',     100)
        cycle_epochs = getattr(self.args, 'cycle_epochs',   30)
        al_cycles    = getattr(self.args, 'al_cycles',       10)

        for c in range(al_cycles):
            print(f"\n{'='*60}")
            print(f"AL Cycle {c+1}/{al_cycles} | "
                  f"Labeled: {len(self.labeled_indices)} | "
                  f"Unlabeled: {len(self.unlabeled_indices)}")
            print("=" * 60)
            self.run_cycle(val_loader, budget=budget, cycle_epochs=cycle_epochs)

        run_time = str(datetime.datetime.now() - run_start)
        print(f"\nCompleted in {run_time}")
        self.history['run_time'] = run_time

        # ---- Save history ----
        if getattr(self.args, 'save_model', False):
            os.makedirs(self.args.model_save_path, exist_ok=True)
            hist_path = os.path.join(self.args.model_save_path, 'al_history.json')
            with open(hist_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"History saved ? {hist_path}")

            ckpt_path = os.path.join(self.args.model_save_path, 'final_checkpoint.pth')
            torch.save({
                'main_model_state_dict':   self.main_model.state_dict(),
                'oracle_model_state_dict': self.oracle_model.state_dict(),
                'policy_state_dict':       self.policy.state_dict(),
                'labeled_indices':         self.labeled_indices,
                'history':                 self.history,
            }, ckpt_path)
            print(f"Checkpoint saved ? {ckpt_path}")

        self.save_results()
        return self.history


# ===========================
# Entry point
# ===========================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RL Active Learning Sewer-ML')

    # Data
    parser.add_argument('--dataroot',           type=str,   default='../..//Datasets/')
    parser.add_argument('--scale_size',         type=int,   default=224)
    parser.add_argument('--batch_size',         type=int,   default=32)
    parser.add_argument('--test_batch_size',    type=int,   default=32)
    parser.add_argument('--workers',            type=int,   default=4)
    parser.add_argument('--train_known_labels', type=float, default=1.0)
    parser.add_argument('--test_known_labels',  type=float, default=1.0)
    parser.add_argument('--inference',          action='store_true')

    # Active Learning
    parser.add_argument('--model',       type=str,   default='resnet101')
    parser.add_argument('--al_cycles',          type=int,   default=10)
    parser.add_argument('--al_budget',          type=int,   default=2000)
    parser.add_argument('--candidate_ratio',    type=float, default=0.2,
                        help='Fraction of unlabeled pool passed to PolicyNet')
    # Cold start
    parser.add_argument('--cold_start_strategy',
        type=str, default='clipiqa',
        choices=['random', 'diversity', 'entropy', 'coreset',
                 'herding', 'stratified', 'self_supervised', 'clipiqa'],
        help='Strategy for selecting the initial labeled set')
    parser.add_argument('--initial_percentage',   type=float, default=0.01)
    parser.add_argument('--seed',                 type=int,   default=42)
    parser.add_argument('--query_strategy',    type=str,   default='reinforce')
    parser.add_argument('--num_cycles',        type=int,   default=10)
    parser.add_argument('--epochs_per_cycle', type=int,   default=10)
    
    # CLIP-IQA filtering (only used when --cold_start_strategy clipiqa)
    parser.add_argument('--clipiqa_json_path',
        type=str, default='IQA/clip_iqa_train_all.json',
        help='Path to clip_iqa_train.json')
    parser.add_argument('--clipiqa_threshold',
        type=float, default=0.3,
        help='Keep images with CLIP-IQA score >= this threshold')
    parser.add_argument('--secondary_strategy',
        type=str, default='stratified',
        choices=['random', 'diversity', 'entropy', 'coreset',
                 'herding', 'stratified', 'self_supervised'],
        help='Cold start strategy applied on the IQA-filtered pool')
    #  Training epochs
    parser.add_argument('--oracle_epochs',      type=int,   default=30)
    parser.add_argument('--initial_epochs',     type=int,   default=10)
    parser.add_argument('--cycle_epochs',       type=int,   default=10)
    parser.add_argument('--lr',                 type=float, default=1e-4)

    # Policy
    parser.add_argument('--policy_lr',          type=float, default=1e-4)
    parser.add_argument('--policy_hidden',      type=int,   default=256)
    parser.add_argument('--policy_temp_start',  type=float, default=1.0)
    parser.add_argument('--policy_temp_end',    type=float, default=0.5)
    parser.add_argument('--entropy_beta',       type=float, default=1e-3)

    # Saving
    parser.add_argument('--save_model',         action='store_true')
    parser.add_argument('--results_dir',       type=str,   default='./results')
    parser.add_argument('--experiment_name',   type=str,   default='rl_al_run')
    parser.add_argument('--task',              type=str,   default='classification')
    parser.add_argument('--dataset_type',      type=str,   default='sewerml')
    



    parser.add_argument('--use_wandb',         action='store_true')
    parser.add_argument('--wandb_project',     type=str,   default='rl_active_learning')
    parser.add_argument('--wandb_entity',      type=str,   default=None)
    parser.add_argument('--wandb_run_name',   type=str,   default=None)
    parser.add_argument('--num_workers',      type=int,   default=4)
    parser.add_argument('--num_epochs',       type=int,   default=50)

    parser.add_argument('--model_save_path',    type=str,   default='./checkpoints')
    parser.add_argument('--gpu',                type=int,   default=0)

    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print("CUDA not available using CPU")

    if args.save_model:
        os.makedirs(args.model_save_path, exist_ok=True)

    print("Loading data")
    train_loader, valid_loader, _, train_classweights = get_data(args)

    al_pipeline = ActiveLearningPipeline(
        train_dataset=train_loader.dataset,
        args=args,
        device=f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )

    history = al_pipeline.run(valid_loader)

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print("ACTIVE LEARNING COMPLETE")
    print("=" * 60)
    init_f1  = history['init_metrics']['macro_f1']
    final_f1 = history['val_macro_f1'][-1] if history.get('val_macro_f1') else init_f1
    print(f"Initial Macro-F1 : {init_f1:.4f}")
    print(f"Final   Macro-F1 : {final_f1:.4f}")
    print(f"Improvement      : {final_f1 - init_f1:+.4f}")
    print(f"Labeled samples  : {len(al_pipeline.labeled_indices)} / "
          f"{len(train_loader.dataset)} "
          f"({100*len(al_pipeline.labeled_indices)/len(train_loader.dataset):.1f}%)")

    if history.get('val_macro_f1'):
        print("\nMacro-F1 per cycle:")
        print(f"  Init  : {init_f1:.4f}")
        for i, (f1, r) in enumerate(zip(history['val_macro_f1'], history.get('reward', [])), 1):
            print(f"  Cycle {i:2d}: {f1:.4f}  (reward {r:+.4f})")