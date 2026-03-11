import numpy as np

def labels_needed_for_target(row, target=0.95):

    labeled = row["labeled_curve"]
    f1 = row["F1_cycles"]

    if not labeled or not f1:
        return None

    full = max(f1)

    threshold = target * full

    for l, v in zip(labeled, f1):

        if v >= threshold:
            return l

    return None