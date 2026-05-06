"""
Metrics for binary RB classification.

Key metric = F2 (recall-weighted), because missing an RB defect is much
worse than a false alarm. This mirrors the CIW-F2 motivation from the
multi-label task.

Also reports:
    - AP / AUPRC       : threshold-free, good for tracking across cycles
    - Precision        : at 0.5 and at best-F2 threshold
    - Recall           : at 0.5 and at best-F2 threshold
    - F1               : at 0.5 and at best-F2 threshold
    - Positive rate    : fraction of samples predicted positive
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def binary_metrics(y_true, y_score, threshold=0.5, beta=2.0):
    """
    Compute a full report for binary classification.

    Args:
        y_true    : np.ndarray [N], 0/1 integer labels
        y_score   : np.ndarray [N], float probabilities in [0, 1]
        threshold : decision threshold for P/R/F1/F2
        beta      : beta for F-beta score (default 2 = F2)

    Returns:
        dict with fixed-threshold and best-F2-threshold metrics.
    """
    y_true   = np.asarray(y_true).astype(np.int32)
    y_score  = np.asarray(y_score).astype(np.float32)
    y_pred_t = (y_score >= threshold).astype(np.int32)

    n_pos_true = int(y_true.sum())
    n_pos_pred = int(y_pred_t.sum())

    # Threshold-free
    try:
        ap = float(average_precision_score(y_true, y_score))
    except ValueError:
        ap = 0.0
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = 0.0

    # At fixed threshold
    prec = float(precision_score(y_true, y_pred_t, zero_division=0))
    rec  = float(recall_score(y_true,    y_pred_t, zero_division=0))
    f1   = float(f1_score(y_true,        y_pred_t, zero_division=0))
    f2   = float(fbeta_score(y_true,     y_pred_t, beta=beta, zero_division=0))

    # Best-F2 threshold search over the PR curve
    best_f2, best_thresh, best_prec, best_rec = 0.0, threshold, prec, rec
    if n_pos_true > 0:
        precs, recs, threshs = precision_recall_curve(y_true, y_score)
        # fbeta per point on the curve
        beta2 = beta * beta
        with np.errstate(divide="ignore", invalid="ignore"):
            fbetas = (1 + beta2) * precs * recs / (beta2 * precs + recs + 1e-12)
        fbetas = np.nan_to_num(fbetas, nan=0.0)
        # threshs has length len(precs) - 1; align
        if len(threshs) > 0:
            idx = int(np.argmax(fbetas[:-1])) if len(fbetas) > 1 else 0
            best_f2     = float(fbetas[idx])
            best_thresh = float(threshs[idx])
            best_prec   = float(precs[idx])
            best_rec    = float(recs[idx])

    return {
        # Counts
        "n":            int(len(y_true)),
        "n_pos_true":   n_pos_true,
        "n_pos_pred":   n_pos_pred,
        "pos_rate_true": n_pos_true / max(1, len(y_true)),
        "pos_rate_pred": n_pos_pred / max(1, len(y_true)),
        # Threshold-free
        "AP":  ap,
        "AUC": auc,
        # At fixed threshold
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "f2":        f2,
        "threshold": threshold,
        # At best-F2 threshold
        "best_f2":           best_f2,
        "best_f2_threshold": best_thresh,
        "best_f2_precision": best_prec,
        "best_f2_recall":    best_rec,
    }


def format_metrics(m, prefix=""):
    """One-line human-readable summary."""
    return (
        "{pfx}AP: {AP:.4f} | AUC: {AUC:.4f} | "
        "F2@0.5: {f2:.4f} (P={precision:.4f} R={recall:.4f}) | "
        "best-F2: {best_f2:.4f} @ t={best_f2_threshold:.3f} "
        "(P={best_f2_precision:.4f} R={best_f2_recall:.4f}) | "
        "pos_true={n_pos_true} pos_pred={n_pos_pred}"
    ).format(pfx=prefix, **m)
