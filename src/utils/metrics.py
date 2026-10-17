"""
Standardized Evaluation Metrics Module.
Calculates Accuracy, Balanced Accuracy, Precision, Recall, Specificity,
FPR, FNR, Macro-F1, Weighted-F1, ROC-AUC, PR-AUC, MCC, Brier score,
and confusion matrix.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
    classification_report,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute full suite of binary evaluation metrics.
    Ensures safe evaluation even if one class is unpredicted.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Confusion matrix elements: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Fundamental rates
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))

    # Composite scores
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    attack_f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Rigorous explicit F2 calculation: (1 + 2^2) * (P * R) / (2^2 * P + R)
    # Never populated from FNR or any ambiguous fallback
    denom_f2 = (4.0 * precision + recall)
    f2_val = float((5.0 * precision * recall) / denom_f2) if denom_f2 > 0 else 0.0
    
    mcc = float(matthews_corrcoef(y_true, y_pred))

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "detection_rate": recall,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "attack_f1": attack_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "f2": f2_val,
        # Backward-compatible aliases
        "f1": attack_f1,
        "f1_score": macro_f1,
        "f2_score": f2_val,
        "matthews_corrcoef": mcc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
    }

    # Probability-based metrics
    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(float)
        # Ensure y_prob contains 1D probabilities of positive class
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.5

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = 0.0

        try:
            metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        except Exception:
            metrics["brier_score"] = 1.0

    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
) -> Dict[str, Any]:
    """Compute multiclass accuracy, macro/weighted F1, and per-class reports."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
    }
