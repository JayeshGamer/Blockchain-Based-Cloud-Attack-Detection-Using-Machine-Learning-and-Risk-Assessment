"""
Unsupervised Anomaly Detector for Attack Tolerance Evidence.
Trained strictly on legitimate benign baseline telemetry.
Outputs normalized anomaly scores in [0, 1].
"""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score


class AnomalyEvidenceDetector:
    """
    Unsupervised Anomaly Detector generating normalized evidence scores in [0, 1].
    Trained strictly on benign network traffic to establish legitimate operational baselines.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        max_samples: int = 5000,
        random_seed: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_seed = random_seed

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        self.min_score_: float = 0.0
        self.max_score_: float = 1.0
        self.is_fitted_: bool = False

    def fit(self, X_benign: np.ndarray) -> "AnomalyEvidenceDetector":
        """
        Fit detector on legitimate baseline traffic only.
        Computes percentile anchors for robust [0, 1] normalization strictly from benign training data.
        """
        if len(X_benign) == 0:
            raise ValueError("Training data for anomaly detector cannot be empty.")

        actual_max_samples = min(self.max_samples, len(X_benign))
        self.model.max_samples = actual_max_samples
        self.model.fit(X_benign)

        # score_samples returns opposite of anomaly score (higher = normal, lower = anomalous)
        raw_benign_scores = -self.model.score_samples(X_benign)
        self.min_score_ = float(np.percentile(raw_benign_scores, 2))
        self.max_score_ = float(np.percentile(raw_benign_scores, 98))

        if self.max_score_ <= self.min_score_:
            self.max_score_ = self.min_score_ + 1.0

        self.is_fitted_ = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute normalized anomaly score in [0, 1].
        0 denotes typical benign traffic; 1 denotes extreme volumetric/statistical anomalies.
        """
        if not self.is_fitted_:
            return np.full(len(X), 0.5)

        raw_scores = -self.model.score_samples(X)
        norm_scores = (raw_scores - self.min_score_) / (self.max_score_ - self.min_score_)
        return np.clip(norm_scores, 0.0, 1.0)

    def evaluate_independently(self, X_eval: np.ndarray, y_eval: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluates anomaly detection independently on a validation or test set.
        Reports AUROC, PR-AUC, Recall, and FPR.
        """
        scores = self.score(X_eval)
        preds = (scores >= threshold).astype(int)

        y_true = np.asarray(y_eval, dtype=int)
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))

        recall = float(tp / max(1, n_pos))
        fpr = float(fp / max(1, n_neg))

        try:
            auroc = float(roc_auc_score(y_true, scores))
        except Exception:
            auroc = 0.5

        try:
            prauc = float(average_precision_score(y_true, scores))
        except Exception:
            prauc = 0.0

        return {
            "anomaly_auroc": round(auroc, 4),
            "anomaly_pr_auc": round(prauc, 4),
            "anomaly_recall": round(recall, 4),
            "anomaly_fpr": round(fpr, 4),
        }
