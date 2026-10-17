"""
Validation-Calibrated, Drift-Aware Adaptive Threshold Engine.
Complies with Phase 7 Rigor Requirements:
1. Calibrated strictly on validation split (never test labels).
2. Mode 1: Recall-Constrained (guarantees >= 90% Recall if feasible, minimizes FPR).
3. Mode 2: F2-score Optimization (penalizes False Negatives twice as heavily as False Positives).
4. Protected benign background tracking (prevents attack bursts from contaminating baseline).
5. Contamination protection & drift cooldown freezing.
6. Hard clamping bounds [theta_min, theta_max] and full decision audit logging.
"""

from collections import deque
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, fbeta_score


def optimize_threshold_on_validation(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    mode: str = "recall_constrained",
    min_recall: float = 0.90,
    beta: float = 2.0,
) -> Dict[str, Any]:
    """
    Determines the optimal baseline decision threshold using validation data only.
    Modes:
      - 'recall_constrained': Selects threshold achieving Recall >= min_recall with minimum FPR.
      - 'f_beta': Maximizes F-beta score (e.g., F2-score prioritizing recall over precision).
    """
    y_val = np.asarray(y_val, dtype=int)
    val_scores = np.asarray(val_scores, dtype=float)

    thresholds = np.linspace(0.05, 0.95, 181)
    best_thresh = 0.50
    best_metric = -1.0

    n_pos = np.sum(y_val == 1)
    n_neg = np.sum(y_val == 0)

    if mode == "recall_constrained":
        valid_candidates = []
        for t in thresholds:
            preds = (val_scores >= t).astype(int)
            tp = np.sum((preds == 1) & (y_val == 1))
            fp = np.sum((preds == 1) & (y_val == 0))
            recall = tp / max(1, n_pos)
            fpr = fp / max(1, n_neg)

            if recall >= min_recall:
                valid_candidates.append((t, fpr, recall))

        if valid_candidates:
            # Choose candidate that satisfies recall and minimizes FPR
            valid_candidates.sort(key=lambda x: (x[1], -x[2]))
            best_thresh = valid_candidates[0][0]
            best_metric = valid_candidates[0][2]
        else:
            # If 90% recall is unattainable, pick threshold maximizing recall
            max_rec = -1.0
            for t in thresholds:
                preds = (val_scores >= t).astype(int)
                rec = np.sum((preds == 1) & (y_val == 1)) / max(1, n_pos)
                if rec > max_rec:
                    max_rec = rec
                    best_thresh = t
            best_metric = max_rec

    elif mode == "f_beta":
        best_f = -1.0
        for t in thresholds:
            preds = (val_scores >= t).astype(int)
            f = fbeta_score(y_val, preds, beta=beta, zero_division=0)
            if f > best_f:
                best_f = f
                best_thresh = t
        best_metric = best_f

    # Evaluate performance at selected threshold
    final_preds = (val_scores >= best_thresh).astype(int)
    tp = int(np.sum((final_preds == 1) & (y_val == 1)))
    fp = int(np.sum((final_preds == 1) & (y_val == 0)))
    fn = int(np.sum((final_preds == 0) & (y_val == 1)))
    tn = int(np.sum((final_preds == 0) & (y_val == 0)))

    recall = tp / max(1, n_pos)
    fpr = fp / max(1, n_neg)
    fnr = fn / max(1, n_pos)

    # Establish validated operating bounds around optimal
    theta_min = max(0.10, float(best_thresh - 0.25))
    theta_max = min(0.90, float(best_thresh + 0.25))

    return {
        "mode": mode,
        "optimal_threshold": round(float(best_thresh), 4),
        "validation_recall": round(float(recall), 4),
        "validation_fpr": round(float(fpr), 4),
        "validation_fnr": round(float(fnr), 4),
        "theta_min": round(theta_min, 4),
        "theta_max": round(theta_max, 4),
    }


class AdaptiveThresholdManager:
    """
    Streaming Adaptive Threshold Manager with protected benign baseline.
    Prevents attack score contamination, relaxes upon concept drift, freezes during bursts.
    """

    def __init__(
        self,
        initial_threshold: float = 0.50,
        target_fpr: float = 0.02,
        rolling_window_size: int = 500,
        min_threshold: float = 0.15,
        max_threshold: float = 0.85,
        drift_relaxation_factor: float = 0.85,
        smoothing_alpha: float = 0.05,
        freeze_during_burst: bool = True,
    ):
        self.current_threshold = float(initial_threshold)
        self.target_fpr = target_fpr
        self.rolling_window_size = rolling_window_size
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.drift_relaxation_factor = float(drift_relaxation_factor)
        self.smoothing_alpha = float(smoothing_alpha)
        self.freeze_during_burst = freeze_during_burst

        # Buffer containing strictly high-confidence benign risk scores
        self.benign_risk_buffer: deque = deque(maxlen=self.rolling_window_size)
        self.drift_cooldown = 0
        self.threshold_history: deque = deque(maxlen=1000)

    def trigger_drift_relaxation(self) -> float:
        """Explicitly triggers drift relaxation on threshold."""
        self.drift_cooldown = 15
        self.current_threshold = float(np.clip(self.current_threshold * self.drift_relaxation_factor, self.min_threshold, self.max_threshold))
        return self.current_threshold

    def update(
        self,
        risk_scores: np.ndarray,
        drift_detected: bool = False,
        delayed_labels: Optional[np.ndarray] = None,
        step_index: int = 0,
        labels: Optional[np.ndarray] = None,
    ) -> float:
        """
        Updates the adaptive threshold using only past information.
        Adheres to contamination protection:
        - Only admits samples into the benign buffer if verified benign via delayed labels
          or with risk score below a safe benign ceiling.
        - Freezes updates during severe attack bursts (e.g., >35% attack ratio).
        - Relaxes threshold upon concept drift to catch emerging attack vectors.
        """
        if delayed_labels is None and labels is not None:
            delayed_labels = labels
        old_thresh = self.current_threshold
        reason = "normal_smoothing"

        # 1. Update protected benign baseline
        if delayed_labels is not None and len(delayed_labels) == len(risk_scores):
            # Ground truth available from delayed batch feedback: strictly append confirmed benign
            for r, y in zip(risk_scores, delayed_labels):
                if y == 0:
                    self.benign_risk_buffer.append(float(r))
        else:
            # Prequential heuristic: only append flows with low risk (protected baseline)
            benign_ceiling = max(0.15, self.current_threshold * 0.70)
            for r in risk_scores:
                if r < benign_ceiling:
                    self.benign_risk_buffer.append(float(r))

        # Check for attack burst
        high_risk_count = sum(1 for r in risk_scores if r >= self.current_threshold)
        high_risk_ratio = high_risk_count / max(1, len(risk_scores))

        # 2. Concept drift relaxation
        if drift_detected:
            self.drift_cooldown = 15  # Freeze baseline re-estimation for 15 batches
            computed_target = self.current_threshold * self.drift_relaxation_factor
            reason = "drift_relaxation"
        elif self.drift_cooldown > 0:
            self.drift_cooldown -= 1
            # Maintain heightened sensitivity while drifting
            computed_target = self.current_threshold
            reason = f"drift_cooldown_{self.drift_cooldown}"
        elif self.freeze_during_burst and high_risk_ratio > 0.35:
            # Attack burst in progress: freeze threshold to prevent threshold inflation!
            computed_target = self.current_threshold
            reason = "attack_burst_frozen"
        elif len(self.benign_risk_buffer) >= 50:
            # Compute quantile of protected benign distribution
            quantile_target = (1.0 - self.target_fpr) * 100.0
            computed_target = float(np.percentile(self.benign_risk_buffer, quantile_target))
            reason = "benign_quantile_tracking"
        else:
            computed_target = self.current_threshold
            reason = "insufficient_buffer"

        # 3. Exponential moving average smoothing
        smoothed = (1.0 - self.smoothing_alpha) * self.current_threshold + self.smoothing_alpha * computed_target

        # 4. Strict safety clamping bounds [theta_min, theta_max]
        self.current_threshold = float(np.clip(smoothed, self.min_threshold, self.max_threshold))

        log_entry = {
            "step_index": step_index,
            "old_threshold": round(old_thresh, 4),
            "new_threshold": round(self.current_threshold, 4),
            "reason": reason,
            "drift_detected": drift_detected,
            "buffer_size": len(self.benign_risk_buffer),
        }
        self.threshold_history.append(log_entry)
        return self.current_threshold

    def get_threshold(self) -> float:
        return self.current_threshold
