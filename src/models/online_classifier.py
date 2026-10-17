"""
Online Classifiers for Streaming Attack Detection.
Supports:
1. Online SGD Classifier (Linear incremental learner via partial_fit).
2. River Hoeffding Adaptive Tree (Non-linear incremental decision tree with built-in ADWIN drift adaptation).
3. Probability calibration via Platt scaling fitted strictly on validation data.
4. Active drift response mechanisms (Plasticity Boost, Background Learner Promotion, Model Reset).
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Check River availability
try:
    from river import tree as river_tree
    from river import compose as river_compose
    from river import preprocessing as river_prep
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


class OnlineSGDClassifier:
    """
    Incremental Linear Classifier using Stochastic Gradient Descent.
    Supports prequential (predict-then-update) evaluation.
    Outputs calibrated attack scores or raw decision scores.
    """

    def __init__(
        self,
        loss: str = "log_loss",
        penalty: str = "l2",
        alpha: float = 0.0001,
        learning_rate: str = "optimal",
        random_seed: int = 42,
    ):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        self.classes_ = np.array([0, 1])
        self.model = SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            random_state=self.random_seed,
            warm_start=True,
        )
        self.is_initialized_ = False
        self.sample_count_ = 0
        self.update_count_ = 0
        self.model_version = 1

        # Validation calibrator (Platt scaling)
        self.calibrator_: Optional[LogisticRegression] = None
        self.plasticity_multiplier = 1.0
        self.plasticity_cooldown = 0

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "OnlineSGDClassifier":
        """Incrementally update weights on incoming batch of flows."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)

        # Apply sample weighting if plasticity boost is active
        if self.plasticity_multiplier > 1.0:
            w = np.full(len(X), self.plasticity_multiplier) if sample_weight is None else sample_weight * self.plasticity_multiplier
            if self.plasticity_cooldown > 0:
                self.plasticity_cooldown -= 1
                if self.plasticity_cooldown == 0:
                    self.plasticity_multiplier = 1.0
        else:
            w = sample_weight

        if not self.is_initialized_:
            self.model.partial_fit(X, y, classes=self.classes_, sample_weight=w)
            self.is_initialized_ = True
        else:
            self.model.partial_fit(X, y, sample_weight=w)

        self.sample_count_ += len(X)
        self.update_count_ += 1
        return self

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> "OnlineSGDClassifier":
        """Fit Platt calibrator on validation predictions."""
        if not self.is_initialized_:
            return self
        raw_scores = self.model.decision_function(X_val).reshape(-1, 1)
        self.calibrator_ = LogisticRegression(solver="lbfgs")
        self.calibrator_.fit(raw_scores, y_val)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute attack scores in [0, 1].
        If calibrated on validation, applies Platt scaling; otherwise uses clipped sigmoid.
        """
        if not self.is_initialized_:
            return np.full(len(X), 0.5)

        X = np.asarray(X, dtype=np.float64)
        raw_scores = self.model.decision_function(X)

        if self.calibrator_ is not None:
            # Calibrated probability output
            return self.calibrator_.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

        # Numerically stable sigmoid attack score
        clipped = np.clip(raw_scores, -25.0, 25.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns 2D probabilities [p_benign, p_attack]."""
        p_atk = self.predict_score(X)
        return np.column_stack([1.0 - p_atk, p_atk])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_score(X) >= threshold).astype(int)

    def adapt_to_drift(self, strategy: str = "plasticity_boost", boost_factor: float = 2.0) -> None:
        """
        Executes active drift response strategy:
        - plasticity_boost: Temporarily weights new post-drift samples more heavily.
        - full_reset: Reinitializes model weights.
        """
        self.model_version += 1
        if strategy == "plasticity_boost":
            self.plasticity_multiplier = float(boost_factor)
            self.plasticity_cooldown = 15  # Heightened learning for next 15 batches
            logger.info(f"Drift response: Applied plasticity boost (x{boost_factor}) for 15 batches.")
        elif strategy == "full_reset":
            self.model = SGDClassifier(
                loss=self.loss,
                penalty=self.penalty,
                alpha=self.alpha,
                learning_rate=self.learning_rate,
                random_state=self.random_seed + self.model_version,
                warm_start=True,
            )
            self.is_initialized_ = False
            logger.info("Drift response: Model weights reset.")


class RiverHoeffdingAdaptiveTreeOnline:
    """
    Non-linear Online Incremental Learner using River's Hoeffding Adaptive Tree Classifier (HATC).
    HATC natively embeds ADWIN drift detectors in tree leaves, pruning subtrees that drift.
    """

    def __init__(
        self,
        grace_period: int = 100,
        delta: float = 1e-5,
        random_seed: int = 42,
    ):
        if not HAS_RIVER:
            raise RuntimeError("River is required for RiverHoeffdingAdaptiveTreeOnline.")
        self.grace_period = grace_period
        self.delta = delta
        self.random_seed = random_seed
        self.model = river_tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=self.grace_period,
            delta=self.delta,
            seed=self.random_seed,
        )
        self.model_version = 1
        self.feature_names: Optional[List[str]] = None

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "RiverHoeffdingAdaptiveTreeOnline":
        """Incrementally update the tree on a batch of flows."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n, p = X.shape

        if self.feature_names is None:
            self.feature_names = [f"f_{i}" for i in range(p)]

        for i in range(n):
            x_dict = {self.feature_names[j]: float(X[i, j]) for j in range(p)}
            self.model.learn_one(x_dict, int(y[i]))
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Compute attack score in [0, 1]."""
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        if self.feature_names is None:
            self.feature_names = [f"f_{i}" for i in range(p)]

        scores = np.zeros(n, dtype=np.float64)
        for i in range(n):
            x_dict = {self.feature_names[j]: float(X[i, j]) for j in range(p)}
            proba = self.model.predict_proba_one(x_dict)
            scores[i] = float(proba.get(1, 0.0))
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = self.predict_score(X)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_score(X) >= threshold).astype(int)

    def adapt_to_drift(self, strategy: str = "plasticity_boost", boost_factor: float = 2.0, **kwargs) -> None:
        """Tree automatically adapts via internal ADWIN, but can be reset if requested."""
        self.model_version += 1
        if strategy == "full_reset":
            self.model = river_tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=self.grace_period,
                delta=self.delta,
                seed=self.random_seed + self.model_version,
            )


# Default online classifier class pointing to OnlineSGDClassifier
OnlineClassifier = OnlineSGDClassifier
