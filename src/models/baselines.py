"""
Baseline Models for Comparative Evaluation (Protocol A and Ablations):
1. DummyMajorityBaseline: Majority class predictor
2. LogisticRegressionBaseline: Linear static classifier with lbfgs
3. StaticSGDBaseline: Static SGD classifier
4. RandomForestBaseline: Static ensemble decision forest
5. HistGradientBoostingBaseline: Fast gradient boosting ensemble
6. IsolationForestBaseline: Benign-trained unsupervised anomaly detector
7. StaticDSTBaseline: Heuristic non-learning DST baseline representing base paper's static heuristic approach
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier


class DummyMajorityBaseline:
    """Predicts majority class (BENIGN = 0)."""

    def __init__(self, random_seed: int = 42):
        self.model = DummyClassifier(strategy="most_frequent", random_state=random_seed)
        self.name = "Dummy Majority"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyMajorityBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class LogisticRegressionBaseline:
    """Static linear baseline."""

    def __init__(self, random_seed: int = 42, max_iter: int = 500):
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_seed,
            class_weight="balanced",
            solver="lbfgs",
        )
        self.name = "Logistic Regression"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class StaticSGDBaseline:
    """Static linear SGD baseline."""

    def __init__(self, random_seed: int = 42):
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            class_weight="balanced",
            random_state=random_seed,
            max_iter=1000,
        )
        self.name = "Static SGD"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StaticSGDBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.decision_function(X)
        clipped = np.clip(raw, -25.0, 25.0)
        p1 = 1.0 / (1.0 + np.exp(-clipped))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class RandomForestBaseline:
    """Static Random Forest ensemble baseline."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 15, random_seed: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.name = "Random Forest (Static)"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class HistGradientBoostingBaseline:
    """Static Histogram Gradient Boosting baseline."""

    def __init__(self, max_iter: int = 100, random_seed: int = 42):
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            random_state=random_seed,
            class_weight="balanced",
        )
        self.name = "HistGradientBoosting"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class IsolationForestBaseline:
    """
    Unsupervised Anomaly Detector.
    Trained strictly on benign traffic.
    Outputs normalized anomaly scores in [0, 1].
    """

    def __init__(
        self,
        n_estimators: int = 50,
        contamination: float = 0.05,
        random_seed: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_seed,
            n_jobs=-1,
        )
        self.min_score_: float = 0.0
        self.max_score_: float = 1.0
        self.name = "Isolation Forest (Unsupervised)"

    def fit(self, X_benign: np.ndarray) -> "IsolationForestBaseline":
        actual_samples = min(5000, len(X_benign))
        self.model.max_samples = actual_samples
        self.model.fit(X_benign)
        raw_scores = -self.model.score_samples(X_benign)
        self.min_score_ = float(np.percentile(raw_scores, 2))
        self.max_score_ = float(np.percentile(raw_scores, 98))
        if self.max_score_ <= self.min_score_:
            self.max_score_ = self.min_score_ + 1.0
        return self

    def score_anomaly(self, X: np.ndarray) -> np.ndarray:
        raw_scores = -self.model.score_samples(X)
        norm_scores = (raw_scores - self.min_score_) / (self.max_score_ - self.min_score_)
        return np.clip(norm_scores, 0.0, 1.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        anom_score = self.score_anomaly(X)
        return np.column_stack([1.0 - anom_score, anom_score])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class StaticDSTBaseline:
    """
    Simplified Static Heuristic-DST baseline (paper-inspired; not an exact reproduction).
    Combines static rate and volatility indicators via fixed Dempster-Shafer combination
    without any online updates, smart contract consensus, or multi-attribute hierarchical
    trust evaluation as proposed in Benmenzer et al. (2025).
    """

    def __init__(self, uncertainty_budget: float = 0.20):
        self.uncertainty = uncertainty_budget
        self.name = "Simplified Static Heuristic-DST (paper-inspired; not an exact reproduction)"
        self.feature_mins_: Optional[np.ndarray] = None
        self.feature_maxs_: Optional[np.ndarray] = None
        self.feature_diff_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "StaticDSTBaseline":
        self.feature_mins_ = np.percentile(X_train, 1, axis=0)
        self.feature_maxs_ = np.percentile(X_train, 99, axis=0)
        diff = self.feature_maxs_ - self.feature_mins_
        diff[diff == 0] = 1.0
        self.feature_diff_ = diff
        return self

    def _combine_dst(self, m1_atk: float, m2_atk: float) -> Tuple[float, float, float]:
        u = self.uncertainty
        m1_ben = (1.0 - m1_atk) * (1.0 - u)
        m1_atk_val = m1_atk * (1.0 - u)
        m1_u = u

        m2_ben = (1.0 - m2_atk) * (1.0 - u)
        m2_atk_val = m2_atk * (1.0 - u)
        m2_u = u

        k = (m1_ben * m2_atk_val) + (m1_atk_val * m2_ben)
        k = min(0.999, k)
        denom = 1.0 - k

        m_combined_atk = ((m1_atk_val * m2_atk_val) + (m1_atk_val * m2_u) + (m1_u * m2_atk_val)) / denom
        m_combined_ben = ((m1_ben * m2_ben) + (m1_ben * m2_u) + (m1_u * m2_ben)) / denom
        m_combined_u = (m1_u * m2_u) / denom
        return m_combined_atk, m_combined_ben, m_combined_u

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_norm = np.clip((X - self.feature_mins_) / self.feature_diff_, 0.0, 1.0)
        mid = X_norm.shape[1] // 2
        ind1 = np.mean(X_norm[:, :mid], axis=1)
        ind2 = np.mean(X_norm[:, mid:], axis=1)

        combined_probs = []
        for i in range(len(X)):
            m_atk, _, _ = self._combine_dst(float(ind1[i]), float(ind2[i]))
            combined_probs.append(m_atk)

        probs_atk = np.array(combined_probs)
        return np.column_stack([1.0 - probs_atk, probs_atk])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
