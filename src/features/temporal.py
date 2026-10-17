"""
Temporal Deviation Evidence Tracker for Ordered Flow Replay.
Maintains rolling statistical baselines (volumetric rates, duration, IAT, packet ratios, and attack scores).
Features:
1. Protected benign baseline: Prevents attack bursts from contaminating rolling statistics.
2. Computes deviation scores strictly before window insertion (past information only).
3. Evaluates 6 key temporal telemetry signals:
   - Flow Bytes/s
   - Flow Packets/s
   - Flow Duration
   - Flow IAT Mean
   - Forward / Backward Packet Ratio
   - Rolling Attack-Score Deviation
4. Uses hyperbolic tangent of composite Z-scores to map to [0, 1].
"""

from collections import deque
from typing import Dict, List, Optional
import numpy as np


class TemporalDeviationTracker:
    """
    Sliding-window temporal deviation estimator with protected benign baseline.
    Tracks rolling mean and standard deviation across ordered flows.
    """

    def __init__(
        self,
        window_size: int = 200,
        feature_indices: Optional[Dict[str, int]] = None,
        weights: Optional[Dict[str, float]] = None,
        benign_confidence_threshold: float = 0.35,
    ):
        self.window_size = window_size
        self.feature_indices = feature_indices or {}
        self.benign_confidence_threshold = benign_confidence_threshold

        self.weights = weights or {
            "flow_bytes_per_s": 0.25,
            "flow_packets_per_s": 0.25,
            "flow_duration": 0.15,
            "flow_iat_mean": 0.15,
            "packet_ratio": 0.10,
            "rolling_risk": 0.10,
        }

        # Rolling history buffers for each tracked attribute
        self.buffers: Dict[str, deque] = {
            k: deque(maxlen=self.window_size) for k in self.weights.keys()
        }

        self.rolling_means: Dict[str, float] = {k: 0.0 for k in self.weights.keys()}
        self.rolling_stds: Dict[str, float] = {k: 1.0 for k in self.weights.keys()}
        self.total_observed = 0

    def resolve_feature_indices(self, feature_names: List[str]) -> None:
        """Map tracked temporal attributes to column indices in feature vector."""
        name_lower_map = {
            name.lower().replace(" ", "").replace("/", "").replace("_", ""): idx
            for idx, name in enumerate(feature_names)
        }

        # Fuzzy match key telemetry indicators
        for target_key, patterns in [
            ("flow_bytes_per_s", ["flowbytess", "flowbytes"]),
            ("flow_packets_per_s", ["flowpacketss", "flowpackets"]),
            ("flow_duration", ["flowduration", "duration"]),
            ("flow_iat_mean", ["flowiatmean", "iatmean"]),
            ("fwd_packets", ["totalfwdpackets", "fwdpackets"]),
            ("bwd_packets", ["totalbackwardpackets", "bwdpackets"]),
        ]:
            for p in patterns:
                for col_clean, idx in name_lower_map.items():
                    if p in col_clean:
                        self.feature_indices[target_key] = idx
                        break
                if target_key in self.feature_indices:
                    break

    def compute_deviation(
        self,
        x: np.ndarray,
        recent_risk_score: float = 0.0,
        update_window: bool = True,
        is_high_risk: bool = False,
    ) -> float:
        """
        Compute normalized temporal deviation score in [0, 1] for a single flow feature vector.
        Strict past-only computation: evaluates Z-scores BEFORE inserting into rolling buffers.
        Protected baseline: only appends if flow is NOT high-risk (prevents attack contamination).
        """
        if not self.feature_indices:
            return 0.20  # Uninformative prior baseline

        self.total_observed += 1
        z_scores = []
        weights = []

        # Extract basic metrics
        extracted_values: Dict[str, float] = {}
        for key in ["flow_bytes_per_s", "flow_packets_per_s", "flow_duration", "flow_iat_mean"]:
            idx = self.feature_indices.get(key)
            if idx is not None and idx < len(x):
                extracted_values[key] = float(x[idx])

        # Forward/Backward packet ratio
        fwd_idx = self.feature_indices.get("fwd_packets")
        bwd_idx = self.feature_indices.get("bwd_packets")
        if fwd_idx is not None and bwd_idx is not None and fwd_idx < len(x) and bwd_idx < len(x):
            fwd = max(1.0, float(x[fwd_idx]))
            bwd = max(1.0, float(x[bwd_idx]))
            extracted_values["packet_ratio"] = float(fwd / (fwd + bwd))
        else:
            extracted_values["packet_ratio"] = 0.5

        # Rolling risk score deviation
        extracted_values["rolling_risk"] = float(recent_risk_score)

        # Compute Z-scores relative to existing baseline
        for key, val in extracted_values.items():
            if key not in self.weights:
                continue

            buf = self.buffers[key]
            if len(buf) >= 10:
                mean = self.rolling_means[key]
                std = max(1e-5, self.rolling_stds[key])
                z = abs(val - mean) / std
            else:
                z = 0.0

            z_scores.append(z)
            weights.append(self.weights.get(key, 0.16))

        # Protected baseline update: strictly exclude high-risk flows
        should_admit = not is_high_risk
        if update_window and should_admit:
            for key, val in extracted_values.items():
                if key in self.buffers:
                    self.buffers[key].append(val)

            if self.total_observed % 5 == 0:
                for key, buf in self.buffers.items():
                    if len(buf) >= 2:
                        arr = np.array(buf, dtype=np.float64)
                        self.rolling_means[key] = float(np.mean(arr))
                        self.rolling_stds[key] = float(np.std(arr))

        if not z_scores:
            return 0.20

        # Weighted composite Z-score mapped to [0, 1] via hyperbolic tangent
        composite_z = np.average(z_scores, weights=weights)
        norm_score = float(np.tanh(composite_z / 3.0))
        return float(np.clip(norm_score, 0.0, 1.0))

    def compute_batch_deviations(
        self,
        X: np.ndarray,
        recent_risk_scores: Optional[np.ndarray] = None,
        update_window: bool = True,
        high_risk_threshold: float = 0.50,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes deviation scores for a batch of flows sequentially."""
        n = len(X)
        scores = np.zeros(n, dtype=np.float64)
        risks = recent_risk_scores if recent_risk_scores is not None else np.zeros(n)

        for i in range(n):
            r_val = float(risks[i]) if i < len(risks) else 0.0
            is_attack = bool(labels[i] == 1) if labels is not None and i < len(labels) else False
            is_hr = is_attack or (r_val >= high_risk_threshold)
            scores[i] = self.compute_deviation(
                X[i],
                recent_risk_score=r_val,
                update_window=update_window,
                is_high_risk=is_hr,
            )
        return scores
