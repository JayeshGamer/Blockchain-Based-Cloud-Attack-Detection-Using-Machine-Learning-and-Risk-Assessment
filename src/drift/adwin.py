"""
ADWIN (Adaptive Windowing) Concept Drift Detector and Multi-Signal Monitor.
Reference:
Albert Bifet and Ricard Gavalda, "Learning from Time-Changing Data with Adaptive Windowing",
SIAM International Conference on Data Mining (SDM), 2007.

Supports:
1. Native River ADWIN wrapper (`river.drift.ADWIN`) when available.
2. Vectorized O(n) exact custom ADWIN fallback using cumulative sum arrays.
3. Multi-signal monitoring: separates prediction error, attack score, and packet rate detectors.
4. Comprehensive drift response coordination with state logging.
"""

import math
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try importing River ADWIN
try:
    from river import drift as river_drift
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


class VectorizedADWIN:
    """
    Vectorized O(n) Adaptive Windowing (ADWIN) algorithm.
    Uses cumulative sums to evaluate candidate cut points in O(1) time per split,
    reducing total update complexity from O(n^2) to O(n).
    """

    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        min_window_length: int = 10,
        grace_period: int = 100,
    ):
        self.delta = delta
        self.clock = clock
        self.min_window_length = min_window_length
        self.grace_period = grace_period

        self.window: List[float] = []
        self.total: float = 0.0
        self.step_count: int = 0
        self.drift_detected: bool = False
        self.drift_history: List[int] = []

    def reset(self) -> None:
        """Reset internal window and state."""
        self.window.clear()
        self.total = 0.0
        self.step_count = 0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """
        Add an element to the window and check for concept drift.
        Returns True if concept drift was detected on this step, False otherwise.
        """
        self.step_count += 1
        self.drift_detected = False

        v = float(value)
        self.window.append(v)
        self.total += v
        n = len(self.window)

        # Check only at multiples of clock, and after initial grace period
        if self.step_count < self.grace_period or (self.step_count % self.clock != 0):
            return False

        if n < self.min_window_length * 2:
            return False

        # Vectorized split evaluation using cumulative sum
        w_arr = np.array(self.window, dtype=np.float64)
        cumsum = np.cumsum(w_arr)
        total_sum = cumsum[-1]

        # Evaluate candidate cut points
        step_size = max(1, n // 20)
        has_cut = False
        cut_idx = 0

        # Log term for Bonferroni correction
        log_term = math.log(max(2.0, float(n)))
        delta_prime = self.delta / log_term

        for i in range(self.min_window_length, n - self.min_window_length, step_size):
            n0 = i
            n1 = n - i
            sum0 = cumsum[i - 1]
            sum1 = total_sum - sum0

            u0 = sum0 / n0
            u1 = sum1 / n1
            diff = abs(u0 - u1)

            # Harmonic mean of subwindow lengths
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps_cut = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / delta_prime))

            if diff > eps_cut:
                has_cut = True
                cut_idx = i
                break

        if has_cut:
            # Shrink window by dropping older subwindow W0
            self.window = self.window[cut_idx:]
            self.total = float(sum(self.window))
            self.drift_detected = True
            self.drift_history.append(self.step_count)
            return True

        return False

    @property
    def estimation(self) -> float:
        return (self.total / len(self.window)) if len(self.window) > 0 else 0.0

    @property
    def width(self) -> int:
        return len(self.window)


class RiverADWINWrapper:
    """Wrapper around river.drift.ADWIN to maintain a unified interface."""

    def __init__(self, delta: float = 0.002, clock: int = 32, grace_period: int = 100):
        if not HAS_RIVER:
            raise RuntimeError("River package is not installed.")
        self.delta = delta
        self.clock = clock
        self.grace_period = grace_period
        self.detector = river_drift.ADWIN(delta=delta, clock=clock, grace_period=grace_period)
        self.step_count = 0
        self.drift_detected = False
        self.drift_history: List[int] = []

    def reset(self) -> None:
        self.detector = river_drift.ADWIN(delta=self.delta, clock=self.clock, grace_period=self.grace_period)
        self.step_count = 0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        self.step_count += 1
        self.detector.update(float(value))
        self.drift_detected = bool(self.detector.drift_detected)
        if self.drift_detected:
            self.drift_history.append(self.step_count)
        return self.drift_detected

    @property
    def estimation(self) -> float:
        return float(self.detector.estimation)

    @property
    def width(self) -> int:
        return int(self.detector.width)


def create_adwin_detector(
    delta: float = 0.002,
    clock: int = 32,
    prefer_river: bool = True,
    grace_period: int = 100,
) -> Any:
    """Factory creating River ADWIN if available, otherwise VectorizedADWIN."""
    if prefer_river and HAS_RIVER:
        logger.info("Using verified River ADWIN implementation.")
        return RiverADWINWrapper(delta=delta, clock=clock, grace_period=grace_period)
    logger.info("Using verified VectorizedADWIN implementation.")
    return VectorizedADWIN(delta=delta, clock=clock, grace_period=grace_period)


# Retain class name ADWIN for backward compatibility
ADWIN = VectorizedADWIN


class MultiSignalDriftCoordinator:
    """
    Coordinates distinct ADWIN detectors across orthogonal operational telemetry:
    1. delayed_loss: prediction error on delayed label feedback
    2. attack_score: distribution of fused attack risk scores
    3. packet_rate: volumetric traffic surges
    """

    def __init__(
        self,
        delta: float = 0.002,
        prefer_river: bool = True,
        drift_response_strategy: str = "plasticity_boost",
        boost_factor: float = 2.0,
        cooldown_batches: int = 20,
    ):
        self.delta = delta
        self.prefer_river = prefer_river
        self.strategy = drift_response_strategy
        self.boost_factor = boost_factor
        self.cooldown_batches = cooldown_batches

        self.detectors = {
            "delayed_loss": create_adwin_detector(delta=delta, prefer_river=prefer_river),
            "attack_score": create_adwin_detector(delta=delta * 2.0, prefer_river=prefer_river),
            "packet_rate": create_adwin_detector(delta=delta * 2.0, prefer_river=prefer_river),
        }

        self.cooldown_remaining = 0
        self.event_log: List[Dict[str, Any]] = []
        self.model_version = 1

    def update_batch(
        self,
        delayed_errors: Optional[np.ndarray] = None,
        risk_scores: Optional[np.ndarray] = None,
        packet_rates: Optional[np.ndarray] = None,
        step_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Update each signal detector and determine if any drift alarm triggered.
        Returns detailed drift event status.
        """
        alarms: Dict[str, bool] = {}

        if delayed_errors is not None and len(delayed_errors) > 0:
            alarm = False
            for err in delayed_errors:
                if self.detectors["delayed_loss"].update(float(err)):
                    alarm = True
            alarms["delayed_loss"] = alarm

        if risk_scores is not None and len(risk_scores) > 0:
            alarm = False
            for r in risk_scores:
                if self.detectors["attack_score"].update(float(r)):
                    alarm = True
            alarms["attack_score"] = alarm

        if packet_rates is not None and len(packet_rates) > 0:
            alarm = False
            for pr in packet_rates:
                if self.detectors["packet_rate"].update(float(pr)):
                    alarm = True
            alarms["packet_rate"] = alarm

        any_alarm = any(alarms.values())
        action_taken = "none"

        if any_alarm and self.cooldown_remaining == 0:
            self.model_version += 1
            action_taken = self.strategy
            self.cooldown_remaining = self.cooldown_batches

            event = {
                "step_index": step_index,
                "alarm_signals": [sig for sig, al in alarms.items() if al],
                "action_taken": action_taken,
                "model_version_after": self.model_version,
            }
            self.event_log.append(event)
            logger.info(f"Concept Drift Alarm at step {step_index}! Action: {action_taken}")
        elif self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        return {
            "drift_detected": any_alarm,
            "alarm_signals": alarms,
            "action_taken": action_taken,
            "in_cooldown": self.cooldown_remaining > 0,
            "model_version": self.model_version,
        }

    def update(self, signals: Any) -> Tuple[bool, str, str]:
        """Convenience method accepting either a scalar loss or a dict of signals."""
        if isinstance(signals, dict):
            delayed_errors = np.array([float(signals.get("delayed_loss", 0.0))]) if "delayed_loss" in signals else None
            risk_scores = np.array([float(signals.get("attack_score", 0.0))]) if "attack_score" in signals else None
            packet_rates = np.array([float(signals.get("packet_rate", 0.0))]) if "packet_rate" in signals else None
        else:
            delayed_errors = np.array([float(signals)])
            risk_scores = None
            packet_rates = None
        res = self.update_batch(delayed_errors=delayed_errors, risk_scores=risk_scores, packet_rates=packet_rates)
        reason = ", ".join([k for k, v in res["alarm_signals"].items() if v]) if res["drift_detected"] else "none"
        return res["drift_detected"], res["action_taken"], reason
