"""
Unit tests for adaptive threshold manager.
Verifies clamping bounds [theta_min, theta_max] and concept drift response.
"""

import numpy as np
import pytest
from src.risk.adaptive_threshold import AdaptiveThresholdManager


def test_threshold_bounds_invariants():
    min_t = 0.20
    max_t = 0.85
    manager = AdaptiveThresholdManager(min_threshold=min_t, max_threshold=max_t, initial_threshold=0.50)

    # Feed extremely low risks: threshold must never dip below min_threshold
    low_risks = np.full(100, 0.01)
    for _ in range(5):
        t = manager.update(low_risks, drift_detected=False)
        assert t >= min_t

    # Feed extremely high risks: threshold must never exceed max_threshold
    high_risks = np.full(100, 0.99)
    for _ in range(10):
        t = manager.update(high_risks, drift_detected=False)
        assert t <= max_t


def test_threshold_drift_relaxation():
    manager = AdaptiveThresholdManager(initial_threshold=0.60)
    baseline_t = manager.get_threshold()

    # Trigger drift alarm
    new_t = manager.update(np.full(20, 0.60), drift_detected=True)
    # Threshold should decrease to become more sensitive
    assert new_t < baseline_t
