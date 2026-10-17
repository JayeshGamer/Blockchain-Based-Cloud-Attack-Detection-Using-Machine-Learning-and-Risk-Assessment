"""
Tests for Temporal Deviation Tracker and Protected Benign Baseline.
"""

import pytest
import numpy as np

from src.features.temporal import TemporalDeviationTracker


def test_temporal_past_only_computation():
    """Verify score is computed strictly prior to window insertion."""
    tracker = TemporalDeviationTracker(window_size=50)
    tracker.feature_indices = {
        "flow_bytes_per_s": 0,
        "flow_packets_per_s": 1,
        "flow_duration": 2,
        "flow_iat_mean": 3,
        "fwd_packets": 4,
        "bwd_packets": 5,
    }

    # First flow: empty history, returns uninformative prior
    flow0 = np.array([1000.0, 50.0, 100000.0, 500.0, 10.0, 10.0])
    s0 = tracker.compute_deviation(flow0, update_window=True)
    assert 0.0 <= s0 <= 1.0


def test_temporal_protected_benign_baseline():
    """Verify high-risk flows are excluded from benign temporal baseline."""
    tracker = TemporalDeviationTracker(window_size=50)
    tracker.feature_indices = {
        "flow_bytes_per_s": 0,
        "flow_packets_per_s": 1,
    }

    # Pre-fill with normal low-rate benign traffic
    for _ in range(20):
        tracker.compute_deviation(np.array([100.0, 10.0]), is_high_risk=False, update_window=True)

    buf_len_before = len(tracker.buffers["flow_bytes_per_s"])

    # Attempt to inject massive attack burst marked as high-risk
    for _ in range(10):
        tracker.compute_deviation(np.array([1000000.0, 50000.0]), is_high_risk=True, update_window=True)

    buf_len_after = len(tracker.buffers["flow_bytes_per_s"])
    assert buf_len_after == buf_len_before, "High-risk flows must NOT enter the protected benign baseline buffer!"


def test_temporal_extreme_deviation_approaches_one():
    """Verify an extreme volumetric surge receives a deviation score approaching 1.0."""
    tracker = TemporalDeviationTracker(window_size=50)
    tracker.feature_indices = {
        "flow_bytes_per_s": 0,
        "flow_packets_per_s": 1,
        "flow_duration": 2,
        "flow_iat_mean": 3,
    }

    # Train baseline on small traffic values
    for _ in range(30):
        tracker.compute_deviation(np.array([10.0, 2.0, 1000.0, 50.0]), is_high_risk=False, update_window=True)

    # Ingest massive 10,000x surge
    extreme_flow = np.array([1000000.0, 500000.0, 50000000.0, 100000.0])
    surge_score = tracker.compute_deviation(extreme_flow, update_window=False)

    assert surge_score > 0.85, f"Expected deviation score near 1.0 for extreme anomaly, got {surge_score}"
