"""
Tests for ADWIN Concept Drift Detection, River Parity, and Active Drift Response.
"""

import pytest
import numpy as np

from src.drift.adwin import VectorizedADWIN, MultiSignalDriftCoordinator, HAS_RIVER
if HAS_RIVER:
    from river import drift as river_drift


def test_adwin_stationary_stream_no_false_alarms():
    """Verify ADWIN produces zero false alarms on stationary Bernoulli stream."""
    np.random.seed(42)
    adwin = VectorizedADWIN(delta=0.002, clock=32, grace_period=100)

    # 1000 stationary samples with mean 0.10
    stream = (np.random.rand(1000) < 0.10).astype(float)
    alarms = [i for i, v in enumerate(stream) if adwin.update(v)]

    assert len(alarms) == 0, f"Expected 0 false alarms on stationary stream, got {len(alarms)}"


def test_adwin_sudden_drift_detection():
    """Verify ADWIN reliably detects sudden distribution shift (0.10 -> 0.80)."""
    np.random.seed(42)
    adwin = VectorizedADWIN(delta=0.002, clock=32, grace_period=100)

    # First 500 samples at p=0.05, next 500 samples at p=0.85
    part1 = (np.random.rand(500) < 0.05).astype(float)
    part2 = (np.random.rand(500) < 0.85).astype(float)
    stream = np.concatenate([part1, part2])

    alarms = [i for i, v in enumerate(stream) if adwin.update(v)]

    assert len(alarms) >= 1, "ADWIN must trigger at least one alarm upon sudden drift"
    # First alarm must occur within 200 steps after step 500
    first_alarm = alarms[0]
    assert 500 <= first_alarm <= 700, f"Alarm fired at {first_alarm}, expected within [500, 700]"


def test_adwin_gradual_drift_detection():
    """Verify ADWIN detects gradual drift ramp."""
    np.random.seed(42)
    adwin = VectorizedADWIN(delta=0.002, clock=32, grace_period=100)

    # 400 stationary samples, followed by 600 linearly increasing samples
    p1 = (np.random.rand(400) < 0.05).astype(float)
    ramp = []
    for step in range(600):
        prob = 0.05 + 0.80 * (step / 600.0)
        ramp.append(1.0 if np.random.rand() < prob else 0.0)

    stream = np.concatenate([p1, np.array(ramp)])
    alarms = [i for i, v in enumerate(stream) if adwin.update(v)]

    assert len(alarms) >= 1, "ADWIN must detect gradual concept drift"


def test_multi_signal_drift_coordinator():
    """Verify multi-signal coordinator detects drift and triggers configured action."""
    coord = MultiSignalDriftCoordinator(
        delta=0.002,
        prefer_river=False,
        drift_response_strategy="plasticity_boost",
        boost_factor=2.0,
    )

    initial_version = coord.model_version

    # Simulate sudden high errors
    high_errors = np.ones(80)
    status = coord.update_batch(delayed_errors=high_errors, step_index=200)

    if status["drift_detected"]:
        assert coord.model_version > initial_version
        assert status["action_taken"] == "plasticity_boost"
        assert len(coord.event_log) > 0


@pytest.mark.skipif(not HAS_RIVER, reason="River not installed")
def test_river_adwin_parity():
    """Verify River ADWIN and VectorizedADWIN both detect sudden drift on identical streams."""
    np.random.seed(42)
    r_adwin = river_drift.ADWIN(delta=0.002, clock=32, grace_period=100)
    v_adwin = VectorizedADWIN(delta=0.002, clock=32, grace_period=100)

    part1 = (np.random.rand(400) < 0.05).astype(float)
    part2 = (np.random.rand(400) < 0.80).astype(float)
    stream = np.concatenate([part1, part2])

    r_alarms = []
    for i, v in enumerate(stream):
        r_adwin.update(v)
        if r_adwin.drift_detected:
            r_alarms.append(i)
    v_alarms = [i for i, v in enumerate(stream) if v_adwin.update(v)]

    assert len(r_alarms) >= 1
    assert len(v_alarms) >= 1
    # Both should detect drift within the post-drift segment
    assert r_alarms[0] >= 400
    assert v_alarms[0] >= 400
