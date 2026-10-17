"""
Unit tests for simulated defense activation engine.
"""

import pytest
from src.simulation.defense import SimulatedDefenseEngine


def test_staged_defense_activation_levels():
    engine = SimulatedDefenseEngine(
        low_threshold=0.35,
        medium_threshold=0.55,
        high_threshold=0.75,
        critical_global_threshold=0.80,
    )

    # Low risk flow
    ev_low = engine.evaluate_node("node_01", "cluster_0", node_risk=0.20, current_threshold=0.50, replay_index=1)
    assert ev_low["action"] == "LOG_MONITOR"

    # Medium risk flow
    ev_med = engine.evaluate_node("node_01", "cluster_0", node_risk=0.60, current_threshold=0.50, replay_index=2)
    assert ev_med["action"] == "RATE_LIMIT"

    # High risk flow
    ev_high = engine.evaluate_node("node_01", "cluster_0", node_risk=0.85, current_threshold=0.50, replay_index=3)
    assert ev_high["action"] == "TRAFFIC_FILTER_ISOLATE"


def test_global_defense_trigger():
    engine = SimulatedDefenseEngine(critical_global_threshold=0.80)

    # Below threshold
    res_normal = engine.evaluate_global(global_risk=0.65, cluster_risks={"c0": 0.70, "c1": 0.60}, replay_index=10)
    assert res_normal is None

    # Exceeding threshold
    res_crit = engine.evaluate_global(global_risk=0.88, cluster_risks={"c0": 0.92, "c1": 0.84}, replay_index=11)
    assert res_crit is not None
    assert res_crit["action"] == "GLOBAL_REROUTE_ALERT"
    assert res_crit["cluster_id"] == "c0"
