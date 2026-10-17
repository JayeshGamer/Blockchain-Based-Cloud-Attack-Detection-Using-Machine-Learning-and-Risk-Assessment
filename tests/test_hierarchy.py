"""
Unit tests for 3-tier hierarchical cloud risk aggregation.
"""

import pytest
from src.risk.hierarchical_risk import HierarchicalRiskEngine


def test_hierarchical_structure_initialization():
    engine = HierarchicalRiskEngine(num_nodes=10, num_clusters=2, random_seed=42)

    assert len(engine.nodes) == 10
    assert len(engine.clusters) == 2
    for cid, members in engine.clusters.items():
        assert len(members) == 5


def test_hierarchical_risk_aggregation_propagation():
    engine = HierarchicalRiskEngine(num_nodes=4, num_clusters=2, random_seed=42)

    # Inject high risk into node_00
    engine.update_node("node_00", risk_score=0.90, belief=0.85, plausibility=0.95, uncertainty=0.10)
    # Inject normal risk into other nodes
    for nid in ["node_01", "node_02", "node_03"]:
        engine.update_node(nid, risk_score=0.10, belief=0.05, plausibility=0.20, uncertainty=0.15)

    res = engine.aggregate_hierarchy()

    # The cluster containing node_00 must have higher risk than the other cluster
    c0 = engine.nodes["node_00"].cluster_id
    c1 = [c for c in engine.clusters if c != c0][0]

    assert res["cluster_risks"][c0] > res["cluster_risks"][c1]
    assert res["global_risk"] > 0.10
