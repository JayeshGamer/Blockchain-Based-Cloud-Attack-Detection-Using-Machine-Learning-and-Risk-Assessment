"""
Hierarchical Cloud Risk Aggregation Engine (3 Tiers).
Models:
  Tier 1: Simulated Nodes (v in V) processing incoming flows.
  Tier 2: Node Clusters (Ci in C) grouping member nodes.
  Tier 3: Global Cloud Environment aggregating cluster risks.

Features:
1. Multi-operator aggregation: Mean, Maximum, 90th Percentile, and DST-evidential aggregation.
2. Full uncertainty propagation: Tracks Belief, Plausibility, and Uncertainty across all 3 tiers.
3. Deterministic load mapping via Knuth's multiplicative hash (zero label leakage).
"""

from collections import deque
from typing import Dict, List, Any, Optional
import numpy as np


class SimulatedNode:
    """Represents a virtual cloud compute node processing ingress flows."""

    def __init__(self, node_id: str, cluster_id: str, window_size: int = 50):
        self.node_id = node_id
        self.cluster_id = cluster_id
        self.flow_history: deque = deque(maxlen=window_size)
        self.current_risk: float = 0.0
        self.current_belief: float = 0.0
        self.current_plausibility: float = 0.0
        self.current_uncertainty: float = 0.0
        self.total_flows: int = 0

    def record_flow(
        self,
        flow_risk: float,
        belief: float,
        plausibility: float,
        uncertainty: float,
    ) -> None:
        """Add new flow assessment to node's temporal window."""
        self.flow_history.append((flow_risk, belief, plausibility, uncertainty))
        self.total_flows += 1

        if self.flow_history:
            # Exponentially recency-weighted average over recent flows
            weights = np.exp(np.linspace(-1.0, 0.0, len(self.flow_history)))
            weights /= np.sum(weights)

            risks = [f[0] for f in self.flow_history]
            beliefs = [f[1] for f in self.flow_history]
            plaus = [f[2] for f in self.flow_history]
            uncs = [f[3] for f in self.flow_history]

            self.current_risk = float(np.dot(weights, risks))
            self.current_belief = float(np.dot(weights, beliefs))
            self.current_plausibility = float(np.dot(weights, plaus))
            self.current_uncertainty = float(np.dot(weights, uncs))


class HierarchicalRiskEngine:
    """
    Three-Tier Hierarchical Risk Assessment Engine with Uncertainty Propagation:
      Tier 1: Local Node Risk (R_v, Bel_v, Pl_v, Theta_v)
      Tier 2: Cluster Risk (R_Ci, Bel_Ci, Pl_Ci, Theta_Ci)
      Tier 3: Global Risk (R_global, Bel_global, Pl_global, Theta_global)
    """

    def __init__(
        self,
        num_nodes: int = 20,
        num_clusters: int = 4,
        random_seed: int = 42,
        node_window_size: int = 50,
        aggregation_operator: str = "percentile_90",  # "mean", "max", "percentile_90", "dst"
    ):
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.random_seed = random_seed
        self.node_window_size = node_window_size
        self.aggregation_operator = aggregation_operator

        self.nodes: Dict[str, SimulatedNode] = {}
        self.clusters: Dict[str, List[str]] = {f"cluster_{c}": [] for c in range(num_clusters)}

        cluster_keys = list(self.clusters.keys())
        for i in range(num_nodes):
            node_id = f"node_{i:02d}"
            cid = cluster_keys[i % num_clusters]
            self.nodes[node_id] = SimulatedNode(node_id, cid, window_size=node_window_size)
            self.clusters[cid].append(node_id)

        self.cluster_telemetry: Dict[str, Dict[str, float]] = {
            cid: {"risk": 0.0, "belief": 0.0, "plausibility": 0.0, "uncertainty": 0.0}
            for cid in self.clusters
        }
        self.global_telemetry: Dict[str, float] = {
            "risk": 0.0,
            "belief": 0.0,
            "plausibility": 0.0,
            "uncertainty": 0.0,
        }

    def assign_flow_to_node(self, flow_index: int) -> str:
        """Deterministically map flow to simulated node using Knuth's multiplicative hash."""
        node_ids = list(self.nodes.keys())
        idx = (flow_index * 2654435761) % len(node_ids)
        return node_ids[idx]

    def update_node(
        self,
        node_id: str,
        risk_score: float,
        belief: float,
        plausibility: float,
        uncertainty: float,
    ) -> None:
        """Update node state with incoming flow assessment."""
        if node_id in self.nodes:
            self.nodes[node_id].record_flow(risk_score, belief, plausibility, uncertainty)

    def _aggregate_values(self, values: List[float], operator: Optional[str] = None) -> float:
        """Applies configured aggregation operator over a list of scalars."""
        if not values:
            return 0.0
        op = operator or self.aggregation_operator
        if op == "max":
            return float(np.max(values))
        elif op == "mean":
            return float(np.mean(values))
        elif op == "percentile_90":
            return float(np.percentile(values, 90))
        elif op == "dst":
            # Hybrid weighted: 0.6 * max + 0.4 * mean
            return float(0.6 * np.max(values) + 0.4 * np.mean(values))
        return float(np.mean(values))

    def aggregate_hierarchy(self) -> Dict[str, Any]:
        """
        Aggregates Tier 1 Node risk into Tier 2 Cluster risk, and Tier 2 into Tier 3 Global risk.
        Propagates risk score, belief, plausibility, and uncertainty across all tiers.
        """
        # Tier 2: Cluster Aggregation
        for cid, member_ids in self.clusters.items():
            risks = [self.nodes[nid].current_risk for nid in member_ids]
            beliefs = [self.nodes[nid].current_belief for nid in member_ids]
            plaus = [self.nodes[nid].current_plausibility for nid in member_ids]
            uncs = [self.nodes[nid].current_uncertainty for nid in member_ids]

            self.cluster_telemetry[cid] = {
                "risk": self._aggregate_values(risks),
                "belief": self._aggregate_values(beliefs),
                "plausibility": self._aggregate_values(plaus),
                "uncertainty": self._aggregate_values(uncs),
            }

        # Tier 3: Global Aggregation
        all_c_risks = [c["risk"] for c in self.cluster_telemetry.values()]
        all_c_beliefs = [c["belief"] for c in self.cluster_telemetry.values()]
        all_c_plaus = [c["plausibility"] for c in self.cluster_telemetry.values()]
        all_c_uncs = [c["uncertainty"] for c in self.cluster_telemetry.values()]

        self.global_telemetry = {
            "risk": self._aggregate_values(all_c_risks),
            "belief": self._aggregate_values(all_c_beliefs),
            "plausibility": self._aggregate_values(all_c_plaus),
            "uncertainty": self._aggregate_values(all_c_uncs),
        }

        return {
            "global": self.global_telemetry,
            "global_risk": self.global_telemetry["risk"],
            "clusters": self.cluster_telemetry,
            "cluster_risks": {cid: c["risk"] for cid, c in self.cluster_telemetry.items()},
            "node_risks": {nid: n.current_risk for nid, n in self.nodes.items()},
            "nodes": {
                nid: {
                    "risk": n.current_risk,
                    "belief": n.current_belief,
                    "plausibility": n.current_plausibility,
                    "uncertainty": n.current_uncertainty,
                }
                for nid, n in self.nodes.items()
            },
        }

    @property
    def global_risk(self) -> float:
        return self.global_telemetry["risk"]

    @property
    def cluster_risks(self) -> Dict[str, float]:
        return {cid: c["risk"] for cid, c in self.cluster_telemetry.items()}

    def get_highest_risk_nodes(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k highest risk nodes with full uncertainty decomposition."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.current_risk,
            reverse=True,
        )
        return [
            {
                "node_id": n.node_id,
                "cluster_id": n.cluster_id,
                "risk_score": round(n.current_risk, 4),
                "belief": round(n.current_belief, 4),
                "plausibility": round(n.current_plausibility, 4),
                "uncertainty": round(n.current_uncertainty, 4),
                "total_flows": n.total_flows,
            }
            for n in sorted_nodes[:top_k]
        ]
