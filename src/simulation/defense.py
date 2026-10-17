"""
Simulated Cloud Defense Activation Engine.
Simulates 7 staged defensive countermeasures:
  1. LOG
  2. INCREASED_MONITORING
  3. RATE_LIMITING
  4. TRAFFIC_FILTERING
  5. TEMPORARY_NODE_ISOLATION
  6. CLUSTER_REROUTING
  7. GLOBAL_ALERT

Pure simulation mode: does NOT touch host network or OS firewall.
Evaluates defense effectiveness: True Interventions, Benign Interruptions, Missed Attacks, and Response Delay.
"""

from collections import deque
from typing import Dict, List, Any, Optional, Deque
import time
import numpy as np


class SimulatedDefenseEngine:
    """
    Simulated Defense Coordinator for Cloud Attack Tolerance.
    Maps assessed hierarchical risk and active thresholds to 7 staged countermeasures.
    """

    def __init__(
        self,
        low_threshold: float = 0.35,
        medium_threshold: float = 0.55,
        high_threshold: float = 0.75,
        critical_global_threshold: float = 0.80,
    ):
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.critical_global_threshold = critical_global_threshold
        self.defense_events: Deque[Dict[str, Any]] = deque(maxlen=1000)

        self.action_counts: Dict[str, int] = {
            "LOG": 0,
            "INCREASED_MONITORING": 0,
            "RATE_LIMITING": 0,
            "TRAFFIC_FILTERING": 0,
            "TEMPORARY_NODE_ISOLATION": 0,
            "CLUSTER_REROUTING": 0,
            "GLOBAL_ALERT": 0,
            "LOG_MONITOR": 0,
            "RATE_LIMIT": 0,
            "TRAFFIC_FILTER_ISOLATE": 0,
            "GLOBAL_REROUTE_ALERT": 0,
        }

        # Quality metrics tracking
        self.stats = {
            "true_attack_interventions": 0,
            "incorrect_benign_interruptions": 0,
            "missed_attack_interventions": 0,
            "total_evaluations": 0,
            "nodes_isolated": set(),
        }

    def evaluate_flow_and_node(
        self,
        event_id: int,
        replay_index: int,
        source_day: str,
        node_id: str,
        cluster_id: str,
        ml_score: float,
        anomaly_score: float = 0.0,
        temporal_score: float = 0.0,
        dst_belief: float = 0.0,
        dst_plausibility: float = 0.0,
        dst_uncertainty: float = 0.0,
        conflict: float = 0.0,
        current_threshold: float = 0.5,
        drift_status: bool = False,
        node_risk: Optional[float] = None,
        ground_truth_is_attack: Optional[int] = None,
        fused_risk_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates and selects one of the 7 simulated defensive countermeasures.
        Captures full evidential context in the event record.
        """
        if node_risk is None:
            node_risk = fused_risk_score if fused_risk_score is not None else float(ml_score)
        # Dynamic boundaries based on active threshold
        t_iso = max(0.80, current_threshold + 0.15)
        t_filter = current_threshold
        t_ratelimit = current_threshold * 0.80
        t_monitor = current_threshold * 0.60

        if node_risk >= t_iso:
            action = "TEMPORARY_NODE_ISOLATION"
            category = "CRITICAL"
            explanation = f"Node risk ({node_risk:.4f}) breached isolation boundary ({t_iso:.4f})."
            self.stats["nodes_isolated"].add(node_id)
        elif node_risk >= t_filter:
            action = "TRAFFIC_FILTERING"
            category = "HIGH"
            explanation = f"Risk ({node_risk:.4f}) exceeded active defense threshold ({t_filter:.4f})."
        elif node_risk >= t_ratelimit:
            action = "RATE_LIMITING"
            category = "MEDIUM"
            explanation = f"Elevated node risk ({node_risk:.4f}) triggered volumetric rate-limiting."
        elif node_risk >= t_monitor:
            action = "INCREASED_MONITORING"
            category = "ELEVATED"
            explanation = f"Telemetry drift ({node_risk:.4f}) warranted heightened sampling frequency."
        else:
            action = "LOG"
            category = "NORMAL"
            explanation = f"Normal operation ({node_risk:.4f}); event recorded in audit buffer."

        event = {
            "event_id": event_id,
            "replay_index": replay_index,
            "source_day": source_day,
            "simulated_node": node_id,
            "simulated_cluster": cluster_id,
            "ml_score": round(float(ml_score), 4),
            "anomaly_score": round(float(anomaly_score), 4),
            "temporal_score": round(float(temporal_score), 4),
            "dst_belief": round(float(dst_belief), 4),
            "dst_plausibility": round(float(dst_plausibility), 4),
            "dst_uncertainty": round(float(dst_uncertainty), 4),
            "conflict": round(float(conflict), 4),
            "current_threshold": round(float(current_threshold), 4),
            "drift_status": bool(drift_status),
            "risk_category": category,
            "selected_action": action,
            "explanation": explanation,
            "timestamp": time.time(),
        }

        self.defense_events.append(event)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.stats["total_evaluations"] += 1

        # Track quality against ground truth if provided
        if ground_truth_is_attack is not None:
            is_active_intervention = action in ["RATE_LIMITING", "TRAFFIC_FILTERING", "TEMPORARY_NODE_ISOLATION"]
            if ground_truth_is_attack == 1:
                if is_active_intervention:
                    self.stats["true_attack_interventions"] += 1
                else:
                    self.stats["missed_attack_interventions"] += 1
            else:
                if is_active_intervention:
                    self.stats["incorrect_benign_interruptions"] += 1

        return event

    def evaluate_cluster_and_global(
        self,
        replay_index: int,
        source_day: str,
        global_risk: float,
        cluster_risks: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Evaluates macro-tier cluster traffic rerouting and global alarms."""
        event = None
        # Check individual clusters for rerouting
        for cid, crisk in cluster_risks.items():
            if crisk >= 0.75:
                action = "CLUSTER_REROUTING"
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
                event = {
                    "event_id": f"CLUST_{replay_index}_{cid}",
                    "replay_index": replay_index,
                    "source_day": source_day,
                    "simulated_node": "N/A",
                    "simulated_cluster": cid,
                    "selected_action": action,
                    "risk_category": "HIGH_CLUSTER",
                    "explanation": f"Cluster {cid} risk ({crisk:.4f}) triggered traffic rerouting.",
                    "timestamp": time.time(),
                }
                self.defense_events.append(event)

        # Check global environment alarm
        if global_risk >= self.critical_global_threshold:
            action = "GLOBAL_ALERT"
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            event = {
                "event_id": f"GLOBAL_{replay_index}",
                "replay_index": replay_index,
                "source_day": source_day,
                "simulated_node": "ALL",
                "simulated_cluster": "ALL",
                "selected_action": action,
                "risk_category": "CRITICAL_GLOBAL",
                "explanation": f"Network-wide cloud risk ({global_risk:.4f}) triggered global alert.",
                "timestamp": time.time(),
            }
            self.defense_events.append(event)

        return event

    def get_defense_quality_metrics(self) -> Dict[str, Any]:
        """Return simulated defense effectiveness summary."""
        tot_eval = max(1, self.stats["total_evaluations"])
        return {
            "true_attack_interventions": self.stats["true_attack_interventions"],
            "incorrect_benign_interruptions": self.stats["incorrect_benign_interruptions"],
            "missed_attack_interventions": self.stats["missed_attack_interventions"],
            "unique_nodes_isolated": len(self.stats["nodes_isolated"]),
            "action_counts": dict(self.action_counts),
        }

    def process_batch(
        self,
        risk_scores: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        source_day: str = "Stream",
    ) -> List[Dict[str, Any]]:
        """
        Processes a batch of flow risk scores and updates defense statistics.
        """
        events = []
        for i, score in enumerate(risk_scores):
            node_idx = i % 20
            cluster_idx = node_idx % 4
            y_val = int(y_true[i]) if y_true is not None and i < len(y_true) else None
            ev = self.evaluate_flow_and_node(
                event_id=i,
                replay_index=self.stats["total_evaluations"],
                source_day=source_day,
                node_id=f"node_{node_idx}",
                cluster_id=f"cluster_{cluster_idx}",
                ml_score=float(score),
                temporal_score=float(score),
                fused_risk_score=float(score),
                ground_truth_is_attack=y_val,
            )
            events.append(ev)
        return events

    def evaluate_node(
        self,
        node_id: str,
        cluster_id: str,
        node_risk: float,
        current_threshold: float,
        replay_index: int = 0,
    ) -> Dict[str, Any]:
        """Backward-compatible evaluation interface for single node."""
        t_high = max(self.high_threshold, current_threshold)
        t_med = t_high * 0.75

        if node_risk >= t_high:
            action = "TRAFFIC_FILTER_ISOLATE"
            category = "HIGH"
            reason = f"Node risk ({node_risk:.4f}) exceeded high defense threshold ({t_high:.4f})"
        elif node_risk >= t_med:
            action = "RATE_LIMIT"
            category = "MEDIUM"
            reason = f"Node risk ({node_risk:.4f}) exceeded medium threshold ({t_med:.4f})"
        else:
            action = "LOG_MONITOR"
            category = "NORMAL"
            reason = f"Normal risk ({node_risk:.4f}) below defensive threshold ({t_med:.4f})"

        event = {
            "timestamp": time.time(),
            "replay_index": replay_index,
            "entity_type": "NODE",
            "entity_id": node_id,
            "cluster_id": cluster_id,
            "risk_score": round(node_risk, 4),
            "threshold_applied": round(t_high, 4),
            "action": action,
            "reason": reason,
        }
        self.defense_events.append(event)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        return event

    def evaluate_global(
        self,
        global_risk: float,
        cluster_risks: Dict[str, float],
        replay_index: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Backward-compatible evaluation interface for global cloud environment."""
        if global_risk >= self.critical_global_threshold:
            action = "GLOBAL_REROUTE_ALERT"
            highest_cluster = max(cluster_risks.items(), key=lambda x: x[1])[0] if cluster_risks else "cluster_0"
            reason = (
                f"Global cloud risk ({global_risk:.4f}) breached critical threshold "
                f"({self.critical_global_threshold:.4f})."
            )
            event = {
                "timestamp": time.time(),
                "replay_index": replay_index,
                "entity_type": "GLOBAL",
                "entity_id": "global_controller",
                "cluster_id": highest_cluster,
                "risk_score": round(global_risk, 4),
                "threshold_applied": round(self.critical_global_threshold, 4),
                "action": action,
                "reason": reason,
            }
            self.defense_events.append(event)
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            return event
        return None

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.defense_events)[-limit:]
