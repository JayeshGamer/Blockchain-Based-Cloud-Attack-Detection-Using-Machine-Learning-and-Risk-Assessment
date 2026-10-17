"""
Explainability Module for Adaptive ML-DST.
Provides:
1. Global permutation feature importance
2. Per-alert feature contribution vectors
3. Evidential decision provenance (Belief, Plausibility, Uncertainty, Conflict)
Note on causality: Feature attributions quantify model prediction sensitivity,
and do not assert definitive real-world physical causation.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.inspection import permutation_importance


class DecisionExplainer:
    """
    Explainer for Adaptive ML-DST decisions.
    Generates transparent diagnostic summaries for security operations teams.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def compute_global_importance(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_repeats: int = 5,
        random_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Compute global permutation feature importance on validation split.
        Measures the increase in prediction error when feature values are permuted.
        """
        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=random_seed,
            n_jobs=-1,
        )

        importances = []
        for idx, name in enumerate(self.feature_names):
            importances.append({
                "feature": name,
                "importance_mean": float(result.importances_mean[idx]),
                "importance_std": float(result.importances_std[idx]),
            })

        # Sort descending by importance
        importances.sort(key=lambda x: x["importance_mean"], reverse=True)
        return importances

    def explain_instance(
        self,
        x_vector: np.ndarray,
        weights: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Compute local linear attribution scores: contribution_i = x_i * w_i.
        Highlights the top features driving the current alert score.
        """
        x_vec = np.asarray(x_vector, dtype=float).flatten()
        w_vec = np.asarray(weights, dtype=float).flatten()

        min_len = min(len(x_vec), len(w_vec), len(self.feature_names))
        contributions = []

        for i in range(min_len):
            contrib = float(x_vec[i] * w_vec[i])
            contributions.append({
                "feature": self.feature_names[i],
                "value": round(float(x_vec[i]), 4),
                "weight": round(float(w_vec[i]), 4),
                "contribution": round(contrib, 4),
            })

        # Sort by absolute magnitude of contribution
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions[:top_k]

    def build_alert_explanation(
        self,
        flow_index: int,
        flow_vector: np.ndarray,
        weights: np.ndarray,
        fusion_result: Dict[str, Any],
        active_threshold: float,
        drift_detected: bool,
        defense_action: str,
    ) -> Dict[str, Any]:
        """
        Build an end-to-end transparent explanation package for a triggered alert.
        """
        top_features = self.explain_instance(flow_vector, weights, top_k=5)

        return {
            "flow_index": flow_index,
            "decision": {
                "defense_action": defense_action,
                "risk_score": round(fusion_result["risk_score"], 4),
                "active_threshold": round(active_threshold, 4),
                "risk_breached": fusion_result["risk_score"] >= active_threshold,
            },
            "evidence_sources": {
                "ml_attack_score": round(float(fusion_result["sources"].get("ml_attack_score", fusion_result["sources"].get("ml_probability", 0.0))), 4),
                "anomaly_score": round(float(fusion_result["sources"].get("anomaly_score", 0.0)), 4),
                "temporal_deviation_score": round(float(fusion_result["sources"].get("temporal_deviation", 0.0)), 4),
            },
            "dst_uncertainty_decomposition": {
                "belief_attack": round(fusion_result["belief"], 4),
                "plausibility_attack": round(fusion_result["plausibility"], 4),
                "epistemic_uncertainty": round(fusion_result["uncertainty"], 4),
                "evidence_conflict_k": round(fusion_result["conflict"], 4),
            },
            "concept_drift_state": {
                "drift_alert_active": drift_detected,
            },
            "top_contributing_features": top_features,
            "methodological_notice": (
                "Feature contributions indicate statistical sensitivity within the model representation; "
                "they do not imply direct physical causation in network packet generation."
            ),
        }
