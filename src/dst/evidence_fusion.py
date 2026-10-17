"""
Dempster-Shafer Theory (DST) Evidence Fusion Engine with Shafer Reliability Discounting.
Frame of Discernment: Omega = {Benign, Attack}

Supports:
1. Basic Probability Assignment (BPA) generation from evidence sources.
2. Classical Shafer Reliability Discounting:
     m^alpha(A) = r_i * m(A)
     m^alpha(B) = r_i * m(B)
     m^alpha(Theta) = 1 - r_i + r_i * m(Theta)
3. Standard Dempster Rule of Combination and Yager Conflict-Reallocation Rule.
4. Calculation of Belief Bel(Attack), Plausibility Pl(Attack), Epistemic Uncertainty m(Theta),
   Pignistic probability BetP(Attack), and Conflict coefficient K.
5. Strict mass conservation, commutativity, associativity within numeric tolerance, and NaN rejection.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np


class DSTEvidenceFusion:
    """
    Dempster-Shafer Evidential Reasoning Engine with Classical Shafer Reliability Discounting.
    Fuses three orthogonal evidence sources:
      - ML Attack Score (s_ML in [0, 1])
      - Unsupervised Anomaly Score (s_anom in [0, 1])
      - Temporal Deviation Score (s_temp in [0, 1])
    """

    def __init__(
        self,
        uncertainty_budget: float = 0.15,
        source_reliabilities: Optional[Dict[str, float]] = None,
        conflict_threshold: float = 0.90,
        conflict_rule: str = "yager",  # "yager" or "dempster"
    ):
        self.u_budget = float(np.clip(uncertainty_budget, 0.01, 0.50))
        # Default source reliabilities calibrated from validation holdout
        self.reliabilities = source_reliabilities or {
            "ml_attack_score": 0.85,
            "anomaly_score": 0.75,
            "temporal_deviation": 0.70,
        }
        self.conflict_threshold = float(conflict_threshold)
        self.conflict_rule = conflict_rule.lower()

    def create_bpa(self, p_attack: float, source_name: Optional[str] = None) -> Dict[str, float]:
        """
        Creates a Basic Probability Assignment (BPA) with Shafer reliability discounting.
        Raw BPA:
          m_raw(Attack) = p * (1 - u)
          m_raw(Benign) = (1 - p) * (1 - u)
          m_raw(Theta) = u
        Discounted BPA under reliability r in [0, 1]:
          m(Attack) = r * m_raw(Attack)
          m(Benign) = r * m_raw(Benign)
          m(Theta) = 1 - r + r * m_raw(Theta)
        """
        if np.isnan(p_attack) or np.isinf(p_attack):
            raise ValueError(f"Invalid input score to BPA: {p_attack}")

        p = float(np.clip(p_attack, 0.0, 1.0))
        u = self.u_budget

        m_raw_atk = p * (1.0 - u)
        m_raw_ben = (1.0 - p) * (1.0 - u)
        m_raw_theta = u

        # Apply Shafer reliability discounting if source reliability is defined
        r = self.reliabilities.get(source_name, 1.0) if source_name else 1.0
        r = float(np.clip(r, 0.0, 1.0))

        m_atk = r * m_raw_atk
        m_ben = r * m_raw_ben
        m_theta = 1.0 - r + r * m_raw_theta

        # Enforce sum = 1.0
        tot = m_atk + m_ben + m_theta
        return {
            "Attack": float(m_atk / tot),
            "Benign": float(m_ben / tot),
            "Theta": float(m_theta / tot),
        }

    def combine_two_bpas(
        self,
        m1: Dict[str, float],
        m2: Dict[str, float],
        use_yager: Optional[bool] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Combines two independent BPAs using Dempster's rule or Yager's rule.
        Frame: {Attack, Benign, Theta}
        Returns:
            (combined_bpa, conflict_factor_K)
        """
        for m in (m1, m2):
            for k in ("Attack", "Benign", "Theta"):
                if k not in m or np.isnan(m[k]):
                    raise ValueError(f"Invalid or NaN mass in BPA: {m}")

        # Compute conflict K: mass on disjoint intersection (Attack & Benign)
        k = float((m1["Attack"] * m2["Benign"]) + (m1["Benign"] * m2["Attack"]))

        # Conjunctive consensus masses before normalization
        # Attack & Attack, Attack & Theta, Theta & Attack
        raw_atk = (m1["Attack"] * m2["Attack"]) + (m1["Attack"] * m2["Theta"]) + (m1["Theta"] * m2["Attack"])
        # Benign & Benign, Benign & Theta, Theta & Benign
        raw_ben = (m1["Benign"] * m2["Benign"]) + (m1["Benign"] * m2["Theta"]) + (m1["Theta"] * m2["Benign"])
        # Theta & Theta
        raw_theta = m1["Theta"] * m2["Theta"]

        apply_yager = (self.conflict_rule == "yager" or k >= self.conflict_threshold) if use_yager is None else use_yager

        if apply_yager:
            # Yager's Rule: Allocate conflicting mass K to epistemic uncertainty Theta
            # m(A) = raw_atk, m(B) = raw_ben, m(Theta) = raw_theta + K
            m_atk = raw_atk
            m_ben = raw_ben
            m_theta = raw_theta + k
        else:
            # Standard Dempster Rule: Normalize by 1 - K
            denom = max(1e-9, 1.0 - k)
            m_atk = raw_atk / denom
            m_ben = raw_ben / denom
            m_theta = raw_theta / denom

        tot = m_atk + m_ben + m_theta
        combined = {
            "Attack": float(m_atk / tot),
            "Benign": float(m_ben / tot),
            "Theta": float(m_theta / tot),
        }
        return combined, k

    def fuse(
        self,
        ml_prob: float,
        anomaly_score: float,
        temporal_score: float,
    ) -> Dict[str, Any]:
        """
        Fuses three orthogonal discounted evidence sources:
        1. Calibrated ML Attack Score
        2. Unsupervised Anomaly Score
        3. Rolling Temporal Deviation Score
        """
        # Formulate discounted BPAs
        bpa_ml = self.create_bpa(ml_prob, source_name="ml_attack_score")
        bpa_anom = self.create_bpa(anomaly_score, source_name="anomaly_score")
        bpa_temp = self.create_bpa(temporal_score, source_name="temporal_deviation")

        # Stage 1: Combine ML and Anomaly
        bpa_12, k1 = self.combine_two_bpas(bpa_ml, bpa_anom)

        # Stage 2: Combine composite with Temporal Deviation
        bpa_123, k2 = self.combine_two_bpas(bpa_12, bpa_temp)

        # Cumulative conflict metric
        total_conflict = float(1.0 - (1.0 - k1) * (1.0 - k2))

        # Evidential measures
        bel_attack = bpa_123["Attack"]
        pl_attack = bpa_123["Attack"] + bpa_123["Theta"]
        uncertainty = bpa_123["Theta"]

        # Pignistic Probability Transformation BetP(Attack)
        betp_attack = float(bel_attack + 0.5 * uncertainty)

        return {
            "risk_score": float(np.clip(betp_attack, 0.0, 1.0)),
            "belief": float(bel_attack),
            "plausibility": float(pl_attack),
            "uncertainty": float(uncertainty),
            "conflict": float(total_conflict),
            "high_conflict_warning": bool(total_conflict >= self.conflict_threshold),
            "bpa": bpa_123,
            "sources": {
                "ml_attack_score": float(ml_prob),
                "anomaly_score": float(anomaly_score),
                "temporal_deviation": float(temporal_score),
            },
        }

    def fuse_batch(
        self,
        ml_probs: np.ndarray,
        anomaly_scores: np.ndarray,
        temporal_scores: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Vectorized/batched fusion across arrays of flows."""
        n = len(ml_probs)
        risks = np.zeros(n, dtype=np.float64)
        beliefs = np.zeros(n, dtype=np.float64)
        plausibilities = np.zeros(n, dtype=np.float64)
        uncertainties = np.zeros(n, dtype=np.float64)
        conflicts = np.zeros(n, dtype=np.float64)

        for i in range(n):
            res = self.fuse(
                float(ml_probs[i]),
                float(anomaly_scores[i]),
                float(temporal_scores[i]),
            )
            risks[i] = res["risk_score"]
            beliefs[i] = res["belief"]
            plausibilities[i] = res["plausibility"]
            uncertainties[i] = res["uncertainty"]
            conflicts[i] = res["conflict"]

        return {
            "risk_scores": risks,
            "beliefs": beliefs,
            "plausibilities": plausibilities,
            "uncertainties": uncertainties,
            "conflicts": conflicts,
        }


# Canonical Hyperparameter Tuning Grid for DST Source Reliabilities
DEFAULT_DST_TUNING_GRID = {
    "ml_attack_score": [0.70, 0.75, 0.80, 0.85, 0.90],
    "anomaly_score": [0.60, 0.65, 0.70, 0.75, 0.80],
    "temporal_deviation": [0.50, 0.60, 0.70, 0.75, 0.80],
}


def calibrate_dst_reliabilities(
    y_val: np.ndarray,
    ml_scores: np.ndarray,
    anom_scores: np.ndarray,
    temp_scores: np.ndarray,
    tuning_grid: Optional[Dict[str, List[float]]] = None,
    uncertainty_budget: float = 0.15,
) -> Dict[str, Any]:
    """
    Performs validation-based estimation of DST source reliabilities across a defined grid.
    Selects the combination minimizing validation Brier score with validation PR-AUC as tiebreaker.
    
    If validation data lacks attack labels (e.g., pure benign cold-start), returns documented
    validation-tuned fixed hyperparameters: r_ml=0.85, r_anom=0.75, r_temp=0.70.
    """
    y_val_arr = np.asarray(y_val, dtype=int)
    grid = tuning_grid or DEFAULT_DST_TUNING_GRID

    # Check if both classes exist in validation data
    unique_labels = np.unique(y_val_arr)
    if len(unique_labels) < 2:
        # Fallback to validation-tuned default parameters with documented grid
        return {
            "best_reliabilities": {
                "ml_attack_score": 0.85,
                "anomaly_score": 0.75,
                "temporal_deviation": 0.70,
            },
            "status": "fixed_validation_tuned_hyperparameters",
            "tuning_grid": grid,
            "best_brier_score": None,
        }

    from sklearn.metrics import brier_score_loss, average_precision_score

    best_brier = float("inf")
    best_pr_auc = -1.0
    best_rels = {
        "ml_attack_score": 0.85,
        "anomaly_score": 0.75,
        "temporal_deviation": 0.70,
    }

    # Evaluate grid
    for r_ml in grid["ml_attack_score"]:
        for r_anom in grid["anomaly_score"]:
            for r_temp in grid["temporal_deviation"]:
                rels = {
                    "ml_attack_score": float(r_ml),
                    "anomaly_score": float(r_anom),
                    "temporal_deviation": float(r_temp),
                }
                engine = DSTEvidenceFusion(
                    uncertainty_budget=uncertainty_budget,
                    source_reliabilities=rels,
                )
                fused = engine.fuse_batch(ml_scores, anom_scores, temp_scores)
                risks = fused["risk_scores"]
                brier = float(brier_score_loss(y_val_arr, risks))
                pr_auc = float(average_precision_score(y_val_arr, risks))

                if (brier < best_brier - 1e-4) or (abs(brier - best_brier) <= 1e-4 and pr_auc > best_pr_auc):
                    best_brier = brier
                    best_pr_auc = pr_auc
                    best_rels = rels

    return {
        "best_reliabilities": best_rels,
        "status": "validation_estimated",
        "tuning_grid": grid,
        "best_brier_score": round(best_brier, 4),
        "best_pr_auc": round(best_pr_auc, 4),
    }
