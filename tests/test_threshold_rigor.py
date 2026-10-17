"""
Tests for Validation-Calibrated Adaptive Threshold Engine.
"""

import pytest
import numpy as np

from src.risk.adaptive_threshold import (
    optimize_threshold_on_validation,
    AdaptiveThresholdManager,
)


def test_threshold_recall_constrained_optimization():
    """Verify Mode 1 finds threshold achieving >= 90% recall on validation holdout."""
    np.random.seed(42)
    n = 1000
    y_val = np.array([0] * 700 + [1] * 300)
    # Benign scores distributed near 0.15, attack scores near 0.70
    scores_benign = np.random.beta(2, 8, size=700)
    scores_attack = np.random.beta(7, 3, size=300)
    scores = np.concatenate([scores_benign, scores_attack])

    res = optimize_threshold_on_validation(
        y_val,
        scores,
        mode="recall_constrained",
        min_recall=0.90,
    )

    assert res["validation_recall"] >= 0.90, f"Expected recall >= 0.90, got {res['validation_recall']}"
    assert res["validation_fnr"] <= 0.10, "FNR must be <= 10%"
    assert 0.15 <= res["optimal_threshold"] <= 0.85


def test_threshold_f2_optimization():
    """Verify Mode 2 maximizes F2 score."""
    y_val = np.array([0] * 500 + [1] * 100)
    scores = np.array([0.10] * 500 + [0.80] * 100)

    res = optimize_threshold_on_validation(y_val, scores, mode="f_beta", beta=2.0)
    assert res["validation_recall"] == 1.0
    assert res["validation_fpr"] == 0.0


def test_protected_benign_buffer_excludes_attack_scores():
    """Verify adaptive threshold only updates benign buffer from low-risk scores."""
    mgr = AdaptiveThresholdManager(initial_threshold=0.50, min_threshold=0.20, max_threshold=0.85)

    # Ingest a batch of high-risk attack scores (0.85 - 0.95)
    attack_scores = np.array([0.85, 0.90, 0.92, 0.88, 0.95])
    mgr.update(attack_scores, drift_detected=False)

    # The protected benign buffer should NOT contain these high-risk scores
    for b_score in mgr.benign_risk_buffer:
        assert b_score < 0.50 * 0.70, f"Attack score {b_score} contaminated benign buffer!"


def test_attack_burst_freezes_threshold():
    """Verify severe attack bursts freeze threshold to prevent threshold inflation."""
    mgr = AdaptiveThresholdManager(
        initial_threshold=0.50,
        freeze_during_burst=True,
    )
    # Pre-fill buffer with normal benign scores
    mgr.benign_risk_buffer.extend([0.05, 0.08, 0.10, 0.06, 0.09] * 10)

    curr_before = mgr.get_threshold()

    # Ingest severe burst: 80% attack scores
    burst = np.array([0.85] * 80 + [0.05] * 20)
    mgr.update(burst, drift_detected=False)

    # Threshold must freeze or not blow up
    curr_after = mgr.get_threshold()
    assert curr_after <= 0.55, f"Threshold inflated during attack burst: {curr_after}"


def test_drift_relaxation_lowers_threshold():
    """Verify concept drift alarm triggers threshold relaxation."""
    mgr = AdaptiveThresholdManager(
        initial_threshold=0.60,
        drift_relaxation_factor=0.80,
    )
    t_before = mgr.get_threshold()
    mgr.update(np.array([0.20, 0.30]), drift_detected=True)
    t_after = mgr.get_threshold()

    assert t_after < t_before, "Concept drift alarm must relax (lower) the threshold to heighten sensitivity"
