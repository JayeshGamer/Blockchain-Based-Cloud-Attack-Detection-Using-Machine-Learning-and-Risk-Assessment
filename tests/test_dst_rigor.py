"""
Tests for Dempster-Shafer Evidence Fusion, Shafer Reliability Discounting, and Conflict Handling.
"""

import pytest
import numpy as np

from src.dst.evidence_fusion import DSTEvidenceFusion


def test_dst_mass_conservation():
    """Verify BPA and combined masses strictly sum to 1.0."""
    dst = DSTEvidenceFusion(uncertainty_budget=0.15)
    for p in [0.0, 0.25, 0.50, 0.75, 1.0]:
        bpa = dst.create_bpa(p)
        assert pytest.approx(sum(bpa.values()), abs=1e-6) == 1.0


def test_dst_shafer_reliability_discounting():
    """Verify classical Shafer discounting shifts mass to uncertainty Theta."""
    dst_reliable = DSTEvidenceFusion(
        uncertainty_budget=0.10,
        source_reliabilities={"src_a": 1.0},
    )
    dst_unreliable = DSTEvidenceFusion(
        uncertainty_budget=0.10,
        source_reliabilities={"src_a": 0.50},
    )

    bpa_rel = dst_reliable.create_bpa(0.90, source_name="src_a")
    bpa_unrel = dst_unreliable.create_bpa(0.90, source_name="src_a")

    # Lower reliability must result in significantly higher uncertainty Theta
    assert bpa_unrel["Theta"] > bpa_rel["Theta"]
    # Theoretical discounted Theta: 1 - r + r * u = 1 - 0.5 + 0.5 * 0.10 = 0.55
    assert pytest.approx(bpa_unrel["Theta"], abs=1e-3) == 0.55


def test_dst_commutativity():
    """Verify combination is commutative: m1 (+) m2 == m2 (+) m1."""
    dst = DSTEvidenceFusion()
    bpa1 = dst.create_bpa(0.80)
    bpa2 = dst.create_bpa(0.30)

    comb12, k12 = dst.combine_two_bpas(bpa1, bpa2)
    comb21, k21 = dst.combine_two_bpas(bpa2, bpa1)

    assert pytest.approx(k12, abs=1e-6) == k21
    assert pytest.approx(comb12["Attack"], abs=1e-6) == comb21["Attack"]
    assert pytest.approx(comb12["Benign"], abs=1e-6) == comb21["Benign"]
    assert pytest.approx(comb12["Theta"], abs=1e-6) == comb21["Theta"]


def test_dst_associativity():
    """Verify combination is associative within numerical tolerance: (m1 (+) m2) (+) m3 == m1 (+) (m2 (+) m3)."""
    dst = DSTEvidenceFusion(conflict_rule="dempster")
    bpa1 = dst.create_bpa(0.70)
    bpa2 = dst.create_bpa(0.60)
    bpa3 = dst.create_bpa(0.80)

    # (1 + 2) + 3
    comb12, _ = dst.combine_two_bpas(bpa1, bpa2)
    comb_12_3, _ = dst.combine_two_bpas(comb12, bpa3)

    # 1 + (2 + 3)
    comb23, _ = dst.combine_two_bpas(bpa2, bpa3)
    comb_1_23, _ = dst.combine_two_bpas(bpa1, comb23)

    assert pytest.approx(comb_12_3["Attack"], abs=1e-4) == comb_1_23["Attack"]
    assert pytest.approx(comb_12_3["Benign"], abs=1e-4) == comb_1_23["Benign"]
    assert pytest.approx(comb_12_3["Theta"], abs=1e-4) == comb_1_23["Theta"]


def test_dst_near_total_conflict_yager():
    """Verify Yager's rule allocates high conflict to epistemic uncertainty Theta."""
    dst = DSTEvidenceFusion(conflict_rule="yager", uncertainty_budget=0.01)
    # Total disagreement: Source 1 says Attack=1.0, Source 2 says Attack=0.0 (Benign=1.0)
    bpa_atk = {"Attack": 0.99, "Benign": 0.0, "Theta": 0.01}
    bpa_ben = {"Attack": 0.0, "Benign": 0.99, "Theta": 0.01}

    combined, k = dst.combine_two_bpas(bpa_atk, bpa_ben, use_yager=True)
    assert k > 0.95, "Conflict factor must be high for contradictory evidence"
    # Under Yager, conflicting mass is reallocated to Theta
    assert combined["Theta"] > 0.90, "Yager's rule must reallocate high conflict to Theta"


def test_dst_nan_rejection():
    """Verify invalid and NaN inputs are rejected."""
    dst = DSTEvidenceFusion()
    with pytest.raises(ValueError):
        dst.create_bpa(float("nan"))
    with pytest.raises(ValueError):
        dst.create_bpa(float("inf"))
