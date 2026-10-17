"""
Unit tests for Dempster-Shafer Theory evidence fusion engine.
Verifies:
1. Mass conservation: m(Attack) + m(Benign) + m(Theta) == 1.0
2. Combination rule commutativity: m1 + m2 == m2 + m1
3. Epistemic uncertainty bounds: Bel <= Pl
4. High-conflict stabilization (k -> 1)
"""

import pytest
import numpy as np
from src.dst.evidence_fusion import DSTEvidenceFusion


def test_bpa_mass_conservation():
    fusion = DSTEvidenceFusion(uncertainty_budget=0.15)
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        bpa = fusion.create_bpa(p)
        total = bpa["Attack"] + bpa["Benign"] + bpa["Theta"]
        assert pytest.approx(total, abs=1e-6) == 1.0
        assert bpa["Theta"] == pytest.approx(0.15, abs=1e-6)


def test_dempster_combination_commutativity():
    fusion = DSTEvidenceFusion()
    bpa1 = fusion.create_bpa(0.8)
    bpa2 = fusion.create_bpa(0.3)

    res12, k12 = fusion.combine_two_bpas(bpa1, bpa2)
    res21, k21 = fusion.combine_two_bpas(bpa2, bpa1)

    assert pytest.approx(k12, abs=1e-6) == k21
    assert pytest.approx(res12["Attack"], abs=1e-6) == res21["Attack"]
    assert pytest.approx(res12["Benign"], abs=1e-6) == res21["Benign"]
    assert pytest.approx(res12["Theta"], abs=1e-6) == res21["Theta"]


def test_belief_plausibility_ordering():
    fusion = DSTEvidenceFusion()
    res = fusion.fuse(ml_prob=0.85, anomaly_score=0.70, temporal_score=0.60)

    bel = res["belief"]
    pl = res["plausibility"]
    unc = res["uncertainty"]
    risk = res["risk_score"]

    assert 0.0 <= bel <= pl <= 1.0
    assert pytest.approx(pl - bel, abs=1e-5) == unc
    assert bel <= risk <= pl


def test_extreme_conflict_stabilization():
    fusion = DSTEvidenceFusion()
    # Source 1 completely certain Attack (0.999), Source 2 completely certain Benign (0.001)
    res = fusion.fuse(ml_prob=0.999, anomaly_score=0.001, temporal_score=0.50)

    assert res["conflict"] > 0.50
    assert not np.isnan(res["risk_score"])
    assert not np.isinf(res["risk_score"])
    assert 0.0 <= res["risk_score"] <= 1.0
