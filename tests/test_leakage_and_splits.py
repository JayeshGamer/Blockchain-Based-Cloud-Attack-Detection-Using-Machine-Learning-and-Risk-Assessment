"""
Tests for Data Sanitization, Global Deduplication, Hash Isolation, and Anti-Leakage.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data.audit import normalize_label, ATTACK_FAMILY_MAPPING
from src.data.deduplication import compute_row_hashes
from src.data.pipeline import create_hash_disjoint_splits, generate_split_integrity_report
from src.features.preprocessor import NetworkDataPreprocessor, DEFAULT_CONSTANT_COLUMNS


def test_label_normalization():
    """Verify all unicode corrupted variants and casing normalize properly."""
    assert normalize_label("BENIGN") == "BENIGN"
    assert normalize_label("Web Attack \ufffd XSS") == "Web Attack - XSS"
    assert normalize_label("Web Attack  Brute Force") == "Web Attack - Brute Force"
    assert normalize_label("Web Attack \ufffd Sql Injection") == "Web Attack - Sql Injection"
    assert normalize_label("  DDoS  ") == "DDoS"
    assert normalize_label("Heartbleed") == "Heartbleed"


def test_global_deduplication_hashing():
    """Verify identical numeric vectors produce identical SHA-256 hashes."""
    row1 = np.array([[1.0, 2.0, 3.5, np.nan, 100.0]])
    row2 = np.array([[1.0, 2.0, 3.5, np.nan, 100.0]])
    row3 = np.array([[1.0, 2.0, 3.5, 0.0, 100.0]])

    h1 = compute_row_hashes(row1)[0]
    h2 = compute_row_hashes(row2)[0]
    h3 = compute_row_hashes(row3)[0]

    assert h1 == h2, "Identical rows must produce identical hashes"
    assert h1 != h3, "Distinct rows must produce distinct hashes"


def test_hash_disjoint_split_isolation():
    """Verify train, validation, and test splits have ZERO hash overlap."""
    np.random.seed(42)
    n = 600
    feats = np.random.randn(n, 10)
    hashes = compute_row_hashes(feats)
    labels = ["BENIGN"] * 400 + ["DoS Hulk"] * 150 + ["PortScan"] * 50

    df = pd.DataFrame(feats, columns=[f"feat_{i}" for i in range(10)])
    df["row_hash"] = hashes
    df["NormalizedLabel"] = labels
    df["source_file"] = "test.csv"
    df["source_day"] = "Monday"
    df["original_row_index"] = list(range(n))

    tr, val, te = create_hash_disjoint_splits(df, test_size=0.2, val_size=0.1, random_seed=42)

    tr_hashes = set(tr["row_hash"])
    val_hashes = set(val["row_hash"])
    te_hashes = set(te["row_hash"])

    assert len(tr_hashes.intersection(val_hashes)) == 0, "Train and Val must be hash-disjoint"
    assert len(tr_hashes.intersection(te_hashes)) == 0, "Train and Test must be hash-disjoint"
    assert len(val_hashes.intersection(te_hashes)) == 0, "Val and Test must be hash-disjoint"


def test_zero_leakage_feature_extraction():
    """Verify features DataFrame strictly excludes labels and metadata."""
    df = pd.DataFrame({
        "Flow Duration": [10, 20],
        "Total Fwd Packets": [2, 4],
        "source_file": ["f1.csv", "f2.csv"],
        "source_day": ["Monday", "Tuesday"],
        "original_row_index": [100, 101],
        "row_hash": ["h1", "h2"],
        "NormalizedLabel": ["BENIGN", "DDoS"],
        "AttackFamily": ["BENIGN", "DDoS"],
    })

    prep = NetworkDataPreprocessor()
    feats, y_bin, y_multi, y_fam = prep.extract_and_clean_labels(df)

    assert "NormalizedLabel" not in feats.columns
    assert "source_file" not in feats.columns
    assert "source_day" not in feats.columns
    assert "original_row_index" not in feats.columns
    assert "row_hash" not in feats.columns
    assert "AttackFamily" not in feats.columns
    assert list(feats.columns) == ["Flow Duration", "Total Fwd Packets"]
    assert list(y_bin) == [0, 1]


def test_training_only_scaler_and_imputer():
    """Verify median imputation and scaling parameters are strictly learned from training."""
    prep = NetworkDataPreprocessor()
    df_train = pd.DataFrame({
        "FeatA": [10.0, np.nan, 30.0],
        "FeatB": [100.0, 200.0, 300.0],
        "Label": ["BENIGN", "BENIGN", "DDoS"],
    })
    X_tr_feats, y_bin, y_multi, _ = prep.extract_and_clean_labels(df_train)
    prep.fit(X_tr_feats, y_multi)

    # Median of [10.0, 30.0] is 20.0
    assert prep.impute_values_["FeatA"] == 20.0
    assert prep.is_fitted_ is True

    # Test transform on unseen data with NaN
    df_test = pd.DataFrame({
        "FeatA": [np.nan],
        "FeatB": [150.0],
    })
    transformed = prep.transform(df_test)
    assert not np.isnan(transformed).any(), "Transformed output must have zero NaNs"
