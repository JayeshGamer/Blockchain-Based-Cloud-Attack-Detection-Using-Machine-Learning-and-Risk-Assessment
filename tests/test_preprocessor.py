"""
Unit tests for preprocessing and zero-leakage constraints.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.preprocessor import NetworkDataPreprocessor


def test_preprocessor_zero_leakage_and_scaling():
    # Synthetic dataframe with typical CICIDS2017 quirks: infs, nans, constant columns, whitespace
    data = {
        " Destination Port ": [80, 443, 8080, 22, 53],
        "Flow Duration": [1000, 2000, 3000, 4000, 5000],
        "Flow Bytes/s": [10.5, np.inf, 30.2, -np.inf, np.nan],
        "Bwd Avg Bulk Rate": [0.0, 0.0, 0.0, 0.0, 0.0],  # constant
        "Fwd Header Length": [32, 32, 32, 32, 32],
        "Fwd Header Length.1": [32, 32, 32, 32, 32],     # duplicate
        " Label ": ["BENIGN", "DDoS", "BENIGN", "PortScan", "BENIGN"],
    }
    df = pd.DataFrame(data)

    prep = NetworkDataPreprocessor(drop_constant=True)
    X_df, y_bin, y_multi, y_fam = prep.extract_and_clean_labels(df)

    assert len(y_bin) == 5
    assert y_bin.tolist() == [0, 1, 0, 1, 0]
    assert "Label" not in X_df.columns

    prep.fit(X_df, y_multi)
    X_scaled = prep.transform(X_df)

    # Constant column and duplicate column pruned
    assert "Bwd Avg Bulk Rate" not in prep.feature_names_
    assert "Fwd Header Length.1" not in prep.feature_names_

    # Clean array
    assert not np.isnan(X_scaled).any()
    assert not np.isinf(X_scaled).any()
    assert X_scaled.shape[0] == 5
    assert X_scaled.shape[1] == len(prep.feature_names_)
