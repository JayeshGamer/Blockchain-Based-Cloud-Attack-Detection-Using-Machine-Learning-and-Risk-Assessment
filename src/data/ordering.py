"""
Unified Chronological Replay Ordering & Stream Fingerprinting Engine.
Ensures identical file/day-and-row chronological ordering across Protocol B
and all chronological ablation variants.
"""

import hashlib
import json
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.data.deduplication import CHRONOLOGICAL_FILES

logger = logging.getLogger(__name__)

# Canonical chronological day rank: Monday -> Friday
CHRONOLOGICAL_DAY_RANKS: Dict[str, int] = {
    day: idx for idx, (_, day) in enumerate(CHRONOLOGICAL_FILES)
}

# Source file to day rank mapping
CHRONOLOGICAL_FILE_RANKS: Dict[str, int] = {
    fname: idx for idx, (fname, _) in enumerate(CHRONOLOGICAL_FILES)
}


def order_chronological_stream(
    df: pd.DataFrame,
    strict_row_index: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Orders a DataFrame strictly according to the declared file/day-and-row replay order:
      1. Chronological day rank (Monday -> Tuesday -> Wednesday -> Thursday Morning ->
         Thursday Afternoon -> Friday Morning -> Friday PortScan -> Friday DDoS).
      2. Original row index within each source file.

    Parameters:
        df: Input DataFrame containing traffic flows.
        strict_row_index: If True, requires 'original_row_index' column.

    Returns:
        (df_ordered, stream_fingerprint)
        stream_fingerprint: SHA-256 digest of ordered (source_day, original_row_index) sequence.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    # Validate metadata columns
    if "source_day" not in df.columns and "source_file" not in df.columns:
        raise ValueError(
            "DataFrame lacks ordering metadata: at least 'source_day' or 'source_file' must be present. "
            "Replaying an un-ordered or arbitrarily shuffled stream violates chronological validity."
        )

    if strict_row_index and "original_row_index" not in df.columns:
        raise ValueError(
            "DataFrame missing required 'original_row_index' column. "
            "Replay order requires preserving original within-file row index sequence."
        )

    df_copy = df.copy()

    # Determine day rank
    if "source_day" in df_copy.columns:
        # Check for unknown source days
        unknown_days = set(df_copy["source_day"].dropna().unique()) - set(CHRONOLOGICAL_DAY_RANKS.keys())
        if unknown_days:
            raise ValueError(f"Unknown source_day values detected: {unknown_days}. Must match declared sequence.")
        df_copy["day_rank"] = df_copy["source_day"].map(CHRONOLOGICAL_DAY_RANKS)
    else:
        unknown_files = set(df_copy["source_file"].dropna().unique()) - set(CHRONOLOGICAL_FILE_RANKS.keys())
        if unknown_files:
            raise ValueError(f"Unknown source_file values detected: {unknown_files}.")
        df_copy["day_rank"] = df_copy["source_file"].map(CHRONOLOGICAL_FILE_RANKS)

    # Sort by day rank, then by original row index
    sort_cols = ["day_rank"]
    if "original_row_index" in df_copy.columns:
        sort_cols.append("original_row_index")

    df_ordered = df_copy.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)

    # Compute deterministic cryptographic stream fingerprint
    # Fingerprint incorporates day_rank, original_row_index, and row_hash if present
    hasher = hashlib.sha256()
    ranks = df_ordered["day_rank"].to_numpy(dtype=np.int64)
    hasher.update(ranks.tobytes())

    if "original_row_index" in df_ordered.columns:
        row_indices = df_ordered["original_row_index"].to_numpy(dtype=np.int64)
        hasher.update(row_indices.tobytes())

    if "row_hash" in df_ordered.columns:
        hash_concat = "".join(df_ordered["row_hash"].astype(str).tolist())
        hasher.update(hash_concat.encode("utf-8"))

    stream_fingerprint = hasher.hexdigest()

    logger.info(
        f"Ordered stream: {len(df_ordered):,} flows | "
        f"First day: {df_ordered.iloc[0].get('source_day')} | "
        f"Last day: {df_ordered.iloc[-1].get('source_day')} | "
        f"Fingerprint: {stream_fingerprint[:16]}..."
    )

    return df_ordered, stream_fingerprint


def verify_stream_ordering(df: pd.DataFrame) -> bool:
    """
    Verifies that a DataFrame is monotonically non-decreasing in day_rank,
    and monotonically non-decreasing in original_row_index within each day.
    """
    if "day_rank" not in df.columns:
        if "source_day" in df.columns:
            ranks = df["source_day"].map(CHRONOLOGICAL_DAY_RANKS).values
        else:
            return False
    else:
        ranks = df["day_rank"].values

    day_diffs = np.diff(ranks)
    if np.any(day_diffs < 0):
        return False

    if "original_row_index" in df.columns:
        # Group by day and check monotonicity within each day
        for _, group in df.groupby(ranks):
            idx_diffs = np.diff(group["original_row_index"].values)
            if np.any(idx_diffs < 0):
                return False

    return True


def save_stream_manifest(
    df_ordered: pd.DataFrame,
    fingerprint: str,
    output_path: str,
) -> Dict[str, Any]:
    """Saves ordered stream metadata and fingerprint manifest."""
    day_counts = df_ordered["source_day"].value_counts().to_dict() if "source_day" in df_ordered.columns else {}
    manifest = {
        "stream_length": len(df_ordered),
        "stream_fingerprint": fingerprint,
        "per_day_counts": day_counts,
        "ordering_convention": "declared_file_day_and_original_row_index",
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
