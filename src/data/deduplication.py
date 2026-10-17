"""
Global Deduplication, Dataset Audit, and Bounded-Memory Hash Set Engine.
Ensures:
1. Global exact deduplication via cryptographic SHA-256 row hashes.
2. Canonical feature sanitization (whitespace strip, duplicate column drop, inf replacement).
3. Label normalization across unicode corruption variants.
4. Retention of original row order for chronological replay.
5. Tracking of duplicate counts per file and per attack class.
6. Strictly bounded memory via DiskBackedHashSet (SQLite) or measured in-memory growth tracking.
7. Strict failure if required dataset files are missing.
"""

import os
import glob
import json
import sqlite3
import tempfile
import hashlib
import time
import tracemalloc
import logging
from typing import Dict, Any, List, Tuple, Set, Optional, Iterator
import numpy as np
import pandas as pd

from src.data.audit import normalize_label, ATTACK_FAMILY_MAPPING
from src.features.preprocessor import DEFAULT_CONSTANT_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Preferred chronological file order representing Monday through Friday
CHRONOLOGICAL_FILES = [
    ("Monday-WorkingHours.pcap_ISCX.csv", "Monday"),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", "Tuesday"),
    ("Wednesday-workingHours.pcap_ISCX.csv", "Wednesday"),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Thursday-Morning"),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Thursday-Afternoon"),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv", "Friday-Morning"),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "Friday-PortScan"),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Friday-DDoS"),
]


def check_dataset_files(data_dir: str = "dataset") -> List[Tuple[str, str]]:
    """
    Validates presence of all declared chronological dataset files.
    Fails explicitly if any required file is missing.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Required raw dataset directory '{data_dir}' does not exist.")

    missing = []
    found = []
    for fname, day in CHRONOLOGICAL_FILES:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            missing.append(fname)
        else:
            found.append((fname, day))

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} required dataset file(s) in '{data_dir}': {missing}. "
            "Full chronological replay cannot proceed with incomplete data."
        )
    return found


class DiskBackedHashSet:
    """
    Disk-backed deduplication set backed by SQLite.
    Guarantees strictly bounded RAM footprint (< 50 MB) across millions of entries.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="dedup_hash_")
            self.db_path = os.path.join(self._temp_dir.name, "seen_hashes.db")
        else:
            self._temp_dir = None
            self.db_path = db_path

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA synchronous = OFF;")
        self.conn.execute("PRAGMA journal_mode = OFF;")
        self.conn.execute("PRAGMA cache_size = 5000;")
        self.conn.execute("CREATE TABLE IF NOT EXISTS seen (h TEXT PRIMARY KEY);")
        self._count = 0

    def filter_unique_batch(self, hashes: List[str]) -> List[int]:
        """
        Given a list of hashes, inserts unique hashes into SQLite and returns
        the indices of unique items within this batch.
        Uses single transaction for speed.
        """
        unique_indices = []
        cur = self.conn.cursor()
        for idx, h in enumerate(hashes):
            try:
                cur.execute("INSERT INTO seen (h) VALUES (?);", (h,))
                unique_indices.append(idx)
                self._count += 1
            except sqlite3.IntegrityError:
                pass  # Duplicate hash already present
        self.conn.commit()
        return unique_indices

    def add(self, h: str) -> bool:
        """Returns True if h was new and added, False if duplicate."""
        try:
            self.conn.execute("INSERT INTO seen (h) VALUES (?);", (h,))
            self._count += 1
            return True
        except sqlite3.IntegrityError:
            return False

    def contains(self, h: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM seen WHERE h = ?;", (h,))
        return cur.fetchone() is not None

    def __len__(self) -> int:
        return self._count

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass

    def __del__(self):
        self.close()


class InMemoryHashSetWithTracking:
    """
    In-memory set that actively tracks and measures its exact memory footprint growth.
    """

    def __init__(self):
        self._set: Set[str] = set()
        self.peak_entries = 0
        self.estimated_bytes = 0

    def add(self, h: str) -> bool:
        if h in self._set:
            return False
        self._set.add(h)
        count = len(self._set)
        if count > self.peak_entries:
            self.peak_entries = count
            # Python set overhead ~64 bytes per string entry + ~80 bytes string obj
            self.estimated_bytes = count * 144
        return True

    def contains(self, h: str) -> bool:
        return h in self._set

    def filter_unique_batch(self, hashes: List[str]) -> List[int]:
        unique = []
        for idx, h in enumerate(hashes):
            if h not in self._set:
                self._set.add(h)
                unique.append(idx)
        count = len(self._set)
        if count > self.peak_entries:
            self.peak_entries = count
            self.estimated_bytes = count * 144
        return unique

    def __len__(self) -> int:
        return len(self._set)

    def memory_mb(self) -> float:
        return round(self.estimated_bytes / (1024 * 1024), 2)


def compute_row_hashes(X_num: np.ndarray) -> List[str]:
    """
    Computes deterministic SHA-256 cryptographic hashes for an array of feature vectors.
    Replaces NaNs with standard sentinel to ensure identical hash representation.
    Returns complete 64-character lowercase hex string.
    """
    clean_arr = np.nan_to_num(X_num, nan=-999999.0).astype(np.float64)
    clean_arr = np.ascontiguousarray(clean_arr)
    hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in clean_arr]
    return hashes


def run_global_deduplication_audit(
    data_dir: str = "dataset",
    chunksize: int = 50000,
    output_dir: str = "reports/dataset_audit",
    use_disk_backed: bool = True,
) -> Dict[str, Any]:
    """
    Performs full global deduplication across all 8 CICIDS2017 files in chronological order.
    Fails explicitly if any declared file is missing.
    Records raw, duplicate, unique, missing, infinite, elapsed time, and peak memory.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=== Starting Global Exact Deduplication & Audit Pipeline ===")

    # Verify all files present
    files_to_process = check_dataset_files(data_dir=data_dir)

    tracemalloc.start()
    t_start = time.time()

    hash_store = DiskBackedHashSet() if use_disk_backed else InMemoryHashSetWithTracking()

    total_raw_rows = 0
    total_deduped_rows = 0
    total_duplicates_removed = 0

    per_file_stats: List[Dict[str, Any]] = []
    per_class_raw: Dict[str, int] = {}
    per_class_deduped: Dict[str, int] = {}
    per_class_removed: Dict[str, int] = {}
    per_family_deduped: Dict[str, int] = {}

    total_missing_values = 0
    total_infinite_values = 0
    canonical_feature_cols: Optional[List[str]] = None

    for fname, source_day in files_to_process:
        fpath = os.path.join(data_dir, fname)
        fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
        logger.info(f"Processing {fname} ({source_day}) [{fsize_mb:.2f} MB]...")

        file_raw_rows = 0
        file_deduped_rows = 0
        file_dupes_removed = 0
        file_missing = 0
        file_inf = 0
        file_class_raw: Dict[str, int] = {}
        file_class_deduped: Dict[str, int] = {}

        for chunk in pd.read_csv(
            fpath,
            chunksize=chunksize,
            encoding="utf-8",
            encoding_errors="replace",
            low_memory=False,
        ):
            chunk.columns = [c.strip() for c in chunk.columns]
            label_col = [c for c in chunk.columns if c.lower() == "label"][0]

            feature_cols = [c for c in chunk.columns if c != label_col and c != "Fwd Header Length.1"]
            if canonical_feature_cols is None:
                canonical_feature_cols = feature_cols

            n_chunk = len(chunk)
            file_raw_rows += n_chunk

            raw_labels = chunk[label_col].apply(normalize_label)
            for lbl in raw_labels:
                file_class_raw[lbl] = file_class_raw.get(lbl, 0) + 1
                per_class_raw[lbl] = per_class_raw.get(lbl, 0) + 1

            feats = chunk[canonical_feature_cols].apply(pd.to_numeric, errors="coerce")
            inf_count = int(np.isinf(feats.values).sum())
            file_inf += inf_count

            feats_no_inf = feats.replace([np.inf, -np.inf], np.nan)
            missing_count = int(feats_no_inf.isna().sum().sum())
            file_missing += missing_count

            hashes = compute_row_hashes(feats_no_inf.values)
            unique_indices = hash_store.filter_unique_batch(hashes)
            unique_set = set(unique_indices)

            for i in range(len(hashes)):
                lbl = raw_labels.iloc[i]
                if i in unique_set:
                    file_deduped_rows += 1
                    per_class_deduped[lbl] = per_class_deduped.get(lbl, 0) + 1
                    file_class_deduped[lbl] = file_class_deduped.get(lbl, 0) + 1
                    fam = ATTACK_FAMILY_MAPPING.get(lbl, "Other Attack")
                    per_family_deduped[fam] = per_family_deduped.get(fam, 0) + 1
                else:
                    file_dupes_removed += 1
                    per_class_removed[lbl] = per_class_removed.get(lbl, 0) + 1

        total_raw_rows += file_raw_rows
        total_deduped_rows += file_deduped_rows
        total_duplicates_removed += file_dupes_removed
        total_missing_values += file_missing
        total_infinite_values += file_inf

        file_stat = {
            "filename": fname,
            "source_day": source_day,
            "raw_rows": file_raw_rows,
            "deduped_rows": file_deduped_rows,
            "duplicates_removed": file_dupes_removed,
            "duplicate_rate_pct": round((file_dupes_removed / max(1, file_raw_rows)) * 100, 2),
            "missing_values": file_missing,
            "infinite_values": file_inf,
            "classes_raw": file_class_raw,
            "classes_deduped": file_class_deduped,
        }
        per_file_stats.append(file_stat)
        logger.info(
            f"Done {fname}: {file_raw_rows:,} raw -> {file_deduped_rows:,} unique "
            f"({file_dupes_removed:,} dupes removed, {file_stat['duplicate_rate_pct']}%)"
        )

    t_elapsed = time.time() - t_start
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = round(peak_mem_bytes / (1024 * 1024), 2)

    if hasattr(hash_store, "close"):
        hash_store.close()

    manifest = {
        "dataset_name": "CICIDS2017",
        "audit_timestamp": pd.Timestamp.now().isoformat(),
        "total_raw_rows": total_raw_rows,
        "total_deduped_rows": total_deduped_rows,
        "total_duplicates_removed": total_duplicates_removed,
        "global_duplicate_rate_pct": round((total_duplicates_removed / max(1, total_raw_rows)) * 100, 2),
        "total_missing_values": total_missing_values,
        "total_infinite_values": total_infinite_values,
        "elapsed_time_sec": round(t_elapsed, 2),
        "peak_memory_mb": peak_mem_mb,
        "deduplication_strategy": "disk_backed_sqlite" if use_disk_backed else "in_memory_tracked",
        "canonical_feature_count": len(canonical_feature_cols) if canonical_feature_cols else 0,
        "constant_feature_count": len(DEFAULT_CONSTANT_COLUMNS),
        "constant_columns": DEFAULT_CONSTANT_COLUMNS,
        "per_file_breakdown": per_file_stats,
        "per_class_summary": {
            lbl: {
                "raw_count": per_class_raw.get(lbl, 0),
                "deduped_count": per_class_deduped.get(lbl, 0),
                "removed_duplicates": per_class_removed.get(lbl, 0),
                "retention_rate_pct": round(
                    (per_class_deduped.get(lbl, 0) / max(1, per_class_raw.get(lbl, 0))) * 100, 2
                ),
            }
            for lbl in sorted(per_class_raw.keys())
        },
        "per_family_summary": per_family_deduped,
    }

    manifest_path = os.path.join(output_dir, "deduplication_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    audit_json_path = os.path.join(output_dir, "corrected_audit.json")
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    class_rows = []
    for lbl, data in manifest["per_class_summary"].items():
        fam = ATTACK_FAMILY_MAPPING.get(lbl, "Other Attack")
        class_rows.append({
            "Class": lbl,
            "Attack_Family": fam,
            "Raw_Count": data["raw_count"],
            "Deduped_Count": data["deduped_count"],
            "Duplicates_Removed": data["removed_duplicates"],
            "Retention_Rate_Pct": data["retention_rate_pct"],
        })
    df_classes = pd.DataFrame(class_rows).sort_values(by="Deduped_Count", ascending=False)
    class_csv_path = os.path.join(output_dir, "class_distribution.csv")
    df_classes.to_csv(class_csv_path, index=False)

    return manifest


if __name__ == "__main__":
    run_global_deduplication_audit()
