"""
Data Loader and Streaming Replay Engine for CICIDS2017.
Supports:
1. Chunked full dataset loading
2. Minority-preserving stratified sampling (for lightweight mode)
3. Stratified train/val/test splits (conventional baseline)
4. Chronological/day-ordered streaming generator for prequential evaluation
"""

import os
import glob
import logging
from typing import Generator, Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.audit import normalize_label, ATTACK_FAMILY_MAPPING

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_stratified_sample(
    data_dir: str = "dataset",
    total_samples: int = 50000,
    random_seed: int = 42,
    preserve_all_rare: bool = True,
    rare_threshold: int = 2500,
) -> pd.DataFrame:
    """
    Builds a stratified sample of CICIDS2017 while preserving 100% of minority attack classes
    (Heartbleed, Infiltration, Sql Injection, XSS, Brute Force, Bot).
    The remaining quota is allocated proportionally across larger attack classes and BENIGN.
    """
    np.random.seed(random_seed)
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Step 1: Read all rows for rare attacks, and sample from larger files
    rare_dfs = []
    large_chunks = []
    
    # Target allocations: e.g. 70% Benign, 30% Attack in the sampled set to ensure balanced learning
    target_benign = int(total_samples * 0.70)
    target_common_attack = int(total_samples * 0.30)
    
    benign_collected = []
    common_attack_collected = []

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        # Read chunks
        for chunk in pd.read_csv(fpath, chunksize=30000, encoding="utf-8", encoding_errors="replace", low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            label_col = [c for c in chunk.columns if c.lower() == "label"][0]
            chunk["NormalizedLabel"] = chunk[label_col].apply(normalize_label)

            # Separate rare attacks, common attacks, and benign
            is_benign = chunk["NormalizedLabel"] == "BENIGN"
            
            # Check rare classes
            counts = chunk["NormalizedLabel"].value_counts()
            for lbl, cnt in counts.items():
                if lbl != "BENIGN":
                    sub = chunk[chunk["NormalizedLabel"] == lbl]
                    if cnt <= rare_threshold or preserve_all_rare and lbl in [
                        "Heartbleed", "Infiltration", "Web Attack - Sql Injection",
                        "Web Attack - XSS", "Web Attack - Brute Force", "Bot"
                    ]:
                        rare_dfs.append(sub)
                    else:
                        common_attack_collected.append(sub)

            benign_rows = chunk[is_benign]
            if len(benign_rows) > 0:
                benign_collected.append(benign_rows)

    # Combine rare attacks
    df_rare = pd.concat(rare_dfs, ignore_index=True) if rare_dfs else pd.DataFrame()
    logger.info(f"Preserved {len(df_rare):,} rare attack instances without downsampling.")

    # Sample benign rows
    df_all_benign = pd.concat(benign_collected, ignore_index=True)
    n_sample_benign = min(len(df_all_benign), target_benign)
    df_benign_sampled = df_all_benign.sample(n=n_sample_benign, random_state=random_seed)

    # Sample common attacks
    df_all_common_attacks = pd.concat(common_attack_collected, ignore_index=True)
    quota_left = max(1000, total_samples - len(df_rare) - len(df_benign_sampled))
    n_sample_attack = min(len(df_all_common_attacks), quota_left)
    df_attack_sampled = df_all_common_attacks.sample(n=n_sample_attack, random_state=random_seed)

    # Combine all parts
    df_final = pd.concat([df_rare, df_benign_sampled, df_attack_sampled], ignore_index=True)
    # Shuffle
    df_final = df_final.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Clean temporary column
    df_final = df_final.drop(columns=["NormalizedLabel"])
    logger.info(f"Built stratified sample: {len(df_final):,} total rows with preserved rare classes.")
    return df_final


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into stratified train, validation, and test splits.
    Preserves class proportions across all three splits.
    """
    label_candidates = [c for c in df.columns if c.strip().lower() in ["label", "normalizedlabel"]]
    if not label_candidates:
        label_candidates = [c for c in df.columns if "label" in c.lower()]
    label_col = label_candidates[0]
    labels = df[label_col].apply(normalize_label)

    # For classes with fewer than 3 instances, we force at least 1 in train
    rare_mask = labels.map(labels.value_counts()) < 3
    df_ultra_rare = df[rare_mask]
    df_regular = df[~rare_mask]
    labels_regular = labels[~rare_mask]

    # Split train+val vs test
    train_val_df, test_df = train_test_split(
        df_regular,
        test_size=test_size,
        stratify=labels_regular,
        random_state=random_seed,
    )

    # Split train vs val
    adj_val_size = val_size / (1.0 - test_size)
    labels_tv = train_val_df[label_col].apply(normalize_label)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adj_val_size,
        stratify=labels_tv,
        random_state=random_seed,
    )

    # Include ultra-rare in train
    if len(df_ultra_rare) > 0:
        train_df = pd.concat([train_df, df_ultra_rare], ignore_index=True)

    logger.info(
        f"Data split completed: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}"
    )
    return train_df, val_df, test_df


def stream_ordered_replay(
    csv_files: Optional[List[str]] = None,
    data_dir: str = "dataset",
    batch_size: int = 1000,
) -> Generator[pd.DataFrame, None, None]:
    """
    Chronological / ordered flow replay generator.
    Streams network flows sequentially across files to simulate real-time production ingress.
    Yields batches of size `batch_size`.
    """
    if csv_files is None:
        # Natural chronological day order of CICIDS2017:
        # Monday (Normal) -> Tuesday (Brute Force) -> Wednesday (DoS/Heartbleed)
        # -> Thursday Morning (Web Attacks) -> Thursday Afternoon (Infiltration)
        # -> Friday Morning (Bot) -> Friday Afternoon PortScan -> Friday Afternoon DDoS
        preferred_order = [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        ]
        csv_files = [os.path.join(data_dir, f) for f in preferred_order if os.path.exists(os.path.join(data_dir, f))]

    current_batch = []
    current_count = 0

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        logger.info(f"Streaming from chronological file: {fname}")
        for chunk in pd.read_csv(fpath, chunksize=batch_size, encoding="utf-8", encoding_errors="replace", low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk["SourceFile"] = fname
            yield chunk
