"""
Data preparation script:
Builds and saves the stratified, minority-preserved sample for rapid development,
baseline training, ablation experiments, and testing.
"""

import os
import logging
from src.data.loader import load_stratified_sample
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_cached_samples(output_dir: str = "data/processed") -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "stratified_sample.csv.gz")

    if os.path.exists(out_path):
        logger.info(f"Sample already exists at {out_path}")
        return

    logger.info("Generating stratified, minority-preserving sample (50,000 flows)...")
    df_sample = load_stratified_sample("dataset", total_samples=50000, random_seed=42)
    df_sample.to_csv(out_path, index=False, compression="gzip")
    logger.info(f"Successfully saved {len(df_sample):,} rows to {out_path}")


if __name__ == "__main__":
    prepare_cached_samples()
