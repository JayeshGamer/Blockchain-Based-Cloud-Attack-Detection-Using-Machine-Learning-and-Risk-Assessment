"""
CLI Prediction Command for Saved Model Inference.
Accepts a CSV of raw flow features, applies saved preprocessing,
and outputs flow identifiers, attack scores, thresholds, and predictions without retraining.

Usage:
    python scripts/predict.py --input data/example_input.csv --output data/example_output.csv
    python scripts/predict.py --input data/example_input.csv --model models/online_sgd_initial.joblib
"""

import os
import sys
import argparse
import logging
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.predictor import OnlineSGDPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run inference using exported Online SGD model bundle.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV containing precomputed network flow features.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write output predictions CSV. Defaults to <input_name>_predictions.csv",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/online_sgd_final_online.joblib",
        help="Path to trained .joblib model bundle. Defaults to models/online_sgd_final_online.joblib",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold override (e.g., 0.50 or calibrated threshold).",
    )
    parser.add_argument(
        "--flow-id-col",
        type=str,
        default=None,
        help="Column name to use for flow IDs. If not specified, auto-detected.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    output_path = args.output
    if not output_path:
        base, _ = os.path.splitext(args.input)
        output_path = f"{base}_predictions.csv"

    logger.info("=" * 65)
    logger.info("ADAPTIVE ML-DST: SAVED MODEL BATCH INFERENCE")
    logger.info("=" * 65)
    logger.info(f"Input file:  {args.input}")
    logger.info(f"Model file:  {args.model}")
    logger.info(f"Output file: {output_path}")

    try:
        predictor = OnlineSGDPredictor(model_path=args.model, default_threshold=args.threshold)
    except Exception as e:
        logger.error(f"Failed to load model bundle: {e}")
        sys.exit(1)

    logger.info("Reading input data...")
    df_in = pd.read_csv(args.input, low_memory=False)
    logger.info(f"Read {len(df_in):,} flows with {len(df_in.columns)} columns.")

    try:
        results_df = predictor.predict_df(
            df_in,
            flow_id_col=args.flow_id_col,
            threshold_override=args.threshold,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    results_df.to_csv(output_path, index=False)

    n_total = len(results_df)
    n_attacks = int((results_df["prediction"] == 1).sum())
    n_benign = n_total - n_attacks
    pct_attacks = (n_attacks / max(1, n_total)) * 100

    logger.info("Inference Completed Successfully:")
    logger.info(f"  Total Flows Evaluated: {n_total:,}")
    logger.info(f"  Predicted Attacks:     {n_attacks:,} ({pct_attacks:.2f}%)")
    logger.info(f"  Predicted Benign:      {n_benign:,} ({100 - pct_attacks:.2f}%)")
    logger.info(f"  Threshold Applied:     {predictor.threshold:.4f}")
    logger.info(f"  Predictions Saved to:  {output_path}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
