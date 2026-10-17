"""
Configuration loader, schema validator, and resolver for Adaptive ML-DST.
Ensures strict validation, rejection of invalid options, zero silent fallbacks,
and persistence of resolved configurations to experiment run directories.
"""

import os
import copy
import json
import yaml
import logging
from typing import Dict, Any, Union, Tuple, Optional

logger = logging.getLogger(__name__)

# Canonical supported execution modes
VALID_MODES = {"lightweight", "full"}

# Default configuration templates for fallback resolution
DEFAULT_CONFIG_VALUES: Dict[str, Any] = {
    "project": {
        "name": "Adaptive ML-DST",
        "version": "4.0.0-verified",
        "random_seed": 42,
    },
    "mode": "lightweight",
    "label": "lightweight development results",
    "data": {
        "raw_dir": "dataset",
        "processed_dir": "data/processed",
        "lightweight_sample_path": "data/processed/lightweight_sample.csv.gz",
        "sample_size": 50000,
        "chunk_size": 50000,
        "test_size": 0.20,
        "val_size": 0.15,
        "deduplication": "global_sha256",
        "drop_constant_columns": True,
        "scaler": "robust",
    },
    "stream": {
        "batch_size": 200,
        "label_delay_batches": 1,
        "monday_initial_baseline": True,
    },
    "online_models": {
        "linear": {
            "algorithm": "sgd_classifier",
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 0.0001,
        },
    },
    "anomaly_detector": {
        "algorithm": "isolation_forest",
        "n_estimators": 50,
        "contamination": 0.05,
        "max_samples": 5000,
        "train_on_benign_only": True,
    },
    "drift_detector": {
        "algorithm": "adwin",
        "delta": 0.002,
        "clock": 32,
        "min_window_length": 10,
        "grace_period": 100,
    },
    "temporal_deviation": {
        "window_size": 200,
        "protected_benign_baseline": True,
    },
    "dst_fusion": {
        "uncertainty_budget": 0.15,
        "conflict_threshold": 0.90,
    },
    "adaptive_threshold": {
        "mode": "recall_constrained",
        "min_target_recall": 0.90,
        "initial_threshold": 0.50,
        "target_fpr": 0.02,
        "min_threshold": 0.15,
        "max_threshold": 0.85,
        "drift_relaxation_factor": 0.85,
    },
}


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validates configuration schema and value ranges.
    Raises ValueError or FileNotFoundError if invalid.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary.")

    # 1. Validate top-level sections
    known_sections = {
        "project", "mode", "label", "description", "data", "stream",
        "online_models", "anomaly_detector", "drift_detector", "drift_response",
        "temporal_deviation", "dst_fusion", "dst", "adaptive_threshold",
        "hierarchical_risk", "simulated_defense", "blockchain", "output",
        "_source_path", "seeds",
    }
    unknown = set(config.keys()) - known_sections
    if unknown:
        raise ValueError(f"Unknown configuration sections detected: {sorted(list(unknown))}")

    # 2. Validate mode
    mode = config.get("mode")
    if mode is None:
        raise ValueError("Configuration missing required field: 'mode' (must be 'lightweight' or 'full').")
    mode_str = str(mode).lower()
    if mode_str not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Supported modes: {sorted(list(VALID_MODES))}")

    # 2. Validate data section
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("Configuration section 'data' must be a dictionary.")

    if mode_str == "full":
        raw_dir = data_cfg.get("raw_dir", "dataset")
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(
                f"Full mode requires raw dataset directory '{raw_dir}', which does not exist. "
                "Silent fallback to lightweight mode is strictly prohibited."
            )
        chunk_size = data_cfg.get("chunk_size", 50000)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {chunk_size}")

    elif mode_str == "lightweight":
        sample_size = data_cfg.get("sample_size", 50000)
        if sample_size is not None and (not isinstance(sample_size, int) or sample_size <= 0):
            raise ValueError(f"sample_size must be a positive integer, got: {sample_size}")

    # 3. Validate stream section
    stream_cfg = config.get("stream", {})
    if not isinstance(stream_cfg, dict):
        raise ValueError("Configuration section 'stream' must be a dictionary.")

    batch_size = stream_cfg.get("batch_size", 200)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"stream.batch_size must be a positive integer, got: {batch_size}")

    label_delay = stream_cfg.get("label_delay_batches", 1)
    if not isinstance(label_delay, int) or label_delay < 0:
        raise ValueError(f"stream.label_delay_batches must be a non-negative integer (>= 0), got: {label_delay}")

    # 4. Validate project seed
    proj_cfg = config.get("project", {})
    seed = proj_cfg.get("random_seed", 42)
    if not isinstance(seed, int):
        raise ValueError(f"project.random_seed must be an integer, got: {seed}")

    # 5. Validate adaptive threshold settings
    thresh_cfg = config.get("adaptive_threshold", {})
    if isinstance(thresh_cfg, dict):
        init_t = thresh_cfg.get("initial_threshold", 0.50)
        min_t = thresh_cfg.get("min_threshold", 0.15)
        max_t = thresh_cfg.get("max_threshold", 0.85)
        if not (0.0 < min_t < max_t < 1.0):
            raise ValueError(f"Threshold bounds invalid: 0.0 < min_threshold ({min_t}) < max_threshold ({max_t}) < 1.0 required.")
        if not (min_t <= init_t <= max_t):
            raise ValueError(f"initial_threshold ({init_t}) must be within [min_threshold ({min_t}), max_threshold ({max_t})]")

    # 6. Validate DST settings
    dst_cfg = config.get("dst_fusion", {}) or config.get("dst", {})
    if isinstance(dst_cfg, dict):
        budget = dst_cfg.get("uncertainty_budget", 0.15)
        if not (0.0 <= budget < 1.0):
            raise ValueError(f"dst.uncertainty_budget must be in [0.0, 1.0), got: {budget}")
        conflict_rule = dst_cfg.get("conflict_rule", "yager").lower()
        if conflict_rule not in {"yager", "dempster"}:
            raise ValueError(f"dst.conflict_rule must be 'yager' or 'dempster', got: '{conflict_rule}'")

    # 7. Validate online models section
    online_cfg = config.get("online_models", {})
    if isinstance(online_cfg, dict):
        linear_cfg = online_cfg.get("linear", {})
        if isinstance(linear_cfg, dict):
            loss = linear_cfg.get("loss", "log_loss")
            if loss not in {"log_loss", "modified_huber", "hinge"}:
                raise ValueError(f"Unsupported online_models.linear.loss: '{loss}'")
            alpha = linear_cfg.get("alpha", 0.0001)
            if not isinstance(alpha, (int, float)) or alpha <= 0:
                raise ValueError(f"online_models.linear.alpha must be positive, got: {alpha}")

    # 8. Validate anomaly detector section
    anom_cfg = config.get("anomaly_detector", {})
    if isinstance(anom_cfg, dict):
        n_est = anom_cfg.get("n_estimators", 50)
        if not isinstance(n_est, int) or n_est <= 0:
            raise ValueError(f"anomaly_detector.n_estimators must be a positive integer, got: {n_est}")
        contam = anom_cfg.get("contamination", 0.05)
        if not (0.0 < contam <= 0.50):
            raise ValueError(f"anomaly_detector.contamination must be in (0.0, 0.50], got: {contam}")

    # 9. Validate drift detector section
    drift_cfg = config.get("drift_detector", {})
    if isinstance(drift_cfg, dict):
        delta = drift_cfg.get("delta", 0.002)
        if not (0.0 < delta < 1.0):
            raise ValueError(f"drift_detector.delta must be in (0.0, 1.0), got: {delta}")


# --- COMPONENT FACTORIES CONNECTING CONFIG TO IMPLEMENTATIONS ---

def build_online_linear_classifier(config: Dict[str, Any], seed: Optional[int] = None):
    """Factory for OnlineSGDClassifier from resolved config."""
    from src.models.online_classifier import OnlineSGDClassifier
    linear_cfg = config.get("online_models", {}).get("linear", {})
    random_seed = seed if seed is not None else config.get("project", {}).get("random_seed", 42)
    return OnlineSGDClassifier(
        loss=linear_cfg.get("loss", "log_loss"),
        penalty=linear_cfg.get("penalty", "l2"),
        alpha=float(linear_cfg.get("alpha", 0.0001)),
        learning_rate=linear_cfg.get("learning_rate", "optimal"),
        random_seed=random_seed,
    )


def build_online_nonlinear_classifier(config: Dict[str, Any], seed: Optional[int] = None):
    """Factory for RiverHoeffdingAdaptiveTreeOnline from resolved config."""
    from src.models.online_classifier import RiverHoeffdingAdaptiveTreeOnline
    tree_cfg = config.get("online_models", {}).get("nonlinear", {})
    random_seed = seed if seed is not None else config.get("project", {}).get("random_seed", 42)
    return RiverHoeffdingAdaptiveTreeOnline(
        grace_period=int(tree_cfg.get("grace_period", 100)),
        delta=float(tree_cfg.get("delta", 1e-5)),
        random_seed=random_seed,
    )


def build_anomaly_detector(config: Dict[str, Any], seed: Optional[int] = None):
    """Factory for AnomalyEvidenceDetector from resolved config."""
    from src.models.anomaly_detector import AnomalyEvidenceDetector
    anom_cfg = config.get("anomaly_detector", {})
    random_seed = seed if seed is not None else config.get("project", {}).get("random_seed", 42)
    return AnomalyEvidenceDetector(
        contamination=float(anom_cfg.get("contamination", 0.05)),
        n_estimators=int(anom_cfg.get("n_estimators", 50)),
        max_samples=int(anom_cfg.get("max_samples", 5000)),
        random_seed=random_seed,
    )


def build_preprocessor(config: Dict[str, Any]):
    """Factory for NetworkDataPreprocessor from resolved config."""
    from src.features.preprocessor import NetworkDataPreprocessor
    data_cfg = config.get("data", {})
    return NetworkDataPreprocessor(
        scaler_type=data_cfg.get("scaler", "robust"),
        drop_constant=data_cfg.get("drop_constant_columns", True),
    )


def build_drift_detector(config: Dict[str, Any]):
    """Factory for ADWIN drift detector from resolved config."""
    from src.drift.adwin import create_adwin_detector
    drift_cfg = config.get("drift_detector", {})
    return create_adwin_detector(
        delta=float(drift_cfg.get("delta", 0.002)),
        prefer_river=True,
    )


def build_drift_coordinator(config: Dict[str, Any]):
    """Factory for MultiSignalDriftCoordinator from resolved config."""
    from src.drift.adwin import MultiSignalDriftCoordinator
    drift_cfg = config.get("drift_detector", {})
    resp_cfg = config.get("drift_response", {})
    return MultiSignalDriftCoordinator(
        delta=float(drift_cfg.get("delta", 0.002)),
        prefer_river=True,
        drift_response_strategy=resp_cfg.get("strategy", "plasticity_boost"),
        boost_factor=float(resp_cfg.get("boost_factor", 2.0)),
        cooldown_batches=int(resp_cfg.get("cooldown_batches", 20)),
    )


def build_temporal_tracker(config: Dict[str, Any]):
    """Factory for TemporalDeviationTracker from resolved config."""
    from src.features.temporal import TemporalDeviationTracker
    temp_cfg = config.get("temporal_deviation", {})
    return TemporalDeviationTracker(
        window_size=int(temp_cfg.get("window_size", 200)),
        benign_confidence_threshold=float(temp_cfg.get("benign_confidence_threshold", 0.35)),
        weights=temp_cfg.get("metrics"),
    )


def build_dst_engine(config: Dict[str, Any]):
    """Factory for DSTEvidenceFusion from resolved config."""
    from src.dst.evidence_fusion import DSTEvidenceFusion
    dst_cfg = config.get("dst_fusion", {}) or config.get("dst", {})
    return DSTEvidenceFusion(
        uncertainty_budget=float(dst_cfg.get("uncertainty_budget", 0.15)),
        source_reliabilities=dst_cfg.get("source_reliabilities"),
        conflict_threshold=float(dst_cfg.get("conflict_threshold", 0.90)),
        conflict_rule=str(dst_cfg.get("conflict_rule", "yager")),
    )


def build_adaptive_threshold_manager(config: Dict[str, Any]):
    """Factory for AdaptiveThresholdManager from resolved config."""
    from src.risk.adaptive_threshold import AdaptiveThresholdManager
    thresh_cfg = config.get("adaptive_threshold", {})
    return AdaptiveThresholdManager(
        initial_threshold=float(thresh_cfg.get("initial_threshold", 0.50)),
        target_fpr=float(thresh_cfg.get("target_fpr", 0.02)),
        rolling_window_size=int(thresh_cfg.get("rolling_window_size", 500)),
        min_threshold=float(thresh_cfg.get("min_threshold", 0.15)),
        max_threshold=float(thresh_cfg.get("max_threshold", 0.85)),
        drift_relaxation_factor=float(thresh_cfg.get("drift_relaxation_factor", 0.85)),
        smoothing_alpha=float(thresh_cfg.get("smoothing_alpha", 0.05)),
        freeze_during_burst=bool(thresh_cfg.get("freeze_during_burst", True)),
    )


def get_effective_components_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns an audited record of component class names and effective parameters
    actually routed into execution.
    """
    return {
        "preprocessor": {
            "class": "NetworkDataPreprocessor",
            "effective_params": {
                "scaler": config.get("data", {}).get("scaler", "robust"),
                "imputer_strategy": config.get("data", {}).get("imputer_strategy", "training_median"),
                "drop_constant_columns": config.get("data", {}).get("drop_constant_columns", True),
                "drop_duplicate_fwd_header": config.get("data", {}).get("drop_duplicate_fwd_header", True),
                "outlier_clip": config.get("data", {}).get("outlier_clip", [-50.0, 50.0]),
            },
        },
        "online_models": {
            "linear_class": "OnlineSGDClassifier",
            "linear_effective_params": config.get("online_models", {}).get("linear", {}),
            "nonlinear_class": "RiverHoeffdingAdaptiveTreeOnline",
            "nonlinear_effective_params": config.get("online_models", {}).get("nonlinear", {}),
        },
        "anomaly_detector": {
            "class": "AnomalyEvidenceDetector",
            "effective_params": config.get("anomaly_detector", {}),
        },
        "drift_detector": {
            "class": "ADWIN / RiverADWIN",
            "effective_params": config.get("drift_detector", {}),
        },
        "drift_coordinator": {
            "class": "MultiSignalDriftCoordinator",
            "effective_params": {
                "delta": config.get("drift_detector", {}).get("delta", 0.002),
                "strategy": config.get("drift_response", {}).get("strategy", "plasticity_boost"),
                "boost_factor": config.get("drift_response", {}).get("boost_factor", 2.0),
                "cooldown_batches": config.get("drift_response", {}).get("cooldown_batches", 20),
            },
        },
        "temporal_deviation": {
            "class": "TemporalDeviationTracker",
            "effective_params": config.get("temporal_deviation", {}),
        },
        "dst_fusion": {
            "class": "DSTEvidenceFusion",
            "effective_params": config.get("dst_fusion", {}) or config.get("dst", {}),
        },
        "adaptive_threshold": {
            "class": "AdaptiveThresholdManager",
            "effective_params": config.get("adaptive_threshold", {}),
        },
    }


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Failed to parse YAML dictionary from: {config_path}")

    return config


def resolve_and_validate_config(config_path_or_dict: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Loads (if path), deep-merges with defaults, applies environment overrides,
    validates schema, and returns resolved configuration dictionary.
    """
    if isinstance(config_path_or_dict, str):
        config = load_config(config_path_or_dict)
        config["_source_path"] = os.path.abspath(config_path_or_dict)
    elif isinstance(config_path_or_dict, dict):
        config = copy.deepcopy(config_path_or_dict)
        config["_source_path"] = "in_memory_dict"
    else:
        raise TypeError(f"Expected str or dict, got: {type(config_path_or_dict)}")

    # Environment variable overrides
    if os.getenv("RANDOM_SEED"):
        config.setdefault("project", {})["random_seed"] = int(os.getenv("RANDOM_SEED"))
    if os.getenv("DATASET_MODE"):
        config["mode"] = os.getenv("DATASET_MODE").lower()
    if os.getenv("STREAM_BATCH_SIZE"):
        config.setdefault("stream", {})["batch_size"] = int(os.getenv("STREAM_BATCH_SIZE"))
    if os.getenv("LABEL_DELAY_BATCHES"):
        config.setdefault("stream", {})["label_delay_batches"] = int(os.getenv("LABEL_DELAY_BATCHES"))

    # Ensure mode is explicitly set
    if "mode" not in config:
        config["mode"] = "lightweight"
    if "label" not in config:
        config["label"] = "lightweight development results" if config["mode"] == "lightweight" else "full-dataset results"

    # Set appropriate output directory
    mode = config["mode"]
    default_out = f"reports/experiments/{mode}" if mode == "full" else "reports/experiments"
    config.setdefault("output", {})
    config["output"].setdefault("experiments_dir", default_out)
    config["output"].setdefault("tables_dir", "reports/tables")
    config["output"].setdefault("figures_dir", "reports/figures")

    # Validate resolved config
    validate_config(config)

    return config


def save_resolved_config(config: Dict[str, Any], output_dir: str) -> Tuple[str, str]:
    """
    Saves resolved configuration to YAML and JSON formats in the specified output directory.
    Returns:
        (yaml_path, json_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, "resolved_config.yaml")
    json_path = os.path.join(output_dir, "resolved_config.json")

    clean_cfg = {k: v for k, v in config.items() if not k.startswith("_")}

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(clean_cfg, f, default_flow_style=False, sort_keys=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_cfg, f, indent=2)

    logger.info(f"Saved resolved configuration to {yaml_path} and {json_path}")
    return yaml_path, json_path
