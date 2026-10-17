"""
Inference module for Adaptive ML-DST system.
Provides saved-model loading, schema validation, and inference pipelines.
"""
from src.inference.predictor import OnlineSGDPredictor

__all__ = ["OnlineSGDPredictor"]
