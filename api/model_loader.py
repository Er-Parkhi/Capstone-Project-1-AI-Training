
from __future__ import annotations

import joblib

from src.config import MODEL_DIR

MODEL_PATH = MODEL_DIR / "fraud_detection_pipeline.pkl"


def train_model_if_missing():
    """Train and persist the default model when the packaged artifact is absent."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    # Import lazily to avoid startup cost in environments that already have a model.
    from src.train import main as train_main

    train_main()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model training did not create {MODEL_PATH}."
        )

    return MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        train_model_if_missing()

    return joblib.load(MODEL_PATH)

