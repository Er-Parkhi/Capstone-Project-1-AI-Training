from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.config import MODEL_DIR

DEFAULT_MODEL_PATH = MODEL_DIR / "fraud_detection_pipeline.pkl"


def save_model(model: Any, model_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a trained fraud-detection pipeline to disk.

    Parameters
    ----------
    model:
        A fitted sklearn pipeline or estimator to save.
    model_path:
        Destination file path. Defaults to the project model directory.
    """
    if model is None:
        raise ValueError("A trained model object is required before saving.")

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


if __name__ == "__main__":
    # This module is intended to be called by the training workflow after a model
    # has been fitted. Keep the script side-effect free unless explicitly invoked.
    print(f"Model save utility ready: {DEFAULT_MODEL_PATH}")
