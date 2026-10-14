from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from src.config import MODEL_DIR
from src.feature_engineering import select_model_features

MODEL_PATH = MODEL_DIR / "fraud_detection_pipeline.pkl"


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the persisted fraud-detection pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Run the training script first."
        )
    return joblib.load(path)


def _normalize_payload(payload: Mapping[str, Any] | pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    elif isinstance(payload, Mapping):
        frame = pd.DataFrame([dict(payload)])
    elif isinstance(payload, list):
        frame = pd.DataFrame(payload)
    else:
        raise TypeError("Payload must be a mapping, dataframe, or list of mappings.")

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required transaction columns: {missing}")

    return frame


def _model_name(model: Any) -> str:
    if hasattr(model, "named_steps") and "model" in getattr(model, "named_steps", {}):
        estimator = model.named_steps["model"]
        name = estimator.__class__.__name__
    else:
        name = model.__class__.__name__
    return name.replace("Classifier", "").replace("Regressor", "")


def predict_transaction(payload: Mapping[str, Any] | pd.DataFrame | list[dict[str, Any]], model=None):
    """Return the fraud prediction and probability for a single transaction."""
    if model is None:
        model = load_model()

    frame = _normalize_payload(payload)
    features = select_model_features(frame)
    probability = float(model.predict_proba(features)[0, 1])
    prediction = bool(probability >= 0.5)

    return {
        "is_fraud": prediction,
        "fraud_probability": probability,
        "model_name": _model_name(model),
    }


if __name__ == "__main__":
    example = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    print(predict_transaction(example))
