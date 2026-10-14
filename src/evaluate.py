from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import DATA_PATH, FIGURE_DIR, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.feature_engineering import select_model_features
from src.preprocessing import build_preprocessor


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the fraud dataset from disk."""
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")
    return frame


def prepare_training_data(frame: pd.DataFrame):
    """Return features and target vectors for model evaluation."""
    data = select_model_features(frame)
    target = frame[TARGET_COLUMN].astype(int)
    return data, target


def build_model() -> Pipeline:
    """Create the standard fraud-detection pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(
    frame: pd.DataFrame | None = None,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Train a model, evaluate it on a holdout split, and return the metrics."""
    data = load_dataset() if frame is None else frame.copy()
    X, y = prepare_training_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4, zero_division=0)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_labels = ["Legit", "Fraud"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    return {
        "model": model,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    result = evaluate_model()
    print("Evaluation metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value:.4f}")
    print("\nClassification report:\n")
    print(result["classification_report"])
