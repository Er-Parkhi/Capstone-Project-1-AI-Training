
import json

import time

import warnings
 
import joblib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import pandas as pd
 
from sklearn.base import clone

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    average_precision_score,

    classification_report,

    confusion_matrix,

    f1_score,

    precision_score,

    recall_score,

    roc_auc_score,

)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
 
from src.config import (

    DATA_PATH,

    FIGURE_DIR,

    MODEL_DIR,

    NUMERIC_FEATURES,

    CATEGORICAL_FEATURES,

    TARGET_COLUMN,

    RANDOM_STATE,

    TEST_SIZE,

)

from src.preprocessing import build_preprocessor
 
warnings.filterwarnings("ignore", category=UserWarning)
 
 
def main():

    print("Loading sampled dataset...")

    df = pd.read_csv(DATA_PATH)
 
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = df[feature_columns]

    y = df[TARGET_COLUMN].astype(int)
 
    print(f"Dataset shape: {df.shape}")

    print(f"Fraud cases: {int(y.sum()):,}")

    print(f"Fraud rate: {y.mean() * 100:.5f}%")
 
    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y,

    )
 
    print(f"Training rows: {len(X_train):,}")

    print(f"Test rows: {len(X_test):,}")

    print(f"Training fraud cases: {int(y_train.sum()):,}")

    print(f"Test fraud cases: {int(y_test.sum()):,}")
 
    # HistGradientBoosting does not support class_weight in this version.

    # For this baseline, use the four models with native class weighting

    # where available; HistGradientBoosting is included with default weights.

    models = {

        "LogisticRegression": LogisticRegression(

            max_iter=500,

            class_weight="balanced",

            solver="liblinear",

            random_state=RANDOM_STATE,

        ),

        "DecisionTree": DecisionTreeClassifier(

            class_weight="balanced",

            max_depth=12,

            min_samples_leaf=5,

            random_state=RANDOM_STATE,

        ),

        "RandomForest": __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(

            n_estimators=1000,

            max_depth=20,

            min_samples_leaf=5,

            class_weight="balanced_subsample",

            n_jobs=-1,

            random_state=RANDOM_STATE,

        ),

        "ExtraTrees": ExtraTreesClassifier(

            n_estimators=100,

            max_depth=16,

            min_samples_leaf=2,

            class_weight="balanced",

            n_jobs=-1,

            random_state=RANDOM_STATE,

        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(

            max_iter=100,

            max_leaf_nodes=15,

            learning_rate=0.1,

            random_state=RANDOM_STATE,

        ),

    }
 
    results = []

    best_model = None

    best_name = None

    best_score = -1.0

    best_predictions = None

    best_probabilities = None
 
    preprocessor = build_preprocessor()
 
    for name, estimator in models.items():

        print(f"\n{'=' * 65}")

        print(f"Training: {name}")

        start = time.time()
 
        pipeline = Pipeline(

            steps=[

                ("preprocessor", clone(preprocessor)),

                ("model", estimator),

            ]

        )
 
        try:

            pipeline.fit(X_train, y_train)
 
            predictions = pipeline.predict(X_test)

            probabilities = pipeline.predict_proba(X_test)[:, 1]
 
            precision = precision_score(

                y_test, predictions, zero_division=0

            )

            recall = recall_score(

                y_test, predictions, zero_division=0

            )

            f1 = f1_score(y_test, predictions, zero_division=0)

            pr_auc = average_precision_score(y_test, probabilities)

            roc_auc = roc_auc_score(y_test, probabilities)
 
            elapsed = time.time() - start
 
            row = {

                "model": name,

                "precision": precision,

                "recall": recall,

                "f1": f1,

                "pr_auc": pr_auc,

                "roc_auc": roc_auc,

                "training_seconds": round(elapsed, 2),

            }

            results.append(row)
 
            print(json.dumps(row, indent=2))

            print("\nClassification report:")

            print(classification_report(

                y_test,

                predictions,

                target_names=["Non-fraud", "Fraud"],

                zero_division=0,

            ))
 
            # PR-AUC is the primary baseline ranking metric for this

            # extremely imbalanced classification problem.

            if pr_auc > best_score:

                best_score = pr_auc

                best_name = name

                best_model = pipeline

                best_predictions = predictions.copy()

                best_probabilities = probabilities.copy()
 
        except Exception as exc:

            print(f"FAILED: {name}: {type(exc).__name__}: {exc}")
 
    if not results:

        raise RuntimeError("All models failed. See errors above.")
 
    results_df = pd.DataFrame(results).sort_values(

        by="pr_auc", ascending=False

    )

    results_path = MODEL_DIR / "model_comparison.csv"

    results_df.to_csv(results_path, index=False)
 
    metrics_path = MODEL_DIR / "best_model_metrics.json"

    metrics = {

        "selected_model": best_name,

        "selection_metric": "PR-AUC (average precision)",

        "test_pr_auc": float(best_score),

        "test_precision": float(

            precision_score(y_test, best_predictions, zero_division=0)

        ),

        "test_recall": float(

            recall_score(y_test, best_predictions, zero_division=0)

        ),

        "test_f1": float(

            f1_score(y_test, best_predictions, zero_division=0)

        ),

        "test_roc_auc": float(

            roc_auc_score(y_test, best_probabilities)

        ),

        "test_rows": int(len(y_test)),

        "test_fraud_cases": int(y_test.sum()),

        "random_state": RANDOM_STATE,

        "test_size": TEST_SIZE,

    }
 
    model_path = MODEL_DIR / "fraud_detection_pipeline.pkl"

    joblib.dump(best_model, model_path)

    metrics_path.write_text(

        json.dumps(metrics, indent=2),

        encoding="utf-8",

    )
 
    cm = confusion_matrix(y_test, best_predictions)

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm)

    ax.set_title(f"Confusion Matrix — {best_name}")

    ax.set_xlabel("Predicted label")

    ax.set_ylabel("Actual label")

    ax.set_xticks([0, 1], ["Non-fraud", "Fraud"])

    ax.set_yticks([0, 1], ["Non-fraud", "Fraud"])
 
    for (i, j), value in __import__("numpy").ndenumerate(cm):

        ax.text(j, i, f"{value:,}", ha="center", va="center")
 
    fig.colorbar(image, ax=ax)

    fig.tight_layout()

    fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=150)

    plt.close(fig)
 
    print(f"\n{'=' * 65}")

    print("MODEL COMPARISON (sorted by PR-AUC)")

    print(results_df.to_string(index=False))

    print("\nSelected model:", best_name)

    print("Saved pipeline:", model_path)

    print("Saved metrics:", metrics_path)

    print("Saved comparison:", results_path)

    print("Saved confusion matrix:", FIGURE_DIR / "confusion_matrix.png")

    print("\nTraining run complete.")
 
 
if __name__ == "__main__":

    main()

