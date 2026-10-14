"""Leakage-conscious comparison of imbalance strategies for transaction fraud."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
TARGET_DEFAULT = "isFraud"
NUMERIC = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
           "oldbalanceDest", "newbalanceDest"]
CATEGORICAL = ["type"]


def make_preprocessor():
    numeric = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = ImbPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ], remainder="drop")


def build_candidates():
    # SMOTE is applied after encoding, only to the training fold.
    return {
        "class_weight_logistic": (
            "none", LogisticRegression(
                max_iter=1500, class_weight="balanced", solver="liblinear",
                random_state=SEED
            )
        ),
        "random_oversampling_logistic": (
            RandomOverSampler(random_state=SEED),
            LogisticRegression(max_iter=1500, solver="liblinear", random_state=SEED)
        ),
        "random_undersampling_logistic": (
            RandomUnderSampler(random_state=SEED),
            LogisticRegression(max_iter=1500, solver="liblinear", random_state=SEED)
        ),
        "smote_logistic": (
            SMOTE(random_state=SEED, k_neighbors=3),
            LogisticRegression(max_iter=1500, solver="liblinear", random_state=SEED)
        ),
        "smoteenn_logistic": (
            SMOTEENN(random_state=SEED),
            LogisticRegression(max_iter=1500, solver="liblinear", random_state=SEED)
        ),
        "class_weight_random_forest": (
            "none", RandomForestClassifier(
                n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample",
                n_jobs=-1, random_state=SEED
            )
        ),
    }


def build_pipeline(sampler, estimator):
    steps = [("preprocess", make_preprocessor())]
    if sampler != "none":
        steps.append(("resample", sampler))
    steps.append(("model", estimator))
    return ImbPipeline(steps)


def choose_threshold(y_true, probabilities):
    # F2 favours recall while still penalising false positives.
    thresholds = np.unique(np.r_[np.arange(0.05, 0.951, 0.01), 0.5])
    scores = [fbeta_score(y_true, probabilities >= t, beta=2, zero_division=0)
              for t in thresholds]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def metrics_at_threshold(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "average_precision_pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/fraud_sample.csv")
    parser.add_argument("--target", default=TARGET_DEFAULT)
    parser.add_argument("--output-dir", default="reports/imbalance")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20,
                        help="Fraction of the remaining development data.")
    args = parser.parse_args()

    data_path = Path(args.data)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found. Columns: {list(df.columns)}")
    required = set(NUMERIC + CATEGORICAL + [args.target])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    # Do not use identifier columns or the pre-existing fraud flag as predictors.
    X = df[NUMERIC + CATEGORICAL].copy()
    y = pd.to_numeric(df[args.target], errors="raise").astype(int)
    if set(y.unique()) - {0, 1}:
        raise ValueError("Target must be binary and encoded as 0/1.")
    if y.nunique() != 2:
        raise ValueError("Dataset must contain both target classes.")

    # Three disjoint stratified splits: train, validation for threshold/model selection,
    # and test reserved for a single final estimate.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=args.validation_size, random_state=SEED, stratify=y_dev
    )
    if int(y_train.sum()) < 4:
        raise ValueError("Too few fraud examples for SMOTE; use more training data.")

    rows, fitted = [], {}
    for name, (sampler, estimator) in build_candidates().items():
        print(f"Training {name} ...", flush=True)
        pipe = build_pipeline(sampler, estimator)
        pipe.fit(X_train, y_train)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        threshold, val_f2 = choose_threshold(y_val, val_prob)
        val_metrics = metrics_at_threshold(y_val, val_prob, threshold)
        rows.append({"model": name, "validation_f2": val_f2, **{
            f"validation_{k}": v for k, v in val_metrics.items()
        }})
        fitted[name] = (pipe, threshold)

    results = pd.DataFrame(rows).sort_values(
        ["validation_average_precision_pr_auc", "validation_f2"], ascending=False
    )
    results.to_csv(out / "model_comparison.csv", index=False)
    winner = str(results.iloc[0]["model"])
    model, threshold = fitted[winner]

    # Test exactly once after selection; do not use test metrics to select the model.
    test_prob = model.predict_proba(X_test)[:, 1]
    final_metrics = metrics_at_threshold(y_test, test_prob, threshold)
    final_metrics.update({
        "selected_model": winner,
        "validation_f2": float(results.iloc[0]["validation_f2"]),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_fraud_count": int(y_train.sum()),
        "validation_fraud_count": int(y_val.sum()),
        "test_fraud_count": int(y_test.sum()),
        "selection_metric": "validation PR-AUC, with validation F2 as tie-breaker",
        "threshold_selection_metric": "validation F2 (beta=2)",
    })
    (out / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2))
    (out / "selected_threshold.json").write_text(json.dumps({
        "threshold": threshold, "selected_model": winner
    }, indent=2))
    joblib.dump(model, "models/fraud_detection_pipeline.pkl")
    Path("models/fraud_threshold.json").write_text(json.dumps({"threshold": threshold}, indent=2))

    pred = (test_prob >= threshold).astype(int)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, pred, labels=[0, 1]),
        display_labels=["Non-fraud", "Fraud"]
    )
    disp.plot(values_format=",", cmap="viridis")
    plt.title(f"Untouched test confusion matrix — {winner}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close()

    print("\nSelected model:", winner)
    print(json.dumps(final_metrics, indent=2))
    print(f"\nArtifacts saved under {out.resolve()} and models/.")


if __name__ == "__main__":
    main()
