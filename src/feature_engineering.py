from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the transaction frame with derived fraud-risk features.

    The project already defines the core columns expected by the model. These
    engineered columns are intentionally lightweight and safe to compute even when
    some values are missing or zero.
    """
    if df is None:
        raise ValueError("A dataframe is required for feature engineering.")

    data = df.copy()

    for column in REQUIRED_COLUMNS:
        if column not in data.columns:
            data[column] = 0.0 if column != "type" else "UNKNOWN"

    data["type"] = data["type"].fillna("UNKNOWN").astype(str).str.upper()

    numeric_cols = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    data["amount_to_org_balance"] = np.divide(
        data["amount"],
        data["oldbalanceOrg"].replace(0, np.nan),
        out=np.zeros(len(data), dtype=float),
        where=~pd.isna(data["oldbalanceOrg"]) & (data["oldbalanceOrg"] != 0),
    )
    data["amount_to_dest_balance"] = np.divide(
        data["amount"],
        data["oldbalanceDest"].replace(0, np.nan),
        out=np.zeros(len(data), dtype=float),
        where=~pd.isna(data["oldbalanceDest"]) & (data["oldbalanceDest"] != 0),
    )

    data["orig_balance_delta"] = data["newbalanceOrig"] - data["oldbalanceOrg"]
    data["dest_balance_delta"] = data["newbalanceDest"] - data["oldbalanceDest"]
    data["orig_net_change"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["dest_net_change"] = data["oldbalanceDest"] - data["newbalanceDest"]
    data["is_large_transfer"] = (data["amount"] > 10000).astype(int)
    data["is_cashout"] = data["type"].eq("CASH_OUT").astype(int)
    data["is_transfer"] = data["type"].eq("TRANSFER").astype(int)
    data["is_cashin"] = data["type"].eq("CASH_IN").astype(int)

    return data


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the final feature table for the fraud model."""
    enriched = add_transaction_features(df)
    feature_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "amount_to_org_balance",
        "amount_to_dest_balance",
        "orig_balance_delta",
        "dest_balance_delta",
        "orig_net_change",
        "dest_net_change",
        "is_large_transfer",
        "is_cashout",
        "is_transfer",
        "is_cashin",
    ]
    return enriched[feature_columns]
