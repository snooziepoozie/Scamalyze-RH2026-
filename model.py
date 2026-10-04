from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.preprocessing import StandardScaler
import joblib
import os

import pipeline as pl

MODEL_PATH = Path(__file__).resolve().with_name("fraud_model.pt")
SCALER_PATH = Path(__file__).resolve().with_name("scaler.joblib")

# Features the model is trained on. Note: only 'amount' exists on Solana;
# the other four are off-chain and must be imputed for Solana input.
FEATURE_COLS = [
    "amount",
    "time_since_last_transaction",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
]
SOLANA_MISSING = FEATURE_COLS[1:]  # the four that Solana can't supply


class FraudNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)


# ── Load the trained model + scaler once, reuse (module-level state) ────────
_model = None
_scaler = None


def _load():
    global _model, _scaler
    if _model is None:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise FileNotFoundError(
                "Train the model first (run train_model) — "
                "fraud_model.pt / scaler.joblib not found."
            )
        _scaler = joblib.load(SCALER_PATH)
        _model = FraudNet(len(FEATURE_COLS))
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model.eval()
    return _model, _scaler


def _prepare_features(df: pd.DataFrame, is_solana: bool) -> np.ndarray:
    """Build the feature matrix the model expects from whatever columns exist."""
    feats = pd.DataFrame(index=df.index)
    feats["amount"] = df["amount"].fillna(0.0)
    for col in SOLANA_MISSING:
        # Solana has none of these; Tiger data has them. Impute 0 when absent.
        feats[col] = df[col].fillna(0.0) if col in df.columns else 0.0
    return feats[FEATURE_COLS].values


# ── The single entry point you asked for ───────────────────────────────────
def run_model(
    filters: dict | None = None,
    use_solana: bool = False,
    solana_df: pd.DataFrame | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Score transactions for fraud.

    use_solana=False  -> pull filtered rows from Tiger (via pipeline) and score.
    use_solana=True   -> score the provided solana_df (transaction_type,
                         merchant_category, location are ignored by design).

    Returns the input rows plus 'fraud_probability' and 'predicted_fraud'.
    """
    if use_solana:
        if solana_df is None or solana_df.empty:
            raise ValueError("use_solana=True requires a non-empty solana_df.")
        data = solana_df.copy()
    else:
        data = pl.get_filtered_export(filters or {}, max_rows=100_000)

    if data.empty:
        return data.assign(fraud_probability=[], predicted_fraud=[])

    model, scaler = _load()
    X = scaler.transform(_prepare_features(data, use_solana))

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze(-1)
        probs = torch.sigmoid(logits).numpy()

    out = data.copy()
    out["fraud_probability"] = probs
    out["predicted_fraud"] = probs > threshold
    return out


def get_solana_transactions(wallet: str, limit: int = 100) -> pd.DataFrame:
    """Fetch a wallet's parsed transactions from Helius into the app's schema.

    Returns the 8 columns the model/dashboard expect. The three off-chain
    columns (transaction_type is real-ish; merchant_category, location) are
    filled with placeholders since Solana has no equivalent — the model
    ignores them anyway.
    """
    import requests

    api_key = os.environ.get("HELIUS_API_KEY")
    if not api_key:
        raise RuntimeError("Set HELIUS_API_KEY in your .env file.")

    url = f"https://api.helius.xyz/v0/addresses/{wallet}/transactions"
    resp = requests.get(url, params={"api-key": api_key, "limit": limit}, timeout=15)
    resp.raise_for_status()
    txns = resp.json()

    rows = []
    for tx in txns:
        sig = tx.get("signature")
        ts = (
            pd.to_datetime(tx.get("timestamp"), unit="s")
            if tx.get("timestamp")
            else pd.NaT
        )
        tx_type = tx.get("type", "UNKNOWN")

        transfers = tx.get("nativeTransfers") or []
        if transfers:
            for t in transfers:
                rows.append(
                    {
                        "transaction_id": sig,
                        "timestamp": ts,
                        "sender_account": t.get("fromUserAccount", "unknown"),
                        "receiver_account": t.get("toUserAccount", "unknown"),
                        "amount": (t.get("amount", 0) or 0) / 1e9,  # lamports -> SOL
                        "transaction_type": tx_type,
                        "merchant_category": None,  # no on-chain equivalent
                        "location": None,  # no on-chain equivalent
                        "is_fraud": False,  # unlabeled — placeholder
                    }
                )
        else:
            # Keep non-transfer transactions (swaps, etc.) as a single row.
            rows.append(
                {
                    "transaction_id": sig,
                    "timestamp": ts,
                    "sender_account": tx.get("feePayer", "unknown"),
                    "receiver_account": "unknown",
                    "amount": (tx.get("fee", 0) or 0) / 1e9,
                    "transaction_type": tx_type,
                    "merchant_category": None,
                    "location": None,
                    "is_fraud": False,
                }
            )

    cols = [
        "transaction_id",
        "timestamp",
        "sender_account",
        "receiver_account",
        "amount",
        "transaction_type",
        "merchant_category",
        "location",
        "is_fraud",
    ]
    return pd.DataFrame(rows, columns=cols)


class FocalLoss(nn.Module):
    def __init__(self, alpha=10, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), targets.float(), reduction="none"
        )
        pt = torch.exp(-bce)
        return (self.alpha * ((1 - pt) ** self.gamma) * bce).mean()
