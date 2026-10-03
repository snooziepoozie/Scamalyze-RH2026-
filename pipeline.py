# pipeline.py
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import kagglehub

# Friendly dropdown label -> actual filename
DATASETS = {
    "HI-Small": "HI-Small_Trans.csv",
    "HI-Medium": "HI-Medium_Trans.csv",
    "LI-Small": "LI-Small_Trans.csv",
    "LI-Medium": "LI-Medium_Trans.csv",
}

TRANS_DTYPES = {
    "From Bank": "int32",
    "To Bank": "int32",
    "Account": "string",
    "Account.1": "string",
    "Amount Received": "float64",
    "Amount Paid": "float64",
    "Receiving Currency": "category",
    "Payment Currency": "category",
    "Payment Format": "category",
    "Is Laundering": "int8",
}
NUMERIC_COLS = [
    "amt_paid_log",
    "amt_recv_log",
    "amt_diff",
    "cross_currency",
    "hour",
    "dayofweek",
]
CAT_COLS = ["pay_currency", "recv_currency", "pay_format"]


def load_data(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset_name!r}. Options: {list(DATASETS)}")
    dataset_dir = Path(
        kagglehub.dataset_download(
            "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
        )
    )
    path = next(
        p for p in dataset_dir.rglob("*.csv") if p.name == DATASETS[dataset_name]
    )
    return pd.read_csv(path, dtype=TRANS_DTYPES, parse_dates=["Timestamp"])


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["amt_paid_log"] = np.log1p(df["Amount Paid"])
    out["amt_recv_log"] = np.log1p(df["Amount Received"])
    out["amt_diff"] = (df["Amount Paid"] - df["Amount Received"]).abs()
    out["cross_currency"] = (
        df["Payment Currency"].astype(str) != df["Receiving Currency"].astype(str)
    ).astype(int)
    ts = pd.to_datetime(df["Timestamp"])
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["pay_currency"] = df["Payment Currency"].astype(str)
    out["recv_currency"] = df["Receiving Currency"].astype(str)
    out["pay_format"] = df["Payment Format"].astype(str)
    return out


class FraudNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainedBundle:
    """Everything you need to make a prediction later — keep these together."""

    model: FraudNet
    preprocessor: ColumnTransformer
    metrics: dict
    n_features: int


def train_model(
    df: pd.DataFrame, epochs: int = 5, lr: float = 1e-3, progress_cb=None
) -> TrainedBundle:
    """Pure training function. progress_cb(epoch, total, loss) is optional —
    the Streamlit app passes one to drive a progress bar; the notebook omits it."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = df.sort_values("Timestamp")
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
        ]
    )
    X_train = pre.fit_transform(engineer(train)).astype("float32")
    X_test = pre.transform(engineer(test)).astype("float32")
    y_train = train["Is Laundering"].to_numpy("float32")
    y_test = test["Is Laundering"].to_numpy("float32")
    n_features = X_train.shape[1]

    train_dl = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1)
        ),
        batch_size=4096,
        shuffle=True,
    )

    model = FraudNet(n_features).to(device)
    pos = y_train.sum()
    neg = len(y_train) - pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        epoch_loss = running / len(train_dl.dataset)
        if progress_cb:
            progress_cb(epoch + 1, epochs, epoch_loss)

    # evaluate
    model.eval()
    with torch.no_grad():
        probs = (
            torch.sigmoid(model(torch.from_numpy(X_test).to(device)))
            .cpu()
            .numpy()
            .ravel()
        )
    preds = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    metrics = {
        "precision": p,
        "recall": r,
        "f1": f1,
        "auc_pr": average_precision_score(y_test, probs),
        "test_positives": int(y_test.sum()),
    }

    return TrainedBundle(model, pre, metrics, n_features)


def predict_one(bundle: TrainedBundle, raw_txn: dict) -> float:
    """raw_txn is a dict with the same raw columns as the CSV (minus label/IDs)."""
    device = next(bundle.model.parameters()).device
    X = bundle.preprocessor.transform(engineer(pd.DataFrame([raw_txn]))).astype(
        "float32"
    )
    bundle.model.eval()
    with torch.no_grad():
        return torch.sigmoid(bundle.model(torch.from_numpy(X).to(device))).item()
