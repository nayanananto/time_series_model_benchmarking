# seasonal_train_lstm.py
"""
Seasonal LSTM training script for Repo B (model training repo).

It:
  - merges existing data/wind_data.csv (Repo B) with latest hourly CSV from Repo A,
  - keeps only a rolling window of history (max_history_days),
  - trains a direct multi-horizon univariate LSTM on the last train_days,
  - outputs a single models/lstm_model.pkl bundle: {"model_bytes", "scaler", "meta"}.

Assumptions:
  - Both repos use a file named data/wind_data.csv.
  - The file in Repo A has at least: datetime, wind_speed, wind_deg (fewer columns).
  - The file in Repo B has at least: datetime, wind_speed (often more columns like wind_deg, temperature, humidity).
"""

import argparse
import io
import os
import pickle

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ---------- utils ----------

def set_reproducible(seed: int = 42):
    """Roughly deterministic TF + numpy + Python for repeatable training."""
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass


def fetch_hourly_from_url(url: str, time_col: str, target_col: str) -> pd.DataFrame:
    """
    Download the hourly wind CSV from Repo A and return a DataFrame.

    - Uses sep=None so pandas auto-detects ',' vs '\\t'.
    - Requires at least [time_col, target_col] to be present.
    - Keeps all columns from Repo A (e.g. datetime, wind_speed, wind_deg).
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    text = resp.text
    if not text.strip():
        raise RuntimeError(f"Hourly URL {url} returned empty response")

    first_line = text.splitlines()[0]
    # Guard against HTML / 404 pages
    if ("<!DOCTYPE html" in first_line) or ("<html" in first_line.lower()):
        raise RuntimeError(
            f"Hourly URL {url} does not look like CSV. "
            f"First line: {first_line!r}"
        )

    buf = io.StringIO(text)
    # Let pandas sniff the delimiter (handles ',', '\\t', ';', etc.)
    df = pd.read_csv(buf, sep=None, engine="python")

    missing = [c for c in (time_col, target_col) if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Hourly CSV from {url} is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def load_and_merge_history(
    history_path: str,
    hourly_url: str | None,
    time_col: str,
    target_col: str,
    max_history_days: int | None,
) -> pd.DataFrame:
    """
    Load Repo B history (data/wind_data.csv), optionally fetch new hourly data from Repo A,
    append, deduplicate by timestamp, clip to max_history_days.

    - History in Repo B may have more columns (wind_deg, temperature, humidity, ...).
    - Hourly CSV in Repo A may have fewer columns (datetime, wind_speed, wind_deg).
    - We keep the union of columns; univariate LSTM only uses target_col for training.
    """
    # --- load history from Repo B ---
    if os.path.exists(history_path):
        # auto-detect separator (comma or tab)
        hist = pd.read_csv(history_path, sep=None, engine="python")
        if time_col not in hist.columns or target_col not in hist.columns:
            raise ValueError(
                f"{history_path} must contain at least [{time_col}, {target_col}]. "
                f"Cols found: {list(hist.columns)}"
            )
    else:
        hist = pd.DataFrame(columns=[time_col, target_col])

    # --- load new hourly from Repo A ---
    if hourly_url:
        new_df = fetch_hourly_from_url(hourly_url, time_col, target_col)
        # union of columns, missing values filled with NaN
        all_df = pd.concat([hist, new_df], ignore_index=True, sort=False)
    else:
        all_df = hist.copy()

    if all_df.empty:
        raise ValueError("No data after merging history and hourly CSV.")

    all_df[time_col] = pd.to_datetime(all_df[time_col], errors="coerce")
    all_df = all_df.dropna(subset=[time_col, target_col])
    all_df[time_col] = all_df[time_col].dt.floor("H")

    # ensure unique hourly timestamps (keep latest row for each datetime)
    all_df = (
        all_df.sort_values(time_col)
        .drop_duplicates(time_col, keep="last")
        .reset_index(drop=True)
    )

    if max_history_days is not None and max_history_days > 0:
        last_time = all_df[time_col].max()
        cutoff = last_time - pd.Timedelta(days=max_history_days)
        all_df = all_df[all_df[time_col] >= cutoff].copy()
        all_df = all_df.reset_index(drop=True)

    return all_df


def make_seq_multi(series: np.ndarray, lookback: int, horizon: int):
    """
    Build multi-step supervised sequences for a univariate series.

    series: shape (N, 1)

    Returns:
      X: (num_samples, lookback, 1)
      Y: (num_samples, horizon)
    """
    X, Y = [], []
    for i in range(lookback, len(series) - horizon + 1):
        X.append(series[i - lookback : i, 0])
        Y.append(series[i : i + horizon, 0])
    X = np.asarray(X)
    Y = np.asarray(Y)
    if len(X) == 0:
        raise ValueError("Not enough data for requested lookback & horizon.")
    return X.reshape((X.shape[0], X.shape[1], 1)), Y


def train_univariate_lstm(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    lookback: int,
    horizon: int,
    train_days: int,
    units: int,
    epochs: int,
    batch_size: int,
    dropout: float,
    loss: str,
    seed: int,
):
    """
    Train a direct multi-horizon univariate LSTM on last train_days of data.

    Uses only df[target_col] for training, but df may contain many extra columns.
    """
    set_reproducible(seed)

    df = df.sort_values(time_col).reset_index(drop=True)
    last_time = df[time_col].max()
    start_time = last_time - pd.Timedelta(days=train_days)
    train_df = df[df[time_col].between(start_time, last_time)].copy()

    if train_df.shape[0] < (lookback + horizon + 10):
        raise ValueError(
            f"Not enough rows in last {train_days} days for lookback={lookback}, "
            f"horizon={horizon}. Have only {train_df.shape[0]} rows."
        )

    values = train_df[target_col].astype(float).values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, Y = make_seq_multi(scaled, lookback, horizon)

    model = Sequential()
    model.add(LSTM(units, input_shape=(lookback, 1)))
    if dropout and dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(horizon))

    keras_loss = tf.keras.losses.Huber() if loss.lower() == "huber" else "mse"
    model.compile(optimizer="adam", loss=keras_loss)

    es = EarlyStopping(
        monitor="loss",
        patience=3,
        min_delta=1e-4,
        restore_best_weights=True,
    )

    model.fit(
        X,
        Y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=False,
        callbacks=[es],
    )

    meta = {
        "time_col": time_col,
        "target_col": target_col,
        "lookback": lookback,
        "horizon": horizon,
        "train_days": train_days,
        "units": units,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout": dropout,
        "loss": loss.lower(),
        "seed": seed,
        "train_start": train_df[time_col].min().isoformat(),
        "train_end": train_df[time_col].max().isoformat(),
    }
    return model, scaler, meta


def main():
    ap = argparse.ArgumentParser(
        "Seasonal LSTM training using wind-data-pipeline repo output."
    )

    ap.add_argument(
        "--history",
        type=str,
        default="data/wind_data.csv",
        help="Historical CSV in Repo B (will be updated).",
    )
    ap.add_argument(
        "--hourly_url",
        type=str,
        required=True,
        help="Raw CSV URL from your wind-data-pipeline repo (Repo A).",
    )
    ap.add_argument("--time_col", type=str, default="datetime")
    ap.add_argument("--target_col", type=str, default="wind_speed")

    ap.add_argument(
        "--max_history_days",
        type=int,
        default=730,
        help="Keep at most this many days of history (e.g. 730 ~ 2 years).",
    )
    ap.add_argument(
        "--train_days",
        type=int,
        default=180,
        help="Train on last N days from merged history.",
    )
    ap.add_argument("--lookback", type=int, default=168, help="Lookback window (hours).")
    ap.add_argument(
        "--horizon", type=int, default=168, help="Forecast horizon (hours)."
    )
    ap.add_argument("--units", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    ap.add_argument("--seed", type=int, default=202)

    ap.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory where lstm_model.pkl goes.",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default="lstm_model.pkl",
        help="Filename inside output_dir.",
    )

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.history), exist_ok=True)

    # 1) Merge existing history with new hourly data from Repo A
    df_hist = load_and_merge_history(
        history_path=args.history,
        hourly_url=args.hourly_url,
        time_col=args.time_col,
        target_col=args.target_col,
        max_history_days=args.max_history_days,
    )
    df_hist.to_csv(args.history, index=False)
    print(f"Merged history saved to {args.history} ({len(df_hist)} rows).")
    print("History head:\n", df_hist.head())
    print("History tail:\n", df_hist.tail())

    # 2) Train LSTM on last train_days
    model, scaler, meta = train_univariate_lstm(
        df=df_hist,
        time_col=args.time_col,
        target_col=args.target_col,
        lookback=args.lookback,
        horizon=args.horizon,
        train_days=args.train_days,
        units=args.units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        loss=args.loss,
        seed=args.seed,
    )

    # 3) Export bundle as lstm_model.pkl
    temp_model_path = os.path.join(args.output_dir, "_temp_lstm.keras")
    model.save(temp_model_path)

    with open(temp_model_path, "rb") as f:
        model_bytes = f.read()
    try:
        os.remove(temp_model_path)
    except OSError:
        pass

    bundle = {
        "model_bytes": model_bytes,
        "scaler": scaler,
        "meta": meta,
    }

    out_path = os.path.join(args.output_dir, args.model_name)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved pretrained LSTM bundle to {out_path}")
    print("Meta:", meta)


if __name__ == "__main__":
    main()
