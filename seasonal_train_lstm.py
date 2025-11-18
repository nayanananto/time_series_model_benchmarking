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

import os
import pandas as pd

import os
import pandas as pd

def fetch_wind_data(remote_url, local_path="data/wind_data.csv", hist_path="data/historical_wind.csv"):
    """
    Use historical_wind exactly like the reference function.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    df_remote = pd.read_csv(remote_url, parse_dates=["datetime"])
    df_hist = pd.read_csv(hist_path, parse_dates=["datetime"])

    out = (
        pd.concat([df_remote, df_hist], ignore_index=True)
          .drop_duplicates(subset=["datetime"], keep="last")
          .sort_values("datetime")
          .reset_index(drop=True)
    )

    out.to_csv(local_path, index=False)

    print("[DEBUG] Merged remote + historical_wind")
    print("[DEBUG] rows:", len(out))
    print("[DEBUG] datetime range:", out["datetime"].min(), "→", out["datetime"].max())

    return out



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
    hourly_url: str | None,
    time_col: str,
    target_col: str,
    max_history_days: int | None,
    history_path: str = "data/wind_data.csv",
    hist_path: str = "data/historical_wind.csv",
) -> pd.DataFrame:
    """
    1. If hourly_url is provided, refresh data/wind_data.csv using:
       - remote hourly file
       - historical_wind.csv

    2. Load data/wind_data.csv

    3. Optionally trim to a rolling window of max_history_days.
    """

    if hourly_url:
        # ✅ This line ensures merged data is always correct (no 2024 cut-off)
        fetch_wind_data(
            remote_url=hourly_url,
            local_path=history_path,
            hist_path=hist_path,
        )

    # Now load the merged history
    df = pd.read_csv(history_path, parse_dates=[time_col])
    df = df.dropna(subset=[time_col, target_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    print("[DEBUG] After load, rows:", len(df))
    print("[DEBUG] datetime range:", df[time_col].min(), "→", df[time_col].max())

    # Apply rolling window if requested
    if max_history_days and max_history_days > 0:
        last_time = df[time_col].max()
        cutoff = last_time - pd.Timedelta(days=max_history_days)
        df = df[df[time_col] >= cutoff].copy().reset_index(drop=True)
        print("[DEBUG] After max_history_days trim, rows:", len(df))
        print("[DEBUG] After max_history_days trim, datetime range:",
              df[time_col].min(), "→", df[time_col].max())

    return df



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
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser(
        description="Seasonal LSTM training using wind-data-pipeline repo output."
    )

    ap.add_argument("--hourly_url", type=str, required=True,
                    help="Remote hourly CSV from Repo A (Open-Meteo pipeline).")
    ap.add_argument("--time_col", type=str, default="datetime")
    ap.add_argument("--target_col", type=str, default="wind_speed")

    ap.add_argument("--max_history_days", type=int, default=0,
                    help="If >0, keep only this many days of history (rolling window).")
    ap.add_argument("--train_days", type=int, default=180,
                    help="Number of days from the end to use for training.")

    ap.add_argument("--lookback", type=int, default=168,
                    help="Number of past hours for each LSTM input window.")
    ap.add_argument("--horizon", type=int, default=168,
                    help="Number of future hours to predict directly.")

    ap.add_argument("--units", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--loss", type=str, default="mse")

    ap.add_argument("--output_dir", type=str, default="models")
    ap.add_argument("--model_name", type=str, default="lstm_model.pkl")

    args = ap.parse_args()

    time_col = args.time_col
    target_col = args.target_col

    # ------------------------------------------------------------------
    # 1) Build merged dataset EXACTLY like the reference fetch_wind_data
    # ------------------------------------------------------------------
    # This should be defined above:
    #   def fetch_wind_data(remote_url, local_path="data/wind_data.csv",
    #                       hist_path="data/historical_wind.csv"): ...
    df_all = fetch_wind_data(
        remote_url=args.hourly_url,
        local_path="data/wind_data.csv",
        hist_path="data/historical_wind.csv",
    )

    # --------------------------------------------------
    # 2) Basic cleaning + rolling window on full history
    # --------------------------------------------------
    df_all[time_col] = pd.to_datetime(df_all[time_col])
    df_all = df_all.dropna(subset=[time_col, target_col])
    df_all = df_all.sort_values(time_col).reset_index(drop=True)

    print("[DEBUG] After fetch_wind_data merge")
    print("[DEBUG] rows:", len(df_all))
    print("[DEBUG] datetime range:",
          df_all[time_col].min(), "→", df_all[time_col].max())

    if args.max_history_days and args.max_history_days > 0:
        last_time = df_all[time_col].max()
        cutoff = last_time - pd.Timedelta(days=args.max_history_days)
        df_all = df_all[df_all[time_col] >= cutoff].copy().reset_index(drop=True)
        print("[DEBUG] After max_history_days trim")
        print("[DEBUG] rows:", len(df_all))
        print("[DEBUG] datetime range:",
              df_all[time_col].min(), "→", df_all[time_col].max())

    # ---------------------------------------------------------
    # 3) Restrict to last train_days for actual model training
    # ---------------------------------------------------------
    if args.train_days and args.train_days > 0:
        last_time = df_all[time_col].max()
        cutoff_train = last_time - pd.Timedelta(days=args.train_days)
        df_train = df_all[df_all[time_col] >= cutoff_train].copy().reset_index(drop=True)
    else:
        df_train = df_all.copy()

    print("[DEBUG] Training window rows:", len(df_train))
    print("[DEBUG] Training window datetime range:",
          df_train[time_col].min(), "→", df_train[time_col].max())

    # ------------------------------
    # 4) Scale target & make windows
    # ------------------------------
    y = df_train[target_col].values.reshape(-1, 1)

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)

    def make_windows(series: np.ndarray, lookback: int, horizon: int):
        """
        series: shape (N, 1) after scaling
        returns X: (num_samples, lookback, 1), y: (num_samples, horizon)
        """
        X_list, y_list = [], []
        for i in range(len(series) - lookback - horizon + 1):
            past = series[i : i + lookback]
            future = series[i + lookback : i + lookback + horizon]
            X_list.append(past)
            y_list.append(future.ravel())
        return np.array(X_list), np.array(y_list)

    X, Y = make_windows(y_scaled, args.lookback, args.horizon)

    if len(X) == 0:
        raise ValueError(
            f"Not enough data to build windows: "
            f"N={len(y_scaled)}, lookback={args.lookback}, horizon={args.horizon}"
        )

    print("[DEBUG] Windowed dataset shapes:", X.shape, Y.shape)

    # ----------------------
    # 5) Build & train LSTM
    # ----------------------
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(args.lookback, 1)),
            tf.keras.layers.LSTM(args.units, return_sequences=False),
            tf.keras.layers.Dropout(args.dropout),
            tf.keras.layers.Dense(args.horizon),
        ]
    )

    model.compile(optimizer="adam", loss=args.loss)
    model.summary()

    history = model.fit(
        X,
        Y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    # ------------------------------
    # 6) Serialize model into bytes
    # ------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to a temporary .keras file then read bytes
    tmp_model_path = os.path.join(args.output_dir, "_tmp_lstm.keras")
    model.save(tmp_model_path)

    with open(tmp_model_path, "rb") as f:
        model_bytes = f.read()

    # Clean up temp file (optional)
    try:
        os.remove(tmp_model_path)
    except OSError:
        pass

    # ------------
    # 7) Meta info
    # ------------
    meta = {
        "time_col": time_col,
        "target_col": target_col,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "units": args.units,
        "dropout": args.dropout,
        "loss": args.loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_start": str(df_train[time_col].min()),
        "train_end": str(df_train[time_col].max()),
        "rows_total": int(len(df_all)),
        "rows_train": int(len(df_train)),
    }

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
