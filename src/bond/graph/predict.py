from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PCE_FILE = BASE_DIR / "pce_releases.csv"
DATA_FILE = BASE_DIR / "training_data.csv"
PREDICTIONS_FILE = BASE_DIR / "predictions.csv"
MODEL_FILE = BASE_DIR / "tnx_direction_model.keras"
PLOT_FILE = BASE_DIR / "tnx_direction_plot.png"

MARKET_TZ = "America/New_York"
MARKET_OPEN = "09:30"
FEATURE_END = "10:30"

DOWNLOAD_PERIOD = "60d"
DOWNLOAD_INTERVAL = "5m"

EPOCHS = 100
BATCH_SIZE = 16
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------------------------
# Starter PCE data (BEA Core PCE YoY %)
# --------------------------------------------------
# These are realistic values but are NOT verified against the BEA.
# Replace with actual BEA releases when you have them.
STARTER_PCE = """\
release_date,pce_yoy
2024-02-29,2.8
2024-03-29,2.8
2024-04-26,2.8
2024-05-31,2.7
2024-06-28,2.6
2024-07-26,2.6
2024-08-30,2.7
2024-09-27,2.7
2024-10-31,2.8
2024-11-27,2.8
2024-12-20,2.8
2025-01-31,2.6
2025-02-28,2.6
2025-03-28,2.7
2025-04-30,2.6
2025-05-30,2.6
2025-06-27,2.7
2025-07-31,2.6
2025-08-29,2.7
2025-09-26,2.7
2025-10-31,2.7
2025-11-26,2.6
2025-12-19,2.7
2026-01-30,2.6
2026-02-27,2.6
2026-03-27,2.6
2026-04-30,2.5
2026-05-29,2.5
2026-06-26,2.6
2026-07-31,2.6
2026-08-28,2.6
2026-09-25,2.5
2026-10-30,2.5
"""


# --------------------------------------------------
# Date helper
# --------------------------------------------------

def naive_midnight(values):
    """Return tz-naive, midnight-normalized Timestamps/Index."""
    ts = pd.to_datetime(values)

    if isinstance(ts, pd.Series):
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
        return ts.dt.normalize()

    idx = pd.DatetimeIndex(ts)
    if idx.tz is not None:
        idx = idx.tz_convert(MARKET_TZ).tz_localize(None)
    return idx.normalize()


# --------------------------------------------------
# Create (or repair) the PCE input file
# --------------------------------------------------

def _pce_file_is_valid(path):
    """True if the file already has the two required columns."""
    if not path.exists():
        return False
    try:
        head = pd.read_csv(path, encoding="utf-8-sig", nrows=1)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    cols = {
        c.replace("\ufeff", "").strip().lower()
        for c in head.columns
    }
    return {"release_date", "pce_yoy"}.issubset(cols)


def create_input_files():
    if _pce_file_is_valid(PCE_FILE):
        return

    if PCE_FILE.exists():
        backup = PCE_FILE.with_suffix(".csv.bak")
        try:
            PCE_FILE.replace(backup)
            print(f"Existing PCE file had wrong format; backed up to: {backup}")
        except OSError as exc:
            raise RuntimeError(
                f"Could not back up existing {PCE_FILE}: {exc}. "
                "Move or delete the file manually and re-run."
            ) from exc

    PCE_FILE.write_text(STARTER_PCE, encoding="utf-8")
    print(
        f"Wrote starter PCE file: {PCE_FILE}\n"
        "  -> Values are PLACEHOLDERS. Verify against BEA release "
        "calendar before trusting predictions."
    )


# --------------------------------------------------
# Load published PCE values
# --------------------------------------------------

def load_pce():
    try:
        pce = pd.read_csv(PCE_FILE, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"{PCE_FILE} is empty. Add a header 'release_date,pce_yoy' "
            "and at least one row."
        ) from exc

    # Normalize header names: strip BOM, whitespace, lowercase.
    pce.columns = (
        pce.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    aliases = {
        "release_date": "release_date",
        "date":         "release_date",
        "pce_yoy":      "pce_yoy",
        "pce":          "pce_yoy",
        "yoy":          "pce_yoy",
    }
    pce = pce.rename(
        columns={c: aliases[c] for c in pce.columns if c in aliases}
    )

    missing = {"release_date", "pce_yoy"} - set(pce.columns)
    if missing:
        raise ValueError(
            f"{PCE_FILE} is missing column(s): {sorted(missing)}. "
            f"Found columns: {list(pce.columns)}."
        )

    if pce.empty:
        raise ValueError(
            f"{PCE_FILE} has a header but no data rows."
        )

    pce["release_date"] = naive_midnight(pce["release_date"])
    pce["pce_yoy"] = pd.to_numeric(pce["pce_yoy"], errors="coerce")

    pce = (
        pce.dropna(subset=["release_date", "pce_yoy"])
        .sort_values("release_date")
        .drop_duplicates("release_date", keep="last")
        .reset_index(drop=True)
    )

    if pce.empty:
        raise ValueError(
            f"{PCE_FILE} had no parseable rows after cleaning."
        )

    print(
        f"Loaded {len(pce)} PCE releases "
        f"({pce['release_date'].min().date()} -> "
        f"{pce['release_date'].max().date()})"
    )
    return pce


# --------------------------------------------------
# Download intraday market data
# --------------------------------------------------

def download_intraday(ticker):
    print(f"Downloading {ticker}...")

    data = yf.download(
        ticker,
        period=DOWNLOAD_PERIOD,
        interval=DOWNLOAD_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise RuntimeError(
            f"No intraday data received for {ticker}. "
            "Check whether Yahoo Finance supports this symbol."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        raise ValueError(f"Missing Close column for {ticker}")

    result = pd.DataFrame(
        {"close": pd.to_numeric(data["Close"], errors="coerce")}
    )
    result.index = pd.to_datetime(result.index)

    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")

    result.index = result.index.tz_convert(MARKET_TZ)
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()
    result = result.dropna(subset=["close"])

    return result


# --------------------------------------------------
# Average prices during the first hour
# --------------------------------------------------

def average_first_hour(data, feature_name):
    rows = []

    for day, group in data.groupby(data.index.date):
        start = pd.Timestamp(f"{day} {MARKET_OPEN}", tz=MARKET_TZ)
        end = pd.Timestamp(f"{day} {FEATURE_END}", tz=MARKET_TZ)

        window = group.loc[
            (group.index >= start) & (group.index < end),
            "close",
        ]

        if len(window) < 8:
            continue

        rows.append({
            "date": pd.Timestamp(day),
            feature_name: float(window.mean()),
            f"{feature_name}_at_1030": float(window.iloc[-1]),
            f"{feature_name}_bar_count": len(window),
        })

    columns = [
        "date",
        feature_name,
        f"{feature_name}_at_1030",
        f"{feature_name}_bar_count",
    ]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------
# Download daily TNX closing values
# --------------------------------------------------

def download_daily_tnx():
    print("Downloading daily TNX closing values...")

    data = yf.download(
        "^TNX",
        period=DOWNLOAD_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise RuntimeError("No daily TNX data received.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")

    dates = naive_midnight(data.index)

    result = pd.DataFrame({
        "date": dates,
        "TNX_close": close.to_numpy(),
    })

    return (
        result.dropna()
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


# --------------------------------------------------
# Build the daily training dataset
# --------------------------------------------------

def build_dataset():
    zn_data = download_intraday("ZN=F")
    move_data = download_intraday("^MOVE")
    tnx_data = download_intraday("^TNX")

    zn = average_first_hour(zn_data, "ZN")
    move = average_first_hour(move_data, "MOVE")
    tnx = average_first_hour(tnx_data, "TNX")

    print(
        f"  ZN days: {len(zn)}, MOVE days: {len(move)}, "
        f"TNX days: {len(tnx)}"
    )

    daily = zn.merge(move, on="date", how="inner")
    daily = daily.merge(tnx, on="date", how="inner")
    daily = daily.merge(download_daily_tnx(), on="date", how="inner")

    pce = load_pce()

    daily["date"] = naive_midnight(daily["date"])
    pce["date"] = naive_midnight(pce["release_date"])

    daily = daily.sort_values("date")
    pce = pce[["date", "pce_yoy"]].sort_values("date")

    daily = pd.merge_asof(
        daily, pce, on="date", direction="backward",
    )

    features = ["pce_yoy", "ZN", "MOVE", "TNX"]

    daily["target_down"] = (
        daily["TNX_close"] < daily["TNX_at_1030"]
    ).astype(int)

    daily = daily.replace([np.inf, -np.inf], np.nan)
    daily = daily.dropna(
        subset=features + ["TNX_close", "TNX_at_1030"]
    )

    daily.to_csv(DATA_FILE, index=False)

    print(f"\nSaved training dataset: {DATA_FILE}")
    print(f"Usable trading days: {len(daily)}")
    if len(daily):
        print("\nTarget distribution:")
        print(
            daily["target_down"]
            .map({0: "Up or unchanged", 1: "Down"})
            .value_counts()
        )

    return daily, features


# --------------------------------------------------
# Train and evaluate the neural network
# --------------------------------------------------

def train_model(daily, features):
    if len(daily) < 30:
        raise ValueError(
            f"Only {len(daily)} trading days are available. "
            "At least 30 are required."
        )

    X = daily[features].to_numpy(dtype=np.float32)
    y = daily["target_down"].to_numpy(dtype=np.int32)

    split = int(len(daily) * 0.80)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "The training period contains only one class. "
            "More historical data is needed."
        )
    if len(X_test) == 0:
        raise ValueError("There are no test observations.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(features),)),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.20,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    majority_class = int(pd.Series(y_train).mode().iloc[0])
    baseline = np.full(len(y_test), majority_class, dtype=np.int32)

    print("\n--- Chronological test results ---")
    print(f"Training observations: {len(y_train)}")
    print(f"Test observations: {len(y_test)}")
    print(f"Model accuracy:    {accuracy_score(y_test, predictions):.3f}")
    print(f"Baseline accuracy: {accuracy_score(y_test, baseline):.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["TNX up or unchanged", "TNX down"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    result = daily.iloc[split:][
        ["date", "TNX_at_1030", "TNX_close", "target_down"]
    ].copy()

    result["predicted_class"] = predictions
    result["probability_up_or_unchanged"] = probabilities[:, 0]
    result["probability_down"] = probabilities[:, 1]
    result["baseline_class"] = baseline
    result["actual_label"] = result["target_down"].map({
        0: "Up or unchanged", 1: "Down",
    })
    result["predicted_label"] = result["predicted_class"].map({
        0: "Up or unchanged", 1: "Down",
    })

    result.to_csv(PREDICTIONS_FILE, index=False)
    model.save(MODEL_FILE)

    print(f"\nSaved predictions: {PREDICTIONS_FILE}")
    print(f"Saved model: {MODEL_FILE}")

    plt.figure(figsize=(12, 5))
    plt.plot(
        result["date"], result["target_down"],
        marker="o", linestyle="", label="Actual (1=down)",
    )
    plt.plot(
        result["date"], result["predicted_class"],
        marker="x", linestyle="", label="Predicted (1=down)",
    )
    plt.yticks([0, 1], ["Up/unchanged", "Down"])
    plt.title("TNX direction: actual vs predicted")
    plt.xlabel("Date")
    plt.ylabel("Direction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=120)
    print(f"Saved plot: {PLOT_FILE}")

    try:
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
    except Exception as exc:
        print(f"(Plot display skipped: {exc})")
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print(f"Working directory: {BASE_DIR}")

    create_input_files()

    try:
        daily, features = build_dataset()
        train_model(daily, features)

    except Exception as exc:
        print(f"\nError: {exc}")
        print(
            "\nCheck your internet connection, data availability, "
            "PCE CSV contents, and installed packages."
        )


if __name__ == "__main__":
    main()