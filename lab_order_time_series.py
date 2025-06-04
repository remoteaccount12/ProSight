import os
from datetime import datetime



from typing import List, Tuple

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from pytorch_lightning.loggers import TensorBoardLogger

import pickle


# -------------------------------
# Configuration
# -------------------------------
DATA_FILE = "ediagno_metrics_2025_06_04.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"
PLOTS_DIR = "plots"
MODEL_FILE = os.path.join(MODEL_DIR, "tft_model.pth")
PARAMS_FILE = os.path.join(MODEL_DIR, "tft_model_params.json")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")


GROUP_COLS = ["status", "is_droplet"]
METRICS = ["placed_gmv", "delivered_gmv", "cancelled_gmv"]
TARGET_COL = "delivered_gmv"

MODEL_PARAMS = dict(
    input_chunk_length=14,
    output_chunk_length=7,
    hidden_size=16,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=32,
    n_epochs=5,
    random_state=42,
)


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_series() -> List[TimeSeries]:
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])

    grouped = (
        df.groupby(GROUP_COLS + ["date"], as_index=False)[METRICS].sum()
    )

    series_list = TimeSeries.from_group_dataframe(
        grouped,
        group_cols=GROUP_COLS,
        time_col="date",
        static_cols=GROUP_COLS,
        value_cols=METRICS,
        freq="D",
    )
    return series_list


def create_model(logger: TensorBoardLogger) -> TFTModel:
    return TFTModel(log_tensorboard=True, tensorboard_logger=logger, **MODEL_PARAMS)


def train_model(series_list: List[TimeSeries]) -> Tuple[TFTModel, Scaler]:
    """Train the TFT model and persist both model and scaler."""
    ensure_dirs()
    logger = TensorBoardLogger(LOG_DIR, name="tft")

    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = TFTModel.load(MODEL_FILE)
        with open(SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)

    else:
        model = create_model(logger)
        scaler = Scaler()
        series_scaled = [scaler.fit_transform(s) for s in series_list]
        model.fit(series_scaled, verbose=True)
        model.save(MODEL_FILE)

        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)
        with open(PARAMS_FILE, "w") as f:
            import json
            json.dump(MODEL_PARAMS, f, indent=2)
    return model, scaler


def forecast_and_plot(
    model: TFTModel,
    series_list: List[TimeSeries],
    scaler: Scaler | None = None,
    horizon: int = 7,
):
    """Use the trained model to forecast and save result plots."""
    if scaler is None:
        if os.path.exists(SCALER_FILE):
            with open(SCALER_FILE, "rb") as f:
                scaler = pickle.load(f)
        else:
            scaler = Scaler()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join(PLOTS_DIR, timestamp)
    os.makedirs(plot_dir, exist_ok=True)




        scaled_series = [scaler.transform(s) for s in series_list]

    forecasts = [model.predict(horizon, s) for s in scaled_series]

    for i, (orig, pred) in enumerate(zip(series_list, forecasts)):
        df_orig = orig.pd_dataframe()
        df_pred = pred.pd_dataframe()

        status = orig.static_covariates["status"].iloc[0]
        droplet = orig.static_covariates["is_droplet"].iloc[0]
        filename = f"series_{i}_{status}_{droplet}.png"
        fig_path = os.path.join(plot_dir, filename)

        ax = df_orig[TARGET_COL].plot(label="actual")
        df_pred[TARGET_COL].plot(ax=ax, label="forecast")
        ax.legend()
        fig = ax.get_figure()
        fig.savefig(fig_path)
        ax.cla()


import numpy as np

def backtest_and_save(
    model: TFTModel,
    series_list: List[TimeSeries],

    scaler: Scaler | None = None,
    test_fraction: float = 0.2,
    z_thresh: float = 3.0,
):
    """Backtest using the trained model and record anomalies."""
    if scaler is None:
        if os.path.exists(SCALER_FILE):
            with open(SCALER_FILE, "rb") as f:
                scaler = pickle.load(f)
        else:
            scaler = Scaler()

    anomalies = []

    for idx, series in enumerate(series_list):
        status = series.static_covariates["status"].iloc[0]
        droplet = series.static_covariates["is_droplet"].iloc[0]

        series = scaler.transform(series)
        train, val = series.split_before(1 - test_fraction)

        pred = model.predict(len(val), train)

        resid = (val - pred).pd_dataframe()
        mean = resid.mean()
        std = resid.std()
        z = ((resid - mean) / std).abs()
        anom = z[z > z_thresh].dropna(how="all")

        for time, row in anom.iterrows():
            anomalies.append(
                {
                    "series": idx,
                    "time": time,
                    "status": status,

                    "is_droplet": droplet,
                    **row.to_dict(),
                }
            )

    if anomalies:
        import pandas as pd

        df_anom = pd.DataFrame(anomalies)
        df_anom.to_csv(os.path.join(PLOTS_DIR, "anomalies.csv"), index=False)

        counts = (
            df_anom.groupby(["status", "is_droplet"]).size().reset_index(name="count")
        )
        counts.to_csv(os.path.join(PLOTS_DIR, "anomaly_counts.csv"), index=False)


def main():
    series_list = load_series()

    model, scaler = train_model(series_list)
    forecast_and_plot(model, series_list, scaler)
    backtest_and_save(model, series_list, scaler)



if __name__ == "__main__":
    main()
