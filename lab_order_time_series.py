import os
from datetime import datetime
from typing import List

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from pytorch_lightning.loggers import TensorBoardLogger


# -------------------------------
# Configuration
# -------------------------------
DATA_FILE = "ediagno_metrics_2025_06_04.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"
PLOTS_DIR = "plots"
MODEL_FILE = os.path.join(MODEL_DIR, "tft_model.pth")
PARAMS_FILE = os.path.join(MODEL_DIR, "tft_model_params.json")

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
    df[METRICS] = df[METRICS].fillna(0)

    grouped = (
        df.groupby(GROUP_COLS + ["date"], as_index=False)[METRICS].sum()
    )

    series_list = TimeSeries.from_group_dataframe(
        grouped,
        group_cols=GROUP_COLS,
        time_col="date",
        value_cols=METRICS,
        freq="D",
    )
    min_len = MODEL_PARAMS["input_chunk_length"] + MODEL_PARAMS["output_chunk_length"]
    series_list = [s for s in series_list if len(s) >= min_len]
    return series_list


def create_model(logger: TensorBoardLogger) -> TFTModel:
    return TFTModel(
        log_tensorboard=True,
        pl_trainer_kwargs={"logger": logger},
        use_static_covariates=False,
        add_relative_index=True,
        **MODEL_PARAMS,
    )


def train_model(series_list: List[TimeSeries]) -> TFTModel:
    ensure_dirs()
    logger = TensorBoardLogger(LOG_DIR, name="tft")

    if os.path.exists(MODEL_FILE):
        model = TFTModel.load(MODEL_FILE)
    else:
        model = create_model(logger)
        scaler = Scaler()
        series_scaled = [scaler.fit_transform(s) for s in series_list]
        model.fit(series_scaled, verbose=True)
        model.save(MODEL_FILE)
        with open(PARAMS_FILE, "w") as f:
            import json
            json.dump(MODEL_PARAMS, f, indent=2)
    return model


def forecast_and_plot(model: TFTModel, series_list: List[TimeSeries], horizon: int = 7):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join(PLOTS_DIR, timestamp)
    os.makedirs(plot_dir, exist_ok=True)

    scaler = Scaler()
    scaled_series = [scaler.fit_transform(s) for s in series_list]
    forecasts = [model.predict(horizon, s) for s in scaled_series]

    for i, (orig, pred) in enumerate(zip(series_list, forecasts)):
        df_orig = orig.pd_dataframe()
        df_pred = pred.pd_dataframe()
        fig_path = os.path.join(plot_dir, f"series_{i}.png")
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
    test_fraction: float = 0.2,
    z_thresh: float = 3.0,
):
    """Backtest the model and record detected anomalies.

    Parameters
    ----------
    model : TFTModel
        The trained TFT model.
    series_list : List[TimeSeries]
        List of time series to evaluate.
    test_fraction : float, optional
        Fraction of the series to reserve for validation, by default ``0.2``.
    z_thresh : float, optional
        Z-score threshold to flag anomalies, by default ``3.0``.
    """

    anomalies = []
    scaler = Scaler()

    for idx, series in enumerate(series_list):
        series = scaler.fit_transform(series)
        status = series.static_covariates["status"].iloc[0]
        is_droplet = series.static_covariates["is_droplet"].iloc[0]

        train, val = series.split_before(1 - test_fraction)
        model.fit([train], verbose=False)
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
                    "is_droplet": is_droplet,
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
    model = train_model(series_list)
    forecast_and_plot(model, series_list)
    backtest_and_save(model, series_list)


if __name__ == "__main__":
    main()
