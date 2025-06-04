# ProSight Timeseries Forecasting

This project demonstrates using the [Darts](https://github.com/unit8co/darts) library to model lab order metrics with a Temporal Fusion Transformer (TFT).

## Requirements
- Python 3.12
- `darts` and its dependencies (PyTorch Lightning, PyTorch)

Install the required packages with:
```bash
pip install darts[torch] plotly
```

## Usage
The main script is `lab_order_time_series.py`. It expects the CSV data file `ediagno_metrics_2025_06_04.csv` in the project root. Run:
```bash
python3 lab_order_time_series.py
```
The script will:
1. Train a TFT model (or load an existing one from `models/`).
2. Save model parameters and a scaler for reuse.
3. Generate forecasts and save plots under `plots/<timestamp>/`.
4. Backtest the model and output detected anomalies to CSV files.

Training logs are written to `logs/` for visualization in TensorBoard.