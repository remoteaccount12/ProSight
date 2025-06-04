# ProSight

ProSight provides time-series analysis tools for forecasting lab order metrics.

## Requirements
- Python 3.10 or higher
- pandas
- darts
- torch
- pytorch-lightning

Install the dependencies using:
```bash
pip install pandas darts torch pytorch-lightning
```

## Running the script

The `lab_order_time_series.py` script expects a metrics file named
`ediagno_metrics_2025_06_04.csv` in the project root. Execute it with:
```bash
python lab_order_time_series.py
```
This trains a TFT model and saves forecasts and anomaly reports under `models/`,
`logs/`, and `plots/`.
