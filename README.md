# FAN Stock Forecasting

This project implements a **Fourier Analysis Network (FAN)** to forecast stock prices and 30-day Return on Investment (ROI). It uses a modern data stack with **Polars**, **DuckDB**, and **PyTorch Lightning**.

## Features

- **Model**: Custom Fourier Analysis Network (FAN) architecture.
- **Data Pipeline**: High-performance ETL using Polars and DuckDB.
- **Training**: PyTorch Lightning loop with MLFlow logging.
- **Monitoring**: Drift detection with Evidently.ai.
- **Infrastructure**: Job submission scripts for `advisor.sh`.

## Project Structure

```
.
├── data/
│   ├── raw/            # Place Kaggle CSV files here
│   └── stocks.duckdb   # Generated DuckDB database
├── src/
│   ├── data/           # ETL and Lightning DataModule
│   ├── model/          # FAN implementation
│   └── utils/          # Logging and drift detection
├── scripts/            # Helper scripts (e.g., job submission)
└── train.py            # Main training entry point
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   Download stock market data (e.g., from Kaggle) and place the CSV files in `data/raw/`.
   
   Run the ETL pipeline to create the DuckDB database:
   ```bash
   python src/data/etl.py
   ```

## Usage

### Local Training

To train the model locally:
```bash
python train.py
```
This will start the training loop, log metrics to MLFlow (local dir), and save checkpoints.

### Cloud Training (Advisor)

To submit a training job using Advisor:

1. Ensure the `advisor` binary is in the project root or your PATH.
2. Run the submission script:
   ```bash
   ./scripts/submit_job.sh
   ```

## Monitoring

- **MLFlow**: Run `mlflow ui` to view training metrics.
- **Evidently**: Run `src/utils/drift.py` to generate data drift reports in `reports/`.