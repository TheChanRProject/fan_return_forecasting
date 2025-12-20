import polars as pl
import duckdb
import glob
import os
from pathlib import Path

def process_data(raw_dir: str = "data/raw", db_path: str = "stocks.duckdb"):
    print("Starting ETL pipeline...")
    
    # 1. Connect/Create DuckDB
    con = duckdb.connect(db_path)
    
    # 2. List CSV files
    files = glob.glob(os.path.join(raw_dir, "*.csv"))
    # files = files[:5] # Limit removed
    
    if not files:
        print("No CSV files found in data/raw. Skipping ETL.")
        return

    # 3. Process each file
    master_df_list = []
    
    print(f"Found {len(files)} files. Processing...")

    for f in files:
        try:
            # Infer ticker from filename (e.g. AAPL.csv)
            ticker = Path(f).stem
            
            # Read CSV
            df = pl.read_csv(f, ignore_errors=True)
            
            # Standardize column names
            df = df.rename({c: c.lower() for c in df.columns})
            
            # Helper to check if col exists
            required_cols = ["date", "close", "open", "high", "low", "volume"]
            if not all(col in df.columns for col in required_cols):
                # print(f"Skipping {ticker}: Missing columns")
                continue

            # Cast date - Format is DD-MM-YYYY based on checking head of AAPL.csv
            df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%d-%m-%Y"))
            
            # Enforce types for numeric columns to prevent schema mismatch during concatenation
            # Volume might be int in some files and float in others
            df = df.with_columns([
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64)
            ])
            
            # Sort by date
            df = df.sort("date")
            
            # --- Feature Engineering ---
            
            # 1. Daily Returns
            df = df.with_columns([
                ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1)).alias("daily_return"),
                (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")
            ])
            
            # 2. Simple Moving Averages (SMA)
            df = df.with_columns([
                pl.col("close").rolling_mean(window_size=10).alias("sma_10"),
                pl.col("close").rolling_mean(window_size=30).alias("sma_30"),
                pl.col("close").rolling_mean(window_size=50).alias("sma_50")
            ])
            
            # 3. Volatility (Rolling Std Dev of Returns)
            df = df.with_columns(
                pl.col("daily_return").rolling_std(window_size=20).alias("volatility_20d")
            )

            # 4. RSI (Relative Strength Index) - 14 days
            # Simple approximation using rolling mean for gains/losses
            delta = pl.col("close").diff()
            up = delta.clip(lower_bound=0)
            down = delta.clip(upper_bound=0).abs()
            
            roll_up = up.rolling_mean(window_size=14)
            roll_down = down.rolling_mean(window_size=14)
            
            rs = roll_up / roll_down
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            df = df.with_columns(rsi.alias("rsi_14"))
            
            # 5. Target Generation
            # Calculate 30-day forward Close (Target)
            # And Calculate Return
            df = df.with_columns([
                pl.col("close").shift(-30).alias("target_close"),
                pl.lit(ticker).alias("ticker")
            ])
            
            # Drop nulls created by rolling windows and shifting
            # We lose the first 50 days (max SMA) and last 30 days (target)
            df = df.drop_nulls()
            
            # Calculate ROI: (Target - Current) / Current
            df = df.with_columns(
                ((pl.col("target_close") - pl.col("close")) / pl.col("close")).alias("target_roi")
            )
            
            if len(df) > 0:
                master_df_list.append(df)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if not master_df_list:
        print("No valid data processed.")
        return

    # 4. Concatenate
    print("Concatenating dataframes...")
    master_tbl = pl.concat(master_df_list)
    
    # 5. Save to DuckDB
    print(f"Saving to {db_path}...")
    # Convert to Arrow -> DuckDB
    con.execute("CREATE OR REPLACE TABLE price_data AS SELECT * FROM master_tbl")
    
    print(f"ETL Complete. Saved {len(master_tbl)} rows to {db_path}.")
    con.close()

if __name__ == "__main__":
    process_data()
