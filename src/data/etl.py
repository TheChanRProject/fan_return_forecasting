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
    files = files[:5] # Limit for now for testing if there are too many
    
    if not files:
        print("No CSV files found in data/raw. Skipping ETL.")
        return

    # 3. Process each file
    # We will append to a master table
    master_df_list = []
    
    for f in files:
        try:
            # Infer ticker from filename (e.g. AAPL.csv)
            ticker = Path(f).stem
            
            df = pl.read_csv(f, ignore_errors=True)
            
            # Standardize column names
            df = df.rename({c: c.lower() for c in df.columns})
            
            # Helper to check if col exists
            if "date" not in df.columns or "close" not in df.columns:
                continue

            # Cast date
            df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")) # Adjust format if needed
            
            # Sort by date
            df = df.sort("date")
            
            # Calculate 30-day forward Close (Target)
            # And Calculate Return
            df = df.with_columns([
                pl.col("close").shift(-30).alias("target_close"),
                pl.lit(ticker).alias("ticker")
            ])
            
            # Drop nulls (last 30 days)
            df = df.drop_nulls()
            
            # Calculate ROI: (Target - Current) / Current
            df = df.with_columns(
                ((pl.col("target_close") - pl.col("close")) / pl.col("close")).alias("target_roi")
            )
            
            master_df_list.append(df)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if not master_df_list:
        print("No valid data processed.")
        return

    # 4. Concatenate
    master_tbl = pl.concat(master_df_list)
    
    # 5. Save to DuckDB
    # Convert to Arrow -> DuckDB
    con.execute("CREATE OR REPLACE TABLE price_data AS SELECT * FROM master_tbl")
    
    print(f"ETL Complete. Saved {len(master_tbl)} rows to {db_path}.")
    con.close()

if __name__ == "__main__":
    process_data()
