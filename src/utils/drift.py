from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.data_structure import DataStructure
import pandas as pd
import duckdb

def run_drift_check(reference_data, current_data, output_path="drift_report.html"):
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    # evidently expects pandas DFs
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(output_path)
    print(f"Drift report saved to {output_path}")

def check_drift_on_db(db_path="stocks.duckdb"):
    con = duckdb.connect(db_path, read_only=True)
    # Load recent data vs older data
    # Just for demo: split first half vs second half
    df = con.execute("SELECT * FROM price_data").fetchdf()
    mid = len(df) // 2
    ref = df.iloc[:mid]
    curr = df.iloc[mid:]
    
    run_drift_check(ref, curr, "reports/data_drift.html")

if __name__ == "__main__":
    check_drift_on_db()
