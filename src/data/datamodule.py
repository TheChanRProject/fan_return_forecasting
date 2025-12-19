import pytorch_lightning as pl
import duckdb
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class StockDataset(Dataset):
    def __init__(self, db_path, split="train", seq_len=60):
        self.db_path = db_path
        self.seq_len = seq_len
        self.split = split
        
        # Load data from DuckDB
        con = duckdb.connect(db_path, read_only=True)
        # Simple split logic: 80% train, 10% val, 10% test
        # In really, we should split by date.
        # Let's just load everything and split in memory for simplicity/speed with this small data.
        
        # We need sequences of (seq_len) -> target
        # Let's load ticker, date, close, target_close, target_roi
        # We need to normalize data per ticker or globally.
        # For simplicity, let's just return raw values and handle normalization here or in model.
        # Ideally, input should be Returns, not Prices, for stationarity.
        # But user asked for Price forecasting too.
        # Let's stick to the plan: Input = Close prices (normalized?)
        
        # For this prototype:
        # Load all data, group by ticker.
        df = con.execute("SELECT ticker, close, target_close, target_roi FROM price_data ORDER BY ticker, date").fetchdf()
        con.close()
        
        self.samples = []
        
        for ticker, group in df.groupby("ticker"):
            # Create sliding windows
            values = group["close"].values
            targets_price = group["target_close"].values
            targets_roi = group["target_roi"].values
            
            if len(values) <= seq_len:
                continue
                
            # Create indices
            # We can use numpy sliding_window_view but let's be explicit
            for i in range(len(values) - seq_len):
                seq = values[i : i + seq_len]
                # Target is specific to the last step of the sequence?
                # The ETL prepared target_close/roi aligned with the row.
                # So if we take row[i+seq_len-1], it has the target for 30 days ahead.
                
                # Check alignment in ETL: 
                # df = df.with_columns(pl.col("close").shift(-30).alias("target_close"))
                # So row `t` has prices at `t` and target at `t+30`.
                # If input sequence is [t-59 ... t], we want to predict target for t.
                
                # So we take the target from the LAST element of the sequence.
                idx_last = i + seq_len - 1
                
                tgt_p = targets_price[idx_last]
                tgt_r = targets_roi[idx_last]
                
                # Simple normalization (divide by first element of sequence)
                norm_factor = seq[0]
                seq_norm = seq / norm_factor
                
                # We also need to destandardize the output price if we want real metrics
                # But for the model regression, better to predict normalized price target?
                # Target Price is also normalized by the same factor
                tgt_p_norm = tgt_p / norm_factor
                
                self.samples.append({
                    "x": seq_norm.astype(np.float32), 
                    "y": np.array([tgt_p_norm, tgt_r], dtype=np.float32),
                    "norm": norm_factor
                })
                
        # Split
        n = len(self.samples)
        if split == "train":
            self.samples = self.samples[:int(0.8*n)]
        elif split == "val":
            self.samples = self.samples[int(0.8*n):int(0.9*n)]
        else:
            self.samples = self.samples[int(0.9*n):]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # x shape: (seq_len, 1) if input_dim=1
        return torch.tensor(item["x"]).unsqueeze(-1), torch.tensor(item["y"])

class StockDataModule(pl.LightningDataModule):
    def __init__(self, db_path="stocks.duckdb", batch_size=32, seq_len=60):
        super().__init__()
        self.db_path = db_path
        self.batch_size = batch_size
        self.seq_len = seq_len

    def setup(self, stage=None):
        self.train_ds = StockDataset(self.db_path, "train", self.seq_len)
        self.val_ds = StockDataset(self.db_path, "val", self.seq_len)
        self.test_ds = StockDataset(self.db_path, "test", self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
