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
        
        # Load all features
        # Features: close, daily_return, log_return, sma_10, sma_30, sma_50, volatility_20d, rsi_14
        query = """
        SELECT 
            ticker, 
            close, daily_return, log_return, 
            sma_10, sma_30, sma_50, 
            volatility_20d, rsi_14,
            target_close, target_roi 
        FROM price_data 
        ORDER BY ticker, date
        """
        df = con.execute(query).fetchdf()
        con.close()
        
        self.samples = []
        
        # Pre-convert to numpy for speed
        # Groups: (ticker, dataframe)
        for ticker, group in df.groupby("ticker"):
            # Extract feature arrays
            # Shape: (N, 8)
            # 0:close, 1:daily_ret, 2:log_ret, 3:sma10, 4:sma30, 5:sma50, 6:vol, 7:rsi
            feature_cols = ["close", "daily_return", "log_return", "sma_10", "sma_30", "sma_50", "volatility_20d", "rsi_14"]
            data_arr = group[feature_cols].values.astype(np.float32)
            
            targets_price = group["target_close"].values.astype(np.float32)
            targets_roi = group["target_roi"].values.astype(np.float32)
            
            n_rows = len(data_arr)
            if n_rows <= seq_len:
                continue
                
            # Create indices
            for i in range(n_rows - seq_len):
                # Input sequence: [i, i+seq_len)
                seq = data_arr[i : i + seq_len]
                
                # Target at step i+seq_len-1 (which corresponds to t+30 horizon)
                idx_last = i + seq_len - 1
                tgt_p = targets_price[idx_last]
                tgt_r = targets_roi[idx_last]
                
                # Normalization
                # Price-like columns: close(0), sma10(3), sma30(4), sma50(5) -> Divide by seq[0, 0] (Start Close)
                norm_price = seq[0, 0]
                if norm_price == 0: norm_price = 1e-8 # Safety
                
                seq_norm = seq.copy()
                seq_norm[:, [0, 3, 4, 5]] /= norm_price
                
                # RSI(7) -> / 100.0
                seq_norm[:, 7] /= 100.0
                
                # Others (returns, vol) -> Keep as is
                
                # Target Price Normalization
                tgt_p_norm = tgt_p / norm_price
                
                self.samples.append({
                    "x": seq_norm, 
                    "y": np.array([tgt_p_norm, tgt_r], dtype=np.float32),
                    "norm": norm_price
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
        # x shape: (seq_len, num_features) = (seq_len, 8)
        # Model expects correct input_dim
        return torch.tensor(item["x"]), torch.tensor(item["y"])

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
