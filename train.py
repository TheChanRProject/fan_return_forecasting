import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model.lit_fan import LitFAN
from src.data.datamodule import StockDataModule
import sys

def train():
    pl.seed_everything(42)
    
    # Params
    DB_PATH = "stocks.duckdb"
    SEQ_LEN = 30 
    BATCH_SIZE = 64 # Increased batch size
    
    dm = StockDataModule(DB_PATH, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    
    # Model
    # Input dim 8: close, daily_ret, log_ret, sma10, sma30, sma50, vol, rsi
    model = LitFAN(input_dim=8, d_model=32, forecast_horizon=30, num_layers=2)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu", # Use "gpu" or "mps" if available
        devices=1,
        default_root_dir="logs",
        callbacks=[ModelCheckpoint(monitor="val_loss")]
    )
    
    # Train
    trainer.fit(model, dm)

if __name__ == "__main__":
    train()
