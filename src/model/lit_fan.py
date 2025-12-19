import torch
import torch.nn as nn
import pytorch_lightning as pl
import mlflow
from src.model.fan import FAN

class LitFAN(pl.LightningModule):
    def __init__(self, input_dim, d_model=64, forecast_horizon=30, output_dim=2, num_layers=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = FAN(input_dim, d_model, forecast_horizon, output_dim, num_layers)
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # (batch, 2)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        
        # Log specific metrics for Price and ROI
        # Assuming y[:, 0] is Price, y[:, 1] is ROI
        price_mae = torch.nn.functional.l1_loss(y_hat[:, 0], y[:, 0])
        roi_mae = torch.nn.functional.l1_loss(y_hat[:, 1], y[:, 1])
        
        self.log("val_price_mae", price_mae)
        self.log("val_roi_mae", roi_mae)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_start(self):
        mlflow.pytorch.autolog()
