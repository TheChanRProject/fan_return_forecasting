import torch
import torch.nn as nn
import torch.fft

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_real = nn.Linear(in_features, out_features)
        self.fc_imag = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x is complex: (batch, seq, features)
        real = x.real
        imag = x.imag
        out_real = self.fc_real(real) - self.fc_imag(imag)
        out_imag = self.fc_real(imag) + self.fc_imag(real)
        return torch.complex(out_real, out_imag)

class FBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.complex_linear = ComplexLinear(d_model, d_model)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model) - Real domain
        res = x
        
        # 1. FFT
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        
        # 2. Filter in Frequency Domain
        x_fft = self.complex_linear(x_fft)
        
        # 3. IFFT
        x_ifft = torch.fft.irfft(x_fft, n=x.size(1), dim=1, norm='ortho')
        
        # 4. Residual + Norm + Activation
        return self.layernorm(self.activation(x_ifft) + res)

class FAN(nn.Module):
    def __init__(self, input_dim, d_model, forecast_horizon, output_dim=2, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.blocks = nn.ModuleList([
            FBlock(d_model) for _ in range(num_layers)
        ])
        
        # Flatten and project to output
        # Alternative: Global Average Pooling or just taking the last step
        # Here we'll just project the last time step for simplicity, or use a decoder.
        # Let's use a simple MLP decoder on the flattened representation for now.
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim) # Predicts Price and ROI for the target date
        )
        
        self.forecast_horizon = forecast_horizon

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
            
        # Take the last time step representation
        last_step = x[:, -1, :]
        return self.output_head(last_step)
