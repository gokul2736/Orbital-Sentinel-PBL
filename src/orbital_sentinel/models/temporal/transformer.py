"""Transformer-based temporal model for CDM sequence risk prediction."""

import math

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for TransformerRiskModel. "
            "Install it with: pip install torch"
        )


if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for transformer input."""

        def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class TransformerRiskModel(nn.Module):
        """Transformer model for predicting collision risk from CDM sequences.

        Input: (batch, seq_len, input_size) tensor of CDM features
        Output: (batch, 1) risk prediction (log10 collision probability)
        """

        def __init__(
            self,
            input_size: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.input_size = input_size
            self.d_model = d_model

            self.input_projection = nn.Linear(input_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x):
            """x: (batch, seq_len, features) -> (batch, 1)"""
            x = self.input_projection(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x[:, -1, :]
            return self.fc(x)

    def train_transformer(
        model, train_loader, val_loader, epochs=50, lr=1e-3, device=None
    ) -> dict:
        """Train the Transformer model and return training history."""
        _check_torch()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()
                if batch_y.ndim == 1:
                    batch_y = batch_y.unsqueeze(1)

                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train = float(np.mean(train_losses))
            history["train_loss"].append(avg_train)

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device).float()
                    batch_y = batch_y.to(device).float()
                    if batch_y.ndim == 1:
                        batch_y = batch_y.unsqueeze(1)
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                    val_losses.append(loss.item())

            avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
            history["val_loss"].append(avg_val)
            scheduler.step(avg_val)

        return history

else:

    class TransformerRiskModel:
        def __init__(self, *args, **kwargs):
            _check_torch()

    def train_transformer(*args, **kwargs):
        _check_torch()
