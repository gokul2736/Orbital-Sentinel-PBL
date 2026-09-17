"""PyTorch GRU model for temporal CDM sequence risk prediction."""

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
            "PyTorch is required for GRURiskModel. "
            "Install it with: pip install torch"
        )


if TORCH_AVAILABLE:

    class GRURiskModel(nn.Module):
        """GRU model for predicting collision risk from CDM sequences.

        Input: (batch, seq_len, input_size) tensor of CDM features
        Output: (batch, 1) risk prediction (log10 collision probability)
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            """x: (batch, seq_len, features) -> (batch, 1)"""
            gru_out, h_n = self.gru(x)
            last_hidden = h_n[-1]
            out = self.dropout(last_hidden)
            return self.fc(out)

    def train_gru(
        model, train_loader, val_loader, epochs=50, lr=1e-3, device=None
    ) -> dict:
        """Train the GRU model and return training history."""
        _check_torch()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

        return history

else:

    class GRURiskModel:
        def __init__(self, *args, **kwargs):
            _check_torch()

    def train_gru(*args, **kwargs):
        _check_torch()
