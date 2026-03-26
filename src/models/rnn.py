from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class _ChordRecurrentBase(nn.Module):
    """Shared recurrent frame classifier with one logit vector per timestep."""

    recurrent_cls: type[nn.RNNBase]

    def __init__(
        self,
        num_classes: int = 25,
        input_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = self.recurrent_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, _ = self.recurrent(packed)
            output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.size(1))
        else:
            output, _ = self.recurrent(x)
        return self.classifier(self.dropout(output))

    def predict(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x, lengths=lengths).argmax(dim=-1)


class ChordRNN(_ChordRecurrentBase):
    """Vanilla RNN over frame sequences."""

    recurrent_cls = nn.RNN


class ChordLSTM(_ChordRecurrentBase):
    """LSTM over frame sequences."""

    recurrent_cls = nn.LSTM
