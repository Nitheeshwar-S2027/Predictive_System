import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerEncoderBlock(nn.Module):
    """
    A minimal one-layer Transformer Encoder block using nn.TransformerEncoderLayer.
    Used as the final layer before the dense output network in hybrid models.
    """

    def __init__(self, d_model, nhead, dim_feedforward):
        super(TransformerEncoderBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        return self.transformer_encoder(x)


class HybridDNN(nn.Module):
    """
    Hybrid Deep Neural Network: Dense Embedding -> Transformer -> Dense Output.
    """

    def __init__(self, input_size, output_size=1):
        super(HybridDNN, self).__init__()
        # Embed the input features into a latent space (32 dimensions)
        self.embedding = nn.Linear(input_size, 32)
        # Apply the single-layer Transformer encoder
        self.transformer = TransformerEncoderBlock(
            d_model=32, nhead=1, dim_feedforward=64
        )
        # Final dense layers for prediction
        self.net = nn.Sequential(
            nn.Linear(32 * 1, 64), nn.ReLU(), nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.embedding(x)
        # Unsqueeze to add sequence dimension (Batch, 1, Feature_dim) for the transformer
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        return self.net(x)


class HybridCNN(nn.Module):
    """
    Hybrid Convolutional Neural Network: 1D CNN -> Transformer -> Dense Output.
    """

    def __init__(self, input_size, output_size=1):
        super(HybridCNN, self).__init__()
        # 1D Convolutional layers for local feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.transformer = TransformerEncoderBlock(
            d_model=32, nhead=1, dim_feedforward=64
        )
        self.net = nn.Sequential(
            nn.Linear(32 * 1, 64), nn.ReLU(), nn.Linear(64, output_size)
        )

    def forward(self, x):
        # Reshape for Conv1D: (Batch, 1, input_size)
        x = x.unsqueeze(1)
        x = self.conv(x)
        # Reshape for Transformer: (Batch, 1, 32)
        x = x.squeeze(-1)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        return self.net(x)


class HybridGRU(nn.Module):
    """
    Hybrid Gated Recurrent Unit Network: GRU -> Transformer -> Dense Output.
    """

    def __init__(self, input_size, output_size=1):
        super(HybridGRU, self).__init__()
        # GRU layer, processes feature vector as a sequence of single-dimension inputs
        self.gru = nn.GRU(input_size=1, hidden_size=32, batch_first=True)
        self.transformer = TransformerEncoderBlock(
            d_model=32, nhead=1, dim_feedforward=64
        )
        self.net = nn.Sequential(
            nn.Linear(32 * 1, 64), nn.ReLU(), nn.Linear(64, output_size)
        )

    def forward(self, x):
        # Reshape for GRU: (Batch, Seq_len=input_size, Feature_dim=1)
        x = x.unsqueeze(2)
        # Get the final hidden state (h_n)
        _, h_n = self.gru(x)
        # Reshape final hidden state for transformer
        x = h_n[-1].unsqueeze(1)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        return self.net(x)
