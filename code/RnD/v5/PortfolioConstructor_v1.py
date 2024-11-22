import torch
import torch.nn as nn
import math
from typing import List

# SREM
class SREM(nn.Module):
    def __init__(self, num_features: int, d_model: int, nhead: int, num_trans_layers: int, dim_feedforward: int, dropout: float):
        super(SREM, self).__init__()
        self.compressor = nn.Linear(num_features, d_model)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.TE = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=num_trans_layers
        )

    def forward(self, X):
        # Handle input shape: (batch_size, seq_length, num_symbols, num_features)
        batch_size, seq_length, num_symbols, num_features = X.shape

        # Reshape input for linear layer: (batch_size, seq_length, num_symbols * num_features)
        X = X.reshape(batch_size, seq_length, num_symbols * num_features)

        # Pass through compressor: (batch_size, seq_length, d_model)
        X = self.compressor(X)

        # Permute for Transformer: (seq_length, batch_size, d_model)
        X = X.permute(1, 0, 2)

        # Pass through Transformer Encoder
        W = self.TE(X)

        # Average across time dimension: (batch_size, d_model)
        r = W.mean(dim=0)
        return r

# CAAN
class CAAN(nn.Module):
    def __init__(self, input_dim, q_k_v_dim):
        super(CAAN, self).__init__()
        self.q_k_v_dim = q_k_v_dim

        self.W_Q = nn.Linear(input_dim, q_k_v_dim)
        self.W_K = nn.Linear(input_dim, q_k_v_dim)
        self.W_V = nn.Linear(input_dim, q_k_v_dim)
        self.W_O = nn.Linear(q_k_v_dim, input_dim)

        self.scorer = nn.Sequential(
            nn.Linear(q_k_v_dim, int(q_k_v_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(q_k_v_dim / 2), 1),
        )

    def forward(self, x):
        query = self.W_Q(x)
        key = self.W_K(x)
        value = self.W_V(x)

        # Attention mechanism
        attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.q_k_v_dim)
        attn = attn.softmax(dim=-1).unsqueeze(-1)

        # Weighted sum
        x = torch.sum(value.unsqueeze(0) * attn, dim=1)

        # Score computation
        scores = self.scorer(x).squeeze()
        return scores

# PortfolioConstructor
class PortfolioConstructor(nn.Module):
    def __init__(self, device, symbol_universe: List[str], num_features: int, d_model: int, nheads: int, num_transformer_layers: int, long_only: bool = True, portfolio_size: int = 10, q_k_v_dim: int = 30):
        super(PortfolioConstructor, self).__init__()
        self.device = device
        self.symbol_universe = symbol_universe
        self.number_assets = len(self.symbol_universe)
        self.long_only = long_only
        self.num_features = num_features
        self.d_model = d_model
        self.portfolio_size = portfolio_size
        self.nheads = nheads
        self.num_transformer_layers = num_transformer_layers
        self.q_k_v_dim = q_k_v_dim

        self.SREM = SREM(
            num_features=self.num_features,
            d_model=self.d_model,
            nhead=self.nheads,
            num_trans_layers=self.num_transformer_layers,
            dim_feedforward=128,
            dropout=0.1
        )
        self.CAAN = CAAN(
            input_dim=self.d_model,
            q_k_v_dim=self.q_k_v_dim
        )

    def portfolio_creator(self, scores):
        if self.long_only:
            rank = torch.argsort(scores, descending=True)
            long_symbols = rank[:self.portfolio_size]

            long_scores = torch.full_like(scores, -float('inf'))
            long_scores[long_symbols] = scores[long_symbols]

            allocations = long_scores.softmax(dim=0)
            port_symbols_idx = long_symbols.tolist()

        else:
            raise NotImplementedError("Shorting is not yet implemented.")

        return port_symbols_idx, allocations

    def forward(self, x):
        # Ensure input has batch size
        if len(x.shape) == 3:  # (seq_length, num_symbols, num_features)
            x = x.unsqueeze(0)  # Add batch dimension

        latent_rep = self.SREM(x)
        scores = self.CAAN(latent_rep)
        port_symbols_idx, allocations = self.portfolio_creator(scores)
        return torch.tensor(port_symbols_idx), allocations
