import numpy as np
import math 

import torch
import torch.nn as nn
from typing import List

class SREM(nn.Module):
    def __init__(
            self,
            seq_len,
            multihead_dim,
            num_trans_layers,
            latent_dim):
        super(SREM, self).__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.multihead_dim = multihead_dim
        self.num_trans_layers = num_trans_layers

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.seq_len,
            nhead = self.multihead_dim,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer = self.transformer_encoder_layer,
            num_layers = self.num_trans_layers,
        )

        self.ff = nn.Linear(
            in_features = self.seq_len,
            out_features = self.latent_dim
        )

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.ff(x)
        x = self.relu(x)

        return x
    

class CAAN(nn.Module):
    def __init__(
            self,
            input_dim,
            q_k_v_dim,
            portfolio_size : int = 10
            ):
        
        super(CAAN, self).__init__()

        # self.d_model = d_model
        self.q_k_v_dim = q_k_v_dim
        # self.d_v = d_v

        self.W_Q = nn.Linear(input_dim, q_k_v_dim)
        self.W_K = nn.Linear(input_dim, q_k_v_dim)

        self.portfolio_size = portfolio_size

        # standard practice to use two smaller weight matrices for Values weights
        self.W_V = nn.Linear(input_dim, q_k_v_dim)
        self.W_O = nn.Linear(q_k_v_dim, input_dim)

        self.normalizer = nn.LayerNorm(self.q_k_v_dim)

        self.scorer = nn.Linear(q_k_v_dim, 1)

    def forward(self, x):

        query = self.W_Q(x)
        key = self.W_K(x)
        value = self.W_V(x)

        attn = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(self.q_k_v_dim)
        attn = attn.softmax(dim = -1).unsqueeze(-1)

        x = torch.sum(value.unsqueeze(0) * attn, dim = 1)
        scores = self.scorer(x).squeeze() 
        
        return scores
    


class PortfolioConstructor(nn.Module):
    def __init__(
            self,
            device,
            symbol_universe : List[str],
            seq_length :int,
            multihead_dim : int,
            num_transformer_layers : int,
            long_only : bool = True,
            latent_dim : int = 30,
            portfolio_size :int = 10,
            q_k_v_dim :int = 15):
        
        super(PortfolioConstructor, self).__init__()
        
        self.device = device

        self.symbol_universe = symbol_universe
        self.number_assets = len(self.symbol_universe)
        self.long_only = long_only

        self.portfolio_size = portfolio_size
        self.seq_length = seq_length
        self.multihead_dim = multihead_dim
        self.num_transformer_layers = num_transformer_layers
        self.latent_dim = latent_dim

        self.q_k_v_dim = q_k_v_dim

        self.SREM = SREM(
            seq_len = self.seq_length,
            multihead_dim = self.multihead_dim,
            num_trans_layers = self.num_transformer_layers,
            latent_dim = self.latent_dim
        )

        self.CAAN = CAAN(
            input_dim = self.latent_dim,
            q_k_v_dim = self.q_k_v_dim
        )

    def portfolio_creator(self, scores):
        
        if self.long_only:
            if self.portfolio_size != 0:
                num_winners = self.portfolio_size
                rank = torch.argsort(scores)

                long_sqs = set(rank.detach().cpu().numpy()[-num_winners:])

                long_mask = torch.Tensor([0 if i in long_sqs else 1 for i in range(rank.shape[0])]).to(self.device)
                long_scores = scores - (1e9 * long_mask)
                long_portfolio = long_scores.softmax(dim = 0)

                allocations = long_portfolio
                port_symbols = list(long_sqs)
            
            return port_symbols, allocations
        else:   
            if self.portfolio_size != 0:
                num_winners = int(self.portfolio_size * 0.5)
                rank = torch.argsort(scores)

                long_sqs = set(rank.detach().cpu().numpy()[-num_winners:])
                short_sqs = set(rank.detach().cpu().numpy()[:num_winners])

                long_mask = torch.Tensor([0 if i in long_sqs else 1 for i in range(rank.shape[0])]).to(self.device)
                long_scores = scores - (1e9 * long_mask)
                long_portfolio = long_scores.softmax(dim = 0)
    
                short_mask = torch.Tensor([0 if i in short_sqs else 1 for i in range(rank.shape[0])]).to(self.device)
                short_scores = 1-scores - (1e9 * short_mask)
                short_portfolio = short_scores.softmax(dim = 0)

                allocations = long_portfolio - short_portfolio
                port_symbols = list(long_sqs) + list(short_sqs)
            
            return port_symbols, allocations

    def forward(self, x):

        latent_rep = self.SREM(x)
        print(f"latent rep : {latent_rep}")
        scores = self.CAAN(latent_rep)
        print(f"scores : {scores}")
        
        port_symbols, allocations = self.portfolio_creator(scores)

        return  port_symbols, allocations