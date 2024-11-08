import numpy as np
import math 

import torch
import torch.nn as nn
from typing import List

class SREM(nn.Module):
    def __init__(
            self,
            num_features: int,
            d_model:int,
            nhead:int,
            num_trans_layers:int,
            dim_feedforward:int,
            dropout:float
            ):
        
        super(SREM, self).__init__()

        self.compressor = nn.Linear(num_features, d_model)

        transofrmer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout)
        
        self.TE = nn.TransformerEncoder(
            encoder_layer = transofrmer_encoder_layer,
            num_layers = num_trans_layers
        )

    def forward(self, X):
        X = self.compressor(X)
        W = self.TE(X)
        r = W.mean(dim = 0)
        return r
    

class CAAN(nn.Module):
    def __init__(
            self,
            input_dim,
            q_k_v_dim,
            ):
        
        super(CAAN, self).__init__()

        # self.d_model = d_model
        self.q_k_v_dim = q_k_v_dim
        # self.d_v = d_v

        self.W_Q = nn.Linear(input_dim, q_k_v_dim)
        self.W_K = nn.Linear(input_dim, q_k_v_dim)

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
            num_features :int,
            d_model:int,
            nheads : int,
            num_transformer_layers : int,
            long_only : bool = True,
            portfolio_size :int = 10,
            q_k_v_dim :int = 15):
        
        super(PortfolioConstructor, self).__init__()
        
        self.device = device

        self.symbol_universe = symbol_universe
        self.number_assets = len(self.symbol_universe)
        self.long_only = long_only

        self.num_features = num_features
        self.d_model = d_model
        self.portfolio_size = portfolio_size
        self.seq_length = num_features
        self.nheads = nheads
        self.num_transformer_layers = num_transformer_layers

        self.q_k_v_dim = q_k_v_dim

        self.SREM = SREM(
            num_features = self.num_features,
            d_model = self.d_model,
            nhead = self.nheads,
            num_trans_layers = self.num_transformer_layers,
            dim_feedforward= 128,
            dropout = 0.1
        )

        self.CAAN = CAAN(
            input_dim = self.d_model,
            q_k_v_dim = self.q_k_v_dim
        )

        self.layer_norm = nn.LayerNorm(self.num_features)

    def portfolio_creator(self, scores):
        
        if self.long_only:
            num_winners = self.portfolio_size
            rank = torch.argsort(scores)

            long_sqs = set(rank.detach().cpu().numpy()[-num_winners:])

            long_mask = torch.Tensor([0 if i in long_sqs else 1 for i in range(rank.shape[0])]).to(self.device)
            long_scores = scores - (1e9 * long_mask)
            long_portfolio = long_scores.softmax(dim = 0)

            allocations = long_portfolio
            port_symbols_idx = list(long_sqs)
            
        else:   
            
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
            port_symbols_idx = list(long_sqs) + list(short_sqs)
        

        portfolio_allocations = [allocation.item() for allocation in allocations if allocation != 0]
        portfolio_symbols = [self.symbol_universe[i] for i in port_symbols_idx]

        # return portfolio_symbols, portfolio_allocations
        return portfolio_symbols, port_symbols_idx, allocations


    def forward(self, x):

        x = self.layer_norm(x)

        latent_rep = self.SREM(x)
        # print(f"latent rep : {latent_rep}")

        scores = self.CAAN(latent_rep)
        # print(f"scores : {scores}")
        
        port_symbols, port_symbols_idx, allocations = self.portfolio_creator(scores)

        # return  port_symbols, port_symbols_idx , allocations
        return torch.tensor(port_symbols_idx) , allocations

        # return scores.softmax(dim = 0)