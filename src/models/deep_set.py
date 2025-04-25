#!/usr/bin/env python3
import torch
import torch.nn as nn

class DeepSet(nn.Module):
    def __init__(self, input_dim, phi_hidden_dims, rho_hidden_dims, output_dim=1, aggregator='sum'):
        super(DeepSet, self).__init__()
        # Phi network applied to each element in the set
        phi_layers = []
        prev_dim = input_dim
        for h in phi_hidden_dims:
            phi_layers.append(nn.Linear(prev_dim, h))
            phi_layers.append(nn.ReLU())
            prev_dim = h
        self.phi = nn.Sequential(*phi_layers)
        # Rho network applied after aggregation
        rho_layers = []
        prev_dim = phi_hidden_dims[-1]
        for h in rho_hidden_dims:
            rho_layers.append(nn.Linear(prev_dim, h))
            rho_layers.append(nn.ReLU())
            prev_dim = h
        rho_layers.append(nn.Linear(prev_dim, output_dim))
        self.rho = nn.Sequential(*rho_layers)
        self.aggregator = aggregator

    def forward(self, x):
        # x: (batch_size, set_size, input_dim)
        batch_size, set_size, input_dim = x.size()
        x_flat = x.reshape(batch_size * set_size, input_dim)
        phi_out = self.phi(x_flat)
        phi_out = phi_out.view(batch_size, set_size, -1)
        if self.aggregator == 'sum':
            agg = phi_out.sum(dim=1)
        elif self.aggregator == 'mean':
            agg = phi_out.mean(dim=1)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        out = self.rho(agg)
        return out.squeeze(-1) 
