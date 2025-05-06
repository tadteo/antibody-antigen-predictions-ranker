#!/usr/bin/env python3
import torch
import torch.nn as nn
import math
class DeepSet(nn.Module):
    def __init__(self, input_dim, phi_hidden_dims, rho_hidden_dims, output_dim=1, aggregator='sum'):
        super(DeepSet, self).__init__()
        
        self.aggregator = aggregator
        
        # Phi network applied to each element in the set
        phi_layers = []
        prev_dim = input_dim
        for h in phi_hidden_dims:
            phi_layers.append(nn.Linear(prev_dim, h))
            phi_layers.append(nn.LeakyReLU(0.1))
            phi_layers.append(nn.LayerNorm(h))
            prev_dim = h
        self.phi = nn.Sequential(*phi_layers)

        if self.aggregator.startswith('attn_pool'):
            hidden = phi_hidden_dims[-1]
            self.pool_q = nn.Parameter(torch.randn(hidden))
            self.attn_w = nn.Linear(hidden, hidden, bias=False)
        
        if self.aggregator == 'concat_stats':
            # sum + mean + max + size => 3H + 1
            rho_input_dim = phi_hidden_dims[-1] * 3 + 1
        elif self.aggregator == 'concat_stats_by_set_size':
            # mean + max + size => 2H + 1
            rho_input_dim = phi_hidden_dims[-1] * 2 + 1
        else:
            rho_input_dim = phi_hidden_dims[-1]

        # Rho network applied after aggregation
        rho_layers = []
        prev_dim = rho_input_dim
        for h in rho_hidden_dims:
            rho_layers.append(nn.Linear(prev_dim, h))
            rho_layers.append(nn.ReLU())
            prev_dim = h
        rho_layers.append(nn.Linear(prev_dim, output_dim))
        #No sigmoid here because we want to use Huber loss and it will choke the gradient 
        self.rho = nn.Sequential(*rho_layers)

    def forward(self, x, lengths):
        # x: [batch_size, set_size, input_dim]
        B, N, D = x.shape
        # 1) flat→phi→reshape
        h = self.phi(x.reshape(B*N, D)).view(B, N, -1)  # [B, N, H]

        # build a mask: True where j < lengths[i]
        device = x.device
        arange = torch.arange(N, device=device)[None, :]     # [1, N]
        mask   = (arange < lengths[:, None]).float()         # [B, N]
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1]
        valid = mask.unsqueeze(-1)                    # [B,N,1]

        # sum over padded slots will be zero; 
        if self.aggregator == 'sum':
            agg = (h * valid).sum(dim=1)
        elif self.aggregator == 'sum_by_set_size':
            # divide by the true length
            weighted = h * valid                    # [B, N, H]
            agg     = weighted.sum(dim=1) / counts  # [B, H]
        elif self.aggregator == 'attn_pool':
            # attention pooling without size normalization, with stable softmax
            Q = self.pool_q.unsqueeze(0).expand(B, -1).unsqueeze(1)  # [B,1,H]
            K = self.attn_w(h)                                      # [B,N,H]
            scores = (Q * K).sum(-1) / math.sqrt(K.size(-1))        # [B,N]
            # stability: subtract max per row
            max_scores = scores.max(dim=1, keepdim=True)[0]
            scores = scores - max_scores
            scores = scores.masked_fill(mask == 0, float('-inf'))
            alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
            agg = (alpha * h).sum(dim=1)       

        elif self.aggregator == 'attn_pool_by_set_size':
            # attention pooling with size normalization and stable softmax
            Q = self.pool_q.unsqueeze(0).expand(B, -1).unsqueeze(1)
            K = self.attn_w(h)
            scores = (Q * K).sum(-1) / math.sqrt(K.size(-1))
            max_scores = scores.max(dim=1, keepdim=True)[0]
            scores = scores - max_scores
            scores = scores.masked_fill(mask == 0, float('-inf'))
            alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (alpha * h).sum(dim=1)
            agg = pooled / counts                                   # [B,H]  
        elif self.aggregator == 'concat_stats':
            sum_pool = (h * valid).sum(dim=1)              # [B, H]
            mean_pool = sum_pool / counts                   # [B, H]
            # max pooling (with masking)
            neg_inf = -1e9
            h_masked = h + (1.0 - valid) * neg_inf
            max_pool, _ = h_masked.max(dim=1)               # [B, H]
            size_feat = counts.sqrt()                       # [B, 1]
            # concatenate: [sum, mean, max, sqrt(n)]
            agg = torch.cat([sum_pool, mean_pool, max_pool, size_feat], dim=1)
        elif self.aggregator == 'concat_stats_by_set_size':
            # three statistics, each normalized by set size
            mean1 = (h * valid).sum(dim=1) / counts       # [B,H]
            # max over valid entries
            neg_inf = -1e9
            h_masked = h + (1.0 - valid) * neg_inf
            max_pool, _ = h_masked.max(dim=1)                # [B,H]
            # normalize max by set size
            max_pool = max_pool / counts                    # [B,H]
            size_feat = counts.sqrt()                     # [B,1]
            agg = torch.cat([mean1, max_pool, size_feat], dim=1)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        
        # 3) rho→out
        out = self.rho(agg)   # [B, output_dim]
        return out.squeeze(-1)

def init_weights(module):
    """
    Kaiming-init all Linear layers:
    - LeakyReLU (a=0.1) for phi network
    - ReLU for rho network
    All biases are initialized to zero.
    """
    if isinstance(module, nn.Linear):
        # Check if this layer belongs to phi network (has LayerNorm after it)
        is_phi_layer = any(isinstance(m, nn.LayerNorm) for m in module._forward_hooks.values())
        
        if is_phi_layer:
            # Phi network uses LeakyReLU
            nn.init.kaiming_uniform_(
                module.weight,
                a=0.1,
                nonlinearity='leaky_relu'
            )
        else:
            # Rho network uses ReLU
            nn.init.kaiming_uniform_(
                module.weight,
                nonlinearity='relu'
            )
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)
