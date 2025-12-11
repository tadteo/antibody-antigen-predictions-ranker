#!/usr/bin/env python3
import torch
import torch.nn as nn
import math
class DeepSet(nn.Module):
    def __init__(self, input_dim, phi_hidden_dims, rho_hidden_dims, output_dim=1, aggregator='sum', sample_level_dim=0):
        super(DeepSet, self).__init__()
        
        self.aggregator = aggregator
        self.sample_level_dim = sample_level_dim
        
        # Phi network applied to each element in the set
        phi_layers = []
        prev_dim = input_dim
        for h in phi_hidden_dims:
            phi_layers.append(nn.Linear(prev_dim, h))
            phi_layers.append(nn.LeakyReLU(0.1))
            phi_layers.append(nn.LayerNorm(h))
            phi_layers.append(nn.Dropout(0.1))
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
        elif self.aggregator == 'attn_pool_concat_stats':
            # attn + sum + mean + max + size => 4H + 1
            rho_input_dim = phi_hidden_dims[-1] * 4 + 1
        else:
            rho_input_dim = phi_hidden_dims[-1]

        # Add sample-level features dimension (e.g., iptm and ptm)
        rho_input_dim = rho_input_dim + sample_level_dim

        # Rho network applied after aggregation
        rho_layers = []
        prev_dim = rho_input_dim
        for h in rho_hidden_dims:
            rho_layers.append(nn.Linear(prev_dim, h))
            rho_layers.append(nn.ReLU())
            rho_layers.append(nn.Dropout(0.1))
            prev_dim = h
        rho_layers.append(nn.Linear(prev_dim, output_dim))
        #No sigmoid here because we want to use Huber loss and it will choke the gradient 
        self.rho = nn.Sequential(*rho_layers)

    def forward(self, x, lengths, sample_features=None):
        # x: [batch_size, complex_size, set_size, input_dim]
        # sample_features: [batch_size, complex_size, sample_level_dim] (optional, e.g., iptm and ptm)
        if not torch.isfinite(x).all():
            bad = ~torch.isfinite(x)
            # first bad index in (B,K,N,F)
            idx = torch.nonzero(bad)[0].tolist()
            b, k, n, f = idx
            val = x[b, k, n, f]
            raise RuntimeError(
                f"Non-finite feature BEFORE phi: x[{b},{k},{n},{f}]={val}"
            )
        
        
        B, K, N, F = x.shape
        # print(f"x shape: {x.shape}")
        # print(f"lengths shape: {lengths.shape}")
        # 1) flat→phi→reshape
        h = self.phi(x.reshape(B*K*N, F)).view(B, K, N, -1)  # [B, K, N, F]
        # print(f"h shape: {h.shape}")
        # build a mask: True where j < lengths[i]
        device = x.device
        arange = torch.arange(N, device=device)[None,None, :]     # [1, 1, N]
        # print(f"arange shape: {arange.shape}")
        mask   = (arange < lengths[:, :,None]).float()         # [B, K, N]
        # print(f"mask shape: {mask.shape}")
        counts = mask.sum(dim=2, keepdim=True).clamp(min=1.0)  # [B, K, 1]
        # print(f"counts shape: {counts.shape}")
        valid = mask.unsqueeze(-1)                    # [B,K,N,1]
        
        epsilon = 1e-8 # Epsilon for numerical stability

        # sum over padded slots will be zero; 
        if self.aggregator == 'sum':
            agg = (h * valid).sum(dim=2) # Shape: [B, K, H]
            # Normalize across K samples for each complex
            mu = agg.mean(dim=1, keepdim=True)
            std = agg.std(dim=1, keepdim=True)
            agg = (agg - mu) / (std + epsilon)
        elif self.aggregator == 'sum_by_set_size':
            # divide by the true length
            weighted = h * valid                    # [B, K, N, H]
            agg     = weighted.sum(dim=2) / counts  # [B, K, H]
            # Normalize across K samples for each complex
            mu = agg.mean(dim=1, keepdim=True)
            std = agg.std(dim=1, keepdim=True)
            agg = (agg - mu) / (std + epsilon)
        elif self.aggregator == 'attn_pool':
            # attention pooling without size normalization, with stable softmax
            Q = self.pool_q.unsqueeze(0).expand(B*K, -1).unsqueeze(1)  # [B*K,1,H]
            K_val = self.attn_w(h.view(B*K, N, -1))                     # [B*K,N,H]
            scores = (Q * K_val).sum(-1) / math.sqrt(K_val.size(-1))    # [B*K,N]
            
            current_mask = mask.view(B*K, N) # Mask for B*K samples
            max_scores = scores.max(dim=1, keepdim=True)[0]
            scores = scores - max_scores
            scores = scores.masked_fill(current_mask == 0, float('-inf'))
            alpha = torch.softmax(scores, dim=1).unsqueeze(-1)      # [B*K,N,1]
            agg_flat = (alpha * h.view(B*K,N,-1)).sum(dim=1)         # [B*K,H]
            agg = agg_flat.view(B, K, -1)                           # [B,K,H]
            # Normalize across K samples for each complex
            mu = agg.mean(dim=1, keepdim=True)
            std = agg.std(dim=1, keepdim=True)
            agg = (agg - mu) / (std + epsilon)
        elif self.aggregator == 'attn_pool_by_set_size':
            # attention pooling with size normalization and stable softmax
            Q = self.pool_q.unsqueeze(0).expand(B*K, -1).unsqueeze(1) # [B*K,1,H]
            K_val = self.attn_w(h.view(B*K, N, -1))                    # [B*K,N,H]
            scores = (Q * K_val).sum(-1) / math.sqrt(K_val.size(-1))   # [B*K,N]

            current_mask = mask.view(B*K, N) # Mask for B*K samples
            max_scores = scores.max(dim=1, keepdim=True)[0]
            scores = scores - max_scores
            scores = scores.masked_fill(current_mask == 0, float('-inf'))
            alpha = torch.softmax(scores, dim=1).unsqueeze(-1)   # [B*K,N,1]
            pooled_flat = (alpha * h.view(B*K,N,-1)).sum(dim=1)   # [B*K,H]
            # Reshape counts to [B*K, 1] for division
            current_counts = counts.view(B*K, 1)
            agg_flat = pooled_flat / current_counts              # [B*K,H]  
            agg = agg_flat.view(B, K, -1)                        # [B,K,H]
            # Normalize across K samples for each complex
            mu = agg.mean(dim=1, keepdim=True)
            std = agg.std(dim=1, keepdim=True)
            agg = (agg - mu) / (std + epsilon)
        elif self.aggregator == 'concat_stats':
            sum_pool = (h * valid).sum(dim=2)              # [B, K, H]
            mean_pool = sum_pool / counts                   # [B, K, H]
            neg_inf = -1e9
            h_masked = h + (1.0 - valid) * neg_inf
            max_pool, _ = h_masked.max(dim=2)               # [B, K, H]
            size_feat = counts.sqrt()                       # [B, K, 1]
            
            phi_derived_features = torch.cat([sum_pool, mean_pool, max_pool], dim=2) # [B, K, 3H]
            
            mu = phi_derived_features.mean(dim=1, keepdim=True)
            std = phi_derived_features.std(dim=1, keepdim=True)
            normalized_phi_features = (phi_derived_features - mu) / (std + epsilon)
            
            agg = torch.cat([normalized_phi_features, size_feat], dim=2) # [B, K, 3H+1]
        elif self.aggregator == 'concat_stats_by_set_size':
            mean1 = (h * valid).sum(dim=2) / counts       # [B,K,H]
            neg_inf = -1e9
            h_masked = h + (1.0 - valid) * neg_inf
            max_pool, _ = h_masked.max(dim=2)                # [B,K,H]
            max_pool_norm_by_size = max_pool / counts        # [B,K,H] This was an error in reasoning before, max_pool should not be normalized by counts again if mean1 is already mean. Let's use raw max_pool.
            
            # Let's use mean1 and the original max_pool (before any division by counts)
            phi_derived_features = torch.cat([mean1, max_pool], dim=2) # [B, K, 2H]
            size_feat = counts.sqrt()                                  # [B, K, 1]

            mu = phi_derived_features.mean(dim=1, keepdim=True)
            std = phi_derived_features.std(dim=1, keepdim=True)
            # normalized_phi_features = (phi_derived_features - mu) / (std + epsilon)
            # normalized_phi_features = (phi_derived_features - mu)
            normalized_phi_features = phi_derived_features
            
            agg = torch.cat([normalized_phi_features, size_feat], dim=2) # [B, K, 2H+1]
        elif self.aggregator == 'attn_pool_concat_stats':
            # 1. Calculate Standard Stats
            sum_pool = (h * valid).sum(dim=2)              # [B, K, H]
            mean_pool = sum_pool / counts                  # [B, K, H]
            
            neg_inf = -1e9
            h_masked = h + (1.0 - valid) * neg_inf
            max_pool, _ = h_masked.max(dim=2)              # [B, K, H]
            
            size_feat = counts.sqrt()                      # [B, K, 1]

            # 2. Calculate Attention Pooling
            # Q is broadcasted, K is projected hidden states
            Q = self.pool_q.unsqueeze(0).expand(B*K, -1).unsqueeze(1)   # [B*K,1,H]
            K_val = self.attn_w(h.view(B*K, N, -1))                     # [B*K,N,H]
            scores = (Q * K_val).sum(-1) / math.sqrt(K_val.size(-1))    # [B*K,N]

            current_mask = mask.view(B*K, N)
            max_scores = scores.max(dim=1, keepdim=True)[0] # Stability trick
            scores = scores - max_scores
            scores = scores.masked_fill(current_mask == 0, float('-inf'))
            
            alpha = torch.softmax(scores, dim=1).unsqueeze(-1)          # [B*K,N,1]
            attn_flat = (alpha * h.view(B*K,N,-1)).sum(dim=1)           # [B*K,H]
            attn_pool = attn_flat.view(B, K, -1)                        # [B,K,H]

            # 3. Concatenate (Attn, Sum, Mean, Max) -> 4H
            phi_derived_features = torch.cat([attn_pool, sum_pool, mean_pool, max_pool], dim=2) 

            # 4. Normalize the 4H features
            mu = phi_derived_features.mean(dim=1, keepdim=True)
            std = phi_derived_features.std(dim=1, keepdim=True)
            normalized_features = (phi_derived_features - mu) / (std + epsilon)
            
            # 5. Append Size
            agg = torch.cat([normalized_features, size_feat], dim=2) # [B, K, 4H + 1]
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        
        # print(f"agg shape after aggregation and normalization: {agg.shape}")

        # Concatenate sample-level features (e.g., iptm, ptm) if provided
        if sample_features is not None:
            agg = torch.cat([agg, sample_features], dim=-1)  # [B, K, agg_dim + sample_level_dim]

        # Rho network
        out = self.rho(agg)   # [B, K, output_dim]
        # print(f"out shape: {out.shape}")
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
