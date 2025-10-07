#!/usr/bin/env python3
"""
train_deepset.py

Train a DeepSet on antibody–antigen interchain PAE features.
Usage:
    fabric run --accelerator=gpu --devices=2 train.py --config configs/config.yaml
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dotenv import load_dotenv
import time
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm
from collections import defaultdict
import random

# Lightning Fabric for distributed training

from lightning.fabric import Fabric

from src.data.dataloader import (
    get_dataloader,
    get_eval_dataloader
)
from src.models.deep_set import DeepSet, init_weights

torch.set_float32_matmul_precision("high")

const_eps = 1e-6

def calculate_soft_rho(pred_sig: np.ndarray, true_sig: np.ndarray, tau=0.1):
    """
    pred_sig, true_sig: [K] in [0,1]
    Returns: soft Spearman rho
    """

    K = len(pred_sig)
    # --- soft rank helper (inline to keep everything in this function) ---
    # rank_i ≈ 1 + sum_{j≠i} sigmoid((s_i - s_j)/tau)
    # Since sigmoid(0)=0.5, we can do: 1 + (sum_j sigmoid(...) - 0.5)
    def _soft_rank(scores: np.ndarray, tau: float) -> np.ndarray:
        s = scores.reshape(-1, 1)                  # [K,1]
        D = s - s.T                                # [K,K]
        P = 1.0 / (1.0 + np.exp(-D / tau))         # [K,K]
        return 1.0 + (P.sum(axis=1) - 0.5)         # [K]

    r_pred = _soft_rank(pred_sig, tau)             # [K]
    r_true = _soft_rank(true_sig, tau)             # [K]

    # Safe Pearson correlation between rank vectors
    r_pred = r_pred - r_pred.mean()
    r_true = r_true - r_true.mean()
    denom = (np.sqrt((r_pred * r_pred).sum()) * np.sqrt((r_true * r_true).sum())) + const_eps
    if denom <= const_eps:
        return 1.0 # decided that if all elements are the same, then the correlation is 1.0
    return float((r_pred * r_true).sum() / denom)

def calculate_spearman(pred_sig: np.ndarray, true_sig: np.ndarray):
    """
    pred_sig, true_sig: [K] in [0,1]
    Returns: [K] tensor of true Spearman rho and soft Spearman rho
    """
    ps = pred_sig
    ts = true_sig
    
    constant_prediction = np.max(ps)-np.min(ps) < 1e-2
    constant_target = np.max(ts)-np.min(ts) < 1e-2

    if constant_prediction and constant_target:
        rho = 1.0
        soft_rho = 1.0
    elif constant_prediction and not constant_target:
        rho = 0.0
        soft_rho = 0.0
    elif constant_target and not constant_prediction:
        rho = 0.0
        soft_rho = 0.0
    else:
   
        rho, _ = spearmanr(ps, ts)
        if np.isnan(rho):
            rho = 1.0 # decided that if all elements are the same, then the correlation is 1.0
        soft_rho = calculate_soft_rho(ps, ts)

    return (rho, soft_rho)

 
def pairwise_soft_rank(scores, tau=0.1):
    """
    scores : Tensor [..., K]
    returns soft ranks of same shape
    """
    diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)        # [..., K, K]
    P    = torch.sigmoid(diff / tau)                          # pairwise probs
    # P_ii = sigmoid(0) = 0.5.
    # soft_rank_i = 1 + sum_{j!=i} sigmoid((s_i - s_j)/tau)
    # P.sum(dim=-1)_i = sum_j P_ij = sum_{j!=i} P_ij + P_ii
    # So, 1 + P.sum(dim=-1)_i - P_ii leads to the correct formula.
    soft_rank = 1 + P.sum(dim=-1) - torch.diagonal(P, dim1=-2, dim2=-1)
    return soft_rank

def soft_spearman_loss(pred, target, loss_type='one_minus_rho', tau=0.2):
    """
    pred, target : shape [B, K] in DockQ (0–1) space
    returns scalar 1 - ρ loss per complex [B]
    Assumes K >= 2, which is handled by the caller.
    
    It calculates the spearman soft loss and after it applies a Fisher transform (atanh) or a simple one_minus_rho 
    to the loss to make it in the same order of magnitude thatn the s1smoothloss on the regression loss.
    
    """
    K = pred.size(-1)
    if K < 2:
        # Return a tensor of zeros with the batch dimension if K < 2,
        # ensuring it's on the correct device and requires grad if inputs do.
        # This allows .mean() to work downstream without error.
        return torch.zeros(pred.size(0), device=pred.device, dtype=pred.dtype)

    sr_pred = pairwise_soft_rank(pred,   tau)   # [B,K]
    sr_true = pairwise_soft_rank(target, tau)   # [B,K]
    
    # mse = (sr_pred - sr_true).pow(2).sum(dim=-1) # per complex [B]
    mse = (sr_pred - sr_true).pow(2).mean(dim=-1) # per complex [B]

    denominator = K * (K**2 - 1)
    # If K < 2, denominator could be 0. Handled by K < 2 check above.
    # However, if K is exactly 0 or 1 due to an issue, ensure no division by zero.
    if denominator == 0: # Should not happen if K >= 2 check is effective
        return torch.zeros(pred.size(0), device=pred.device, dtype=pred.dtype)

    rho_soft = 1 - 6 * mse / denominator # per complex [B]
    rho_soft = torch.clamp(rho_soft, min=-1.0 + 1e-6, max=1.0 - 1e-6)

    if loss_type == 'fisher':
        atanh_rho_soft = torch.atanh(rho_soft)
        #softplus of -atanh to keep it bounded to 0 to infinity
        loss = torch.softplus(-atanh_rho_soft) # per complex [B], to be minimized (hence -atanh of rho) range plus minus infinity, minus infinity is best correlation
    elif loss_type == 'one_minus_rho':
        loss = 1.0 - rho_soft #per complex [B], to be minimized (hence 1 - rho) range 0 to 2, 0 is best correlation
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return loss.mean()

def distance_preservation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: [B, K]
    Returns: [B] loss per complex, averaged over batch
    Computes sum of squared errors between pairwise absolute distances of pred and target.
    """
    B, K = pred.shape
    if K < 2:
        return torch.zeros(B, device=pred.device, dtype=pred.dtype)

    # Pairwise absolute distances
    Dp = (pred.unsqueeze(2) - pred.unsqueeze(1)).abs()  # [B,K,K]
    Dt = (target.unsqueeze(2) - target.unsqueeze(1)).abs()  # [B,K,K]

    # Use only upper triangle (i<j)
    mask = torch.triu(torch.ones(K, K, device=pred.device, dtype=torch.bool), diagonal=1)

    # Difference for valid pairs
    diff = Dp[:, mask] - Dt[:, mask]  # [B, M], M = K(K-1)/2

    # Sum squared errors per complex (no mean across batch)
    loss_per_complex = (diff ** 2).sum(dim=1)  # [B]
    return loss_per_complex.mean()

def main():
    # 1) Setup argument parsing
    parser = argparse.ArgumentParser(
        description="Train DeepSet model with per-sample weighting & advanced sampling"
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save model checkpoints. Overrides config file if specified.'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint file to resume training from. Overrides config file if specified.'
    )
    parser.add_argument(
        '--precision', type=str, default="bf16-mixed",
        help='Training precision (32, 16-mixed, bf16-mixed)'
    )
    parser.add_argument(
        '--config_overrides', type=str, nargs='*', default=[],
        help='Paths to YAML files merged over the base config (e.g., configs/schedulers/onecycle.yaml)'
    )
    
    # Parse known arguments (config path, output_dir, resume)
    # and collect unknown arguments for OmegaConf to parse as overrides
    args, unknown_cli_args = parser.parse_known_args()  

    # Initialize Lightning Fabric for distributed training
    fabric = Fabric(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",  # Let Fabric detect from environment
        strategy="auto",  # Let Fabric choose strategy based on devices
        precision=args.precision
    )
    
    # Only launch if not already launched (e.g., when using torchrun directly)
    # fabric run CLI handles launching automatically
    if not fabric._launched:
        fabric.launch()
    
    ### DEBUG STEPS FOR MULTI NODES SYNCING ###
    if torch.cuda.is_available():
        local_dev = torch.cuda.current_device()
        dev_name  = torch.cuda.get_device_name(local_dev)
    else:
        local_dev, dev_name = -1, "CPU"

    fabric.print(
        f"[rank={fabric.global_rank} / world={fabric.world_size}] "
        f"local_dev={local_dev} ({dev_name})  "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    fabric.barrier()

    x = torch.ones((), device=fabric.device) * (fabric.global_rank + 1)
    x_sum = fabric.all_reduce(x.clone(), reduce_op="sum")
    fabric.print(f"[rank={fabric.global_rank}] all_reduce sum = {x_sum.item()}")
    fabric.barrier()

    # Only print on rank 0
    if fabric.global_rank == 0:
        print(f"Initialized Fabric with {fabric.world_size} devices")

    ### END DEBUG STEPS FOR MULTI NODES SYNCING ###

    # Load base config from YAML
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    cfg = OmegaConf.load(args.config)

    # Merge optional override YAMLs (later files win on conflicts)
    if getattr(args, 'config_overrides', None):
        override_confs = [OmegaConf.load(p) for p in args.config_overrides]
        cfg = OmegaConf.merge(cfg, *override_confs)

    # Auto-merge scheduler subconfiguration based on lr_scheduler_type
    try:
        sched_type = cfg.training.get('lr_scheduler_type', 'ReduceLROnPlateau')
        explicit_sched_path = cfg.training.get('scheduler_config_path', None)
        sched_filename_map = {
            'OneCycleLR': 'onecycle.yaml',
            'ReduceLROnPlateau': 'plateau.yaml',
            'WarmupHoldLinear': 'warmup_hold_linear.yaml',
        }
        scheduler_cfg_path = None
        if explicit_sched_path is not None:
            scheduler_cfg_path = explicit_sched_path
        else:
            base_dir = os.path.dirname(os.path.abspath(args.config))
            fname = sched_filename_map.get(sched_type)
            if fname is not None:
                scheduler_cfg_path = os.path.join(base_dir, 'schedulers', fname)
        if scheduler_cfg_path is not None and os.path.exists(scheduler_cfg_path):
            cfg = OmegaConf.merge(cfg, OmegaConf.load(scheduler_cfg_path))
    except Exception as _:
        raise ValueError(f"Invalid scheduler type: {sched_type}")

    # Auto-merge loss subconfiguration based on ranking_loss_type
    add_ranking_loss = cfg.training.get('add_ranking_loss', False)
    
    if add_ranking_loss:
        try:
            ranking_loss_type = cfg.training.get('ranking_loss_type', 'one_minus_rho')  # 'fisher' or 'one_minus_rho'
            explicit_loss_path = cfg.training.get('loss_config_path', None)
            loss_filename_map = {
                'fisher': 'fisher.yaml',
                'one_minus_rho': 'one_minus_rho.yaml',
            }
            loss_cfg_path = None
            if explicit_loss_path is not None:
                loss_cfg_path = explicit_loss_path
            else:
                base_dir = os.path.dirname(os.path.abspath(args.config))
                lname = loss_filename_map.get(ranking_loss_type)
                if lname is not None:
                    loss_cfg_path = os.path.join(base_dir, 'losses', lname)
            if loss_cfg_path is not None and os.path.exists(loss_cfg_path):
                cfg = OmegaConf.merge(cfg, OmegaConf.load(loss_cfg_path))
        except Exception:
            raise ValueError(f"Invalid ranking loss type: {ranking_loss_type}")

    # Merge CLI-provided output_dir and resume into the config.
    # These take precedence over values in the YAML file.
    cli_provided_params = {}
    if args.output_dir is not None:
        cli_provided_params['output_dir'] = args.output_dir
    if args.resume is not None:
        cli_provided_params['resume'] = args.resume
    
    if cli_provided_params:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_provided_params))

    # Parse and merge remaining command line arguments as OmegaConf overrides
    # These should be in key=value format, e.g., training.batch_size=128
    if unknown_cli_args:
        try:
            override_conf = OmegaConf.from_cli(unknown_cli_args)
            cfg = OmegaConf.merge(cfg, override_conf)
        except Exception as e:
            print(f"Warning: Could not parse all CLI overrides with OmegaConf: {e}")
            print(f"Ensure overrides are in 'key=value' format (e.g., training.lr=0.01). Unknown args received: {unknown_cli_args}")

    # 1.a) Initialize W&B login (only on rank 0) if enabled
    if fabric.global_rank == 0:
        load_dotenv()  # will read .env in cwd
        wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
    
    # 2) Paths & hyperparams
    
    manifest_csv        = cfg.data.manifest_file
    if manifest_csv is None:
        raise ValueError("manifest_csv is not set in the config")

    batch_size_per_gpu          = cfg.data.batch_size_per_gpu
    num_workers         = cfg.data.num_workers
    global_batch_size = batch_size_per_gpu * fabric.world_size

    # sampling choices
    samples_per_complex = cfg.data.get('samples_per_complex', 1) 
    bucket_balance      = cfg.training.get('bucket_balance', False)

    # number of epochs should allow to theoretically see al the samples at least once
    # so since in each epoch we sample just M samples per complex, then the number of epochs should be
    # len(manifest_csv) / (batch_size * M)) * actual_number_of_epochs

        
    epochs = cfg.training.epochs
    
    if fabric.global_rank == 0:
        print(f"Number of lines in manifest: {len(open(manifest_csv).readlines())-1}")
        print(f"Batch size-- batch_size_per_gpu: {batch_size_per_gpu}, global_batch_size: {global_batch_size}")
        print(f"Samples per complex: {samples_per_complex}")
        print(f"Number of epochs: {epochs}")

    lr                  = cfg.training.lr
    weight_decay        = cfg.training.weight_decay
    seed                = cfg.training.get('seed', None)
    weighted_loss       = cfg.training.get('weighted_loss', False)
    # Ranking loss parameters
    add_ranking_loss    = cfg.training.get('add_ranking_loss', False)
    add_distance_preservation_loss = cfg.training.get('add_distance_preservation_loss', False)
    ranking_loss_start_epoch = cfg.training.get('ranking_loss_start_epoch', 1)
    distance_loss_start_epoch = cfg.training.get('distance_loss_start_epoch', 1)
    spearman_tau        = cfg.training.get('spearman_tau', 0.02)
    ranking_loss_type   = cfg.training.get('ranking_loss_type', 'one_minus_rho')  # 'fisher' or 'one_minus_rho'
    # Fixed lambda parameters (replacing dynamic EMA-based updating)
    ranking_lambda = cfg.training.get('ranking_lambda', 10.0)
    distance_lambda = cfg.training.get('distance_lambda', 0.1)
    
    feature_transform   = cfg.data.feature_transform
    feature_centering   = cfg.data.get('feature_centering', False)
    use_interchain_ca_distances = cfg.data.get('use_interchain_ca_distances', False)
    use_interchain_pae = cfg.data.get('use_interchain_pae', True)
    use_esm_embeddings = cfg.data.get('use_esm_embeddings', False)
    esm_embedding_dim = cfg.data.get('esm_embedding_dim', 0)
    use_distance_cutoff = cfg.data.get('use_distance_cutoff', False)
    distance_cutoff = cfg.data.get('distance_cutoff', 10.0)
    use_file_cache = cfg.data.get('use_file_cache', True)
    cache_size_mb = cfg.data.get('cache_size_mb', 512)
    max_cached_files = cfg.data.get('max_cached_files', 20)
    print(f"Use interchain PAE: {use_interchain_pae}")
    print(f"Use ESM embeddings: {use_esm_embeddings} (dim={esm_embedding_dim})")

    # adaptive weight: focus more on extreme targets (DockQ near 0 or 1)
    adaptive_weight = cfg.training.get('adaptive_weight', False)
    if fabric.global_rank == 0:
        if adaptive_weight:
            print("Using adaptive weight")
        else:
            print("Not using adaptive weight")
        print(f"Feature transform: {feature_transform}")
        print(f"Feature centering: {feature_centering}")
        print(f"Use interchain Cα distances: {use_interchain_ca_distances}")
        print(f"Use interchain PAE: {use_interchain_pae}")
        print(f"Use ESM embeddings: {use_esm_embeddings} (dim={esm_embedding_dim})")
        print(f"Use distance cutoff: {use_distance_cutoff}")
        if use_distance_cutoff:
            print(f"Distance cutoff: {distance_cutoff} Å")
        print(f"Use HDF5 file cache: {use_file_cache}")
        if use_file_cache:
            print(f"  Cache size: {cache_size_mb} MB per file")
            print(f"  Max cached files: {max_cached_files} per worker")

    # 3) DataLoaders
    #    - train: with our chosen sampler
    is_distributed = fabric is not None and fabric.world_size > 1
    world_size = fabric.world_size
    rank = fabric.global_rank
    
    train_loader = get_dataloader(
        manifest_csv,
        split='train',
        batch_size=global_batch_size,
        num_workers=num_workers,
        samples_per_complex=samples_per_complex,
        bucket_balance=bucket_balance,
        feature_transform=feature_transform,
        feature_centering=feature_centering,
        use_interchain_ca_distances=use_interchain_ca_distances,
        use_interchain_pae=use_interchain_pae,
        use_esm_embeddings=use_esm_embeddings,
        use_distance_cutoff=use_distance_cutoff,
        distance_cutoff=distance_cutoff,
        use_file_cache=use_file_cache,
        cache_size_mb=cache_size_mb,
        max_cached_files=max_cached_files,
        seed=seed,
        distributed=is_distributed,
        world_size=world_size,
        rank=rank
    )
    #    - val: sequential, no special sampling
    val_loader = get_eval_dataloader(
        manifest_csv,
        split='val',
        batch_size=global_batch_size,
        samples_per_complex=samples_per_complex,
        num_workers=num_workers,
        feature_transform=feature_transform,
        feature_centering=feature_centering,
        use_interchain_ca_distances=use_interchain_ca_distances,
        use_interchain_pae=use_interchain_pae,
        use_esm_embeddings=use_esm_embeddings,
        use_distance_cutoff=use_distance_cutoff,
        distance_cutoff=distance_cutoff,
        use_file_cache=use_file_cache,
        cache_size_mb=cache_size_mb,
        max_cached_files=max_cached_files,
        seed=seed,
        distributed=is_distributed,
        world_size=world_size,
        rank=rank
    )

    # Model, device, optimizer
    # Adjust input_dim if we append interchain Cα distances as an extra feature
    input_dim_base   = int(cfg.model.input_dim)
    input_dim        = input_dim_base + (1 if use_interchain_ca_distances else 0) - (0 if use_interchain_pae else 2)
    # Add ESM embedding dimensions (2x since we have embeddings for both residues i and j)
    if use_esm_embeddings:
        input_dim += 2 * esm_embedding_dim
    phi_hidden_dims  = cfg.model.phi_hidden_dims
    rho_hidden_dims  = cfg.model.rho_hidden_dims
    aggregator       = cfg.model.aggregator
    print(f"input_dim: {input_dim} (base={input_dim_base}, ca_dist={use_interchain_ca_distances}, esm={use_esm_embeddings})")
    model = DeepSet(input_dim, phi_hidden_dims, rho_hidden_dims, aggregator=aggregator)
    model.apply(init_weights)

    # Setup device and move model
    device = fabric.device
    model = model.to(device)

    # Add NaN guard hook to catch NaN/Inf values during forward pass
    def _nan_guard_hook(module, inp, out):
        def _has_bad(x):
            return isinstance(x, torch.Tensor) and not torch.isfinite(x).all()
        
        bad_in = any(_has_bad(t) for t in (inp if isinstance(inp, (tuple, list)) else [inp]))
        bad_out = any(_has_bad(t) for t in (out if isinstance(out, (tuple, list)) else [out]))
        
        if bad_in or bad_out:
            print(f"🚨 NaN/Inf detected in {module.__class__.__name__}")
            if bad_in:
                print(f"   Bad input detected")
            if bad_out:
                print(f"   Bad output detected")
            raise RuntimeError(f"NaN/Inf in {module.__class__.__name__}")

    # Register the hook on all modules
    for m in model.modules():
        # Skip tiny containers to reduce noise if you want
        if len(list(m.children())) == 0:  # Only leaf modules
            m.register_forward_hook(_nan_guard_hook)

    # Only print on rank 0
    if fabric.global_rank == 0:
        print(f"Using device: {device}")
        print(f"NaN guard hooks registered on model")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 

    # Setup with Fabric for distributed training
    model, optimizer = fabric.setup(model, optimizer)
    
    train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)
    val_loader = fabric.setup_dataloaders(val_loader, use_distributed_sampler=False)

    # Loss: use weighted Huber (Smooth L1) for robust regression
    # reduction='none' gives you one loss per sample so you can apply your weights
    base_criterion = nn.SmoothL1Loss(reduction='none', beta=cfg.training.smooth_l1_beta)

    # Ensure output_dir is set, either from config or CLI, before it's used.
    if not hasattr(cfg, 'output_dir') or cfg.output_dir is None:
        raise ValueError("output_dir must be specified either in the config file or via --output_dir CLI argument.")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Replace the summary() call with: (only print on rank 0)
    if fabric.global_rank == 0:
        print(f"\nModel Architecture:")
        print(model)
        print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"The length of the train loader is {len(train_loader)}")
    # Initialize learning rate scheduler
    scheduler_type = cfg.training.get('lr_scheduler_type', 'ReduceLROnPlateau')
    if scheduler_type == "OneCycleLR":
        steps_per_epoch = len(train_loader)
        max_lr = float(cfg.training.get('onecycle_max_lr', lr))
        pct_start = float(cfg.training.get('onecycle_pct_start', 0.3))
        div_factor = float(cfg.training.get('onecycle_div_factor', 25.0))
        final_div_factor = float(cfg.training.get('onecycle_final_div_factor', 1e4))
        learning_rate_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy='cos'
        )
        learning_rate_scheduler_str = (
            f"OneCycleLR_maxlr_{max_lr}_epochs_{epochs}_steps_per_epoch_{steps_per_epoch}"
        )
    elif scheduler_type == "WarmupHoldLinear":
        # --- % schedule settings ---
        warmup_pct = float(cfg.training.get('warmup_pct', 0.10))  # 10% warmup
        hold_pct   = float(cfg.training.get('hold_pct',   0.40))  # 40% hold
        # decay_pct = 1 - warmup_pct - hold_pct

        # absolute min lr (clearer than only factor); fallback to factor if not provided
        base_lr     = lr  # your optimizer plateau (max) LR
        min_lr_cfg  = cfg.training.get('min_lr', None)
        min_factor  = float(cfg.training.get('min_lr_factor', 1e-2))
        min_lr      = float(min_lr_cfg) if min_lr_cfg is not None else base_lr * min_factor

        steps_per_epoch    = len(train_loader)
        total_optim_steps  = int(np.ceil(epochs * steps_per_epoch))

        warmup_steps = int(round(warmup_pct * total_optim_steps))
        hold_steps   = int(round(hold_pct   * total_optim_steps))
        decay_steps  = max(1, total_optim_steps - warmup_steps - hold_steps)

        # factors relative to the optimizer's base lr
        peak_fac = 1.0                      # plateau is optimizer lr
        min_fac  = float(min_lr / base_lr)  # final factor

        # optional: start warmup above 0 to avoid Adam "cold start"
        start_factor = float(cfg.training.get('warmup_start_factor', 0.0))  # e.g., 0.1

        def _whl_lambda(step):
            # step is the optimizer step index (0, 1, 2, ...)
            if step < warmup_steps:
                t = step / max(1, warmup_steps)
                return start_factor + (peak_fac - start_factor) * t
            elif step < warmup_steps + hold_steps:
                return peak_fac
            else:
                s = step - warmup_steps - hold_steps
                x = min(1.0, s / max(1, decay_steps))       # 0 -> 1 over decay
                # linear from peak_fac down to min_fac
                return min_fac + (peak_fac - min_fac) * (1.0 - x)

        learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _whl_lambda)
        learning_rate_scheduler_str = (
            f"WarmupHoldLinear_w{warmup_steps}_h{hold_steps}_d{decay_steps}_minf{min_factor}"
        )
    else:
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.training.lr_scheduler_factor,
            patience=cfg.training.lr_scheduler_patience,
            min_lr=cfg.training.min_lr
        )
        learning_rate_scheduler_str = f"ReduceLROnPlateau_factor_{learning_rate_scheduler.factor}_patience_{learning_rate_scheduler.patience}"

    if fabric.global_rank == 0:
        print(f"Learning rate scheduler: {learning_rate_scheduler_str}")

    #phi_hidden_dims and rho_hiddens_dims to string
    phi_hidden_dims_str = '_'.join(str(dim) for dim in phi_hidden_dims)
    rho_hidden_dims_str = '_'.join(str(dim) for dim in rho_hidden_dims)
    name = f"DeepSet_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_aggregator_{aggregator}_lr_scheduler_{learning_rate_scheduler_str}"

    # run_id = os.getenv("SLURM_JOB_ID", "local")
    # Initialize W&B run (only on rank 0)
    if fabric.global_rank == 0:
        wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"), 
            config=OmegaConf.to_container(cfg, resolve=True), # Convert OmegaConf to dict for wandb
            name=name,
            # id=run_id
        )

        # wandb.watch(model, log="all", log_graph=True)
    # Run a forward pass with the sample input (this will be captured by wandb.watch)
    # Create a sample input for wandb to trace the model
    # sample_batch_size = batch_size
    # sample_complex_size = samples_per_complex
    # sample_seq_len = 20
    # sample_input_dim = input_dim
    # sample_x = torch.zeros((sample_batch_size, sample_complex_size, sample_seq_len, sample_input_dim), device=device)
    # sample_lengths = torch.full((sample_batch_size, sample_complex_size), sample_seq_len, device=device)

    # with torch.no_grad():
    #     _ = model(sample_x, sample_lengths)

    # Initialize checkpoint path and save config
    ckpt_output_dir = os.path.join(cfg.output_dir, name)
    os.makedirs(ckpt_output_dir, exist_ok=True)
    with open(os.path.join(ckpt_output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f) # Use OmegaConf to save
    
    #before training, run a forward pass with the biggest complex in the train set
    #to check if the model does not throw a memory error
    # biggest_batch = None
    # for batch in train_loader:
    #     #find the batch with the biggest complex
    #     if biggest_batch is None or batch['lengths'].max() > biggest_batch['lengths'].max():
    #         biggest_batch = batch
    # feats  = biggest_batch['features'].to(device)
    # lengths = biggest_batch['lengths'].to(device)
    # labels = biggest_batch['label'].to(device)
    # weights= biggest_batch['weight'].to(device)
    # complex_id = biggest_batch['complex_id']
    # # print(f"feats shape: {feats.shape}")
    # # print(f"lengths shape: {lengths.shape}")
    # # print(f"labels shape: {labels.shape}")
    # # print(f"weights shape: {weights.shape}")
    # # print(f"complex_id shape: {complex_id.shape}")
    # _ = model(feats, lengths)
    
    # #clean up memory
    # torch.cuda.empty_cache()
    
    # --- resume logic: load checkpoint if requested and bump start_epoch ---
    start_epoch = 1
    resume_path = cfg.get('resume', None) # Get resume path from config
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        # full checkpoint with epoch, model+opt states?
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            # assume it's just a plain model.state_dict()
            model.load_state_dict(checkpoint)
            print("Loaded model weights only; optimizer state not found, starting at epoch 1")
    
    # Training loop
    print(f"Starting training from epoch {start_epoch}")
    
    #Setup variables useful for logging
    #We define a step as a batch (a gradient step)
    step = 0
    

    # Initialize persistent wandb tables for accumulating plots across epochs (only on rank 0)
    if fabric.global_rank == 0:
        all_complexes_table = wandb.Table(columns=["epoch", "image"], log_mode= 'INCREMENTAL')
        per_complex_table = wandb.Table(columns=["epoch", "complex_id", "image"], log_mode= 'INCREMENTAL')


    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Epochs", disable=fabric.global_rank != 0):
        model.train()
        epoch_start = time.time()
        training_start = time.time()
        
        # Initialize epoch-level metrics
        epoch_loss = 0.0
        epoch_avg_dockq = 0.0
        epoch_ranking_loss = 0.0
        epoch_distance_loss = 0.0
        epoch_regression_loss = 0.0
        epoch_grad_norm_reg = 0.0
        epoch_grad_norm_rank = 0.0
        epoch_grad_norm_distance = 0.0
        epoch_batch_count = 0

        # Notify when ranking loss is activated
        if add_ranking_loss and epoch == ranking_loss_start_epoch and fabric.global_rank == 0:
            print(f"Activating ranking (secondary) loss from epoch {epoch}")
        
        # Notify when distance preservation loss is activated
        if add_distance_preservation_loss and epoch == distance_loss_start_epoch and fabric.global_rank == 0:
            print(f"Activating distance preservation (tertiary) loss from epoch {epoch}")

        # Decide whether to use ranking loss this epoch
        add_rank_this_epoch = add_ranking_loss and (epoch >= ranking_loss_start_epoch)
        # Decide whether to use distance preservation loss this epoch
        add_distance_this_epoch = add_distance_preservation_loss and (epoch >= distance_loss_start_epoch)

        for batch in tqdm(train_loader, desc="Training epoch", disable=fabric.global_rank != 0):
            step += 1
            combined_loss_val, current_batch_size, avg_dockq, \
            current_regression_loss, current_base_ranking_loss_val, current_base_distance_loss_val, \
            current_grad_norm_reg, current_grad_norm_rank, current_grad_norm_distance = train_step(
                model, batch, optimizer, base_criterion, device, 
                step,
                weighted_loss, adaptive_weight,
                add_rank_this_epoch, add_distance_this_epoch, spearman_tau, ranking_loss_type,
                ranking_lambda=ranking_lambda, # Use fixed ranking lambda
                distance_lambda=distance_lambda, # Use fixed distance lambda
                fabric=fabric  # Pass fabric for autocast and backward
            )

            # Update learning rate per step for schedulers that are step-wise
            if scheduler_type in ("OneCycleLR", "WarmupHoldLinear"):
                learning_rate_scheduler.step()
            
            # Accumulate metrics for epoch-level logging
            epoch_loss += combined_loss_val
            epoch_avg_dockq += avg_dockq
            epoch_regression_loss += current_regression_loss
            epoch_grad_norm_reg += current_grad_norm_reg
            epoch_grad_norm_rank += current_grad_norm_rank
            epoch_grad_norm_distance += current_grad_norm_distance
            if add_rank_this_epoch:
                epoch_ranking_loss += current_base_ranking_loss_val
            if add_distance_this_epoch:
                epoch_distance_loss += current_base_distance_loss_val
            epoch_batch_count += 1

        # ---- END OF EPOCH TRAINING ----
        training_time = time.time() - training_start
        
        # Calculate average metrics for the epoch
        avg_epoch_loss = epoch_loss / epoch_batch_count
        avg_epoch_dockq = epoch_avg_dockq / epoch_batch_count
        avg_epoch_regression_loss = epoch_regression_loss / epoch_batch_count
        avg_epoch_grad_norm_reg = epoch_grad_norm_reg / epoch_batch_count
        avg_epoch_grad_norm_rank = epoch_grad_norm_rank / epoch_batch_count
        avg_epoch_grad_norm_distance = epoch_grad_norm_distance / epoch_batch_count
        
        # Calculate epoch averages for ranking and distance losses if enabled
        avg_epoch_ranking_loss = epoch_ranking_loss / epoch_batch_count if add_rank_this_epoch else 0.0
        avg_epoch_distance_loss = epoch_distance_loss / epoch_batch_count if add_distance_this_epoch else 0.0

        # ---- VALIDATION ----
        validation_start = time.time()
        fabric.barrier()
        val_loss_scalar = 0.0  # init on all ranks so variable exists
        abs_err_mean_scalar = 0.0
        
        # Run validation on all ranks to avoid hanging
        val_results = run_validation(
            model, val_loader, base_criterion, device, weighted_loss,
            add_ranking_loss=add_rank_this_epoch, 
            add_distance_preservation_loss=add_distance_this_epoch,
            spearman_tau=spearman_tau,
            ranking_lambda=ranking_lambda,
            distance_lambda=distance_lambda,
            ranking_loss_type=ranking_loss_type
        )
        
        val_loss_scalar = float(val_results['loss'])
        validation_time = time.time() - validation_start
        
        # ----  NaN check (tensors, numpy arrays, numpy scalars, floats) ----
        def _has_nan(x):
            import numpy as _np
            import torch as _torch
            if isinstance(x, _torch.Tensor):
                return _torch.isnan(x).any().item()
            if isinstance(x, (np.ndarray, np.generic)):
                return _np.isnan(x).any()
            if isinstance(x, float):
                return np.isnan(x)
            return False

        for key, val in val_results.items():
            if _has_nan(val):
                if fabric.global_rank == 0:
                    print(f"{key}: contains NaN")
                exit()

        pcp = val_results['per_complex_points']
        val_preds  = np.concatenate([pcp[c]['model'] for c in pcp.keys()])
        val_labels = np.concatenate([pcp[c]['true']  for c in pcp.keys()])

        # Check the difference between the predicted and true labels
        errors = val_preds - val_labels
        abs_errors = np.abs(errors)
        abs_err_mean_scalar = float(abs_errors.mean())

        # Only create plots and detailed logging on rank 0
        if fabric.global_rank == 0:
            # Extract complex IDs and per-complex metrics from the returned data
            per_complex_points = val_results['per_complex_points']
            cid = list(per_complex_points.keys())
            
            rho_m = val_results['rho_true_per_complex']  # true rho for model predictions
            delta = val_results['delta_true_rho_per_complex']  # delta true rho per complex
            
            # Calculate baseline true rho by subtracting delta from model rho
            rho_b = rho_m - delta

            n = min(len(cid), len(rho_m), len(rho_b), len(delta))
            
            # Debug information
            print(f"Validation plotting debug:")
            print(f"  - Number of complexes: {len(cid)}")
            print(f"  - val_preds shape: {val_preds.shape}")
            print(f"  - val_labels shape: {val_labels.shape}")
            print(f"  - rho_m length: {len(rho_m)}")
            print(f"  - delta length: {len(delta)}")
            print(f"  - Sample val_preds range: {val_preds.min():.3f} to {val_preds.max():.3f}")
            print(f"  - Sample val_labels range: {val_labels.min():.3f} to {val_labels.max():.3f}")
            
            # =============== TABLE 1: Per-complex prediction vs target ===============
            # Only plot every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                # Create scatter plots for 10 random complexes
                random_cids = random.sample(cid, 10)
                print(f"Random complexes: {random_cids}")
                
                for complex_id in random_cids:
                    if complex_id in per_complex_points:
                        complex_preds = np.array(per_complex_points[complex_id]['model'])
                        complex_targets = np.array(per_complex_points[complex_id]['true'])
                        
                        if len(complex_preds) > 0 and len(complex_targets) > 0:
                            # Create scatter plot for this complex
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(complex_targets, complex_preds, alpha=0.6)
                            ax.plot([0, 1], [0, 1], 'r--', alpha=0.8)  # diagonal line
                            ax.set_xlabel('True DockQ')
                            ax.set_ylabel('Predicted DockQ')
                            ax.set_title(f'Complex {complex_id}: Prediction vs Target (Epoch {epoch})')
                            ax.grid(True, alpha=0.3)
                            
                            # Add single row to table for this complex
                            per_complex_table.add_data(
                                epoch, str(complex_id), wandb.Image(fig)
                            )
                            plt.close(fig)
            
            # =============== TABLE 2: All complexes combined ===============
            # Only plot every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                # Create combined scatter plot with colors by complex
                print(f"Creating all complexes plot with {len(val_preds)} points from {len(cid)} complexes")
                
                # Prepare colors for each complex
                import matplotlib.cm as cm
                
                # Create arrays for plotting with colors by complex
                plot_labels = []
                plot_preds = []
                plot_colors = []
                
                # Generate a color for each complex
                colormap = cm.get_cmap('tab20')  # Good for categorical data
                num_complexes = len(cid)
                
                for idx, complex_id in enumerate(cid):
                    if complex_id in per_complex_points:
                        complex_preds = np.array(per_complex_points[complex_id]['model'])
                        complex_targets = np.array(per_complex_points[complex_id]['true'])
                        
                        if len(complex_preds) > 0 and len(complex_targets) > 0:
                            plot_labels.extend(complex_targets.tolist())
                            plot_preds.extend(complex_preds.tolist())
                            # Assign same color to all points from this complex
                            color = colormap(idx % 20)  # tab20 has 20 colors, cycle if needed
                            plot_colors.extend([color] * len(complex_preds))
                
                plot_labels = np.array(plot_labels)
                plot_preds = np.array(plot_preds)
                
                fig, ax = plt.subplots(figsize=(12, 9))
                scatter = ax.scatter(plot_labels, plot_preds, c=plot_colors, alpha=0.6, s=20, edgecolors='none')
                ax.plot([0, 1], [0, 1], 'r--', alpha=0.8, linewidth=2)  # diagonal line
                ax.set_xlabel('True DockQ', fontsize=12)
                ax.set_ylabel('Predicted DockQ', fontsize=12)
                ax.set_title(f'All Complexes: Prediction vs Target (Epoch {epoch})\n{num_complexes} complexes, {len(plot_preds)} points', fontsize=13)
                ax.grid(True, alpha=0.3)
                
                # Add text annotation with statistics
                correlation = np.corrcoef(plot_labels, plot_preds)[0, 1]
                mae = np.mean(np.abs(plot_preds - plot_labels))
                info_text = f'Complexes: {num_complexes}\nTotal points: {len(plot_preds)}\nCorrelation: {correlation:.3f}\nMAE: {mae:.4f}'
                ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                        verticalalignment='top', fontsize=10)
                
                all_complexes_table.add_data(epoch, wandb.Image(fig))
                plt.close(fig)
            

            
            
            val_log_dict = {
                # Core losses
                "val/loss": val_results['loss'],                     # Combined loss (scalar)
                "val/regression_loss": val_results['regression_loss'],
                "val/ranking_loss": val_results['ranking_loss'],
                "val/distance_loss": val_results['distance_loss'],
                "val/avg_dockq": val_results['avg_dockq'],

                # Histograms (media)
                "val/soft_rho_histogram_per_complex": wandb.Histogram(val_results['rho_soft_per_complex'], num_bins=50),
                "val/true_rho_histogram_per_complex": wandb.Histogram(val_results['rho_true_per_complex'], num_bins=50),
                "val/baseline_rho_soft_histogram_per_complex": wandb.Histogram(val_results['rho_baseline_per_complex'], num_bins=50),
                "val/delta_soft_rho_histogram_per_complex": wandb.Histogram(val_results['delta_soft_rho_per_complex'], num_bins=50),
                "val/delta_true_rho_histogram_per_complex": wandb.Histogram(val_results['delta_true_rho_per_complex'], num_bins=50),
                "val/abs_error_histogram": wandb.Histogram(abs_errors, num_bins=512),

                # Scalar summaries (no lists)
                "val/soft_rho_mean": val_results['soft_rho_mean'],
                "val/true_rho_mean": val_results['true_rho_mean'],
                "val/baseline_rho_soft_mean": val_results['baseline_rho_mean'],
                "val/delta_soft_rho_mean": val_results['delta_soft_rho_mean'],
                "val/delta_true_rho_mean": val_results['delta_true_rho_mean'],
                "val/abs_error_mean": abs_err_mean_scalar,
                "val/abs_error_q25": float(np.percentile(abs_errors, 25)),
                "val/abs_error_q50": float(np.percentile(abs_errors, 50)),
                "val/abs_error_q75": float(np.percentile(abs_errors, 75)),
                "val/abs_error_max":  float(abs_errors.max()),
                "val/epoch": epoch,
            }
            
            # Add ranking loss metrics if enabled
            if add_rank_this_epoch:
                val_log_dict["val/ranking_loss"] = val_results['ranking_loss']
                val_log_dict["val/reg_vs_rank_loss_ratio"] = val_results['regression_loss'] / (val_results['ranking_loss'] + 1e-8)
                val_log_dict["val/ranking_lambda"] = ranking_lambda
            
            # Add distance loss metrics if enabled
            if add_distance_this_epoch:
                val_log_dict["val/distance_loss"] = val_results['distance_loss']
                val_log_dict["val/reg_vs_distance_loss_ratio"] = val_results['regression_loss'] / (val_results['distance_loss'] + 1e-8)
                val_log_dict["val/distance_lambda"] = distance_lambda
            
            # Create training log dictionary
            train_log_dict = {
                "train/loss": avg_epoch_loss,
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/epoch": epoch,
                "train/avg_dockq": avg_epoch_dockq,
                "train/regression_loss": avg_epoch_regression_loss,
                "train/grad_norm_reg": avg_epoch_grad_norm_reg,
                "train/grad_norm_rank": avg_epoch_grad_norm_rank,
                "train/grad_norm_distance": avg_epoch_grad_norm_distance,
                "train/grad_norm_reg_rank_ratio": avg_epoch_grad_norm_reg / (avg_epoch_grad_norm_rank + 1e-8),
                "train/grad_norm_reg_distance_ratio": avg_epoch_grad_norm_reg / (avg_epoch_grad_norm_distance + 1e-8),
                "train/ranking_lambda": ranking_lambda,
                "train/distance_lambda": distance_lambda,
                "train/training_time_seconds": training_time,
                "train/validation_time_seconds": validation_time,
                "train/total_epoch_time_seconds": time.time() - epoch_start,
            }
            
            # Add ranking loss metrics if enabled
            if add_rank_this_epoch:
                train_log_dict["train/ranking_loss"] = avg_epoch_ranking_loss
                train_log_dict["train/reg_vs_rank_loss_ratio"] = avg_epoch_regression_loss / (avg_epoch_ranking_loss + 1e-8)
            
            # Add distance loss metrics if enabled
            if add_distance_this_epoch:
                train_log_dict["train/distance_loss"] = avg_epoch_distance_loss
                train_log_dict["train/reg_vs_distance_loss_ratio"] = avg_epoch_regression_loss / (avg_epoch_distance_loss + 1e-8)
            
            # Combine train and validation logs
            combined_log_dict = {**train_log_dict, **val_log_dict}
            
            wandb.log(combined_log_dict, step=epoch)
            wandb.log({
                "val/per_complex_pred_vs_target": per_complex_table,
                "val/all_complexes_pred_vs_target": all_complexes_table,})
        # broadcast scalar val_loss to every rank (sum of {x,0,0,...} -> x on all)
        val_loss_t = torch.tensor(val_loss_scalar, device=fabric.device, dtype=torch.float32)
        val_loss_t = fabric.all_reduce(val_loss_t, reduce_op="sum")
        avg_val_loss = float(val_loss_t.item())
        if scheduler_type == "ReduceLROnPlateau":
            learning_rate_scheduler.step(avg_val_loss)
        
        # Only print on rank 0
        if fabric.global_rank == 0:
            total_epoch_time = time.time() - epoch_start
            epoch_print_msg = f"Epoch {epoch}/{epochs}, Total time: {total_epoch_time:.2f}s (Train: {training_time:.2f}s, Val: {validation_time:.2f}s), Train loss: {avg_epoch_loss:.8f} —  Val loss: {avg_val_loss:.8f} —  Val abs error mean: {abs_err_mean_scalar:.8f} - regression loss: {avg_epoch_regression_loss:.8f}"
            if add_rank_this_epoch:
                epoch_print_msg += f" - ranking loss: {avg_epoch_ranking_loss:.8f}"
            if add_distance_this_epoch:
                epoch_print_msg += f" - distance loss: {avg_epoch_distance_loss:.8f}"
            print(epoch_print_msg)

        # 7) Save checkpoint (only on rank 0)
        if epoch % 5 == 0:
            fabric.barrier()
            fabric.save(os.path.join(ckpt_output_dir, f"checkpoint_epoch{epoch}.pt"), {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            })

    # Only print on rank 0
    if fabric.global_rank == 0:
        print("Training complete.")

def run_validation(model, val_loader, base_criterion, device, weighted_loss, 
                  add_ranking_loss=False, add_distance_preservation_loss=False, 
                  spearman_tau=0.02, ranking_lambda=1.0, distance_lambda=1.0,
                  ranking_loss_type='one_minus_rho'):
    """
    Enhanced validation function that computes the same losses as training:
    - Regression loss
    - Ranking loss (if enabled)
    - Distance preservation loss (if enabled)
    - Combined loss
    - Average DockQ
    """
    model.eval()
    val_loss   = 0.0
    val_count  = 0
    
    # Additional loss accumulators
    val_regression_loss = 0.0
    val_ranking_loss = 0.0
    val_distance_loss = 0.0
    val_avg_dockq = 0.0
    val_rho_soft_per_complex = []
    val_true_rho_per_complex = []
    val_base_rho_soft_per_complex = [] #the distance between rho and baseline rho
    val_base_true_rho_per_complex = [] #the distance between true rho and baseline true rho
    val_delta_rho_per_complex = [] #the distance between rho and baseline rho

    per_complex_points = defaultdict(lambda: {"true": [], "model": [], "base": [], 'soft_rho': [], 'true_rho': [], 'baseline_soft_rho': [], 'baseline_true_rho': []})

    complex_ids = set()
    with torch.no_grad():
        for batch in val_loader:
            feats  = batch['features'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['label'].to(device) # Already logit transformed
            weights= batch['weight'].to(device)  # still available
            complex_id = batch['complex_id']
            tmp_complex_ids = set(complex_id.flatten().tolist())
            complex_ids.update(tmp_complex_ids)
            ranking_score = batch['ranking_score'].to(device)

            logits = model(feats, lengths)
            logits = torch.clamp(logits, min=-100.0, max=100.0)
            
            # =============== REGRESSION LOSS ===============
            regression_losses = base_criterion(logits, labels)
            if weighted_loss:
                regression_loss = (regression_losses * weights).mean()
            else:
                regression_loss = regression_losses.mean()
            
            # =============== RANKING LOSS (if enabled) ===============
            base_ranking_loss = 0.0
            # =============== DISTANCE PRESERVATION LOSS (if enabled) ===============
            base_distance_loss = 0.0
            pred_sig = torch.sigmoid(logits) # use the predicted dockQ as the predicted label
            true_sig = torch.sigmoid(labels) # use the true dockQ as the true label
            base_sig = torch.clamp(ranking_score, 1e-7, 1 - 1e-7) #using the ranking score as the baseline, aka what AF gives as official ranking

            #append the true, model, and base arrays to the per_complex_points dictionary
            true_sig_flat = true_sig.detach().cpu().numpy().flatten()
            pred_sig_flat = pred_sig.detach().cpu().numpy().flatten()
            base_sig_flat = base_sig.detach().cpu().numpy().flatten()
            
            for c in tmp_complex_ids:
                
                per_complex_points[str(c)]["true"].extend(true_sig_flat.tolist())
                per_complex_points[str(c)]["model"].extend(pred_sig_flat.tolist())
                per_complex_points[str(c)]["base"].extend(base_sig_flat.tolist())

            if add_ranking_loss and logits.size(1) > 1:
                # Compute ranking relevance weight (same as training)
                with torch.no_grad():
                    true_sig_std = true_sig.std(dim=1, keepdim=True) + 1e-6
                    ranking_relevance_weight = torch.clamp(true_sig_std / 0.07, min=0.0, max=1.0)
                if ranking_loss_type == 'one_minus_rho':
                    rho_soft_terms = soft_spearman_loss(pred_sig, true_sig, loss_type='one_minus_rho', tau=spearman_tau)
                    base_spearman_loss_terms = rho_soft_terms
                elif ranking_loss_type == 'fisher':
                    base_spearman_loss_terms = soft_spearman_loss(pred_sig, true_sig, loss_type='fisher', tau=spearman_tau)
                else:
                    raise ValueError(f"Invalid ranking loss type: {ranking_loss_type}")
                base_ranking_loss = (base_spearman_loss_terms * ranking_relevance_weight.squeeze()).mean()
            
            # =============== DISTANCE PRESERVATION LOSS (if enabled) ===============
            if add_distance_preservation_loss and logits.size(1) > 1:
                distance_loss_terms = distance_preservation_loss(pred_sig, true_sig)  # [B]
                base_distance_loss = distance_loss_terms.mean()
            
            # =============== COMBINED LOSS ===============
            combined_loss = regression_loss
            if add_ranking_loss and logits.size(1) > 1:
                combined_loss = combined_loss + ranking_lambda * base_ranking_loss
            if add_distance_preservation_loss and logits.size(1) > 1:
                combined_loss = combined_loss + distance_lambda * base_distance_loss
            
            # =============== AVERAGE DOCKQ ===============
            avg_dockq = torch.sigmoid(labels).mean().item()

            # Accumulate losses
            val_loss += combined_loss.item()
            val_regression_loss += regression_loss.item()
            if add_ranking_loss and logits.size(1) > 1:
                val_ranking_loss += base_ranking_loss.item()
            if add_distance_preservation_loss and logits.size(1) > 1:
                val_distance_loss += base_distance_loss.item()
            val_avg_dockq += avg_dockq
            val_count += feats.size(0)

    #convert values to numpy arrays
    for c in complex_ids:
        per_complex_points[str(c)]["true"] = np.array(per_complex_points[str(c)]["true"])
        per_complex_points[str(c)]["model"] = np.array(per_complex_points[str(c)]["model"])
        per_complex_points[str(c)]["base"] = np.array(per_complex_points[str(c)]["base"])
        
    # Average the losses
    avg_val_loss = val_loss / len(val_loader)
    avg_val_regression_loss = val_regression_loss / len(val_loader)
    avg_val_ranking_loss = val_ranking_loss / len(val_loader) if add_ranking_loss else 0.0
    avg_val_distance_loss = val_distance_loss / len(val_loader) if add_distance_preservation_loss else 0.0
    avg_val_dockq = val_avg_dockq / len(val_loader)


    # =============== SOFT SPEARMAN RHO (per complex) ===============
    soft_rho_per_complex = []
    true_rho_per_complex = []
    baseline_soft_rho_per_complex = []
    baseline_true_rho_per_complex = []
    for c in complex_ids:

        true_rho_per_complex_this, soft_rho_per_complex_this = calculate_spearman(per_complex_points[str(c)]["model"], per_complex_points[str(c)]["true"])
        soft_rho_per_complex.append(soft_rho_per_complex_this)
        true_rho_per_complex.append(true_rho_per_complex_this)

        baseline_true_rho_per_complex_this, baseline_soft_rho_per_complex_this = calculate_spearman(per_complex_points[str(c)]["base"], per_complex_points[str(c)]["true"])
        baseline_soft_rho_per_complex.append(baseline_soft_rho_per_complex_this)
        baseline_true_rho_per_complex.append(baseline_true_rho_per_complex_this)
        
        per_complex_points[str(c)]["soft_rho"].append(soft_rho_per_complex_this)
        per_complex_points[str(c)]["true_rho"].append(true_rho_per_complex_this)
        per_complex_points[str(c)]["baseline_soft_rho"].append(baseline_soft_rho_per_complex_this)
        per_complex_points[str(c)]["baseline_true_rho"].append(baseline_true_rho_per_complex_this)
    
    soft_rho_per_complex = np.array(soft_rho_per_complex)
    true_rho_per_complex = np.array(true_rho_per_complex)
    baseline_soft_rho_per_complex = np.array(baseline_soft_rho_per_complex)
    baseline_true_rho_per_complex = np.array(baseline_true_rho_per_complex)
    delta_soft_rho_per_complex = (soft_rho_per_complex - baseline_soft_rho_per_complex)
    delta_true_rho_per_complex = (true_rho_per_complex - baseline_true_rho_per_complex)

    return {
        'loss': avg_val_loss,
        'regression_loss': avg_val_regression_loss,
        'ranking_loss': avg_val_ranking_loss,
        'distance_loss': avg_val_distance_loss,
        'avg_dockq': avg_val_dockq,

        'rho_soft_per_complex': soft_rho_per_complex,
        'rho_true_per_complex': true_rho_per_complex,
        'rho_baseline_per_complex': baseline_soft_rho_per_complex,
        'soft_rho_mean': float(soft_rho_per_complex.mean()),
        'true_rho_mean': float(true_rho_per_complex.mean()),
        'baseline_rho_mean': float(baseline_soft_rho_per_complex.mean()),
        'delta_soft_rho_per_complex': delta_soft_rho_per_complex,
        'delta_true_rho_per_complex': delta_true_rho_per_complex,
        'delta_soft_rho_mean': float(delta_soft_rho_per_complex.mean()),
        'delta_true_rho_mean': float(delta_true_rho_per_complex.mean()),

        'per_complex_points': dict(per_complex_points),
    }

def train_step(model, batch, optimizer, base_criterion, device,
               step,
               weighted_loss, adaptive_weight,
               add_ranking_loss, add_distance_preservation_loss, spearman_tau, ranking_loss_type,
               ranking_lambda=1.0,          # Fixed ranking lambda
               distance_lambda=1.0,         # Fixed distance lambda
               fabric=None):                # Lightning Fabric for autocast and backward                        
    """
    Fabric-safe training step with fixed lambdas:
      - single forward (under autocast)
      - probe backprops to get ||grad(L_reg)||, ||grad(L_rank)||, ||grad(L_distance)|| without DDP sync
      - build total loss with fixed ranking_lambda and distance_lambda
      - real Fabric backward + optimizer step
    """

    # def grad_norm_via_fabric_probe(loss_tensor, params, fabric, optimizer):
    #     """
    #     Fabric-safe per-loss grad-norm probe:
    #     - uses fabric.no_backward_sync(model) to avoid DDP all-reduce
    #     - uses fabric.backward(loss, retain_graph=True) to satisfy Fabric
    #     - reads .grad, computes global L2, then zeroes grads immediately
    #     - returns a finite Python float (DDP-averaged)
    #     """
    #     if (loss_tensor is None) or (not loss_tensor.requires_grad) or (len(params) == 0):
    #         return 0.0

    #     # 1) local backward w/o sync -> fills p.grad locally
    #     with fabric.no_backward_sync(model):
    #         optimizer.zero_grad(set_to_none=True)
    #         fabric.backward(loss_tensor, retain_graph=True)

    #     # 2) compute L2 over current grads in fp32 (ignore NaN/Inf terms)
    #     total_sq = torch.zeros((), device=params[0].device, dtype=torch.float32)
    #     for p in params:
    #         g = p.grad
    #         if g is not None:
    #             g32 = g.detach().to(torch.float32)
    #             g32 = torch.nan_to_num(g32, nan=0.0, posinf=0.0, neginf=0.0)
    #             total_sq = total_sq + g32.pow(2).sum()
    #     gn = total_sq.sqrt()

    #     # 3) agree across ranks (mean is fine; we only need consistency)
    #     gn = fabric.all_reduce(gn, reduce_op="mean")

    #     # 4) clean up probe grads to avoid contaminating the real step
    #     optimizer.zero_grad(set_to_none=True)

    #     return float(torch.clamp(gn, min=1e-12))

    # --------------- move batch to device ---------------
    feats   = batch['features'].to(device)
    lengths = batch['lengths'].to(device)
    labels  = batch['label'].to(device)
    weights = batch['weight'].to(device)
    # complex_id = batch['complex_id']  # not used here
    
    # print(f"feats shape: {feats.shape}")
    avg_dockq = torch.sigmoid(labels).mean().item()
    params = [p for p in model.parameters() if p.requires_grad]

    # wrappers so this function also runs without Fabric (fallback to plain PyTorch)
    autocast_ctx = fabric.autocast()
    def _backward(loss, **kw):
        fabric.backward(loss, **kw)


    optimizer.zero_grad()

    # ------------------- forward + base losses -------------------
    def compute_losses():
                   
        logits = model(feats, lengths)  # [B, K] predictions in logit space
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        
        # regression loss on logits vs labels (already in logit space)
        regression_losses = base_criterion(logits, labels)

        current_regression_weights = weights
        if adaptive_weight:
            with torch.no_grad():
                # focus more on extremes (DockQ near 0 or 1)
                conf_w = 1.0 + 4.0 * ((torch.sigmoid(labels) < 0.1) | (torch.sigmoid(labels) > 0.9))
                current_regression_weights = current_regression_weights * conf_w

        if weighted_loss:
            regression_loss = (regression_losses * current_regression_weights).mean()
        else:
            regression_loss = regression_losses.mean()

        return logits, regression_loss, current_regression_weights

    with autocast_ctx:
        logits, regression_loss, current_regression_weights = compute_losses()

        # Optional ranking loss (Spearman-soft) if K > 1
        base_ranking_loss_for_log = 0.0
        mean_base_ranking_loss = None
        # Optional distance preservation loss if K > 1
        base_distance_loss_for_log = 0.0
        mean_base_distance_loss = None
        
        if add_ranking_loss and logits.size(1) > 1:
            pred_sig = torch.sigmoid(logits)
            true_sig = torch.sigmoid(labels)
            # Continuous weighting approach

            with torch.no_grad():
                true_sig_std = true_sig.std(dim=1, keepdim=True)
                tmp_eps = 1e-3
                gate = (true_sig_std > const_eps).float()
                ranking_relevance_weight = gate * torch.clamp(true_sig_std / 0.07, min=0.0, max=1.0)
            if ranking_loss_type == 'one_minus_rho':
                rho_soft_terms = soft_spearman_loss(pred_sig, true_sig, loss_type=ranking_loss_type, tau=spearman_tau)
                base_spearman_loss_terms = rho_soft_terms  # [B]
            elif ranking_loss_type == 'fisher':
                base_spearman_loss_terms = soft_spearman_loss(pred_sig, true_sig, loss_type=ranking_loss_type, tau=spearman_tau)  # [B]
            else:
                raise ValueError(f"Invalid ranking loss type: {ranking_loss_type}")
            mean_base_ranking_loss = (base_spearman_loss_terms * ranking_relevance_weight.squeeze()).mean()
            base_ranking_loss_for_log = float(mean_base_ranking_loss.detach())

        if add_distance_preservation_loss and logits.size(1) > 1:
            # pred_sig = logits
            # true_sig = labels
            pred_sig = torch.sigmoid(logits)
            true_sig = torch.sigmoid(labels)
            distance_loss_terms = distance_preservation_loss(pred_sig, true_sig)  # [B]
            mean_base_distance_loss = distance_loss_terms.mean()
            base_distance_loss_for_log = float(mean_base_distance_loss.detach())

    # grad_norm_reg_val  = grad_norm_via_fabric_probe(regression_loss, params, fabric, optimizer)
    # grad_norm_rank_val = grad_norm_via_fabric_probe(mean_base_ranking_loss, params, fabric, optimizer) if (add_ranking_loss and logits.size(1) > 1) else 0.0
    # grad_norm_distance_val = grad_norm_via_fabric_probe(mean_base_distance_loss, params, fabric, optimizer) if (add_distance_preservation_loss and logits.size(1) > 1) else 0.0

    # ------------------- final combined backward + step -------------------
    # Build the combined loss using the fixed lambdas
    total_combined_loss = regression_loss
    if add_ranking_loss and logits.size(1) > 1:
        total_combined_loss = total_combined_loss + ranking_lambda * mean_base_ranking_loss
    if add_distance_preservation_loss and logits.size(1) > 1:
        total_combined_loss = total_combined_loss + distance_lambda * mean_base_distance_loss

    optimizer.zero_grad(set_to_none=True)
    _backward(total_combined_loss)   # Fabric handles AMP/DDP

    # If you later switch to GradScaler, call scaler.unscale_(optimizer) before clipping.
    try:
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
    except Exception:
        # Be resilient; clipping is best-effort and shouldn't crash training.
        pass
    
    optimizer.step()

    # ------------------- return (simplified signature) -------------------
        
    return_values = [
        float(total_combined_loss.detach()),
        feats.size(0),
        avg_dockq,
        float(regression_loss.detach()),
        base_ranking_loss_for_log,
        base_distance_loss_for_log,
        0,
        0,
        0,
    ]

    return tuple(return_values)


if __name__ == '__main__':
    main()
