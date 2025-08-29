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
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
import wandb
from dotenv import load_dotenv
import time
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf

# Lightning Fabric for distributed training

from lightning.fabric import Fabric

from src.data.dataloader import (
    get_dataloader,
    get_eval_dataloader
)
from src.models.deep_set import DeepSet, init_weights

torch.set_float32_matmul_precision("high")

def beta_regression_loss(y_true, z_mu, z_k, T=3.0, eps=1e-3):
    """
    y_true: [B, K] in [0,1]
    z_mu, z_k: [B, K] raw outputs from model
    Returns: (loss_scalar, mu, kappa)
    """
    # Predict mean μ in (eps, 1-eps) using a temperatured sigmoid (flatter near edges)
    mu = eps + (1 - 2*eps) * torch.sigmoid(z_mu / T)
    # Predict concentration κ > 0
    kappa = 1.0 + F.softplus(z_k)

    alpha = mu * kappa
    beta  = (1 - mu) * kappa

    y = y_true.clamp(eps, 1 - eps)  # avoid log(0) at exact edges
    dist = Beta(alpha, beta)
    nll = -dist.log_prob(y)         # [B, K]
    # mean over K then B (like your current reduction); adjust if you weight per-sample
    return nll.mean(), mu, kappa

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

def fisher_spearman_soft_loss(pred, target, tau=0.2):
    """
    pred, target : shape [B, K] in DockQ (0–1) space
    returns scalar 1 - ρ loss per complex [B]
    Assumes K >= 2, which is handled by the caller.
    
    It calculates the spearman soft loss and after it applies a Fisher transform (atanh) 
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
    
    mse = (sr_pred - sr_true).pow(2).sum(dim=-1) # per complex [B]
    
    denominator = K * (K**2 - 1)
    # If K < 2, denominator could be 0. Handled by K < 2 check above.
    # However, if K is exactly 0 or 1 due to an issue, ensure no division by zero.
    if denominator == 0: # Should not happen if K >= 2 check is effective
        return torch.zeros(pred.size(0), device=pred.device, dtype=pred.dtype)

    rho_soft = 1 - 6 * mse / denominator # per complex [B]

    #apply fisher transform
    rho_soft = torch.clamp(rho_soft, min=-1.0 + 1e-6, max=1.0 - 1e-6)
    atanh_rho_soft = torch.atanh(rho_soft)

    return -atanh_rho_soft # per complex [B], to be minimized (hence -athan of rho)

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
        '--precision', type=str, default="32",
        help='Training precision (32, 16-mixed, bf16-mixed)'
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

    # 1.a) Initialize W&B run (only on rank 0)
    if fabric.global_rank == 0:
        load_dotenv()  # will read .env in cwd
        wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
    
    # 2) Paths & hyperparams
    
    manifest_csv        = cfg.data.manifest_file
    if manifest_csv is None:
        raise ValueError("manifest_csv is not set in the config")

    batch_size          = cfg.data.batch_size
    num_workers         = cfg.data.num_workers

    # sampling choices
    samples_per_complex = cfg.data.get('samples_per_complex', 1) 
    bucket_balance      = cfg.training.get('bucket_balance', False)

    # number of epochs should allow to theoretically see al the samples at least once
    # so since in each epoch we sample just M samples per complex, then the number of epochs should be
    # len(manifest_csv) / (batch_size * M)) * actual_number_of_epochs

        
    epochs = cfg.training.epochs
    
    if fabric.global_rank == 0:
        print(f"Number of lines in manifest: {len(open(manifest_csv).readlines())-1}")
        print(f"Batch size: {batch_size}")
        print(f"Samples per complex: {samples_per_complex}")
        print(f"Number of epochs: {epochs}")

    lr                  = cfg.training.lr
    weight_decay        = cfg.training.weight_decay
    seed                = cfg.training.get('seed', None)
    weighted_loss       = cfg.training.get('weighted_loss', False)
    # Ranking loss parameters
    add_ranking_loss    = cfg.training.get('add_ranking_loss', False)
    ranking_loss_weight = cfg.training.get('ranking_loss_weight', 10)
    ranking_loss_start_epoch = cfg.training.get('ranking_loss_start_epoch', 1)
    spearman_tau        = cfg.training.get('spearman_tau', 0.02)
    lambda_ema_alpha    = cfg.training.get('lambda_ema_alpha', 0.9) # For dynamic lambda smoothing

    feature_transform   = cfg.data.feature_transform
    feature_centering   = cfg.data.get('feature_centering', False)
    use_interchain_ca_distances = cfg.data.get('use_interchain_ca_distances', False)

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

    # 3) DataLoaders
    #    - train: with our chosen sampler
    is_distributed = fabric is not None and fabric.world_size > 1
    world_size = fabric.world_size
    rank = fabric.global_rank
    
    train_loader = get_dataloader(
        manifest_csv,
        split='train',
        batch_size=batch_size,
        num_workers=num_workers,
        samples_per_complex=samples_per_complex,
        bucket_balance=bucket_balance,
        feature_transform=feature_transform,
        feature_centering=feature_centering,
        use_interchain_ca_distances=use_interchain_ca_distances,
        seed=seed,
        distributed=is_distributed,
        world_size=world_size,
        rank=rank
    )
    #    - val: sequential, no special sampling
    val_loader = get_eval_dataloader(
        manifest_csv,
        split='val',
        batch_size=batch_size,
        samples_per_complex=samples_per_complex,
        num_workers=num_workers,
        feature_transform=feature_transform,
        feature_centering=feature_centering,
        use_interchain_ca_distances=use_interchain_ca_distances,
        seed=seed,
        distributed=is_distributed,
        world_size=world_size,
        rank=rank
    )

    # Model, device, optimizer
    # Adjust input_dim if we append interchain Cα distances as an extra feature
    input_dim_base   = int(cfg.model.input_dim)
    input_dim        = input_dim_base + (1 if use_interchain_ca_distances else 0)
    phi_hidden_dims  = cfg.model.phi_hidden_dims
    rho_hidden_dims  = cfg.model.rho_hidden_dims
    aggregator       = cfg.model.aggregator

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

    ### DEBUG STEPS FOR MULTI NODES SYNCING ###
    def param_checksum(m):
        s1 = torch.tensor(0.0, device=fabric.device)
        s2 = torch.tensor(0.0, device=fabric.device)
        with torch.no_grad():
            for p in m.parameters():
                if p is not None:
                    pf = p.float()
                    s1 += pf.sum()
                    s2 += (pf**2).sum()
        s1_all = fabric.all_gather(s1).cpu().numpy()
        s2_all = fabric.all_gather(s2).cpu().numpy()
        return s1_all, s2_all

    s1_all, s2_all = param_checksum(model)
    if fabric.global_rank == 0:
        print("INIT checksum s1 per-rank:", s1_all)
        print("INIT checksum s2 per-rank:", s2_all)
    ### END DEBUG STEPS FOR MULTI NODES SYNCING ###
    
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
    name = f"DeepSet_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_betadist_loss_{weighted_loss}_aggregator_{aggregator}_lr_scheduler_{learning_rate_scheduler_str}"

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

        wandb.watch(model, log="all", log_graph=True)
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
    log_interval = len(train_loader)//4
    log_interval_counter = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    num_batches = 0
    last_log_time = time.time()
    running_avg_dockq = 0.0
    running_ranking_loss = 0.0 # For accumulating base ranking loss values
    running_regression_loss = 0.0 # For accumulating regression loss values
    running_grad_norm_reg = 0.0 # For accumulating ||grad(L_reg)||
    running_grad_norm_rank = 0.0 # For accumulating ||grad(L_rank)||
    
    # Initialize dynamic lambda for ranking loss
    dynamic_ranking_lambda = ranking_loss_weight # Starts with the config value

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_start = time.time()
        
        avg_val_loss_per_epoch = 0.0
        log_step_per_epoch = 0
        avg_loss = 0.0  # Initialize avg_loss here
        abs_errors = np.array([0.0])  # Initialize abs_errors here

        # Notify when ranking loss is activated
        if add_ranking_loss and epoch == ranking_loss_start_epoch and fabric.global_rank == 0:
            print(f"Activating ranking (secondary) loss from epoch {epoch}")

        # Decide whether to use ranking loss this epoch
        add_rank_this_epoch = add_ranking_loss and (epoch >= ranking_loss_start_epoch)

        for batch in train_loader:

            combined_loss_val, current_batch_size, avg_dockq, \
            current_regression_loss, current_base_ranking_loss_val, \
            current_grad_norm_reg, current_grad_norm_rank, \
            updated_dynamic_lambda = train_step(
                model, batch, optimizer, base_criterion, device, 
                weighted_loss, adaptive_weight,
                add_rank_this_epoch, spearman_tau, 
                current_ranking_lambda=dynamic_ranking_lambda, # Pass current lambda
                lambda_ema_alpha=lambda_ema_alpha,
                fabric=fabric  # Pass fabric for autocast and backward
            )
            dynamic_ranking_lambda = updated_dynamic_lambda # Update lambda for next step

            # Update learning rate
            if scheduler_type == "OneCycleLR":
                learning_rate_scheduler.step()
            
            running_loss += combined_loss_val # This is now the combined loss
            log_interval_counter += 1
            running_avg_dockq += avg_dockq
            if add_rank_this_epoch:
                running_ranking_loss += current_base_ranking_loss_val # Accumulate base ranking loss
            running_regression_loss += current_regression_loss
            running_grad_norm_reg += current_grad_norm_reg # Accumulate ||grad(L_reg)||
            running_grad_norm_rank += current_grad_norm_rank # Accumulate ||grad(L_rank)||

            # Log every log_interval steps
            if log_interval_counter % log_interval == 0:
                avg_loss = running_loss / log_interval 
                avg_dockq_interval = running_avg_dockq / log_interval 
                avg_regression_loss_interval = running_regression_loss / log_interval
                avg_grad_norm_reg_interval = running_grad_norm_reg / log_interval
                avg_grad_norm_rank_interval = running_grad_norm_rank / log_interval


                elapsed = time.time() - last_log_time
                throughput = log_interval / elapsed if elapsed > 0 else 0.0
                
                log_dict = {
                    "train/loss": avg_loss, # This is combined loss
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/throughput": throughput,
                    "train/epoch": epoch,
                    "train/avg_dockq": avg_dockq_interval,
                    "train/regression_loss": avg_regression_loss_interval,
                    "train/grad_norm_reg": avg_grad_norm_reg_interval, # Log ||grad(L_reg)||
                    "train/grad_norm_rank": avg_grad_norm_rank_interval, # Log ||grad(L_rank)||
                    "train/grad_norm_reg_rank_ratio": avg_grad_norm_reg_interval / (avg_grad_norm_rank_interval + 1e-8),
                    "train/dynamic_ranking_lambda": dynamic_ranking_lambda, # Log current lambda
                }
                if add_rank_this_epoch: 
                    avg_base_ranking_loss_interval = running_ranking_loss / log_interval 
                    log_dict["train/ranking_loss"] = avg_base_ranking_loss_interval # Log L_rank
                    # Ratio of regression loss to base ranking loss (values, not grads)
                    log_dict["train/reg_vs_rank_loss_ratio"] = avg_regression_loss_interval / (avg_base_ranking_loss_interval + 1e-8) 
                
                # Only log on rank 0
                if fabric.global_rank == 0:
                    wandb.log(log_dict, step=log_interval_counter)

                # Reset running stats for next interval
                running_loss = 0.0
                running_avg_dockq = 0.0
                running_regression_loss = 0.0
                running_grad_norm_reg = 0.0
                running_grad_norm_rank = 0.0
                if add_rank_this_epoch:
                    running_ranking_loss = 0.0
                # throughput = 0.0 # throughput is recalculated, not accumulated
                last_log_time = time.time()
                
                # ---- VALIDATION ----
                fabric.barrier()
                val_loss_scalar = 0.0  # init on all ranks so variable exists
                abs_err_mean_scalar = 0.0
                if fabric.global_rank == 0:
                    val_results = run_validation(
                        model, val_loader, base_criterion, device, weighted_loss,
                        add_ranking_loss=add_rank_this_epoch, 
                        spearman_tau=spearman_tau,
                        current_ranking_lambda=dynamic_ranking_lambda
                    )

                    val_loss_scalar = float(val_results['loss'])

                    val_preds = val_results['preds'] 
                    val_labels = val_results['labels']

                    # Check the difference between the predicted and true labels
                    errors = val_preds - val_labels
                    abs_errors = np.abs(errors)

                    abs_err_mean_scalar = float(abs_errors.mean())
                    # Enhanced validation logging to W&B (only on rank 0)
                    val_log_dict = {
                        "val/loss": val_results['loss'],  # Combined loss
                        "val/regression_loss": val_results['regression_loss'],
                        "val/avg_dockq": val_results['avg_dockq'],
                        "val/abs_error": abs_errors,
                        
                        # Distribution metrics:
                        "val/abs_error_histogram": wandb.Histogram(abs_errors, num_bins=512),
                        "val/abs_error_mean": abs_err_mean_scalar,
                        
                        # Quantiles:
                        "val/abs_error_q10": np.percentile(abs_errors, 10),
                        "val/abs_error_q25": np.percentile(abs_errors, 25),
                        "val/abs_error_q50": np.percentile(abs_errors, 50),
                        "val/abs_error_q75": np.percentile(abs_errors, 75),
                        "val/abs_error_q90": np.percentile(abs_errors, 90),
                        "val/abs_error_max": abs_errors.max(),
                        
                        "val/epoch": epoch,
                    }
                    
                    # Add ranking loss metrics if enabled
                    if add_rank_this_epoch:
                        val_log_dict["val/ranking_loss"] = val_results['ranking_loss']
                        val_log_dict["val/reg_vs_rank_loss_ratio"] = val_results['regression_loss'] / (val_results['ranking_loss'] + 1e-8)
                        val_log_dict["val/dynamic_ranking_lambda"] = dynamic_ranking_lambda
                    
                    wandb.log(val_log_dict, step=log_interval_counter)

                # broadcast scalar val_loss to every rank (sum of {x,0,0,...} -> x on all)
                val_loss_t = torch.tensor(val_loss_scalar, device=fabric.device, dtype=torch.float32)
                val_loss_t = fabric.all_reduce(val_loss_t, reduce_op="sum")
                val_loss = float(val_loss_t.item())

                # Now every rank can safely update aggregates
                avg_val_loss_per_epoch += val_loss
                log_step_per_epoch += 1
                
                # Only print on rank 0
                if fabric.global_rank == 0:
                    print(f"Epoch {epoch}/{epochs} step {log_interval_counter%len(train_loader)}/{len(train_loader)} — Train loss: {avg_loss:.8f}")
                
                
                
                #after validation bring the model back to training mode
                fabric.barrier()
                model.train()
        
        avg_val_loss = avg_val_loss_per_epoch / max(log_step_per_epoch, 1)
        if scheduler_type != "OneCycleLR":
            learning_rate_scheduler.step(avg_val_loss)
        
        # Only print on rank 0
        if fabric.global_rank == 0:
            print(f"Epoch {epoch}/{epochs}, Elapsed time: {time.time() - epoch_start:.2f}s, Train loss: {avg_loss:.8f} —  Val loss: {avg_val_loss:.8f} —  Val abs error mean: {abs_errors.mean():.8f}")

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
                  add_ranking_loss=False, spearman_tau=0.02, current_ranking_lambda=1.0):
    """
    Enhanced validation function that computes the same losses as training:
    - Regression loss
    - Ranking loss (if enabled)
    - Combined loss
    - Average DockQ
    """
    model.eval()
    val_preds  = []
    val_labels = []
    val_loss   = 0.0
    val_count  = 0
    
    # Additional loss accumulators
    val_regression_loss = 0.0
    val_ranking_loss = 0.0
    val_avg_dockq = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            feats  = batch['features'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['label'].to(device) # Already logit transformed
            weights= batch['weight'].to(device)  # still available
            complex_id = batch['complex_id']

            # logits = model(feats, lengths)
            # logits = torch.clamp(logits, min=-100.0, max=100.0)
            
            # # =============== REGRESSION LOSS ===============
            # regression_losses = base_criterion(logits, labels)
            # if weighted_loss:
            #     regression_loss = (regression_losses * weights).mean()
            # else:
            #     regression_loss = regression_losses.mean()
            
            z_mu, z_k = model(feats, lengths)
            regression_loss, mu, kappa = beta_regression_loss(labels, z_mu, z_k, T=3.0, eps=1e-3)

            # =============== RANKING LOSS (if enabled) ===============
            base_ranking_loss = 0.0
            # if add_ranking_loss and logits.size(1) > 1:
            #     pred_sig = torch.sigmoid(logits)
            #     true_sig = torch.sigmoid(labels)
            if add_ranking_loss and mu.size(1) > 1:
                pred_sig = mu
                true_sig = labels

                # Compute ranking relevance weight (same as training)
                with torch.no_grad():
                    true_sig_std = true_sig.std(dim=1, keepdim=True) + 1e-6
                    ranking_relevance_weight = torch.clamp(true_sig_std / 0.07, min=0.0, max=1.0)
                
                base_spearman_loss_terms = fisher_spearman_soft_loss(pred_sig, true_sig, tau=spearman_tau)
                base_ranking_loss = (base_spearman_loss_terms * ranking_relevance_weight.squeeze()).mean()
            
            # =============== COMBINED LOSS ===============
            combined_loss = regression_loss
            # if add_ranking_loss and logits.size(1) > 1:
            #     combined_loss = combined_loss + current_ranking_lambda * base_ranking_loss
            if add_ranking_loss and mu.size(1) > 1:
                combined_loss = combined_loss + current_ranking_lambda * base_ranking_loss
            
            # =============== AVERAGE DOCKQ ===============
            avg_dockq = torch.sigmoid(labels).mean().item()
            
            # Accumulate losses
            val_loss += combined_loss.item()
            val_regression_loss += regression_loss.item()
            if add_ranking_loss and logits.size(1) > 1:
                val_ranking_loss += base_ranking_loss.item()
            val_avg_dockq += avg_dockq
            val_count += feats.size(0)

            # convert back labels and logits to 0-1 for metrics
            # Labels are already logit transformed, so apply sigmoid to convert back for comparison
            # labels_original_scale = torch.sigmoid(labels)
            # logits_original_scale = torch.sigmoid(logits)

            # #clipping for extreme sigmoid outputs
            # labels_original_scale = torch.clamp(labels_original_scale, min=1e-7, max=1-1e-7)
            # logits_original_scale = torch.clamp(logits_original_scale, min=1e-7, max=1-1e-7)
            
            # val_preds.extend(logits_original_scale.cpu().numpy().flatten())
            # val_labels.extend(labels_original_scale.cpu().numpy().flatten())

            # metrics: μ is already in 0-1, labels already in 0-1
            val_preds.extend(mu.detach().cpu().numpy().flatten())
            val_labels.extend(labels.cpu().numpy().flatten())

    # Average the losses
    avg_val_loss = val_loss / len(val_loader)
    avg_val_regression_loss = val_regression_loss / len(val_loader)
    avg_val_ranking_loss = val_ranking_loss / len(val_loader) if add_ranking_loss else 0.0
    avg_val_dockq = val_avg_dockq / len(val_loader)
    
    return {
        'loss': avg_val_loss,
        'regression_loss': avg_val_regression_loss,
        'ranking_loss': avg_val_ranking_loss,
        'avg_dockq': avg_val_dockq,
        'preds': np.array(val_preds),
        'labels': np.array(val_labels)
    }

def train_step(model, batch, optimizer, base_criterion, device, 
               weighted_loss, adaptive_weight,
               add_ranking_loss, spearman_tau, 
               current_ranking_lambda,      # Lambda from previous step/initial (EMA smoothed)
               lambda_ema_alpha,            # Smoothing factor for lambda EMA
               fabric=None                  # Lightning Fabric for autocast and backward
               ):
    """
    Fabric-safe training step:
      - single forward (under autocast)
      - two short probe backprops to get ||grad(L_reg)|| and ||grad(L_rank)|| without DDP sync
      - build total loss with current_ranking_lambda
      - real Fabric backward + optimizer step
    """

    def grad_norm_via_fabric_probe(loss_tensor, params, fabric, optimizer):
        """
        Fabric-safe per-loss grad-norm probe:
        - uses fabric.no_backward_sync(model) to avoid DDP all-reduce
        - uses fabric.backward(loss, retain_graph=True) to satisfy Fabric
        - reads .grad, computes global L2, then zeroes grads immediately
        - returns a finite Python float (DDP-averaged)
        """
        if (loss_tensor is None) or (not loss_tensor.requires_grad) or (len(params) == 0):
            return 0.0

        # 1) local backward w/o sync -> fills p.grad locally
        with fabric.no_backward_sync(model):
            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss_tensor, retain_graph=True)

        # 2) compute L2 over current grads in fp32 (ignore NaN/Inf terms)
        total_sq = torch.zeros((), device=params[0].device, dtype=torch.float32)
        for p in params:
            g = p.grad
            if g is not None:
                g32 = g.detach().to(torch.float32)
                g32 = torch.nan_to_num(g32, nan=0.0, posinf=0.0, neginf=0.0)
                total_sq = total_sq + g32.pow(2).sum()
        gn = total_sq.sqrt()

        # 3) agree across ranks (mean is fine; we only need consistency)
        gn = fabric.all_reduce(gn, reduce_op="mean")

        # 4) clean up probe grads to avoid contaminating the real step
        optimizer.zero_grad(set_to_none=True)

        return float(torch.clamp(gn, min=1e-12))

    # --------------- move batch to device ---------------
    feats   = batch['features'].to(device)
    lengths = batch['lengths'].to(device)
    labels  = batch['label'].to(device)
    weights = batch['weight'].to(device)
    # complex_id = batch['complex_id']  # not used here

    avg_dockq = torch.sigmoid(labels).mean().item()
    params = [p for p in model.parameters() if p.requires_grad]

    # wrappers so this function also runs without Fabric (fallback to plain PyTorch)
    autocast_ctx = fabric.autocast()
    def _backward(loss, **kw):
        fabric.backward(loss, **kw)


    optimizer.zero_grad(set_to_none=True)

    # ------------------- forward + base losses -------------------
    def compute_losses():
                   
        # logits = model(feats, lengths)  # [B, K] predictions in logit space

        # # regression loss on logits vs labels (already in logit space)
        # regression_losses = base_criterion(logits, labels)

        # current_regression_weights = weights
        # if adaptive_weight:
        #     with torch.no_grad():
        #         # focus more on extremes (DockQ near 0 or 1)
        #         conf_w = 1.0 + 4.0 * ((torch.sigmoid(labels) < 0.1) | (torch.sigmoid(labels) > 0.9))
        #         current_regression_weights = current_regression_weights * conf_w

        # if weighted_loss:
        #     regression_loss = (regression_losses * current_regression_weights).mean()
        # else:
        #     regression_loss = regression_losses.mean()

        # return logits, regression_loss, current_regression_weights
        z_mu, z_k = model(feats, lengths)  # each [B, K]
        # Beta NLL (returns scalar), also returns μ in [0,1] and κ>0
        regression_loss, mu, kappa = beta_regression_loss(labels, z_mu, z_k, T=3.0, eps=1e-3)

        # If you need per-sample weighting, compute per-sample NLL (mean over K, not reduced):
        # nll = -Beta(mu*kappa, (1-mu)*kappa).log_prob(labels.clamp(1e-3, 1-1e-3))  # [B,K]
        # regression_loss = (nll.mean(dim=1) * weights).mean() if weighted_loss else nll.mean()

        return (mu, regression_loss)  # return μ for ranking and logging


    with autocast_ctx:
        mu, regression_loss = compute_losses()

        # Optional ranking loss (Spearman-soft) if K > 1
        base_ranking_loss_for_log = 0.0
        mean_base_ranking_loss = None
        # if add_ranking_loss and logits.size(1) > 1:
        #     pred_sig = torch.sigmoid(logits)
        #     true_sig = torch.sigmoid(labels)
        if add_ranking_loss and mu.size(1) > 1:
            pred_sig = mu
            true_sig = labels
            with torch.no_grad():
                true_sig_std = true_sig.std(dim=1, keepdim=True) + 1e-6
                ranking_relevance_weight = torch.clamp(true_sig_std / 0.07, min=0.0, max=1.0)
            base_spearman_loss_terms = fisher_spearman_soft_loss(pred_sig, true_sig, tau=spearman_tau)  # [B]
            mean_base_ranking_loss = (base_spearman_loss_terms * ranking_relevance_weight.squeeze()).mean()
            base_ranking_loss_for_log = float(mean_base_ranking_loss.detach())

    grad_norm_reg_val  = grad_norm_via_fabric_probe(regression_loss, params, fabric, optimizer)
    grad_norm_rank_val = grad_norm_via_fabric_probe(mean_base_ranking_loss, params, fabric, optimizer) if (add_ranking_loss and logits.size(1) > 1) else 0.0

    # ------------------- dynamic lambda update (EMA on ideal ratio) -------------------
    updated_lambda_for_next_step = current_ranking_lambda
    if add_ranking_loss and logits.size(1) > 1:

        # compute a safe ratio
        eps_abs = 1e-6
        eps_rel = 1e-3  # relative to reg norm
        denom = max(grad_norm_rank_val, eps_abs, eps_rel * grad_norm_reg_val)

        ratio = grad_norm_reg_val / denom  # >= 1 by construction
        # clamp ratio; keep it sane
        ratio = float(min(max(ratio, 1e-3), 1e3))

        # log-space EMA toward target ratio
        log_lambda   = float(np.log(max(current_ranking_lambda, 1e-8)))
        log_target   = float(np.log(ratio * max(current_ranking_lambda, 1e-8)))
        alpha = lambda_ema_alpha  # e.g. 0.9
        log_new = alpha * log_lambda + (1 - alpha) * log_target

        # optional hysteresis: if rank grad is tiny, decay λ
        tiny_rank = grad_norm_rank_val < max(1e-6, 1e-3 * grad_norm_reg_val)
        if tiny_rank:
            log_new = log_lambda + np.log(0.5)  # decay by 0.5 instead of exploding

        # final clamp for λ
        new_lambda = float(np.exp(log_new))
        new_lambda = float(min(max(new_lambda, 0.2), 10.0))

        updated_lambda_for_next_step = new_lambda

        # Print AFTER clamp so logs match what you actually use
        # print(f"Ideal ratio (clamped): {ratio}")
        # print(f"Current ranking lambda: {current_ranking_lambda}")
        # print(f"Updated lambda for next step: {updated_lambda_for_next_step}")

    # ------------------- final combined backward + step -------------------
    # Build the combined loss using the *current* lambda (as in your code)
    total_combined_loss = regression_loss
    if add_ranking_loss and mu.size(1) > 1:
        total_combined_loss = total_combined_loss + current_ranking_lambda * mean_base_ranking_loss

    optimizer.zero_grad(set_to_none=True)
    _backward(total_combined_loss)   # Fabric handles AMP/DDP

    # If you later switch to GradScaler, call scaler.unscale_(optimizer) before clipping.
    try:
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
    except Exception:
        # Be resilient; clipping is best-effort and shouldn't crash training.
        pass
    
    optimizer.step()

    # ------------------- return (preserve your signature) -------------------
        
    return_values = [
        float(total_combined_loss.detach()),
        feats.size(0),
        avg_dockq,
        float(regression_loss.detach()),
        base_ranking_loss_for_log,
        grad_norm_reg_val,
        grad_norm_rank_val,
        updated_lambda_for_next_step,
    ]

    return tuple(return_values)


if __name__ == '__main__':
    main()
