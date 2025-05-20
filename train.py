#!/usr/bin/env python3
"""
train_deepset.py

Train a DeepSet on antibody–antigen interchain PAE features using our new sampler & weighting.
Usage:
    python train_deepset.py --config configs/config.yaml --output_dir out/
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

from src.data.dataloader import (
    get_dataloader,
    get_eval_dataloader
)
from src.models.deep_set import DeepSet, init_weights

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
    
    # Parse known arguments (config path, output_dir, resume)
    # and collect unknown arguments for OmegaConf to parse as overrides
    args, unknown_cli_args = parser.parse_known_args()

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

    # 1.a) Initialize W&B run
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

    # actual number of epochs
    print(f"Number of lines in manifest: {len(open(manifest_csv).readlines())-1}")
    print(f"Batch size: {batch_size}")
    print(f"Samples per complex: {samples_per_complex}")
    epochs = cfg.training.epochs
    
    print(f"Number of epochs: {epochs}")

    lr                  = cfg.training.lr
    weight_decay        = cfg.training.weight_decay
    seed                = cfg.training.get('seed', None)
    weighted_loss       = cfg.training.get('weighted_loss', False)

    feature_transform   = cfg.data.feature_transform
    feature_centering   = cfg.data.get('feature_centering', False)
    print(f"Feature transform: {feature_transform}")
    print(f"Feature centering: {feature_centering}")
    
    # adaptive weight: focus more on extreme targets (DockQ near 0 or 1)
    adaptive_weight = cfg.training.get('adaptive_weight', False)
    if adaptive_weight:
        print("Using adaptive weight")
    else:
        print("Not using adaptive weight")

    # 3) DataLoaders
    #    - train: with our chosen sampler
    train_loader = get_dataloader(
        manifest_csv,
        split='train',
        batch_size=batch_size,
        num_workers=num_workers,
        samples_per_complex=samples_per_complex,
        bucket_balance=bucket_balance,
        feature_transform=feature_transform,
        feature_centering=feature_centering,
        seed=seed
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
        seed=seed
    )

    # Model, device, optimizer
    input_dim        = cfg.model.input_dim
    phi_hidden_dims  = cfg.model.phi_hidden_dims
    rho_hidden_dims  = cfg.model.rho_hidden_dims
    aggregator       = cfg.model.aggregator

    model  = DeepSet(input_dim, phi_hidden_dims, rho_hidden_dims, aggregator=aggregator)
    model.apply(init_weights)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 

    # Loss: use weighted Huber (Smooth L1) for robust regression
    # reduction='none' gives you one loss per sample so you can apply your weights
    base_criterion = nn.SmoothL1Loss(reduction='none', beta=cfg.training.smooth_l1_beta)

    # Ensure output_dir is set, either from config or CLI, before it's used.
    if not hasattr(cfg, 'output_dir') or cfg.output_dir is None:
        raise ValueError("output_dir must be specified either in the config file or via --output_dir CLI argument.")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Replace the summary() call with:
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

    print(f"Learning rate scheduler: {learning_rate_scheduler_str}")

    #phi_hidden_dims and rho_hiddens_dims to string
    phi_hidden_dims_str = '_'.join(str(dim) for dim in phi_hidden_dims)
    rho_hidden_dims_str = '_'.join(str(dim) for dim in rho_hidden_dims)
    name = f"DeepSet_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_weighted_loss_{weighted_loss}_aggregator_{aggregator}_lr_scheduler_{learning_rate_scheduler_str}"

    # Initialize W&B run
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"), 
        config=OmegaConf.to_container(cfg, resolve=True), # Convert OmegaConf to dict for wandb
        name=name
    )

    wandb.watch(model, log="all", log_graph=True)
    # Run a forward pass with the sample input (this will be captured by wandb.watch)
    # Create a sample input for wandb to trace the model
    sample_batch_size = batch_size
    sample_complex_size = samples_per_complex
    sample_seq_len = 20
    sample_input_dim = cfg.model.input_dim
    sample_x = torch.zeros((sample_batch_size, sample_complex_size, sample_seq_len, sample_input_dim), device=device)
    sample_lengths = torch.full((sample_batch_size, sample_complex_size), sample_seq_len, device=device)

    with torch.no_grad():
        _ = model(sample_x, sample_lengths)

    # Initialize checkpoint path and save config
    ckpt_output_dir = os.path.join(cfg.output_dir, name)
    os.makedirs(ckpt_output_dir, exist_ok=True)
    with open(os.path.join(ckpt_output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f) # Use OmegaConf to save
    
    #clean up memory
    torch.cuda.empty_cache()

    #before training, run a forward pass with the biggest complex in the train set
    #to check if the model does not throw a memory error
    biggest_batch = None
    for batch in train_loader:
        #find the batch with the biggest complex
        if biggest_batch is None or batch['lengths'].max() > biggest_batch['lengths'].max():
            biggest_batch = batch
    feats  = biggest_batch['features'].to(device)
    lengths = biggest_batch['lengths'].to(device)
    labels = biggest_batch['label'].to(device)
    weights= biggest_batch['weight'].to(device)
    complex_id = biggest_batch['complex_id']
    # print(f"feats shape: {feats.shape}")
    # print(f"lengths shape: {lengths.shape}")
    # print(f"labels shape: {labels.shape}")
    # print(f"weights shape: {weights.shape}")
    # print(f"complex_id shape: {complex_id.shape}")
    _ = model(feats, lengths)
    
    #clean up memory
    torch.cuda.empty_cache()
    
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
    log_interval = 150
    log_interval_counter = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    num_batches = 0
    last_log_time = time.time()
    running_avg_dockq = 0.0
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_start = time.time()
        
        avg_val_loss_per_epoch = 0.0
        log_step_per_epoch = 0
        avg_loss = 0.0  # Initialize avg_loss here
        abs_errors = np.array([0.0])  # Initialize abs_errors here

        for batch in train_loader:
            #print the batch size
            # print(f"Batch size: {batch['features'].shape[0]}")
            #print the number of complexes in the batch
            # print(f"Number of complexes in the batch: {len(batch['complex_id'])}")
            # print(f"Batch: {batch}")
            loss, grad_norm, batch_size, avg_dockq = train_step(
                model, batch, optimizer, base_criterion, device, weighted_loss, adaptive_weight
            )

            # Update learning rate
            if scheduler_type == "OneCycleLR":
                learning_rate_scheduler.step()
                
            running_loss += loss
            running_grad_norm += grad_norm
            log_interval_counter += 1
            running_avg_dockq += avg_dockq
            # Log every log_interval steps
            if log_interval_counter % log_interval == 0:
                avg_loss = running_loss / log_interval_counter
                avg_grad_norm = running_grad_norm / log_interval_counter
                avg_dockq = running_avg_dockq / log_interval
                elapsed = time.time() - last_log_time
                throughput = log_interval / elapsed if elapsed > 0 else 0.0
                
                #log training metrics to W&B
                wandb.log({
                    "train/loss": avg_loss,
                    "train/grad_norm": avg_grad_norm,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/throughput": throughput,
                    "train/epoch": epoch,
                    "train/avg_dockq": avg_dockq,
                }, step=log_interval_counter)

                # Reset running stats for next interval
                running_loss = 0.0
                running_grad_norm = 0.0
                running_avg_dockq = 0.0
                throughput = 0.0
                last_log_time = time.time()

                #run validation as part of the logging

                # ---- VALIDATION ----
                val_loss, val_preds, val_labels = run_validation(
                    model, val_loader, base_criterion, device, weighted_loss
                )

                # Check the difference between the predicted and true labels
                errors = val_preds - val_labels
                abs_errors = np.abs(errors)

                # Log validation metrics to W&B
                wandb.log({
                    "val/loss":         val_loss,
                    "val/abs_error":    abs_errors,

                    # new distribution metrics:
                    "val/error_histogram":      wandb.Histogram(errors, num_bins=512),
                    "val/abs_error_histogram":  wandb.Histogram(abs_errors, num_bins=512),
                    "val/abs_error_mean":       abs_errors.mean(),

                    # optionally, some quantiles for quick reference:
                    "val/abs_error_q10": np.percentile(abs_errors, 10),
                    "val/abs_error_q50": np.percentile(abs_errors, 50),
                    "val/abs_error_q90": np.percentile(abs_errors, 90),
                }, step=log_interval_counter)

                avg_val_loss_per_epoch += val_loss
                log_step_per_epoch += 1
                
                print(f"Epoch {epoch}/{epochs} step {log_interval_counter}/{len(train_loader)} — Train loss: {avg_loss:.8f}")
                
                
                
                #after validation bring the model back to training mode
                model.train()
        
        avg_val_loss = avg_val_loss_per_epoch / max(log_step_per_epoch, 1)
        if scheduler_type != "OneCycleLR":
            learning_rate_scheduler.step(avg_val_loss)
        print(f"Epoch {epoch}/{epochs}, Elapsed time: {time.time() - epoch_start:.2f}s, Train loss: {avg_loss:.8f} —  Val loss: {avg_val_loss:.8f} —  Val abs error mean: {abs_errors.mean():.8f}")

        

        # 7) Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(ckpt_output_dir, f"checkpoint_epoch{epoch}.pt"))

    print("Training complete.")

def run_validation(model, val_loader, base_criterion, device, weighted_loss):
    model.eval()
    val_preds  = []
    val_labels = []
    val_loss   = 0.0
    val_count  = 0
    with torch.no_grad():
        for batch in val_loader:
            feats  = batch['features'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['label'].to(device) # Already logit transformed
            weights= batch['weight'].to(device)  # still available
            complex_id = batch['complex_id']

            # transform labels via a clipped logit function
            # epsilon = 1e-6
            # labels = torch.clamp(labels, min=epsilon, max=1-epsilon)
            # labels = torch.log(labels / (1-labels)) # Already done in DataLoader

            logits = model(feats, lengths)
            losses = base_criterion(logits, labels)
            if weighted_loss:
                loss = (losses * weights).mean()
            else:
                loss = losses.mean()

            val_loss += loss.item()
            val_count += feats.size(0)

            # convert back labels and logits to 0-1 for metrics
            # Labels are already logit transformed, so apply sigmoid to convert back for comparison
            labels_original_scale = torch.sigmoid(labels)
            logits_original_scale = torch.sigmoid(logits)

            val_preds.append(logits_original_scale.cpu().numpy())
            val_labels.append(labels_original_scale.cpu().numpy())

    val_loss /= val_count # val_loss should be divided by number of samples, not batches
    val_preds = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)
    return val_loss, val_preds, val_labels

def train_step(model, batch, optimizer, base_criterion, device, weighted_loss, adaptive_weight):
    feats  = batch['features'].to(device)
    lengths = batch['lengths'].to(device)
    labels = batch['label'].to(device) # Already logit transformed
    weights= batch['weight'].to(device)
    complex_id = batch['complex_id']

    avg_dockq = torch.sigmoid(labels).mean().item() # Calculate avg_dockq on original scale for logging
    # Transform labels via a clipped logit function
    # epsilon = 1e-6
    # labels = torch.clamp(labels, min=epsilon, max=1-epsilon)
    # labels = torch.log(labels / (1-labels)) # Already done in DataLoader

    optimizer.zero_grad()
    logits = model(feats, lengths)
    losses = base_criterion(logits, labels)

    if adaptive_weight:
        # Compute adaptive weight: focus more on extreme targets (DockQ near 0 or 1)
        with torch.no_grad():
            # Compute confidence weight: focus more on extreme targets (DockQ near 0 or 1 do this per label)
            # adaptive weight should operate on the logit-transformed labels
            confidence_weight = 1.0 + 4.0 * ((torch.sigmoid(labels) < 0.1) | (torch.sigmoid(labels) > 0.9))
            # print(f"Confidence weight: {confidence_weight}")
            # print(f"Labels: {labels}")
        
        # apply confidence weight to weights
        weights = weights * confidence_weight
    # print(f"Weights: {weights}, weights shape: {weights.shape}")
    # print(f"Losses: {losses}, losses shape: {losses.shape}")

    if weighted_loss:
        loss   = (losses * weights).mean()
    else:
        loss   = losses.mean()

    loss.backward()
    optimizer.step()

    # Compute grad norm
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    return loss.item(), grad_norm, feats.size(0), avg_dockq

if __name__ == '__main__':
    main()
