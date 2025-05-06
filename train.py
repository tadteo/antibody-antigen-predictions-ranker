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
from src.data.dataloader import (
    get_dataloader,
    get_eval_dataloader
)
from src.models.deep_set import DeepSet, init_weights

def main():
    parser = argparse.ArgumentParser(
        description="Train DeepSet model with per-sample weighting & advanced sampling"
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--manifest_csv', type=str, required=True,
        help='Path to manifest CSV file'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint file to resume training from'
    )
    args = parser.parse_args()

    # 1) Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1.a) Initialize W&B run
    load_dotenv()  # will read .env in cwd
    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
    
    # 2) Paths & hyperparams
    manifest_csv        = args.manifest_csv
    batch_size          = cfg['training']['batch_size']
    num_workers         = cfg['training']['num_workers']

    # sampling choices
    samples_per_complex = cfg['num_samples_per_complex']
    bucket_balance      = cfg['training'].get('bucket_balance', False)

    # number of epochs should allow to theoretically see al the samples at least once
    # so since in each epoch we sample just M samples per complex, then the number of epochs should be
    # len(manifest_csv) / (batch_size * M)) * actual_number_of_epochs

    # actual number of epochs
    print(f"Number of lines in manifest: {len(open(manifest_csv).readlines())-1}")
    print(f"Batch size: {batch_size}")
    print(f"Samples per complex: {samples_per_complex}")
    actual_number_of_epochs = int((len(open(manifest_csv).readlines())-1) / (batch_size * samples_per_complex))
    print(f"Actual number of epochs: {actual_number_of_epochs}")
    epochs              = cfg['training']['epochs'] * actual_number_of_epochs
    
    print(f"Number of epochs: {epochs}")

    lr                  = cfg['training']['lr']
    weight_decay        = cfg['training']['weight_decay']
    seed                = cfg['training'].get('seed', None)
    weighted_loss       = cfg['training'].get('weighted_loss', False)

    feature_transform   = cfg['data']['feature_transform']
    print(f"Feature transform: {feature_transform}")


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
        seed=seed
    )
    #    - val: sequential, no special sampling
    val_loader = get_eval_dataloader(
        manifest_csv,
        split='val',
        batch_size=batch_size,
        num_workers=num_workers,
        feature_transform=feature_transform,
        seed=seed
    )

    # 4) Model, device, optimizer
    input_dim        = cfg['model']['input_dim']
    phi_hidden_dims  = cfg['model']['phi_hidden_dims']
    rho_hidden_dims  = cfg['model']['rho_hidden_dims']
    aggregator       = cfg['model']['aggregator']

    model  = DeepSet(input_dim, phi_hidden_dims, rho_hidden_dims, aggregator=aggregator)
    model.apply(init_weights)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 

    # Loss: use weighted Huber (Smooth L1) for robust regression
    # reduction='none' gives you one loss per sample so you can apply your weights
    base_criterion = nn.SmoothL1Loss(reduction='none', beta=cfg['training']['smooth_l1_beta'])

    os.makedirs(args.output_dir, exist_ok=True)

    # Replace the summary() call with:
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters())}")

    # Create a sample input for wandb to trace the model
    sample_batch_size = 1
    sample_seq_len = 20
    sample_input_dim = 3
    sample_x = torch.zeros((sample_batch_size, sample_seq_len, sample_input_dim), device=device)
    sample_lengths = torch.full((sample_batch_size,), sample_seq_len, device=device)

    # Initialize learning rate scheduler
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg['training']['lr_scheduler_factor'],
        patience=cfg['training']['lr_scheduler_patience'],
        min_lr=cfg['training']['min_lr']
    )

    #learning rate scheduler to string
    learning_rate_scheduler_str = f"ReduceLROnPlateau_factor_{learning_rate_scheduler.factor}_patience_{learning_rate_scheduler.patience}"

    #phi_hidden_dims and rho_hiddens_dims to string
    phi_hidden_dims_str = '_'.join(str(dim) for dim in phi_hidden_dims)
    rho_hidden_dims_str = '_'.join(str(dim) for dim in rho_hidden_dims)
    name = f"DeepSet_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_logit_transform_encode_features_{feature_transform}_phi_hidden_dims_{phi_hidden_dims_str}_rho_hidden_dims_{rho_hidden_dims_str}_seed_{seed}_samples_per_complex_{samples_per_complex}_aggregator_{aggregator}_lr_scheduler_{learning_rate_scheduler_str}"

    # Initialize W&B run
    wandb.init(
        project=os.getenv("WANDB_PROJECT"), 
        config=cfg,
        name=name
    )

    wandb.watch(model, log="all", log_graph=True)
    # Run a forward pass with the sample input (this will be captured by wandb.watch)
    with torch.no_grad():
        _ = model(sample_x, sample_lengths)

    # Initialize checkpoint path and save config
    ckpt_output_dir = os.path.join(args.output_dir, name)
    os.makedirs(ckpt_output_dir, exist_ok=True)
    with open(os.path.join(ckpt_output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    
    #before training, run a forward pass with the biggest complex in the train set
    #to check if the model does not throw a memory error
    for batch in train_loader:
        feats  = batch['features'].to(device)
        lengths = batch['lengths'].to(device)
        labels = batch['label'].to(device)
        weights= batch['weight'].to(device)
        _ = model(feats, lengths)

    # --- resume logic: load checkpoint if requested and bump start_epoch ---
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
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
            start_epoch = 1820

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        # ── track training time & grad norms ───────────────────────────────────
        epoch_start    = time.time()
        total_grad_norm = 0.0
        num_batches     = 0
        running_loss = 0.0
        num_samples  = 0

        #average_dockq_per_epoch
        dockq_per_epoch = []

        # ---- TRAINING ----
        for batch in train_loader:
            # unpack the dict that our Dataset returns
            feats  = batch['features'].to(device)  # [B, 4]
            lengths = batch['lengths'].to(device)  # [B]
            labels = batch['label'].to(device)     # [B]
            weights= batch['weight'].to(device)    # [B]

            #debug dataloader
            # print(feats.shape)
            # print(labels.shape)
            # print(weights.shape)
        
            dockq_per_epoch.append(labels.cpu().numpy())


            #tranform labels via a clipped logit function
            epsilon = 1e-6
            labels = torch.clamp(labels, min=epsilon, max=1-epsilon)
            labels = torch.log(labels / (1-labels))

            # zero the parameter gradients
            optimizer.zero_grad()
            logits = model(feats, lengths)                  # [B]

            # per-sample Huber (Smooth L1), then weight, then average
            # logits: [B], labels: [B], weights: [B]
            if weighted_loss:
                losses = base_criterion(logits, labels)
                loss   = (losses * weights).mean()
            else:
                loss   = base_criterion(logits, labels).mean()

            loss.backward()
            # accumulate gradient‐norm for this batch
            sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    sq += p.grad.data.norm(2).item() ** 2
            total_grad_norm += sq ** 0.5
            num_batches     += 1
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            num_samples  += feats.size(0)

        average_dockq_per_epoch = np.mean(dockq_per_epoch)

        train_loss = running_loss / num_samples
        # compute average grad‐norm & throughput for the epoch
        avg_grad_norm = total_grad_norm / num_batches if num_batches else 0.0
        elapsed       = time.time() - epoch_start
        throughput    = num_samples / elapsed if elapsed > 0 else 0.0

        # ---- VALIDATION ----
        model.eval()
        val_preds  = []
        val_labels = []
        val_loss   = 0.0
        val_count  = 0
        with torch.no_grad():
            for batch in val_loader:
                feats  = batch['features'].to(device)
                lengths = batch['lengths'].to(device)
                labels = batch['label'].to(device)
                weights= batch['weight'].to(device)  # still available

                #tranform labels via a clipped logit function
                epsilon = 1e-6
                labels = torch.clamp(labels, min=epsilon, max=1-epsilon)
                labels = torch.log(labels / (1-labels))

                logits = model(feats, lengths)
                losses = base_criterion(logits, labels)
                # per-sample Huber (Smooth L1), then weight, then average
                # logits: [B], labels: [B], weights: [B]
                if weighted_loss:
                    loss   = (losses * weights).mean()
                else:
                    loss   = losses.mean()

                val_loss += loss.item() * feats.size(0)
                val_count+= feats.size(0)
                # collect predictions & true labels
                #convert back labels and logits to 0 1
                labels = torch.sigmoid(labels)
                logits = torch.sigmoid(logits)

                val_preds.append(logits.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_loss /= val_count

        print(f"Epoch {epoch}/{epochs} —  Train loss: {train_loss:.8f} —  Val loss: {val_loss:.8f}")
        
        # average DockQ on val set
        val_preds  = np.concatenate(val_preds) #
        val_labels = np.concatenate(val_labels)
        # Check the difference between the predicted and true labels


        errors = val_preds - val_labels
        abs_errors = np.abs(errors)
        
        #squeeze all errors above 10 nad below -10 to 10
        errors = np.clip(errors, -10, 10)
        abs_errors = np.clip(abs_errors, -10, 10)

        # 6.a) Log metrics to W&B
        wandb.log({
            "train/loss":       train_loss,
            "train/grad_norm":  avg_grad_norm,
            "train/lr":         optimizer.param_groups[0]['lr'],
            "train/throughput": throughput,
            "train/average_dockq": average_dockq_per_epoch,

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
        }, step=epoch)

        # Update learning rate
        learning_rate_scheduler.step(val_loss)

        # 7) Save checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(ckpt_output_dir, f"checkpoint_epoch{epoch}.pt"))

    print("Training complete.")

if __name__ == '__main__':
    main()
