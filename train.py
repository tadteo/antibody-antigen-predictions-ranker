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
from torchsummary import summary
import wandb
from dotenv import load_dotenv
import time
import numpy as np
from datetime import datetime
from src.data.dataloader import (
    get_dataloader,
    get_eval_dataloader
)
from src.models.deep_set import DeepSet

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
    seed                = cfg['training'].get('seed', None)


    


    # 3) DataLoaders
    #    - train: with our chosen sampler
    train_loader = get_dataloader(
        manifest_csv,
        split='train',
        batch_size=batch_size,
        num_workers=num_workers,
        samples_per_complex=samples_per_complex,
        bucket_balance=bucket_balance,
        seed=seed
    )
    #    - val: sequential, no special sampling
    val_loader = get_eval_dataloader(
        manifest_csv,
        split='val',
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 4) Model, device, optimizer
    input_dim        = cfg['model']['input_dim']
    phi_hidden_dims  = cfg['model']['phi_hidden_dims']
    rho_hidden_dims  = cfg['model']['rho_hidden_dims']
    aggregator       = cfg['model']['aggregator']

    model  = DeepSet(input_dim, phi_hidden_dims, rho_hidden_dims, aggregator=aggregator)
    # 4.a) Watch model for gradients/parameters
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) 

    # 5) Loss: we'll compute per-sample losses, then weight them
    base_criterion = nn.BCEWithLogitsLoss(reduction='none')

    os.makedirs(args.output_dir, exist_ok=True)

    # wandb plot model architecture 
    #    input_size = (batch_size, set_size, input_dim)
    #    use batch_size=1 just for shape inference
    model_info = summary(model,
        input_size=(20, 3),
        batch_size=1,
        device=str(device)
    )

    # Initialize W&B run
    wandb.init(
        project=os.getenv("WANDB_PROJECT"), 
        config=cfg,
        name=f"DeepSet_aggregator_{aggregator}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_batch_size_{batch_size}_samples_per_complex_{samples_per_complex}_bucket_balance_{bucket_balance}_seed_{seed}"
    )

    wandb.watch(model, log="all")

    # 4) Convert to HTML and log
    html = "<pre>" + str(model_info) + "</pre>"
    wandb.log({
        "model_summary": wandb.Html(html)
    })
    

    # 6) Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        # ── track training time & grad norms ───────────────────────────────────
        epoch_start    = time.time()
        total_grad_norm = 0.0
        num_batches     = 0
        running_loss = 0.0
        num_samples  = 0

        # ---- TRAINING ----
        for batch in train_loader:
            # unpack the dict that our Dataset returns
            feats  = batch['features'].to(device)  # [B, 4]
            labels = batch['label'].to(device)     # [B]
            weights= batch['weight'].to(device)    # [B]

            #debug dataloader
            # print(feats.shape)
            # print(labels.shape)
            # print(weights.shape)

            # zero the parameter gradients
            optimizer.zero_grad()
            logits = model(feats)                  # [B]

            # per-sample BCE, then weight, then average
            losses = base_criterion(logits, labels)
            loss   = (losses * weights).mean()

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
                labels = batch['label'].to(device)
                weights= batch['weight'].to(device)  # still available

                logits = model(feats)
                losses = base_criterion(logits, labels)
                # here you can choose to weight val as well, or just mean:
                loss   = (losses * weights).mean()

                val_loss += loss.item() * feats.size(0)
                val_count+= feats.size(0)
                # collect predictions & true labels
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_loss /= val_count

        print(f"Epoch {epoch}/{epochs} —  Train loss: {train_loss:.4f} —  Val loss: {val_loss:.4f}")
        
        # average DockQ on val set
        val_preds  = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        # Check the difference between the predicted and true labels
        val_dockq_diff = float(abs(val_preds - val_labels).mean())

        # 6.a) Log metrics to W&B
        wandb.log({
            "train/loss":       train_loss,
            "train/grad_norm":  avg_grad_norm,
            "train/lr":         optimizer.param_groups[0]['lr'],
            "train/throughput": throughput,
            "val/loss":         val_loss,
            "val/dockq_diff":   val_dockq_diff
        }, step=epoch)

        # 7) Save checkpoint
        if epoch % 20 == 0:
            ckpt_path = os.path.join(args.output_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), f"model_epoch{epoch}.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    print("Training complete.")

if __name__ == '__main__':
    main()
