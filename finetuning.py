#!/usr/bin/env python3
"""
fine_tune_deepset.py

Fine-tune a pretrained DeepSet model on specific DockQ regions (low DockQ or high-variance complexes).
Usage:
    python finetuning.py \
        --config configs/config.yaml \
        --manifest_csv data/manifest.csv \
        --checkpoint model_checkpoint.pt \
        --output_dir out/ \
        --focus low_dockq \
        [--dockq_threshold 0.2] \
        # or for high variance:
        --focus high_variance \
        [--std_threshold 0.15]
"""
import argparse
import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from datetime import datetime
from src.data.dataloader import get_dataloader, get_eval_dataloader
from src.models.deep_set import DeepSet, init_weights
import numpy as np
import time

# Global DockQ bin edges
BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.00]
NUM_BUCKETS = len(BIN_EDGES) - 1

def filter_manifest(df, focus, dockq_thresh, std_thresh):
    if focus == 'low_dockq':
        return df[df['label'] <= dockq_thresh].reset_index(drop=True)
    elif focus == 'high_variance':
        # compute per-complex std and filter
        stats = df.groupby('complex_id')['label'].agg(['std']).reset_index()
        high_var = stats[stats['std'] >= std_thresh]['complex_id']
        high_var_df = df[df['complex_id'].isin(high_var)].reset_index(drop=True)
        # reweight the complexes by buckets

        #count number of complexes in each bucket
        bucket_counts = high_var_df['bucket'].value_counts().to_dict()
        #compute weight for each bucket
        high_var_df['weight_bucket'] = high_var_df['bucket'].map(lambda b: 1.0 / bucket_counts[b])
        #divide the weight_bucket by the number of buckets
        high_var_df['weight_bucket'] = high_var_df['weight_bucket'] / NUM_BUCKETS

        #final weight is product of the two:
        high_var_df['weight'] = high_var_df['weight_bucket']

        return high_var_df
    else:
        raise ValueError(f"Unknown focus type: {focus}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepSet on a target subset of DockQ data"
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--manifest_csv', type=str, required=True, help='Original manifest CSV')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained model checkpoint (.pt)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save fine-tuned checkpoints')
    parser.add_argument('--focus', type=str, required=True, choices=['low_dockq', 'high_variance', 'none'], help='Region to focus on')
    parser.add_argument('--dockq_threshold', type=float, default=0.2, help='Threshold for low DockQ')
    parser.add_argument('--std_threshold', type=float, default=0.15, help='Std threshold for high-variance')
    args = parser.parse_args()

    # load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ensure output
    os.makedirs(args.output_dir, exist_ok=True)

    # load and filter manifest
    df = pd.read_csv(args.manifest_csv)
    if args.focus == 'none':
        #use default manifest
        focused_csv = args.manifest_csv
        print(f"Using default manifest")
    else:
        df_focus = filter_manifest(df[df['split']=='train'], args.focus, args.dockq_threshold, args.std_threshold)
        focused_csv = os.path.join(args.output_dir, f"manifest_{args.focus}.csv")
        df_focus.to_csv(focused_csv, index=False)
        print(f"Filtered {len(df_focus)} training samples for focus={args.focus}")

    # training hyperparams
    batch_size    = cfg['training']['batch_size']
    num_workers   = cfg['training']['num_workers']
    lr            = cfg['training']['lr']
    weight_decay  = cfg['training']['weight_decay']
    weighted_loss = cfg['training'].get('weighted_loss', False)
    epochs        = cfg['training']['epochs']
    samples_per_complex = cfg['num_samples_per_complex']
    
    actual_number_of_epochs = int((len(open(focused_csv).readlines())-1) / (batch_size * samples_per_complex))
    epochs = epochs * actual_number_of_epochs

    # dataloaders: use uniform sampling (no bucket_balance)
    if args.focus == 'none' or args.focus == 'low_dockq':
        train_loader = get_dataloader(
            focused_csv, 'train', batch_size=batch_size,
            num_workers=num_workers,
            samples_per_complex=samples_per_complex,
            bucket_balance=False,
            feature_transform=cfg['data']['feature_transform'],
            seed=cfg['training'].get('seed', None)
        )
    elif args.focus == 'high_variance':
        train_loader = get_dataloader(
            focused_csv, 'train', batch_size=batch_size,
            num_workers=num_workers,
            samples_per_complex=samples_per_complex,
            bucket_balance=True,
            feature_transform=cfg['data']['feature_transform'],
            seed=cfg['training'].get('seed', None)
        )
    val_loader = get_eval_dataloader(
        args.manifest_csv, 'val', batch_size=batch_size,
        num_workers=num_workers,
        feature_transform=cfg['data']['feature_transform'],
        seed=cfg['training'].get('seed', None)
    )

    # model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSet(
        cfg['model']['input_dim'],
        cfg['model']['phi_hidden_dims'],
        cfg['model']['rho_hidden_dims'],
        aggregator=cfg['model']['aggregator']
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=cfg['training']['lr_scheduler_factor'],
        patience=cfg['training']['lr_scheduler_patience'],
        min_lr=2e-6
    )
    criterion = nn.SmoothL1Loss(reduction='none', beta=cfg['training']['smooth_l1_beta'])

    # init W&B
    load_dotenv()
    wandb.login(key=os.getenv('WANDB_API_KEY'), relogin=True)
    run_name = f"FineTune_{args.focus}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb.init(project=os.getenv('WANDB_PROJECT'), name=run_name, config=cfg)
    wandb.watch(model)

    # Initialize checkpoint path and save config
    ckpt_output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(ckpt_output_dir, exist_ok=True)
    with open(os.path.join(ckpt_output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # fine-tuning loop
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, n = 0.0, 0
        
        #average_dockq_per_epoch
        dockq_per_epoch = []

        # ---- TRAINING ----
        for batch in train_loader:
            feats = batch['features'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['label'].to(device)

            dockq_per_epoch.append(labels.cpu().numpy())

            # logit-transform
            eps = 1e-6
            labels = torch.clamp(labels, eps, 1-eps)
            labels = torch.log(labels/(1-labels))

            optimizer.zero_grad()
            logits = model(feats, lengths)
            losses = criterion(logits, labels)
            loss = (losses.mean() if not weighted_loss else (losses*batch['weight'].to(device)).mean())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)
            n += feats.size(0)

        train_loss = total_loss / n
        average_dockq_per_epoch = np.mean(dockq_per_epoch)

        # ---- VALIDATION ----
        model.eval()
        val_preds  = []
        val_labels = []
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                feats = batch['features'].to(device)
                lengths = batch['lengths'].to(device)
                labels = batch['label'].to(device)
                eps = 1e-6
                labels = torch.clamp(labels, eps, 1-eps)
                labels = torch.log(labels/(1-labels))
                logits = model(feats, lengths)
                losses = criterion(logits, labels)
                loss = losses.mean()
                val_loss += loss.item() * feats.size(0)
                vn += feats.size(0)

                # collect predictions & true labels
                #convert back labels and logits to 0 1
                labels = torch.sigmoid(labels)
                logits = torch.sigmoid(logits)

                val_preds.append(logits.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_loss /= vn

        print(f"Epoch {epoch}/{epochs}  Train_Loss={train_loss:.6f}  Val_Loss={val_loss:.6f}")
        wandb.log({'train/loss': train_loss, 'val/loss': val_loss}, step=epoch)

        # log metrics to wandb
        val_preds  = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        errors = val_preds - val_labels
        abs_errors = np.abs(errors)
        wandb.log({
            "train/loss":       train_loss,
            "train/lr":         optimizer.param_groups[0]['lr'],
            "train/average_dockq": average_dockq_per_epoch,

            "val/loss":         val_loss,
            "val/abs_error":    abs_errors,

            # new distribution metrics:
            "val/error_histogram":      wandb.Histogram(errors, num_bins=512),
            "val/abs_error_histogram":  wandb.Histogram(abs_errors, num_bins=512),
            "val/abs_error_mean":       abs_errors.mean(),
        }, step=epoch)
        scheduler.step(val_loss)

        # save checkpoint
        if epoch % cfg['training'].get('save_every', 10) == 0:
            ckpt = os.path.join(ckpt_output_dir, f"finetune_{args.focus}_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt)

    print("Fine-tuning complete.")

if __name__ == '__main__':
    main()
