#!/usr/bin/env python
"""
correlation_analysis.py

Compare your DeepSet model's DockQ predictions and the original
ranking_score against ground truth DockQ (label) and TM score.

Generates four scatter plots and prints a summary of Pearson r and MSE.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.models.deep_set import DeepSet
from src.data.dataloader import get_eval_dataloader

def load_model_and_config(model_path: str, device: torch.device) -> DeepSet:
    """Load a DeepSet model from a .pt plus its config.yaml."""
    model_dir = os.path.dirname(model_path)
    cfg_path = os.path.join(model_dir, "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)["model"]
    model = DeepSet(
        input_dim=cfg["input_dim"],
        phi_hidden_dims=cfg["phi_hidden_dims"],
        rho_hidden_dims=cfg["rho_hidden_dims"],
        aggregator=cfg["aggregator"]
    )
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model

def evaluate_model(model: DeepSet, dataloader, device: torch.device):
    """Run the model on all batches, return flattened (preds, labels)."""
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            feats   = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            labels  = batch["label"].to(device)
            logits  = model(feats, lengths)
            preds   = torch.sigmoid(logits)  # in [0,1]
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)

def make_scatter(x, y, xlabel, ylabel, title, outpath):
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, alpha=0.5, s=20)
    plt.plot([0,1],[0,1], 'k--', alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def pearson_r(x, y):
    return np.corrcoef(x, y)[0,1]

def spearman_r(x, y):
    """Compute Spearman rank correlation (rho)."""
    return spearmanr(x, y)[0]

def make_scatter_by_complex(x, y, complex_ids, xlabel, ylabel, title, outpath):
    """Scatter x vs y colored by complex_id."""
    unique = list(dict.fromkeys(complex_ids))
    idx_map = {cid:i for i,cid in enumerate(unique)}
    colors = [idx_map[c] for c in complex_ids]

    plt.figure(figsize=(6,6))
    sc = plt.scatter(x, y, c=colors, cmap='tab20', s=20, alpha=0.6)
    plt.plot([0,1], [0,1], 'k--', alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=10)

    if len(unique) <= 20:
        cbar = plt.colorbar(sc, ticks=range(len(unique)))
        cbar.ax.set_yticklabels(unique)
    else:
        plt.colorbar(sc, label='complex_id index')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Correlation analysis: model vs ranking")
    p.add_argument("--model_path",      required=True, help=".pt file of your trained model")
    p.add_argument("--manifest",        default="/proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/data/manifest_with_ptm_no_normalization.csv", help="CSV manifest with ranking_score, tm_normalized, label, split")
    p.add_argument("--split",           default="test",  help="Which split to use (train/test)")
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--num_workers",     type=int, default=2)
    p.add_argument("--output_dir",      default="correlation_reports")
    args = p.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load model
    model = load_model_and_config(args.model_path, device)

    # 2) Prepare dataloader & run model
    dataloader = get_eval_dataloader(
        args.manifest,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature_transform=True
    )
    preds, labels = evaluate_model(model, dataloader, device)

    # 3) Load manifest to get ranking_score & tm_normalized
    df = pd.read_csv(args.manifest)
    df = df[df["split"] == args.split].reset_index(drop=True)
    ranking     = df["ranking_score"].values
    tm_score    = df["tm_normalized"].values
    dockq       = df["label"].values   # should match `labels`
    complex_ids = df["complex_id"].tolist()

    assert len(preds) == len(ranking) == len(tm_score) == len(dockq) == len(complex_ids), \
        "Lengths mismatch between model output and manifest!"

    # 4) Make four standard scatter plots
    make_scatter(preds, dockq,
                 xlabel="Model prediction (DockQ)",
                 ylabel="True DockQ",
                 title="Model vs DockQ",
                 outpath=os.path.join(args.output_dir, "model_vs_dockq.png"))

    make_scatter(preds, tm_score,
                 xlabel="Model prediction (DockQ)",
                 ylabel="TM normalized",
                 title="Model vs TM score",
                 outpath=os.path.join(args.output_dir, "model_vs_tm.png"))

    make_scatter(ranking, dockq,
                 xlabel="Ranking confidence",
                 ylabel="True DockQ",
                 title="Ranking score vs DockQ",
                 outpath=os.path.join(args.output_dir, "rank_vs_dockq.png"))

    make_scatter(ranking, tm_score,
                 xlabel="Ranking confidence",
                 ylabel="TM normalized",
                 title="Ranking score vs TM score",
                 outpath=os.path.join(args.output_dir, "rank_vs_tm.png"))

    # 4b) Make the same four plots but colored by complex_id
    make_scatter_by_complex(preds, dockq, complex_ids,
                            xlabel="Model prediction (DockQ)",
                            ylabel="True DockQ",
                            title="Model vs DockQ by complex",
                            outpath=os.path.join(args.output_dir, "model_vs_dockq_by_complex.png"))

    make_scatter_by_complex(preds, tm_score, complex_ids,
                            xlabel="Model prediction (DockQ)",
                            ylabel="TM normalized",
                            title="Model vs TM score by complex",
                            outpath=os.path.join(args.output_dir, "model_vs_tm_by_complex.png"))

    make_scatter_by_complex(ranking, dockq, complex_ids,
                            xlabel="Ranking confidence",
                            ylabel="True DockQ",
                            title="Ranking score vs DockQ by complex",
                            outpath=os.path.join(args.output_dir, "rank_vs_dockq_by_complex.png"))

    make_scatter_by_complex(ranking, tm_score, complex_ids,
                            xlabel="Ranking confidence",
                            ylabel="TM normalized",
                            title="Ranking score vs TM score by complex",
                            outpath=os.path.join(args.output_dir, "rank_vs_tm_by_complex.png"))

    # 5) Compute global metrics (Pearson, Spearman, MSE)
    metrics = {
        "Model↔DockQ": {
            "pearson":  pearson_r(preds, dockq),
            "spearman": spearman_r(preds, dockq),
            "mse":      ((preds-dockq)**2).mean()
        },
        "Model↔TM": {
            "pearson":  pearson_r(preds, tm_score),
            "spearman": spearman_r(preds, tm_score),
            "mse":      ((preds-tm_score)**2).mean()
        },
        "Rank↔DockQ": {
            "pearson":  pearson_r(ranking, dockq),
            "spearman": spearman_r(ranking, dockq),
            "mse":      ((ranking-dockq)**2).mean()
        },
        "Rank↔TM": {
            "pearson":  pearson_r(ranking, tm_score),
            "spearman": spearman_r(ranking, tm_score),
            "mse":      ((ranking-tm_score)**2).mean()
        },
    }

    # 6) Print a short report including Spearman
    print("\n=== Correlation & MSE report ===")
    for name, m in metrics.items():
        print(
            f"{name:15s}  Pearson r = {m['pearson']:6.3f}    "
            f"Spearman ρ = {m['spearman']:6.3f}    MSE = {m['mse']:8.4f}"
        )

    print("\nSummary:")
    # Compare which is better for DockQ
    if metrics["Model↔DockQ"]["pearson"] > metrics["Rank↔DockQ"]["pearson"]:
        print("‣ Your model correlates better with true DockQ than the original ranking score.")
    else:
        print("‣ The original ranking score correlates better with true DockQ than your model.")

    # Compare which is better for TM
    if metrics["Model↔TM"]["pearson"] > metrics["Rank↔TM"]["pearson"]:
        print("‣ Your model correlates better with TM normalized than the original ranking score.")
    else:
        print("‣ The original ranking score correlates better with TM normalized than your model.")

    # 7) Per-complex Pearson & Spearman vs DockQ
    unique_cids = df["complex_id"].unique()
    pm_model = []
    pr_rank  = []
    sm_model = []
    sr_rank  = []
    for cid in unique_cids:
        mask = df["complex_id"] == cid
        y = dockq[mask]
        m_ = preds[mask]
        r_ = ranking[mask]
        if len(y) > 1:
            pm_model.append(pearson_r(m_, y))
            pr_rank.append(pearson_r(r_, y))
            sm_model.append(spearman_r(m_, y))
            sr_rank.append(spearman_r(r_, y))
        else:
            pm_model.append(np.nan)
            pr_rank.append(np.nan)
            sm_model.append(np.nan)
            sr_rank.append(np.nan)

    # Bar chart: per-complex Pearson
    ind = np.arange(len(unique_cids))
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(ind,       pm_model, width, label="Model")
    plt.bar(ind+width, pr_rank,  width, label="Ranking")
    plt.xticks(ind+width/2, unique_cids, rotation=90)
    plt.ylabel("Pearson r")
    plt.title("Per-complex Pearson correlation vs DockQ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pearson_per_complex.png"), dpi=150)
    plt.close()

    # Bar chart: per-complex Spearman
    plt.figure(figsize=(10,5))
    plt.bar(ind,       sm_model, width, label="Model")
    plt.bar(ind+width, sr_rank,  width, label="Ranking")
    plt.xticks(ind+width/2, unique_cids, rotation=90)
    plt.ylabel("Spearman ρ")
    plt.title("Per-complex Spearman correlation vs DockQ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "spearman_per_complex.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
