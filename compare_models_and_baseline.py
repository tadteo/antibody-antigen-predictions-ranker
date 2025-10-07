#!/usr/bin/env python
"""
compare_models_and_baseline.py

Simplified tool to compare trained models against baseline ranking scores.

Core functionality:
- Load and evaluate multiple models
- Compare against baseline (AlphaFold ranking confidence)
- Compute correlations and Top-K metrics (K=1,3)
- Generate comparison plots and reports
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
from scipy.stats import spearmanr, pearsonr as scipy_pearsonr
from collections import defaultdict, OrderedDict
import shutil
from tqdm import tqdm

try:
    from src.models.deep_set import DeepSet
    from src.data.dataloader import get_eval_dataloader
except ImportError:
    print("Warning: Could not import from src. Ensure PYTHONPATH is set correctly.")
    raise


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model_and_config(model_path: str, device: torch.device):
    """Load model checkpoint and its config file."""
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config.get('model', {})
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Auto-detect input dimension from checkpoint
    if "phi.0.weight" in state_dict:
        actual_input_dim = state_dict["phi.0.weight"].shape[1]
        model_cfg['input_dim'] = actual_input_dim
    
    model = DeepSet(
        input_dim=model_cfg.get('input_dim'),
        phi_hidden_dims=model_cfg.get('phi_hidden_dims'),
        rho_hidden_dims=model_cfg.get('rho_hidden_dims'),
        aggregator=model_cfg.get('aggregator')
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, config


def evaluate_model(model: torch.nn.Module, dataloader, device: torch.device):
    """Run model inference and return predictions, labels, and complex IDs."""
    all_preds = []
    all_labels = []
    all_complex_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            feats = batch['features'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch["label"].to(device)
            
            batch_complex_ids = batch['complex_id']
            ids_repeated = np.repeat(batch_complex_ids[:, 0], batch_complex_ids.shape[1])
            all_complex_ids.extend(ids_repeated.tolist())
            
            logits = model(feats, lengths)
            preds = torch.sigmoid(logits)
            labels = torch.sigmoid(labels)
            
            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_labels.append(labels.cpu().numpy().reshape(-1))
    
    predictions = np.concatenate(all_preds) if all_preds else np.array([])
    true_labels = np.concatenate(all_labels) if all_labels else np.array([])
    
    return predictions, true_labels, all_complex_ids


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, handling edge cases."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid, y_valid = x[valid_mask], y[valid_mask]
    if len(x_valid) < 2 or np.std(x_valid) == 0 or np.std(y_valid) == 0:
        return np.nan
    return scipy_pearsonr(x_valid, y_valid)[0]


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation, handling edge cases."""
    if len(x) < 2:
        return np.nan
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid, y_valid = x[valid_mask], y[valid_mask]
    if len(x_valid) < 2:
        return np.nan
    return spearmanr(x_valid, y_valid)[0]


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MSE."""
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_valid, y_pred_valid = y_true[valid_mask], y_pred[valid_mask]
    if len(y_true_valid) == 0:
        return np.nan
    return ((y_true_valid - y_pred_valid)**2).mean()


# ============================================================================
# TOP-K METRICS (K=1,3)
# ============================================================================

def compute_topk_metrics(predictions: np.ndarray, true_labels: np.ndarray, 
                         complex_ids: list, quality_threshold: float = 0.23,
                         k_values: list = [1, 3]) -> dict:
    """
    
    Measure RANKING ACCURACY: How good is the top-K selection according to this ranking?
    
    THE KEY QUESTION:
    "If I pick the top K structures according to MODEL vs BASELINE, 
     which selection has better TRUE quality?"
    
    ⚠️ THIS FUNCTION IS CALLED TWICE (once for model, once for baseline):
    
    EXAMPLE for Complex "7km3" with 10 structures:
    
    True DockQ (ground truth):  [0.8, 0.3, 0.6, 0.1, 0.5, 0.2, 0.4, 0.15, 0.25, 0.35]
    Structure IDs:              [ s1,  s2,  s3,  s4,  s5,  s6,  s7,   s8,   s9,  s10]
    
    MODEL ranks them:           [0.9, 0.4, 0.8, 0.2, 0.7, 0.3, 0.6, 0.15, 0.25, 0.5]
    BASELINE ranks them:        [0.7, 0.6, 0.5, 0.4, 0.8, 0.3, 0.2, 0.15, 0.25, 0.9]
    
    Top-5 according to MODEL:   s1, s3, s5, s7, s10  → True DockQ = [0.8, 0.6, 0.5, 0.4, 0.35] → Avg = 0.53
    Top-5 according to BASELINE: s10, s5, s2, s3, s4  → True DockQ = [0.35, 0.5, 0.3, 0.6, 0.1] → Avg = 0.37
    
    MODEL IS BETTER! Its top-5 selection has higher average quality (0.53 vs 0.37)
    
    METRICS COMPUTED:
    1. Mean DockQ of top-K: Average true quality of the K structures you selected
    2. % of true top-K captured: How many of the ACTUAL top-K did you find?
    3. Success rate: % of complexes where top-K contains ≥1 good structure (DockQ > threshold)
    
    Returns dict with various ranking quality metrics
    """
    df = pd.DataFrame({
        'pred': predictions,
        'true': true_labels,
        'complex_id': complex_ids
    })
    
    results = {}
    
    for k in k_values:
        mean_topk_quality = []  # Average true DockQ of top-K selections
        topk_overlap_scores = []  # How many true top-K structures were captured
        success_count = 0  # How many complexes have ≥1 good structure in top-K
        total_complexes = 0
        
        for cid, group in df.groupby('complex_id'):
            if len(group) < k:
                continue
            
            # Top-K by ranking (model or baseline)
            topk_by_ranking = group.nlargest(k, 'pred')
            true_quality_of_topk = topk_by_ranking['true'].values
            mean_topk_quality.append(np.mean(true_quality_of_topk))
            
            # Compare to actual top-K
            actual_topk = group.nlargest(k, 'true')
            overlap = len(set(actual_topk.index) & set(topk_by_ranking.index))
            topk_overlap_scores.append(overlap / k)
            
            # Success: any good structure?
            if (true_quality_of_topk > quality_threshold).any():
                success_count += 1
            
            total_complexes += 1
        
        results[f'Top{k}_mean_quality'] = np.mean(mean_topk_quality) if mean_topk_quality else np.nan
        results[f'Top{k}_overlap'] = np.mean(topk_overlap_scores) if topk_overlap_scores else np.nan
        results[f'Top{k}_success_rate'] = success_count / total_complexes if total_complexes > 0 else np.nan
    
    # Mean rank of best structure
    mean_best_ranks = []
    for cid, group in df.groupby('complex_id'):
        if len(group) < 2:
            continue
        sorted_group = group.sort_values('pred', ascending=False).reset_index(drop=True)
        best_idx = group['true'].idxmax()
        rank = sorted_group.index[sorted_group.index.isin([best_idx])].tolist()
        if rank:
            mean_best_ranks.append(rank[0] + 1)
    
    results['mean_rank_of_best'] = np.mean(mean_best_ranks) if mean_best_ranks else np.nan
    
    return results


def compute_quality_discrimination(predictions: np.ndarray, true_labels: np.ndarray, 
                                   complex_ids: list) -> dict:
    """
    Measure how well the ranking discriminates between quality levels.
    
    THE KEY QUESTION:
    "Is my model better at identifying HIGH vs MEDIUM vs LOW quality structures?"
    
    Quality levels (DockQ thresholds):
    - High quality:   DockQ > 0.80
    - Medium quality: DockQ 0.49-0.80
    - Acceptable:     DockQ 0.23-0.49
    - Poor:           DockQ < 0.23
    
    METRICS:
    1. Precision@K for high-quality structures
    2. Mean score for each quality level (should be monotonically increasing)
    3. AUC for binary classification at each threshold
    """
    from sklearn.metrics import roc_auc_score
    
    df = pd.DataFrame({
        'pred': predictions,
        'true': true_labels,
        'complex_id': complex_ids
    })
    
    results = {}
    thresholds = {'high': 0.80, 'medium': 0.49, 'acceptable': 0.23}
    
    # Mean score for each quality level
    quality_scores = {'high': [], 'medium': [], 'acceptable': [], 'poor': []}
    
    for cid, group in df.groupby('complex_id'):
        high = group[group['true'] > thresholds['high']]
        if len(high) > 0:
            quality_scores['high'].append(high['pred'].mean())
        
        medium = group[(group['true'] > thresholds['medium']) & (group['true'] <= thresholds['high'])]
        if len(medium) > 0:
            quality_scores['medium'].append(medium['pred'].mean())
        
        acceptable = group[(group['true'] > thresholds['acceptable']) & (group['true'] <= thresholds['medium'])]
        if len(acceptable) > 0:
            quality_scores['acceptable'].append(acceptable['pred'].mean())
        
        poor = group[group['true'] <= thresholds['acceptable']]
        if len(poor) > 0:
            quality_scores['poor'].append(poor['pred'].mean())
    
    for level in ['high', 'medium', 'acceptable', 'poor']:
        results[f'mean_score_{level}'] = np.mean(quality_scores[level]) if quality_scores[level] else np.nan
    
    # AUC for binary classification
    for level, threshold in thresholds.items():
        try:
            binary_true = (true_labels > threshold).astype(int)
            if len(np.unique(binary_true)) > 1:
                results[f'auc_{level}'] = roc_auc_score(binary_true, predictions)
            else:
                results[f'auc_{level}'] = np.nan
        except:
            results[f'auc_{level}'] = np.nan
    
    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_scatter(x_values: np.ndarray, y_values: np.ndarray, complex_ids: list,
                xlabel: str, ylabel: str, title: str, outpath: str):
    """Generate scatter plot colored by complex."""
    if len(x_values) == 0:
        return
    
    valid_mask = ~np.isnan(x_values) & ~np.isnan(y_values)
    x_plot = x_values[valid_mask]
    y_plot = y_values[valid_mask]
    complex_ids_plot = [complex_ids[i] for i in range(len(complex_ids)) if valid_mask[i]]
    
    if len(x_plot) == 0:
        return
    
    unique_cids = sorted(list(set(complex_ids_plot)))
    cid_to_idx = {cid: i for i, cid in enumerate(unique_cids)}
    colors = [cid_to_idx[c] for c in complex_ids_plot]
    
    plt.figure(figsize=(7, 6))
    plt.scatter(x_plot, y_plot, c=colors, cmap='tab20', alpha=0.6, s=20)
    plt.plot([0,1], [0,1], 'k--', alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nSamples: {len(x_plot)}", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_combined_scatter(all_data: list, xlabel: str, ylabel: str, 
                         title: str, outpath: str, model_colors: dict):
    """Plot multiple models on single scatter."""
    plt.figure(figsize=(10, 8))
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (x_vals, y_vals, name, _) in enumerate(all_data):
        if len(x_vals) == 0:
            continue
        valid_mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        x_plot, y_plot = x_vals[valid_mask], y_vals[valid_mask]
        if len(x_plot) == 0:
            continue
        
        color = model_colors.get(name, 'gray')
        plt.scatter(x_plot, y_plot, alpha=0.5, s=30, label=name,
                   color=color, marker=markers[i % len(markers)])
    
    plt.plot([0,1], [0,1], 'k--', alpha=0.4, label='Ideal (1:1)')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.83, 1])
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_metric_bars(metrics_data: OrderedDict, title: str, ylabel: str,
                    outpath: str, model_colors: dict, lower_is_better: bool = False):
    """Bar chart for a single metric across models."""
    names = [n for n, v in metrics_data.items() if not np.isnan(v)]
    values = [v for v in metrics_data.values() if not np.isnan(v)]
    colors = [model_colors.get(n, 'grey') for n in names]
    
    if not names:
        return
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=colors)
    plt.ylabel(ylabel, fontsize=10)
    plt.title(title, fontsize=12)
    plt.xticks(rotation=40, ha='right')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}',
                va='bottom', ha='center', fontsize=8)
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ============================================================================
# MAIN LOGIC
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare models against baseline")
    parser.add_argument("--model_list_file", type=str, 
                       default="/proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/configs/models_to_test.txt",
                       help="Text file with model paths, one per line")
    parser.add_argument("--output_dir", type=str, default="comparison_reports",
                       help="Output directory for plots and reports")
    parser.add_argument("--splits", type=str, default="val",
                       help="Comma-separated splits to evaluate (e.g., val,test)")
    parser.add_argument("--manifest_path", type=str, default=None,
                       help="Path to manifest CSV (if not provided, uses first model's config)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    
    # Load model paths
    if not os.path.exists(args.model_list_file):
        print(f"Error: Model list file not found: {args.model_list_file}")
        return
    
    with open(args.model_list_file, 'r') as f:
        model_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    if not model_paths:
        print("No model paths found")
        return
    
    # Generate model IDs and colors
    baseline_id = "Baseline"
    model_ids = {path: f"Model_{i+1}" for i, path in enumerate(model_paths)}
    all_ids = [baseline_id] + list(model_ids.values())
    
    cmap = matplotlib.colormaps.get_cmap('tab10')
    model_colors = {name: cmap(i % 10) for i, name in enumerate(all_ids)}
    
    # Determine manifest path
    manifest_path = args.manifest_path
    if not manifest_path:
        try:
            _, first_config = load_model_and_config(model_paths[0], device)
            manifest_path = first_config['data']['manifest_file']
            print(f"Using manifest from first model: {manifest_path}")
        except Exception as e:
            print(f"Error loading first model config: {e}")
            return
    
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found: {manifest_path}")
        return
    
    # Load manifest
    try:
        main_df = pd.read_csv(manifest_path)
    except Exception as e:
        print(f"Error reading manifest: {e}")
        return
    
    ranking_col = "ranking_confidence" if "ranking_confidence" in main_df.columns else "ranking_score"
    required_cols = ['complex_id', 'label', ranking_col, 'split']
    if not all(col in main_df.columns for col in required_cols):
        print(f"Error: Missing required columns in manifest")
        return
    
    # Collect data for all models and baseline
    all_results = defaultdict(dict)
    
    for split in splits:
        print(f"\n=== Processing split: {split} ===")
        
        # Baseline data
        split_df = main_df[main_df['split'] == split].reset_index(drop=True)
        if not split_df.empty:
            all_results[baseline_id][split] = {
                "predictions": split_df[ranking_col].values,
                "true_dockq": split_df['label'].values,
                "complex_ids": split_df['complex_id'].tolist(),
            }
            #how many samples and how many complexes
            print(f"Baseline: {len(split_df)} samples and {len(split_df['complex_id'].unique())} complexes")
        
        # Model data
        for model_path in model_paths:
            model_id = model_ids[model_path]
            print(f"Evaluating {model_id}...")
            
            try:
                model, config = load_model_and_config(model_path, device)
                data_cfg = config.get('data', {})
                
                dataloader = get_eval_dataloader(
                    manifest_csv=manifest_path,
                    split=split,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    samples_per_complex=data_cfg.get('samples_per_complex', 10),
                    feature_transform=data_cfg.get('feature_transform', True),
                    feature_centering=data_cfg.get('feature_centering', False),
                    use_interchain_ca_distances=data_cfg.get('use_interchain_ca_distances', True),
                    use_interchain_pae=data_cfg.get('use_interchain_pae', True),
                    use_esm_embeddings=data_cfg.get('use_esm_embeddings', False),
                    use_file_cache=data_cfg.get('use_file_cache', True),
                    seed=data_cfg.get('seed', 42)
                )
                
                preds, labels, cids = evaluate_model(model, dataloader, device)
                
                if len(preds) > 0:
                    all_results[model_id][split] = {
                        "predictions": preds,
                        "true_dockq": labels,
                        "complex_ids": cids,
                    }
                    print(f"  {model_id}: {len(preds)} samples")
                
            except Exception as e:
                print(f"  Error with {model_id}: {e}")
    
    # Compute metrics
    print("\n=== Computing Metrics ===")
    all_metrics = defaultdict(lambda: defaultdict(dict))
    
    for model_id, split_data in all_results.items():
        for split, data in split_data.items():
            preds = data['predictions']
            true_dockq = data['true_dockq']
            cids = data['complex_ids']
            
            metrics = all_metrics[model_id][split]
            
            # Correlations
            metrics['Pearson_DockQ'] = pearson_r(preds, true_dockq)
            metrics['Spearman_DockQ'] = spearman_r(preds, true_dockq)
            metrics['MSE_DockQ'] = mean_squared_error(true_dockq, preds)
            
            # Per-complex correlations
            df_temp = pd.DataFrame({'preds': preds, 'true': true_dockq, 'cid': cids})
            per_complex_spearman = []
            for cid, group in df_temp.groupby('cid'):
                if len(group) > 1:
                    rho = spearman_r(group['preds'].values, group['true'].values)
                    if not np.isnan(rho):
                        per_complex_spearman.append(rho)
            
            metrics['Avg_PerComplex_Spearman'] = np.mean(per_complex_spearman) if per_complex_spearman else np.nan
            
            # Top-K metrics
            topk_metrics = compute_topk_metrics(preds, true_dockq, cids, k_values=[1, 3])
            metrics.update(topk_metrics)
            
            # Quality discrimination
            qual_metrics = compute_quality_discrimination(preds, true_dockq, cids)
            metrics.update(qual_metrics)
    
    # Generate plots
    print("\n=== Generating Plots ===")
    for split in splits:
        split_dir = os.path.join(args.output_dir, f"split_{split}")
        os.makedirs(split_dir, exist_ok=True)
        
        combined_data_dockq = []
        
        for model_id in all_results.keys():
            if split not in all_results[model_id]:
                continue
            
            data = all_results[model_id][split]
            preds = data['predictions']
            true_dockq = data['true_dockq']
            cids = data['complex_ids']
            
            # Individual scatter plots
            model_dir = os.path.join(split_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            plot_scatter(
                true_dockq, preds, cids,
                xlabel="True DockQ", ylabel="Prediction",
                title=f"{model_id} vs True DockQ",
                outpath=os.path.join(model_dir, "scatter_dockq.png")
            )
            
            combined_data_dockq.append((true_dockq, preds, model_id, cids))
        
        # Combined scatter
        if combined_data_dockq:
            plot_combined_scatter(
                combined_data_dockq,
                xlabel="True DockQ", ylabel="Prediction",
                title=f"All Models vs True DockQ - Split: {split}",
                outpath=os.path.join(split_dir, "combined_scatter.png"),
                model_colors=model_colors
            )
        
        # Metric bar charts
        metric_names = [
            ('Spearman_DockQ', 'Spearman Correlation', False),
            ('Pearson_DockQ', 'Pearson Correlation', False),
            ('MSE_DockQ', 'Mean Squared Error', True),
            ('Top1_mean_quality', 'Top-1 Mean Quality', False),
            ('Top3_mean_quality', 'Top-3 Mean Quality', False),
            ('Top1_success_rate', 'Top-1 Success Rate', False),
            ('Top3_success_rate', 'Top-3 Success Rate', False),
        ]
        
        for metric_key, metric_title, lower_better in metric_names:
            metric_data = OrderedDict()
            for model_id in all_ids:
                if split in all_metrics[model_id] and metric_key in all_metrics[model_id][split]:
                    metric_data[model_id] = all_metrics[model_id][split][metric_key]
                else:
                    metric_data[model_id] = np.nan
            
            plot_metric_bars(
                metric_data,
                title=f"{metric_title} - Split: {split}",
                ylabel=metric_title,
                outpath=os.path.join(split_dir, f"bar_{metric_key}.png"),
                model_colors=model_colors,
                lower_is_better=lower_better
            )
    
    # Generate summary report
    print("\n=== Summary Report ===")
    report_data = []
    for model_id, split_data in all_metrics.items():
        for split, metrics in split_data.items():
            row = {'Model': model_id, 'Split': split}
            for k, v in metrics.items():
                if isinstance(v, (float, np.floating)):
                    row[k] = f"{v:.4f}" if not np.isnan(v) else "NaN"
                else:
                    row[k] = str(v)
            report_data.append(row)
    
    if report_data:
        df_summary = pd.DataFrame(report_data)
        
        # Save CSV
        csv_path = os.path.join(args.output_dir, "summary.csv")
        df_summary.to_csv(csv_path, index=False)
        print(f"Summary saved to: {csv_path}")
        
        # Print to console
        print("\n" + "="*80)
        print(df_summary.to_string(index=False))
        print("="*80)
        
        # Print best performers
        for split in splits:
            df_split = df_summary[df_summary['Split'] == split]
            if df_split.empty:
                continue
            
            print(f"\n=== Best Performers - Split: {split} ===")
            
            # Convert to numeric for comparison
            for col in ['Spearman_DockQ', 'Top1_mean_quality', 'Top3_mean_quality']:
                if col in df_split.columns:
                    df_split[col] = pd.to_numeric(df_split[col], errors='coerce')
                    if df_split[col].notna().sum() > 0:
                        best_idx = df_split[col].idxmax()
                        best = df_split.loc[best_idx]
                        print(f"  Best {col}: {best['Model']} ({best[col]:.4f})")
    
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
