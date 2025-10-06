#!/usr/bin/env python
"""
compare_models_and_baseline.py

A comprehensive tool to compare multiple trained DeepSet models (or other compatible models)
and a baseline ranking_score score against ground truth DockQ and TM scores.

Functionality:
- Loads models specified in a text file.
- Uses a primary manifest (from the first model's config) for consistent dataset splits
  and ground truth data (DockQ, TM scores, ranking_score baseline).
- Evaluates each model and the baseline across specified data splits (e.g., train, val, test).
- Computes global and per-complex correlation metrics (Pearson r, Spearman ρ) and MSE.
- Generates various visualizations:
    - Individual scatter plots (model/baseline vs. DockQ/TM, colored by complex).
    - Combined scatter plot overlaying all models and baseline.
    - Bar charts for comparing global and per-complex metrics.
- Outputs a summary report highlighting the best performers.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Ensure script can run on headless servers
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr as scipy_pearsonr # Avoid conflict with custom pearson_r
from collections import defaultdict, OrderedDict
import re # Added for get_model_info_from_path
import shutil # For cleaning directories

# Attempt to import from src, assuming PYTHONPATH is set or script is run from root
try:
    from src.models.deep_set import DeepSet
    from src.data.dataloader import get_eval_dataloader
except ImportError:
    print("Warning: Could not import DeepSet or get_eval_dataloader from src.")
    print("Please ensure that the 'src' directory is in your PYTHONPATH or script is run from the project root.")
    # Provide dummy implementations if not found, to allow script structure to be checked
    class DeepSet(torch.nn.Module): # type: ignore
        def __init__(self, **kwargs): super().__init__(); self.dummy = torch.nn.Linear(1,1)
        def forward(self, feats, lengths): return torch.rand(feats.size(0), 1)

    def get_eval_dataloader(manifest_file, split, batch_size, num_workers, feature_transform, feature_centering, **kwargs): # type: ignore
        print(f"Warning: Using dummy get_eval_dataloader for {manifest_file} split {split}")
        # This dummy loader needs to return batches with 'features', 'lengths', 'label'
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self): self.data = [{"features": torch.randn(np.random.randint(5,15), 10), "label": torch.rand(1)} for _ in range(20)]
            def __len__(self): return len(self.data)
            def __getitem__(self, idx):
                item = self.data[idx]
                return {"features": item["features"], "lengths": torch.tensor(len(item["features"])), "label": item["label"]}
        return torch.utils.data.DataLoader(DummyDataset(), batch_size=batch_size, num_workers=num_workers)


# --- Helper Functions ---

def load_model_and_config(model_path: str, device: torch.device):
    """Load a model and its configuration."""
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path} for model {model_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config.get('model', {})
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Auto-detect input_dim from checkpoint weights (same as correlation_analysis.py)
    if "phi.0.weight" in state_dict:
        actual_input_dim = state_dict["phi.0.weight"].shape[1]
        config_input_dim = model_cfg.get('input_dim')
        if actual_input_dim != config_input_dim:
            print(f"  Warning: Config says input_dim={config_input_dim}, "
                  f"but checkpoint has input_dim={actual_input_dim}. "
                  f"Using {actual_input_dim} from checkpoint.")
            model_cfg['input_dim'] = actual_input_dim

    # TODO: Extend this to support different model types if necessary
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

def evaluate_model_on_dataloader(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """
    Run model inference.
    Returns:
        - flat_predictions: np.array of model predictions (sigmoid applied).
        - flat_true_labels_dockq: np.array of true DockQ labels from the batch.
        - flat_complex_ids: list of complex_id strings.
    """
    all_preds_flat = []
    all_labels_flat = [] 
    all_complex_ids_flat = []

    with torch.no_grad():
        for batch in dataloader:
            feats = batch['features'].to(device)
            lengths = batch['lengths'].to(device)
            true_labels_batch = batch["label"].to(device)      # Shape [B, K]
            
            # Handle complex IDs correctly - same as in test_models.py
            batch_complex_ids = batch['complex_id']  # This is a np.array [B, K]
            # Assuming complex_ids[:, 0] is the representative ID for all K samples of a complex
            # Repeat each complex_id K times
            ids_repeated = np.repeat(batch_complex_ids[:, 0], batch_complex_ids.shape[1])
            all_complex_ids_flat.extend(ids_repeated.tolist())
            
            logits = model(feats, lengths) # Shape [B, K]
            preds = torch.sigmoid(logits)  # Shape [B, K]
            
            # Apply sigmoid to labels if they are logits (like in test_models.py)
            # Verify this step based on your dataset's label format.
            # If labels are already probabilities [0,1], this line might be unnecessary.
            true_labels_batch = torch.sigmoid(true_labels_batch)
            
            all_preds_flat.append(preds.cpu().numpy().reshape(-1))
            all_labels_flat.append(true_labels_batch.cpu().numpy().reshape(-1))

    final_preds = np.concatenate(all_preds_flat) if all_preds_flat else np.array([])
    final_labels = np.concatenate(all_labels_flat) if all_labels_flat else np.array([])
    
    # all_complex_ids_flat are already lists
    return final_preds, final_labels, all_complex_ids_flat

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) < 2 or len(y) < 2 or np.std(x) == 0 or np.std(y) == 0 or np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return np.nan
    # Filter out NaNs pairs
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid, y_valid = x[valid_mask], y[valid_mask]
    if len(x_valid) < 2 or np.std(x_valid) == 0 or np.std(y_valid) == 0 : # check again after NaN removal
         return np.nan
    return scipy_pearsonr(x_valid, y_valid)[0]

def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation (rho)."""
    if len(x) < 2 or len(y) < 2 or np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return np.nan
    # Filter out NaNs pairs for spearmanr
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid, y_valid = x[valid_mask], y[valid_mask]
    if len(x_valid) < 2: # spearmanr itself handles std=0 cases by returning nan for correlation
        return np.nan
    return spearmanr(x_valid, y_valid)[0]

def bootstrap_correlation_test(preds1: np.ndarray, preds2: np.ndarray, true_labels: np.ndarray, 
                                n_bootstrap: int = 1000, metric: str = 'spearman') -> dict:
    """
    Bootstrap test to compare two sets of predictions against ground truth.
    
    ⚠️ ANALYSIS LEVEL: GLOBAL (all samples together)
    - Uses ALL samples across ALL complexes together (e.g., 2895 total samples)
    - Computes GLOBAL correlation: how well rankings work across entire dataset
    - This is appropriate for overall performance comparison
    - For PER-COMPLEX analysis, see Win/Tie/Loss metric!
    
    WHY 1000 ITERATIONS?
    - Bootstrap resampling estimates the sampling distribution of the correlation difference
    - 1000 iterations provides stable estimates of confidence intervals and p-values
    - More iterations (e.g., 10000) give slightly more precision but take longer
    - This is standard practice in statistical testing
    
    HOW IT WORKS:
    1. Takes your FULL dataset of predictions (e.g., 2895 samples across 163 complexes)
    2. Randomly resamples WITH REPLACEMENT 1000 times
    3. For each resample, computes: Model_correlation - Baseline_correlation
    4. These 1000 differences form a distribution
    5. P-value = how often the observed difference could occur by chance
    
    IMPORTANT: Samples within a complex are treated as independent observations.
    This is reasonable because:
    - Each AlphaFold3 prediction is a different protein conformation
    - We're testing global ranking ability, not complex-specific performance
    - Per-complex performance is separately evaluated in Win/Tie/Loss analysis
    
    Returns:
        dict with 'diff_mean', 'diff_std', 'p_value', 'ci_lower', 'ci_upper', 
        'model1_corr', 'model2_corr', 'bootstrap_diffs' (for plotting)
    """
    valid_mask = ~np.isnan(preds1) & ~np.isnan(preds2) & ~np.isnan(true_labels)
    p1, p2, y = preds1[valid_mask], preds2[valid_mask], true_labels[valid_mask]
    
    if len(p1) < 10:
        return {'diff_mean': np.nan, 'diff_std': np.nan, 'p_value': np.nan, 
                'ci_lower': np.nan, 'ci_upper': np.nan, 
                'model1_corr': np.nan, 'model2_corr': np.nan}
    
    corr_func = spearman_r if metric == 'spearman' else pearson_r
    
    # Observed difference
    corr1_obs = corr_func(p1, y)
    corr2_obs = corr_func(p2, y)
    diff_obs = corr1_obs - corr2_obs
    
    # Bootstrap
    n = len(p1)
    diffs = []
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        p1_boot, p2_boot, y_boot = p1[indices], p2[indices], y[indices]
        c1 = corr_func(p1_boot, y_boot)
        c2 = corr_func(p2_boot, y_boot)
        if not np.isnan(c1) and not np.isnan(c2):
            diffs.append(c1 - c2)
    
    if len(diffs) == 0:
        return {'diff_mean': diff_obs, 'diff_std': np.nan, 'p_value': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan,
                'model1_corr': corr1_obs, 'model2_corr': corr2_obs}
    
    diffs = np.array(diffs)
    # Two-tailed p-value: proportion of bootstrap samples where |diff| >= |diff_obs|
    p_value = np.mean(np.abs(diffs) >= np.abs(diff_obs))
    
    # 95% confidence interval
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    
    return {
        'diff_mean': diff_obs,
        'diff_std': np.std(diffs),
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'model1_corr': corr1_obs,
        'model2_corr': corr2_obs,
        'bootstrap_diffs': diffs  # For plotting
    }

def compute_topk_metrics(predictions: np.ndarray, true_labels: np.ndarray, 
                         complex_ids: list, quality_threshold: float = 0.23,
                         k_values: list = [1, 3, 5, 10]) -> dict:
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
            
            # Get top-K according to this ranking (model or baseline)
            topk_by_ranking = group.nlargest(k, 'pred')
            
            # What is the TRUE quality of these top-K selections?
            true_quality_of_topk = topk_by_ranking['true'].values
            mean_topk_quality.append(np.mean(true_quality_of_topk))
            
            # What are the ACTUAL top-K structures (ground truth)?
            actual_topk = group.nlargest(k, 'true')
            actual_topk_indices = set(actual_topk.index)
            selected_topk_indices = set(topk_by_ranking.index)
            
            # How many did we capture correctly?
            overlap = len(actual_topk_indices & selected_topk_indices)
            topk_overlap_scores.append(overlap / k)  # Fraction of true top-K captured
            
            # Success: any structure with DockQ > threshold?
            if (true_quality_of_topk > quality_threshold).any():
                success_count += 1
            
            total_complexes += 1
        
        # Store metrics for this K
        results[f'Top{k}_mean_quality'] = np.mean(mean_topk_quality) if mean_topk_quality else np.nan
        results[f'Top{k}_overlap'] = np.mean(topk_overlap_scores) if topk_overlap_scores else np.nan
        results[f'Top{k}'] = success_count / total_complexes if total_complexes > 0 else np.nan
    
    # Additional metric: Mean rank of BEST structure
    mean_best_ranks = []
    for cid, group in df.groupby('complex_id'):
        if len(group) < 2:
            continue
        sorted_group = group.sort_values('pred', ascending=False).reset_index(drop=True)
        best_true_idx = group['true'].idxmax()
        rank = sorted_group.index[sorted_group.index.isin([best_true_idx])].tolist()
        if rank:
            mean_best_ranks.append(rank[0] + 1)
    
    results['mean_rank_of_best_structure'] = np.mean(mean_best_ranks) if mean_best_ranks else np.nan
    results['n_complexes_analyzed'] = len(mean_best_ranks) if mean_best_ranks else 0
    
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
    
    # Define quality thresholds
    thresholds = {
        'high': 0.80,
        'medium': 0.49,
        'acceptable': 0.23
    }
    
    # 1. Mean prediction score for each quality level (per complex, then averaged)
    quality_scores = {
        'high': [],
        'medium': [],
        'acceptable': [],
        'poor': []
    }
    
    for cid, group in df.groupby('complex_id'):
        # High quality structures
        high = group[group['true'] > thresholds['high']]
        if len(high) > 0:
            quality_scores['high'].append(high['pred'].mean())
        
        # Medium quality structures
        medium = group[(group['true'] > thresholds['medium']) & (group['true'] <= thresholds['high'])]
        if len(medium) > 0:
            quality_scores['medium'].append(medium['pred'].mean())
        
        # Acceptable quality structures
        acceptable = group[(group['true'] > thresholds['acceptable']) & (group['true'] <= thresholds['medium'])]
        if len(acceptable) > 0:
            quality_scores['acceptable'].append(acceptable['pred'].mean())
        
        # Poor quality structures
        poor = group[group['true'] <= thresholds['acceptable']]
        if len(poor) > 0:
            quality_scores['poor'].append(poor['pred'].mean())
    
    for level in ['high', 'medium', 'acceptable', 'poor']:
        results[f'mean_score_{level}'] = np.mean(quality_scores[level]) if quality_scores[level] else np.nan
    
    # 2. Precision at top-K for high-quality structures
    for k in [1, 3, 5, 10]:
        precision_high = []
        
        for cid, group in df.groupby('complex_id'):
            if len(group) < k:
                continue
            
            topk = group.nlargest(k, 'pred')
            n_high_quality = (topk['true'] > thresholds['high']).sum()
            precision_high.append(n_high_quality / k)
        
        results[f'precision_high_top{k}'] = np.mean(precision_high) if precision_high else np.nan
    
    # 3. AUC for binary classification at each threshold
    for level, threshold in thresholds.items():
        try:
            binary_true = (true_labels > threshold).astype(int)
            if len(np.unique(binary_true)) > 1:  # Need both classes
                auc = roc_auc_score(binary_true, predictions)
                results[f'auc_{level}'] = auc
            else:
                results[f'auc_{level}'] = np.nan
        except:
            results[f'auc_{level}'] = np.nan
    
    return results

def compute_win_tie_loss(model_preds: np.ndarray, baseline_preds: np.ndarray,
                         true_labels: np.ndarray, complex_ids: list) -> dict:
    """
    For each complex, determine if model wins/ties/loses against baseline
    based on Spearman correlation with ground truth.
    
    Returns dict with 'wins', 'ties', 'losses', 'win_complexes', 'loss_complexes'
    """
    df = pd.DataFrame({
        'model': model_preds,
        'baseline': baseline_preds,
        'true': true_labels,
        'complex_id': complex_ids
    })
    
    wins, ties, losses = 0, 0, 0
    win_complexes, loss_complexes = [], []
    
    for cid, group in df.groupby('complex_id'):
        if len(group) < 2:  # Need at least 2 samples to compute correlation
            continue
        
        model_corr = spearman_r(group['model'].values, group['true'].values)
        baseline_corr = spearman_r(group['baseline'].values, group['true'].values)
        
        if np.isnan(model_corr) or np.isnan(baseline_corr):
            continue
        
        diff = model_corr - baseline_corr
        
        if diff > 0.05:  # Model is better (threshold to avoid noise)
            wins += 1
            win_complexes.append((cid, model_corr, baseline_corr, diff))
        elif diff < -0.05:  # Baseline is better
            losses += 1
            loss_complexes.append((cid, model_corr, baseline_corr, diff))
        else:  # Tie
            ties += 1
    
    total = wins + ties + losses
    
    return {
        'wins': wins,
        'ties': ties,
        'losses': losses,
        'win_rate': wins / total if total > 0 else np.nan,
        'win_complexes': sorted(win_complexes, key=lambda x: x[3], reverse=True)[:10],  # Top 10 wins
        'loss_complexes': sorted(loss_complexes, key=lambda x: x[3])[:10]  # Top 10 losses
    }


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    if len(y_true) == 0 or len(y_pred) == 0 or np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_valid, y_pred_valid = y_true[valid_mask], y_pred[valid_mask]
    if len(y_true_valid) == 0:
        return np.nan
    return ((y_true_valid - y_pred_valid)**2).mean()

def get_model_info_from_path(model_path: str) -> str:
    """Extracts ModelName_YYYY-MM-DD_HH-MM-SS from the model path's parent directory.
       If datetime is not found, returns ModelName.
       If ModelName cannot be determined, returns a sanitized version of the folder name.
    """
    folder_name = os.path.basename(os.path.dirname(model_path))
    
    # Regex to find YYYY-MM-DD_HH-MM-SS pattern
    datetime_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', folder_name)
    datetime_str = datetime_match.group(1) if datetime_match else None

    # Determine ModelName: part before the first underscore, or whole folder name if no underscore
    parts = folder_name.split('_', 1)
    model_name_part = parts[0]
    
    if model_name_part and datetime_str:
        info_str = f"{model_name_part}_{datetime_str}"
    elif model_name_part: # Datetime not found, but model name part exists
        info_str = model_name_part # Do not keep the rest as per user request
    else: # Fallback if folder_name is unusual (e.g., empty or starts with an underscore)
        info_str = folder_name

    # Sanitize and limit length
    info_str_sanitized = info_str.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    if not info_str_sanitized: # If somehow empty after sanitization
        return f"model_{hash(model_path) % 10000:04d}"
        
    return info_str_sanitized[:75] # Limit length

# --- Plotting Functions ---

def plot_scatter_by_complex(
    x_values: np.ndarray, y_values: np.ndarray, complex_ids: list,
    xlabel: str, ylabel: str, title: str, outpath: str,
    ref_line: bool = True
):
    """Generates a scatter plot colored by complex_id."""
    if len(x_values) == 0:
        print(f"Skipping plot {title} due to empty data.")
        return

    # Ensure all arrays have same length
    min_len = min(len(x_values), len(y_values), len(complex_ids))
    if min_len < len(x_values) or min_len < len(y_values) or min_len < len(complex_ids):
        print(f"Warning: Arrays have different lengths. Truncating to {min_len} elements.")
        x_values = x_values[:min_len]
        y_values = y_values[:min_len]
        complex_ids = complex_ids[:min_len]
    
    # Filter out NaN values
    valid_mask = ~np.isnan(x_values) & ~np.isnan(y_values)
    x_plot = x_values[valid_mask]
    y_plot = y_values[valid_mask]
    complex_ids_plot = [complex_ids[i] for i in range(len(complex_ids)) if valid_mask[i]]
    
    if len(x_plot) == 0:
        print(f"Skipping plot {title} - All data points contain NaN values.")
        return
    
    # Create a mapping for complex IDs to colors
    unique_cids = sorted(list(set(complex_ids_plot)))
    cid_to_idx = {cid: i for i, cid in enumerate(unique_cids)}
    colors_mapped = [cid_to_idx.get(c, -1) for c in complex_ids_plot]

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(x_plot, y_plot, c=colors_mapped, cmap='tab20', alpha=0.6, s=20)
    
    if ref_line:
        plt.plot([0,1],[0,1], 'k--', alpha=0.3)  # Explicitly plot 0-1 line

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nSamples: {len(x_plot)}", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add color bar for complex IDs
    if 0 < len(unique_cids) <= 20:
        cbar = plt.colorbar(scatter, ticks=range(len(unique_cids)))
        try:
            cbar.ax.set_yticklabels(unique_cids)
        except Exception as e:
            print(f"Warning: Could not set colorbar labels for {title}: {e}")
            plt.colorbar(scatter, label='Complex ID Index (Labeling Error)')
    elif len(unique_cids) > 0:
        plt.colorbar(scatter, label='Complex ID Index')
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_combined_scatter(
    all_plot_data: list, # List of tuples: (x, y, name, complex_ids)
    xlabel: str, ylabel: str, title: str, outpath: str,
    model_to_color: dict, # Added: model_name -> color mapping
    ref_line: bool = True
):
    """Plots multiple prediction sets on a single scatter plot."""
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # markers list can remain the same, colors will come from model_to_color
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, (x_vals, y_vals, name, _) in enumerate(all_plot_data): 
        if len(x_vals) == 0: continue
        valid_mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        x_plot, y_plot = x_vals[valid_mask], y_vals[valid_mask]
        if len(x_plot) == 0: continue

        color_for_model = model_to_color.get(name, 'gray') # Default to gray if name not in map

        plt.scatter(x_plot, y_plot, alpha=0.5, s=30, label=name, 
                    color=color_for_model, 
                    marker=markers[i % len(markers)]) # Keep marker cycling for visual distinction if colors repeat due to >10 models

    if ref_line:
        plt.plot([0,1],[0,1], 'k--', alpha=0.4, label='Ideal (1:1)')

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    
    if all_plot_data: 
        # Legend will be ordered by how all_plot_data was sorted before calling
        plt.legend(title="Models/Baseline", bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0., fontsize='small', title_fontsize='medium')
    
    plt.tight_layout(rect=[0, 0, 0.83, 1]) 
    plt.savefig(outpath, dpi=200)
    plt.close()

# New helper function to draw a single bar chart subplot
def _draw_single_metric_subplot(ax, metrics_data: OrderedDict, subplot_title: str, 
                                ylabel: str, model_to_color: dict, lower_is_better: bool = False):
    """Draws a single bar chart for a given metric on the provided Axes object."""
    
    plottable_model_names = []
    plottable_values = []
    plottable_colors = []

    for model_name, value in metrics_data.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            plottable_model_names.append(model_name)
            plottable_values.append(value)
            plottable_colors.append(model_to_color.get(model_name, 'grey'))
    
    if not plottable_model_names:
        ax.text(0.5, 0.5, "No plottable data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(subplot_title, fontsize=10, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    bars = ax.bar(plottable_model_names, plottable_values, color=plottable_colors)
    
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(subplot_title, fontsize=10, pad=10)
    
    num_models_to_plot = len(plottable_model_names)
    ax.tick_params(axis='x', labelrotation=40, labelsize=max(7, 9 - num_models_to_plot // 4), labelright=False, labelleft=True)
    # Ensure x-tick labels are set correctly
    ax.set_xticks(range(len(plottable_model_names)))
    ax.set_xticklabels(plottable_model_names, ha="right")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', 
                 va='bottom' if (yval >= 0 and not lower_is_better) or (yval < 0 and lower_is_better) else 'top', 
                 ha='center', fontsize=7)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# New function to create a figure with multiple metric bar subplots
def plot_metric_subplots_figure(figure_metrics_data: list, # List of dicts, each for a subplot
                                overall_figure_title: str, 
                                outpath: str, 
                                model_to_color: dict, 
                                num_cols: int = 2):
    """
    Generates a figure with multiple bar chart subplots for a group of metrics.
    figure_metrics_data: [{'metric_key': str, 'data': OrderedDict, 'ylabel': str, 'lower_is_better': bool, 'subplot_title': str}]
    """
    num_metrics = len(figure_metrics_data)
    if num_metrics == 0:
        print(f"No metrics data provided for figure: {overall_figure_title}")
        return

    num_rows = (num_metrics + num_cols - 1) // num_cols # Calculate rows needed
    
    fig_height_per_row = 4 # Inches
    fig_width = max(10, num_cols * 5) # Adjust overall width based on columns

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, num_rows * fig_height_per_row), squeeze=False)
    axes_flat = axes.flatten()

    for i, metric_info in enumerate(figure_metrics_data):
        ax = axes_flat[i]
        _draw_single_metric_subplot(
            ax=ax, 
            metrics_data=metric_info['data'], 
            subplot_title=metric_info.get('subplot_title', metric_info['metric_key']), # Use metric_key if no title
            ylabel=metric_info['ylabel'], 
            model_to_color=model_to_color, 
            lower_is_better=metric_info['lower_is_better']
        )
    
    # Hide any unused subplots if num_metrics < num_rows * num_cols
    for j in range(num_metrics, num_rows * num_cols):
        fig.delaxes(axes_flat[j])

    fig.suptitle(overall_figure_title, fontsize=14, y=0.99) # y might need adjustment
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle and x-labels
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_bootstrap_distribution(bootstrap_diffs: np.ndarray, observed_diff: float, 
                                 ci_lower: float, ci_upper: float, p_value: float,
                                 model_name: str, split_name: str, outpath: str):
    """Plot the bootstrap distribution of correlation differences."""
    plt.figure(figsize=(10, 6))
    
    # Histogram of bootstrap differences
    plt.hist(bootstrap_diffs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Mark observed difference
    plt.axvline(observed_diff, color='red', linestyle='--', linewidth=2, 
                label=f'Observed Δρ = {observed_diff:.4f}')
    
    # Mark confidence interval
    plt.axvline(ci_lower, color='orange', linestyle=':', linewidth=2, 
                label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    plt.axvline(ci_upper, color='orange', linestyle=':', linewidth=2)
    
    # Mark zero line
    plt.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5, 
                label='Δρ = 0 (no difference)')
    
    plt.xlabel('Difference in Spearman ρ (Model - Baseline)', fontsize=12)
    plt.ylabel('Frequency (out of 1000 bootstrap samples)', fontsize=12)
    plt.title(f'Bootstrap Distribution: {model_name} vs Baseline\n'
              f'Split: {split_name} | p-value = {p_value:.4f}', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add interpretation text
    sig_text = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
    better_text = "Model is BETTER" if observed_diff > 0 else "Baseline is BETTER"
    plt.text(0.02, 0.98, f'{better_text}\n({sig_text}, p={p_value:.4f})',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_topk_comparison(model_topk: dict, baseline_topk: dict, 
                         model_name: str, split_name: str, outpath: str):
    """Plot Top-K success rate comparison."""
    k_labels = ['Top-1', 'Top-3', 'Top-5', 'Top-10']
    k_keys = ['Top1', 'Top3', 'Top5', 'Top10']
    
    model_rates = [model_topk.get(k, np.nan) * 100 for k in k_keys]
    baseline_rates = [baseline_topk.get(k, np.nan) * 100 for k in k_keys]
    
    x = np.arange(len(k_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, model_rates, width, label=model_name, color='steelblue')
    bars2 = ax.bar(x + width/2, baseline_rates, width, label='Baseline', color='coral')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('K (Top-K Predictions)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'Top-K Success Rate: Finding Good Structures (DockQ > 0.23)\n'
                 f'{model_name} vs Baseline | Split: {split_name}', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add interpretation
    improvements = [model_rates[i] - baseline_rates[i] for i in range(len(k_labels))]
    avg_improvement = np.mean([imp for imp in improvements if not np.isnan(imp)])
    
    verdict = "MODEL IS BETTER" if avg_improvement > 2 else ("SIMILAR PERFORMANCE" if avg_improvement > -2 else "BASELINE IS BETTER")
    color = 'green' if avg_improvement > 2 else ('orange' if avg_improvement > -2 else 'red')
    
    ax.text(0.02, 0.98, f'Avg Improvement: {avg_improvement:+.1f}%\n{verdict}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_win_tie_loss(wins: int, ties: int, losses: int, 
                      model_name: str, split_name: str, outpath: str):
    """Plot Win/Tie/Loss pie chart and bar chart."""
    total = wins + ties + losses
    
    # Handle case where no complexes were analyzed
    if total == 0:
        print(f"Warning: No complexes to plot for Win/Tie/Loss analysis. Skipping plot.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    sizes = [wins, ties, losses]
    labels = [f'Model Wins\n{wins} ({wins/total*100:.1f}%)',
              f'Ties\n{ties} ({ties/total*100:.1f}%)',
              f'Model Losses\n{losses} ({losses/total*100:.1f}%)']
    colors = ['#90EE90', '#FFD700', '#FFB6C1']
    explode = (0.05, 0, 0)  # Explode the wins slice
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='', shadow=True, startangle=90)
    ax1.set_title(f'Per-Complex Win/Tie/Loss\n{model_name} vs Baseline', fontsize=12)
    
    # Bar chart
    categories = ['Model\nWins', 'Ties', 'Model\nLosses']
    counts = [wins, ties, losses]
    bars = ax2.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Number of Complexes', fontsize=12)
    ax2.set_title(f'Total Complexes: {total} | Split: {split_name}', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add verdict
    win_rate = wins / total
    if win_rate > 0.55:
        verdict = "✅ MODEL WINS MORE OFTEN"
        vcolor = 'green'
    elif win_rate < 0.45:
        verdict = "❌ BASELINE WINS MORE OFTEN"
        vcolor = 'red'
    else:
        verdict = "⚖️ EVENLY MATCHED"
        vcolor = 'orange'
    
    ax2.text(0.5, 0.95, verdict, transform=ax2.transAxes, fontsize=13, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor=vcolor, alpha=0.3))
    
    plt.suptitle(f'Per-Complex Performance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_per_complex_spearman_comparison(
    baseline_spearman_values: list, 
    model_spearman_values: list, 
    common_complex_ids: list, # For potential future use (e.g., labeling points)
    model_name: str,
    baseline_name: str,
    split_name: str,
    outpath: str
):
    """Generates a scatter plot comparing per-complex Spearman correlations of a model vs. baseline."""
    
    x_data = np.array(baseline_spearman_values)
    y_data = np.array(model_spearman_values)

    # Filter out any NaN pairs that might exist if a complex had undefined correlation for one but not the other
    valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_plot = x_data[valid_mask]
    y_plot = y_data[valid_mask]
    # common_complex_ids_plot = [common_complex_ids[i] for i, v in enumerate(valid_mask) if v]

    if len(x_plot) == 0:
        print(f"Skipping per-complex Spearman comparison for {model_name} vs {baseline_name} on split {split_name} - no valid data points.")
        return

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    plt.scatter(x_plot, y_plot, alpha=0.6, s=30, label=f"{model_name} vs {baseline_name}\n(N={len(x_plot)} complexes)")
    
    # Determine overall min/max for the 1:1 line, considering both axes typically range -1 to 1 for correlation
    # However, Spearman can be NaN, which we filtered. Valid values are -1 to 1.
    min_val = -1.05 # min(np.min(x_plot), np.min(y_plot)) if len(x_plot) > 0 else -1
    max_val = 1.05 # max(np.max(x_plot), np.max(y_plot)) if len(x_plot) > 0 else 1
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4, label='Model = Baseline (1:1)')
    
    plt.xlabel(f"{baseline_name}\nPer-Complex Spearman ρ (vs. True DockQ)", fontsize=10)
    plt.ylabel(f"{model_name}\nPer-Complex Spearman ρ (vs. True DockQ)", fontsize=10)
    plt.title(f"Per-Complex Spearman ρ Comparison - Split: {split_name}", fontsize=12)
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Compare multiple models and a baseline.")
    parser.add_argument("--model_list_file", type=str, default="/proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/configs/models_to_test.txt",
                        help="Text file listing paths to model .pt files, one per line.")
    parser.add_argument("--output_dir", type=str, default="comparison_reports_new",
                        help="Directory to save plots and reports.")
    parser.add_argument("--splits", type=str, default="val",
                        help="Comma-separated list of splits to evaluate (e.g., val,test,train).")
    parser.add_argument("--primary_manifest_path", type=str, default=None,
                        help="Optional: Path to a specific manifest CSV to use for all evaluations. "
                             "If not provided, uses the manifest from the first model's config.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dataloaders (use 1 to avoid partial batch issues).")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloaders (0 for main process only).")
    parser.add_argument("--report_formats", type=str, default="print,csv,md",
                        help="Comma-separated list of report formats (print, csv, md).")
    parser.add_argument("--default_samples_per_complex_eval", type=int, default=1,
                        help="Default K value if not in model config, for dataloader during eval.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_to_eval_original = [s.strip() for s in args.splits.split(',') if s.strip()]
    # We will process original splits first, then potentially add 'test+val'

    # --- 1. Load Model Paths ---
    if not os.path.exists(args.model_list_file):
        print(f"Error: Model list file not found: {args.model_list_file}")
        return
    with open(args.model_list_file, 'r') as f:
        model_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not model_paths:
        print("No model paths found in the list file.")
        return

    # --- 1.5 Generate Fixed Model IDs (Unique per model_path) ---
    model_path_to_fixed_id = {}
    temp_model_ids_for_uniqueness_check = {} # {raw_name: count}
    for path_idx, model_path in enumerate(model_paths):
        raw_model_id = get_model_info_from_path(model_path)
        
        # Ensure uniqueness if different paths produce the same raw_model_id
        # (e.g. copied model folders with same name structure)
        count_for_this_raw_id = temp_model_ids_for_uniqueness_check.get(raw_model_id, 0)
        final_model_id_for_path = raw_model_id
        if count_for_this_raw_id > 0:
            final_model_id_for_path = f"{raw_model_id}_ID{count_for_this_raw_id + 1}" 
        
        temp_model_ids_for_uniqueness_check[raw_model_id] = count_for_this_raw_id + 1
        model_path_to_fixed_id[model_path] = final_model_id_for_path
        
    # --- 1.6 Define a fixed order and color map for all models and baseline ---
    baseline_id = "Baseline (Ranking Confidence)" # Defined as a constant earlier or ensure it is
    
    # Create a sorted list of actual model IDs (excluding baseline initially)
    # Values from model_path_to_fixed_id can be non-unique if get_model_info_from_path is not perfectly unique
    # and the _IDx suffix was added. So, use set to get unique fixed model IDs.
    unique_fixed_model_ids = sorted(list(set(model_path_to_fixed_id.values())))
    
    # Final display order: Baseline first, then other models sorted
    # Ensure baseline_id is not duplicated if it somehow was a result of get_model_info_from_path
    final_model_display_order = [baseline_id]
    for mid in unique_fixed_model_ids:
        if mid != baseline_id:
            final_model_display_order.append(mid)
            
    # Generate colors
    cmap_for_models = matplotlib.colormaps.get_cmap('tab10') # tab10 is good for distinct colors
    model_to_color = {}
    for i, model_id_name in enumerate(final_model_display_order):
        model_to_color[model_id_name] = cmap_for_models(i % cmap_for_models.N) # Cycle through colormap
    # If baseline_id wasn't in unique_fixed_model_ids (it shouldn't be), ensure it gets a color
    if baseline_id not in model_to_color:
        # Assign it the next available color or a default one
        # This case should ideally not happen if baseline_id is handled distinctly from model-generated IDs
        num_assigned_colors = len(model_to_color)
        model_to_color[baseline_id] = cmap_for_models(num_assigned_colors % cmap_for_models.N)
        if baseline_id not in final_model_display_order: # Should have been added first
             final_model_display_order.insert(0, baseline_id) # Ensure it's first for ordering

    # --- Clean existing model-specific output folders ---
    print("Cleaning existing model-specific output folders...")
    fixed_model_ids_to_clean = list(model_path_to_fixed_id.values())
    
    if os.path.exists(args.output_dir):
        for item_name in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item_name)
            if os.path.isdir(item_path) and item_name in fixed_model_ids_to_clean:
                print(f"  Removing existing model folder: {item_path}")
                shutil.rmtree(item_path, ignore_errors=True)
    
    comparative_reports_dir = os.path.join(args.output_dir, "comparative_reports")
    if os.path.exists(comparative_reports_dir):
        print(f"  Removing existing comparative reports: {comparative_reports_dir}")
        shutil.rmtree(comparative_reports_dir, ignore_errors=True)
        
    os.makedirs(comparative_reports_dir, exist_ok=True)
    
    potential_splits_for_dirs = splits_to_eval_original[:]
    if 'test' in splits_to_eval_original and 'val' in splits_to_eval_original:
        potential_splits_for_dirs.append('test+val')

    for split_name in potential_splits_for_dirs:
        safe_split_dir_name = split_name.replace('+', '_plus_')
        os.makedirs(os.path.join(comparative_reports_dir, f"plots_split_{safe_split_dir_name}"), exist_ok=True)

    # --- 2. Determine and Load Primary Manifest ---
    primary_manifest_path = args.primary_manifest_path
    
    if not primary_manifest_path:
        try:
            # Try to load config from the first model to get its manifest path
            # Note: model itself is not kept in memory here, only its config for manifest path
            _, first_model_config = load_model_and_config(model_paths[0], device) 
            primary_manifest_path = first_model_config['data']['manifest_file']
            print(f"Using manifest from first model's config: {primary_manifest_path}")
        except Exception as e:
            print(f"Error loading config for first model {model_paths[0]} to get manifest: {e}")
            print("Please specify --primary_manifest_path or ensure the first model's config.yaml and "
                  "data.manifest_file entry are valid.")
            return
    
    if not os.path.exists(primary_manifest_path):
        print(f"Error: Primary manifest file not found: {primary_manifest_path}")
        return
        
    print(f"Loading primary manifest: {primary_manifest_path}")
    try:
        main_df_original = pd.read_csv(primary_manifest_path)
    except Exception as e:
        print(f"Error reading primary manifest {primary_manifest_path}: {e}")
        return

    # Handle different column names in different manifest versions
    ranking_col = "ranking_confidence" if "ranking_confidence" in main_df_original.columns else "ranking_score"
    required_cols = ['complex_id', 'label', 'tm_normalized', ranking_col, 'split']
    if not all(col in main_df_original.columns for col in required_cols):
        missing = [col for col in required_cols if col not in main_df_original.columns]
        print(f"Error: Primary manifest {primary_manifest_path} is missing required columns: {missing}")
        return

    # --- 3. Data Collection for Models and Baseline ---
    all_results_data = defaultdict(dict) 

    # Load the primary manifest once for baseline and for cases where model-specific manifest might fail
    # but prioritize model-specific manifests for model evaluations.
    main_df_for_baseline_lookup = None
    if primary_manifest_path and os.path.exists(primary_manifest_path):
        try:
            main_df_for_baseline_lookup = pd.read_csv(primary_manifest_path)
            # Use the same ranking_col determined earlier
            required_cols_baseline = ['complex_id', 'label', 'tm_normalized', ranking_col, 'split']
            if not all(col in main_df_for_baseline_lookup.columns for col in required_cols_baseline):
                missing = [col for col in required_cols_baseline if col not in main_df_for_baseline_lookup.columns]
                print(f"Warning: Primary manifest {primary_manifest_path} is missing required columns: {missing}. Baseline might be affected.")
                main_df_for_baseline_lookup = None # Invalidate if columns are missing
        except Exception as e:
            print(f"Warning: Could not read primary manifest {primary_manifest_path}: {e}. Baseline might be affected.")
            main_df_for_baseline_lookup = None
    else:
        print(f"Warning: Primary manifest path not provided or file not found: {primary_manifest_path}. Baseline might be affected if not using model-specific manifests.")


    # PROCESS ORIGINAL SPLITS
    for split in splits_to_eval_original:
        print(f"Processing split: {split}")

        # Baseline data (uses the primary manifest if available)
        baseline_id = "Baseline (Ranking Confidence)"
        if main_df_for_baseline_lookup is not None:
            split_df_for_baseline = main_df_for_baseline_lookup[main_df_for_baseline_lookup['split'] == split].reset_index(drop=True)
            if not split_df_for_baseline.empty:
                all_results_data[baseline_id][split] = {
                    "predictions": split_df_for_baseline[ranking_col].values,  # Use ranking_col
                    "true_dockq": split_df_for_baseline['label'].values, # Assuming these are already [0,1] or this needs sigmoid too
                    "true_tm": split_df_for_baseline['tm_normalized'].values,
                    "complex_ids": split_df_for_baseline['complex_id'].tolist(),
                }
                print(f"  Added baseline data for split '{split}'. Samples: {len(split_df_for_baseline)}")
            else:
                print(f"  Warning: No data for baseline in split '{split}' from primary manifest. Baseline will be skipped for this split.")
        else:
            print(f"  Warning: Primary manifest not available. Baseline will be skipped for split '{split}'.")

        for model_idx, model_path in enumerate(model_paths):
            # Use the pre-generated fixed ID for this model path
            current_fixed_model_id = model_path_to_fixed_id[model_path]

            print(f"  Evaluating model: {current_fixed_model_id} (Path: {os.path.basename(model_path)})")
            try:
                model, model_config = load_model_and_config(model_path, device)
                
                data_cfg = model_config.get('data', {})
                current_samples_per_complex = data_cfg.get('samples_per_complex', args.default_samples_per_complex_eval)
                
                # Use manifest from the model's own config
                model_specific_manifest_path = data_cfg.get('manifest_file')
                if not model_specific_manifest_path or not os.path.exists(model_specific_manifest_path):
                    print(f"    Warning: Manifest file not found or not specified in config for model {current_fixed_model_id} ({model_specific_manifest_path}). Skipping model for this split.")
                    if primary_manifest_path:
                         print(f"    Consider using --primary_manifest_path {primary_manifest_path} if appropriate for this model.")
                    continue
                
                print(f"    Using model-specific manifest: {model_specific_manifest_path}")
                
                # Load this model-specific manifest to get TM scores later
                try:
                    model_manifest_df = pd.read_csv(model_specific_manifest_path)
                    # Filter this model-specific manifest for the current split for TM score lookup
                    split_df_for_tm_lookup = model_manifest_df[model_manifest_df['split'] == split].reset_index(drop=True)
                    if split_df_for_tm_lookup.empty:
                        print(f"    Warning: No data for split '{split}' in model-specific manifest {model_specific_manifest_path}. Cannot get TM scores.")
                        # continue # Or proceed without TM scores if desired
                except Exception as e:
                    print(f"    Error reading model-specific manifest {model_specific_manifest_path} for TM scores: {e}. Skipping TM for this model/split.")
                    split_df_for_tm_lookup = pd.DataFrame() # Empty DF

                dataloader = get_eval_dataloader(
                    manifest_csv=model_specific_manifest_path, # Use model's own manifest for dataloader
                    split=split,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    samples_per_complex=current_samples_per_complex,
                    feature_transform=data_cfg.get('feature_transform', True),
                    feature_centering=data_cfg.get('feature_centering', False),
                    use_interchain_ca_distances=data_cfg.get('use_interchain_ca_distances', False),
                    use_interchain_pae=data_cfg.get('use_interchain_pae', True),
                    seed=data_cfg.get('seed', 42) # Use seed from model's config
                )
                
                model_preds, true_dockq_from_eval, complex_ids_from_eval = evaluate_model_on_dataloader(model, dataloader, device)

                if len(model_preds) == 0:
                    print(f"    Warning: No predictions from model {current_fixed_model_id} on split {split}. Skipping.")
                    continue
                
                # Align TM scores using the model-specific manifest's data for the current split
                aligned_true_tm = np.full_like(true_dockq_from_eval, np.nan) # Default to NaN
                if not split_df_for_tm_lookup.empty:
                    try:
                        complex_id_df = split_df_for_tm_lookup[['complex_id', 'tm_normalized']].drop_duplicates(subset=['complex_id'])
                        complex_id_to_tm = dict(zip(complex_id_df['complex_id'], complex_id_df['tm_normalized']))
                        aligned_true_tm = np.array([complex_id_to_tm.get(cid, np.nan) for cid in complex_ids_from_eval])
                        
                        nan_count = np.isnan(aligned_true_tm).sum()
                        if nan_count > 0:
                            print(f"    Warning: {nan_count}/{len(aligned_true_tm)} complex IDs from dataloader not found in model-specific manifest for TM lookup. Using NaN for those TM values.")
                            missing_cids_tm = [cid for cid in complex_ids_from_eval if cid not in complex_id_to_tm]
                            if missing_cids_tm: print(f"    First few missing complex IDs for TM: {missing_cids_tm[:5]}")
                    except Exception as e:
                        print(f"    Error aligning TM scores using model-specific manifest: {e}")
                        # aligned_true_tm remains NaN as initialized
                else:
                    print(f"    Skipping TM score alignment as model-specific manifest for split '{split}' is empty or failed to load.")

                if not (len(model_preds) == len(true_dockq_from_eval) == len(aligned_true_tm) == len(complex_ids_from_eval)):
                    print(f"    Critical Error: Length mismatch for {current_fixed_model_id} on split {split}. Skipping.")
                    continue
                
                # Store results using the fixed_model_id
                all_results_data[current_fixed_model_id][split] = {
                    "predictions": model_preds,
                    "true_dockq": true_dockq_from_eval,
                    "true_tm": aligned_true_tm,
                    "complex_ids": complex_ids_from_eval,
                    "model_path_source": model_path # Keep track of the original path for reference if needed
                }
                print(f"    Finished evaluation for {current_fixed_model_id} on {split}. Samples: {len(model_preds)}")

            except FileNotFoundError as e:
                 print(f"    Skipping model {current_fixed_model_id} due to FileNotFoundError: {e}")
            except Exception as e:
                print(f"    Error evaluating model {current_fixed_model_id} on split {split}: {e}")
                import traceback
                traceback.print_exc()
    
    # --- 3.5 Combine Test and Val Data if both were processed ---
    data_for_combined_split = False
    if 'test' in splits_to_eval_original and 'val' in splits_to_eval_original:
        print("--- Combining Test and Val data ---")
        combined_split_name = "test+val"
        for model_id in list(all_results_data.keys()): # Iterate over a copy of keys
            if 'test' in all_results_data[model_id] and 'val' in all_results_data[model_id]:
                test_data = all_results_data[model_id]['test']
                val_data = all_results_data[model_id]['val']
                
                # Check if data is valid np.ndarray before concatenation
                if not (isinstance(test_data['predictions'], np.ndarray) and isinstance(val_data['predictions'], np.ndarray) and
                        isinstance(test_data['true_dockq'], np.ndarray) and isinstance(val_data['true_dockq'], np.ndarray) and
                        isinstance(test_data['true_tm'], np.ndarray) and isinstance(val_data['true_tm'], np.ndarray)):
                    print(f"  Skipping {model_id} for test+val: data arrays are not all numpy arrays.")
                    continue

                combined_preds = np.concatenate([test_data['predictions'], val_data['predictions']])
                combined_dockq = np.concatenate([test_data['true_dockq'], val_data['true_dockq']])
                combined_tm = np.concatenate([test_data['true_tm'], val_data['true_tm']])
                combined_cids = test_data['complex_ids'] + val_data['complex_ids']
                
                all_results_data[model_id][combined_split_name] = {
                    "predictions": combined_preds,
                    "true_dockq": combined_dockq,
                    "true_tm": combined_tm,
                    "complex_ids": combined_cids,
                    # model_path is not directly applicable here, or could be a list/concat if needed
                }
                print(f"  Combined test+val data for {model_id}. Total samples: {len(combined_preds)}")
                data_for_combined_split = True
            else:
                print(f"  Skipping {model_id} for test+val: missing data for 'test' or 'val' split.")
        
        if data_for_combined_split:
            splits_to_process_for_metrics_and_plots = splits_to_eval_original + [combined_split_name]
        else:
            splits_to_process_for_metrics_and_plots = splits_to_eval_original
    else:
        splits_to_process_for_metrics_and_plots = splits_to_eval_original

    # --- 4. Metric Computation ---
    all_metrics_summary = defaultdict(lambda: defaultdict(dict)) 
    advanced_comparisons = defaultdict(lambda: defaultdict(dict))  # Store advanced comparison metrics
    
    print("--- Computing Metrics ---")
    # Use splits_to_process_for_metrics_and_plots which may include 'test+val'
    for model_id, model_split_data in all_results_data.items():
        for split, data in model_split_data.items(): # This will now include 'test+val' if present
            if split not in splits_to_process_for_metrics_and_plots: continue # ensure we only process desired splits

            print(f"  Calculating metrics for: {model_id} - Split: {split}")
            preds = data['predictions']
            true_dockq = data['true_dockq']
            true_tm = data['true_tm']
            complex_ids = data['complex_ids']

            if not isinstance(preds, np.ndarray) or preds.size == 0:
                print(f"    No predictions for {model_id} on split {split}, skipping metrics.")
                continue

            metrics_dict = all_metrics_summary[model_id][split]
            metrics_dict['Pearson_DockQ']  = pearson_r(preds, true_dockq)
            metrics_dict['Spearman_DockQ'] = spearman_r(preds, true_dockq)
            metrics_dict['MSE_DockQ']      = mean_squared_error(true_dockq, preds)

            metrics_dict['Pearson_TM']  = pearson_r(preds, true_tm)
            metrics_dict['Spearman_TM'] = spearman_r(preds, true_tm)
            metrics_dict['MSE_TM']      = mean_squared_error(true_tm, preds)

            per_complex_pearson_dockq = {}
            per_complex_spearman_dockq = {}
            df_temp = pd.DataFrame({'preds': preds, 'true_dockq': true_dockq, 'complex_id': complex_ids})
            for cid, group in df_temp.groupby('complex_id'):
                g_preds, g_dockq = group['preds'].values, group['true_dockq'].values
                if len(g_preds) > 1: 
                    per_complex_pearson_dockq[cid] = pearson_r(g_preds, g_dockq)
                    per_complex_spearman_dockq[cid] = spearman_r(g_preds, g_dockq)
                else:
                    per_complex_pearson_dockq[cid] = np.nan
                    per_complex_spearman_dockq[cid] = np.nan
            metrics_dict['PerComplex_Pearson_DockQ_Dict'] = per_complex_pearson_dockq
            metrics_dict['PerComplex_Spearman_DockQ_Dict'] = per_complex_spearman_dockq
            
            # Calculate average of valid per-complex values
            valid_pc_pearson = [v for v in per_complex_pearson_dockq.values() if not np.isnan(v)]
            metrics_dict['Avg_PerComplex_Pearson_DockQ'] = np.mean(valid_pc_pearson) if valid_pc_pearson else np.nan
            
            valid_pc_spearman = [v for v in per_complex_spearman_dockq.values() if not np.isnan(v)]
            metrics_dict['Avg_PerComplex_Spearman_DockQ'] = np.mean(valid_pc_spearman) if valid_pc_spearman else np.nan

    # --- 4.5 Advanced Comparisons: Model vs Baseline ---
    print("\n--- Computing Advanced Comparisons (Model vs Baseline) ---")
    baseline_id = "Baseline (Ranking Confidence)"
    
    for split in splits_to_process_for_metrics_and_plots:
        if baseline_id not in all_results_data or split not in all_results_data[baseline_id]:
            print(f"  Skipping advanced comparisons for split '{split}' - baseline data not available")
            continue
        
        baseline_data = all_results_data[baseline_id][split]
        baseline_preds = baseline_data['predictions']
        baseline_true_dockq = baseline_data['true_dockq']
        baseline_cids = baseline_data['complex_ids']
        
        for model_id in all_results_data.keys():
            if model_id == baseline_id:
                continue  # Skip comparing baseline to itself
            
            if split not in all_results_data[model_id]:
                continue
            
            model_data = all_results_data[model_id][split]
            model_preds = model_data['predictions']
            model_true_dockq = model_data['true_dockq']
            model_cids = model_data['complex_ids']
            
            print(f"  Comparing {model_id} vs Baseline on split '{split}'")
            print(f"    Model has {len(model_preds)} predictions across {len(set(model_cids))} complexes")
            print(f"    Baseline has {len(baseline_preds)} predictions across {len(set(baseline_cids))} complexes")
            
            # Align data by complex_id for fair comparison
            baseline_dict = {cid: (pred, true) for cid, pred, true in zip(baseline_cids, baseline_preds, baseline_true_dockq)}
            
            aligned_baseline_preds = []
            aligned_model_preds = []
            aligned_true_labels = []
            aligned_cids = []
            
            for cid, m_pred, m_true in zip(model_cids, model_preds, model_true_dockq):
                if cid in baseline_dict:
                    b_pred, b_true = baseline_dict[cid]
                    aligned_baseline_preds.append(b_pred)
                    aligned_model_preds.append(m_pred)
                    aligned_true_labels.append(m_true)  # Use model's true label (should match baseline's)
                    aligned_cids.append(cid)
            
            if len(aligned_model_preds) == 0:
                print(f"    Warning: No overlapping samples between {model_id} and baseline for split '{split}'")
                continue
            
            aligned_baseline_preds = np.array(aligned_baseline_preds)
            aligned_model_preds = np.array(aligned_model_preds)
            aligned_true_labels = np.array(aligned_true_labels)
            
            # 1. Bootstrap statistical significance test
            print(f"    Running bootstrap test (Spearman)...")
            bootstrap_result = bootstrap_correlation_test(
                aligned_model_preds, aligned_baseline_preds, aligned_true_labels,
                n_bootstrap=1000, metric='spearman'
            )
            
            # 2. Top-K success rates
            print(f"    Computing Top-K metrics (success rates + mean rank)...")
            model_topk = compute_topk_metrics(aligned_model_preds, aligned_true_labels, aligned_cids)
            baseline_topk = compute_topk_metrics(aligned_baseline_preds, aligned_true_labels, aligned_cids)
            
            # 2.5 Quality discrimination analysis
            print(f"    Computing quality discrimination (high/medium/low quality identification)...")
            model_quality = compute_quality_discrimination(aligned_model_preds, aligned_true_labels, aligned_cids)
            baseline_quality = compute_quality_discrimination(aligned_baseline_preds, aligned_true_labels, aligned_cids)
            
            # 3. Win/Tie/Loss analysis
            print(f"    Computing Win/Tie/Loss counts...")
            wtl_result = compute_win_tie_loss(aligned_model_preds, aligned_baseline_preds, 
                                              aligned_true_labels, aligned_cids)
            
            # Store results
            advanced_comparisons[model_id][split] = {
                'bootstrap': bootstrap_result,
                'model_topk': model_topk,
                'baseline_topk': baseline_topk,
                'model_quality': model_quality,
                'baseline_quality': baseline_quality,
                'win_tie_loss': wtl_result,
                'n_samples': len(aligned_model_preds),
                'n_complexes': len(set(aligned_cids))
            }
            
            print(f"    ✓ Analysis complete: {len(aligned_model_preds)} samples, {len(set(aligned_cids))} complexes")

    # --- 4.6 Generate Advanced Comparison Plots ---
    print("\n--- Generating Advanced Comparison Plots ---")
    for split in splits_to_process_for_metrics_and_plots:
        safe_split_name_for_dir = split.replace('+', '_plus_')
        
        for model_id in advanced_comparisons.keys():
            if split not in advanced_comparisons[model_id]:
                continue
            
            comp = advanced_comparisons[model_id][split]
            
            # Create advanced plots directory for this model
            model_specific_plot_base_dir = os.path.join(args.output_dir, model_id)
            advanced_plot_dir = os.path.join(model_specific_plot_base_dir, 
                                            f"advanced_analysis_split_{safe_split_name_for_dir}")
            os.makedirs(advanced_plot_dir, exist_ok=True)
            
            print(f"  Creating advanced plots for {model_id} on split '{split}'")
            
            # 1. Bootstrap distribution plot
            if 'bootstrap_diffs' in comp['bootstrap']:
                plot_bootstrap_distribution(
                    bootstrap_diffs=comp['bootstrap']['bootstrap_diffs'],
                    observed_diff=comp['bootstrap']['diff_mean'],
                    ci_lower=comp['bootstrap']['ci_lower'],
                    ci_upper=comp['bootstrap']['ci_upper'],
                    p_value=comp['bootstrap']['p_value'],
                    model_name=model_id,
                    split_name=split,
                    outpath=os.path.join(advanced_plot_dir, "bootstrap_distribution.png")
                )
            
            # 2. Top-K success rate comparison
            plot_topk_comparison(
                model_topk=comp['model_topk'],
                baseline_topk=comp['baseline_topk'],
                model_name=model_id,
                split_name=split,
                outpath=os.path.join(advanced_plot_dir, "topk_success_rates.png")
            )
            
            # 3. Win/Tie/Loss visualization
            wtl = comp['win_tie_loss']
            plot_win_tie_loss(
                wins=wtl['wins'],
                ties=wtl['ties'],
                losses=wtl['losses'],
                model_name=model_id,
                split_name=split,
                outpath=os.path.join(advanced_plot_dir, "win_tie_loss_analysis.png")
            )
            
            print(f"    ✓ Saved plots to {advanced_plot_dir}")

    # --- 5. Visualization ---
    print("--- Generating Plots ---")
    for split in splits_to_process_for_metrics_and_plots:
        safe_split_name_for_dir = split.replace('+', '_plus_')
        comparative_split_plot_dir = os.path.join(args.output_dir, "comparative_reports", f"plots_split_{safe_split_name_for_dir}")
        
        combined_scatter_data_dockq = []
        combined_scatter_data_tm = []

        has_data_for_split = any(split in all_results_data[fid] for fid in all_results_data)
        if not has_data_for_split:
            print(f"No data processed for split '{split}'. Skipping plots for this split.")
            continue

        # Iterate using the fixed model IDs from all_results_data keys
        for fixed_model_id in all_results_data.keys(): 
            if fixed_model_id == baseline_id and split == 'test+val': # Baseline handles combined data internally
                if 'test+val' not in all_results_data[baseline_id]: continue # Skip if baseline has no combined data
            elif split not in all_results_data[fixed_model_id]: 
                continue
            
            data = all_results_data[fixed_model_id][split]
            preds = data.get('predictions') # Use .get for safety
            true_dockq = data.get('true_dockq')
            true_tm = data.get('true_tm')
            c_ids = data.get('complex_ids')

            if preds is None or true_dockq is None or true_tm is None or c_ids is None or not isinstance(preds, np.ndarray) or preds.size == 0:
                print(f"    Skipping plots for {fixed_model_id} on split {split} due to missing or empty data arrays.")
                continue

            # Use fixed_model_id for the model-specific directory path
            model_specific_plot_base_dir = os.path.join(args.output_dir, fixed_model_id)
            os.makedirs(model_specific_plot_base_dir, exist_ok=True) # Ensure base model dir exists
            
            model_specific_split_plot_dir = os.path.join(model_specific_plot_base_dir, f"plots_split_{safe_split_name_for_dir}")
            
            if os.path.exists(model_specific_split_plot_dir):
                shutil.rmtree(model_specific_split_plot_dir, ignore_errors=True)
            os.makedirs(model_specific_split_plot_dir, exist_ok=True)

            plot_scatter_by_complex(
                true_dockq, preds, c_ids,
                xlabel="True DockQ", ylabel="Model Prediction",
                title=f"True DockQ vs. {fixed_model_id} Prediction - Split: {split}",
                outpath=os.path.join(model_specific_split_plot_dir, f"true_dockq_vs_prediction_by_complex.png")
            )
            plot_scatter_by_complex(
                true_tm, preds, c_ids,
                xlabel="True TM Score", ylabel="Model Prediction",
                title=f"True TM Score vs. {fixed_model_id} Prediction - Split: {split}",
                outpath=os.path.join(model_specific_split_plot_dir, f"true_tm_vs_prediction_by_complex.png")
            )
            
            # Plot Model vs Baseline Ranking Confidence (if this is not the baseline itself)
            if fixed_model_id != baseline_id and baseline_id in all_results_data and split in all_results_data[baseline_id]:
                baseline_data = all_results_data[baseline_id][split]
                baseline_preds = baseline_data.get('predictions')
                baseline_cids = baseline_data.get('complex_ids')
                
                # Align baseline predictions with current model's samples by complex_id
                if baseline_preds is not None and baseline_cids is not None:
                    # Create a mapping from complex_id to baseline prediction
                    baseline_cid_to_pred = dict(zip(baseline_cids, baseline_preds))
                    # Map model's complex_ids to baseline predictions
                    aligned_baseline = np.array([baseline_cid_to_pred.get(cid, np.nan) for cid in c_ids])
                    
                    # Only plot if we have valid alignments
                    valid_mask = ~np.isnan(aligned_baseline) & ~np.isnan(preds)
                    if valid_mask.sum() > 0:
                        plot_scatter_by_complex(
                            aligned_baseline, preds, c_ids,
                            xlabel="Baseline Ranking Confidence", 
                            ylabel=f"{fixed_model_id} Prediction",
                            title=f"Baseline vs. {fixed_model_id} Predictions - Split: {split}",
                            outpath=os.path.join(model_specific_split_plot_dir, f"baseline_vs_model_prediction_by_complex.png"),
                            ref_line=True
            )
            
            # Plot Per-Complex Spearman Correlation (Model vs Baseline)
            if fixed_model_id != baseline_id and baseline_id in all_metrics_summary and \
               split in all_metrics_summary[baseline_id] and \
               'PerComplex_Spearman_DockQ_Dict' in all_metrics_summary[baseline_id][split] and \
               fixed_model_id in all_metrics_summary and \
               split in all_metrics_summary[fixed_model_id] and \
               'PerComplex_Spearman_DockQ_Dict' in all_metrics_summary[fixed_model_id][split]:
                
                baseline_pc_spearman_dict = all_metrics_summary[baseline_id][split]['PerComplex_Spearman_DockQ_Dict']
                model_pc_spearman_dict = all_metrics_summary[fixed_model_id][split]['PerComplex_Spearman_DockQ_Dict']
                
                # Find common complex IDs and align their Spearman values
                common_cids_for_plot = sorted(list(set(baseline_pc_spearman_dict.keys()) & set(model_pc_spearman_dict.keys())))
                
                aligned_baseline_vals = [baseline_pc_spearman_dict[cid] for cid in common_cids_for_plot]
                aligned_model_vals = [model_pc_spearman_dict[cid] for cid in common_cids_for_plot]
                
                if common_cids_for_plot:
                    plot_per_complex_spearman_comparison(
                        baseline_spearman_values=aligned_baseline_vals,
                        model_spearman_values=aligned_model_vals,
                        common_complex_ids=common_cids_for_plot,
                        model_name=fixed_model_id,
                        baseline_name=baseline_id,
                        split_name=split,
                        outpath=os.path.join(model_specific_split_plot_dir, f"per_complex_spearman_vs_baseline_{safe_split_name_for_dir}.png")
                    )
                else:
                    print(f"    Skipping per-complex Spearman comparison for {fixed_model_id} vs {baseline_id} on split {split} - no common complexes with data.")
            
            # For combined plots, the 'name' field is fixed_model_id
            combined_scatter_data_dockq.append((true_dockq, preds, fixed_model_id, c_ids))
            combined_scatter_data_tm.append((true_tm, preds, fixed_model_id, c_ids))
        
        # Sort combined_scatter_data_dockq and _tm according to final_model_display_order
        # to ensure legend order matches bar chart order.
        if combined_scatter_data_dockq:
            # Create a dict from list of tuples for easier sorting by name
            temp_plot_data_map_dockq = {item[2]: item for item in combined_scatter_data_dockq}
            sorted_plot_data_dockq = [temp_plot_data_map_dockq[model_name] for model_name in final_model_display_order if model_name in temp_plot_data_map_dockq]
            plot_combined_scatter(
                sorted_plot_data_dockq, xlabel="True DockQ", ylabel="Predicted DockQ",
                title=f"All Models & Baseline vs. True DockQ - Split: {split}",
                outpath=os.path.join(comparative_split_plot_dir, f"all_models_vs_dockq_combined_{safe_split_name_for_dir}.png"),
                model_to_color=model_to_color
            )
        if combined_scatter_data_tm:
            temp_plot_data_map_tm = {item[2]: item for item in combined_scatter_data_tm}
            sorted_plot_data_tm = [temp_plot_data_map_tm[model_name] for model_name in final_model_display_order if model_name in temp_plot_data_map_tm]
            plot_combined_scatter(
                sorted_plot_data_tm, xlabel="True TM Score", ylabel="Model Prediction (DockQ Model Output)",
                title=f"All Models & Baseline vs. True TM Score - Split: {split}",
                outpath=os.path.join(comparative_split_plot_dir, f"all_models_vs_tm_combined_{safe_split_name_for_dir}.png"),
                model_to_color=model_to_color
            )

        # Grouped Metric Bar Subplots
        metric_groups = {
            "DockQ_Metrics": [
                # (metric_key, lower_is_better, display_name_for_subplot_title_and_ylabel)
                ('Pearson_DockQ', False, 'Pearson DockQ'), 
                ('Spearman_DockQ', False, 'Spearman DockQ'), 
                ('MSE_DockQ', True, 'MSE DockQ'),
                ('Avg_PerComplex_Pearson_DockQ', False, 'Avg. PC Pearson DockQ'),
                ('Avg_PerComplex_Spearman_DockQ', False, 'Avg. PC Spearman DockQ')
            ],
            "TM_Score_Metrics": [
                ('Pearson_TM', False, 'Pearson TM'), 
                ('Spearman_TM', False, 'Spearman TM'), 
                ('MSE_TM', True, 'MSE TM')
            ]
        }

        for group_name, metrics_in_group in metric_groups.items():
            figure_metrics_data_for_group = []
            for metric_key, lower_is_better, display_name in metrics_in_group:
                current_metric_data_for_key = OrderedDict()
                for model_name_in_order in final_model_display_order:
                    if (model_name_in_order in all_metrics_summary and
                        split in all_metrics_summary[model_name_in_order] and
                        metric_key in all_metrics_summary[model_name_in_order][split]):
                        metric_val = all_metrics_summary[model_name_in_order][split].get(metric_key, np.nan)
                        current_metric_data_for_key[model_name_in_order] = metric_val
                
                # DEBUG PRINT for the specific metric data collected before filtering for the figure group
                print(f"DEBUG: Split: {split}, Metric: {metric_key}, Data collected before subplot add: {dict(current_metric_data_for_key)}")

                if current_metric_data_for_key and not all( (isinstance(v, float) and np.isnan(v)) or v is None for v in current_metric_data_for_key.values()):
                    figure_metrics_data_for_group.append({
                        'metric_key': metric_key,
                        'data': current_metric_data_for_key,
                        'ylabel': display_name.split(' ')[-1], # e.g., "DockQ", "TM", "MSE"
                        'lower_is_better': lower_is_better,
                        'subplot_title': display_name 
                    })
            
            if figure_metrics_data_for_group:
                num_metrics_in_fig = len(figure_metrics_data_for_group)
                num_cols = 2 if num_metrics_in_fig > 1 else 1 # Sensible default for columns
                if num_metrics_in_fig > 4 : num_cols = 3 # For groups with 5 metrics like DockQ
                
                plot_metric_subplots_figure(
                    figure_metrics_data=figure_metrics_data_for_group,
                    overall_figure_title=f"{group_name.replace('_', ' ')} - Split: {split}",
                    outpath=os.path.join(comparative_split_plot_dir, f"grouped_{group_name.lower()}_{safe_split_name_for_dir}.png"),
                    model_to_color=model_to_color,
                    num_cols=num_cols
                )
            else:
                print(f"    Skipping grouped bar chart for {group_name} on split {split} - no valid data for any metrics in the group.")

    # --- 6. Automated Reporting ---
    print("--- Reporting Summary ---")
    report_formats = [fmt.strip().lower() for fmt in args.report_formats.split(',')]
    
    report_data = []
    for model_id, model_split_data in all_metrics_summary.items():
        for split, metrics in model_split_data.items():
            row = {'Model': model_id, 'Split': split}
            # Flatten metrics, excluding the dicts of per-complex values
            for k, v in metrics.items():
                if not isinstance(v, dict): # Exclude PerComplex_..._Dict entries
                    row[k] = f"{v:.4f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else (str(v) if not np.isnan(v) else "NaN")
            report_data.append(row)

    if not report_data:
        print("No summary data to report.")
        return
        
    full_summary_df = pd.DataFrame(report_data)
    
    # Desired column order
    ordered_cols = [
        'Model', 'Split', 
        'Pearson_DockQ', 'Spearman_DockQ', 'MSE_DockQ', 
        'Avg_PerComplex_Pearson_DockQ', 'Avg_PerComplex_Spearman_DockQ',
        'Pearson_TM', 'Spearman_TM', 'MSE_TM'
    ]
    # Ensure all desired columns exist, add others at the end
    final_cols = [col for col in ordered_cols if col in full_summary_df.columns]
    final_cols += [col for col in full_summary_df.columns if col not in final_cols]
    full_summary_df = full_summary_df[final_cols]


    if "print" in report_formats:
        print("=== Overall Performance Summary ===")
        try:
            # Ensure 'test+val' split appears after 'test' and 'val' if present
            if 'test+val' in full_summary_df['Split'].unique():
                split_order = [s for s in splits_to_eval_original if s in full_summary_df['Split'].unique()]
                if 'test' in split_order and 'val' in split_order:
                    # Try to insert 'test+val' after 'val'
                    try:
                        val_idx = split_order.index('val')
                        split_order.insert(val_idx + 1, 'test+val')
                    except ValueError: # if 'val' wasn't processed
                        split_order.append('test+val')
                else: # if 'test' or 'val' are not there, just append
                    split_order.append('test+val')
                
                other_splits = [s for s in full_summary_df['Split'].unique() if s not in split_order]
                final_split_order = split_order + other_splits
                full_summary_df['Split'] = pd.Categorical(full_summary_df['Split'], categories=final_split_order, ordered=True)
                full_summary_df = full_summary_df.sort_values('Split')

            print(full_summary_df.to_string(index=False, na_rep="NaN"))
        except Exception as e:
            print(f"Error printing summary table: {e}. Printing raw DF: {full_summary_df}")

        for split_val in splits_to_eval_original: # Use split_val to avoid conflict with outer 'split'
            df_s = full_summary_df[full_summary_df['Split'] == split_val].copy()
            if df_s.empty: continue
            print(f"--- Best Performers for Split: {split_val} ---")
            
            for metric_col, higher_is_better_flag in [
                ('Pearson_DockQ', True), ('Spearman_DockQ', True), ('MSE_DockQ', False),
                ('Avg_PerComplex_Pearson_DockQ', True), ('Avg_PerComplex_Spearman_DockQ', True),
                ('Pearson_TM', True), ('Spearman_TM', True), ('MSE_TM', False)
            ]:
                if metric_col not in df_s.columns: continue
                
                # Convert to numeric, coercing errors for robust extraction of best
                df_s[metric_col] = pd.to_numeric(df_s[metric_col], errors='coerce')
                
                if df_s[metric_col].notna().sum() == 0: # Skip if all are NaN after conversion
                    print(f"  {metric_col}: All NaN, cannot determine best.")
                    continue

                if higher_is_better_flag:
                    best_idx = df_s[metric_col].idxmax(skipna=True)
                else:
                    best_idx = df_s[metric_col].idxmin(skipna=True)
                
                if pd.notna(best_idx):
                    best_performer = df_s.loc[best_idx]
                    print(f"  Best {metric_col}{'' if higher_is_better_flag else ' (lower is better)'}: "
                          f"{best_performer['Model']} ({best_performer[metric_col]:.4f})")
                else:
                     print(f"  {metric_col}: Could not determine best (all NaN or empty after filtering).")

    if "csv" in report_formats:
        csv_path = os.path.join(args.output_dir, "comparison_performance_summary.csv")
        # Remove existing file if it exists
        if os.path.exists(csv_path):
            os.remove(csv_path)
        full_summary_df.to_csv(csv_path, index=False, na_rep="NaN")
        print(f"Summary report saved to CSV: {csv_path}")

    if "md" in report_formats:
        md_path = os.path.join(args.output_dir, "comparison_performance_summary.md")
        # Remove existing file if it exists
        if os.path.exists(md_path):
            os.remove(md_path)
        try:
            full_summary_df.to_markdown(md_path, index=False)
            print(f"Summary report saved to Markdown: {md_path}")
        except Exception as e:
            print(f"Error saving Markdown report: {e}. Using basic text dump.")
            with open(md_path, 'w') as f:
                 f.write(full_summary_df.to_string(index=False, na_rep="NaN"))


    # --- 7. Advanced Comparison Report ---
    print("\n" + "="*80)
    print("=== COMPREHENSIVE MODEL vs BASELINE ANALYSIS ===")
    print("="*80)
    
    for split in splits_to_process_for_metrics_and_plots:
        models_with_comparisons = [m for m in advanced_comparisons.keys() if split in advanced_comparisons[m]]
        
        if not models_with_comparisons:
            continue
        
        print(f"\n{'='*80}")
        print(f"SPLIT: {split.upper()}")
        print(f"{'='*80}\n")
        
        for model_id in models_with_comparisons:
            comp = advanced_comparisons[model_id][split]
            boot = comp['bootstrap']
            model_topk = comp['model_topk']
            baseline_topk = comp['baseline_topk']
            wtl = comp['win_tie_loss']
            n_samples = comp['n_samples']
            
            print(f"\n{'─'*80}")
            print(f"MODEL: {model_id}")
            print(f"{'─'*80}")
            n_complexes = comp.get('n_complexes', 'Unknown')
            print(f"Dataset: {n_samples} predictions across {n_complexes} unique complexes")
            print(f"(Each complex typically has multiple AlphaFold3 predictions to rank)\n")
            
            # 1. Statistical Significance
            print("📊 STATISTICAL SIGNIFICANCE (Bootstrap Test with 1000 iterations)")
            print("─" * 80)
            print("  Analysis level: GLOBAL (all samples across all complexes)")
            print("  Compares: Model's global correlation vs Baseline's global correlation")
            print("  How it works: Randomly resample your data 1000 times to estimate")
            print("  whether the observed difference could occur by chance.")
            print("  See the bootstrap distribution plot for visualization!\n")
            model_corr = boot['model1_corr']
            baseline_corr = boot['model2_corr']
            diff = boot['diff_mean']
            p_val = boot['p_value']
            ci_low = boot['ci_lower']
            ci_high = boot['ci_upper']
            
            print(f"  Model Spearman ρ:    {model_corr:7.4f}")
            print(f"  Baseline Spearman ρ: {baseline_corr:7.4f}")
            print(f"  Difference (Δρ):     {diff:7.4f}  (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
            print(f"  P-value:             {p_val:7.4f}")
            
            if not np.isnan(p_val):
                if p_val < 0.001:
                    sig_str = "*** HIGHLY SIGNIFICANT"
                elif p_val < 0.01:
                    sig_str = "** VERY SIGNIFICANT"
                elif p_val < 0.05:
                    sig_str = "* SIGNIFICANT"
                else:
                    sig_str = "NOT SIGNIFICANT"
                
                if diff > 0:
                    print(f"\n  ✅ MODEL IS BETTER: {sig_str} (p < {p_val:.4f})")
                    if p_val < 0.05:
                        print(f"     The improvement is statistically significant!")
                    else:
                        print(f"     The improvement is NOT statistically significant.")
                elif diff < 0:
                    print(f"\n  ❌ BASELINE IS BETTER: {sig_str} (p < {p_val:.4f})")
                    if p_val < 0.05:
                        print(f"     The baseline's advantage is statistically significant.")
                    else:
                        print(f"     The difference is NOT statistically significant.")
                else:
                    print(f"\n  ⚖️  PERFORMANCE IS EQUAL")
            
            # 2. Top-K RANKING QUALITY - THE MOST IMPORTANT METRIC!
            print(f"\n{'='*80}")
            print(f"📈 TOP-K RANKING ACCURACY - ⭐ KEY QUESTION: WHICH TOP-K IS BETTER? ⭐")
            print(f"{'='*80}")
            print(f"  Question: 'If I pick top K structures, which ranking gives better quality?'")
            print(f"  Compares: MODEL's top-K selection vs BASELINE's top-K selection")
            print(f"  Per-complex analysis, then averaged across all complexes\n")
            
            # 1. Mean quality of top-K (MOST IMPORTANT!)
            print(f"  1️⃣  MEAN QUALITY OF TOP-K (Higher = Better)")
            print(f"  Average TRUE DockQ of the K structures you selected")
            print(f"  {'-'*80}")
            print(f"  {'K':<6} {'Model Avg':<12} {'Baseline Avg':<13} {'Difference':<12} {'Status'}")
            print(f"  {'-'*6} {'-'*12} {'-'*13} {'-'*12} {'-'*20}")
            
            for k in [1, 3, 5, 10]:
                m_qual = model_topk.get(f'Top{k}_mean_quality', np.nan)
                b_qual = baseline_topk.get(f'Top{k}_mean_quality', np.nan)
                if not np.isnan(m_qual) and not np.isnan(b_qual):
                    diff = m_qual - b_qual
                    status = "✅ Better" if diff > 0.01 else ("❌ Worse" if diff < -0.01 else "⚖️  Similar")
                    print(f"  Top-{k:<2} {m_qual:>10.3f}  {b_qual:>12.3f}  {diff:>+11.3f}  {status}")
            
            # 2. Overlap with true top-K
            print(f"\n  2️⃣  OVERLAP WITH TRUE TOP-K (Higher = Better)")
            print(f"  % of ACTUAL top-K structures that you correctly identified")
            print(f"  {'-'*80}")
            print(f"  {'K':<6} {'Model':<12} {'Baseline':<12} {'Difference':<12} {'Status'}")
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*20}")
            
            for k in [1, 3, 5, 10]:
                m_overlap = model_topk.get(f'Top{k}_overlap', np.nan)
                b_overlap = baseline_topk.get(f'Top{k}_overlap', np.nan)
                if not np.isnan(m_overlap) and not np.isnan(b_overlap):
                    diff = m_overlap - b_overlap
                    status = "✅ Better" if diff > 0.05 else ("❌ Worse" if diff < -0.05 else "⚖️  Similar")
                    print(f"  Top-{k:<2} {m_overlap*100:>10.1f}%  {b_overlap*100:>10.1f}%  {diff*100:>+10.1f}%  {status}")
            
            # 3. Success rate (finding at least one good structure)
            print(f"\n  3️⃣  SUCCESS RATE (Higher = Better)")
            print(f"  % of complexes where top-K contains ≥1 acceptable structure (DockQ > 0.23)")
            print(f"  {'-'*80}")
            print(f"  {'K':<6} {'Model':<12} {'Baseline':<12} {'Difference':<12} {'Status'}")
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*20}")
            
            for k in [1, 3, 5, 10]:
                m_rate = model_topk.get(f'Top{k}', np.nan)
                b_rate = baseline_topk.get(f'Top{k}', np.nan)
                if not np.isnan(m_rate) and not np.isnan(b_rate):
                    diff = m_rate - b_rate
                    status = "✅ Better" if diff > 0.02 else ("❌ Worse" if diff < -0.02 else "⚖️  Similar")
                    print(f"  Top-{k:<2} {m_rate*100:>10.1f}%  {b_rate*100:>10.1f}%  {diff*100:>+10.1f}%  {status}")
            
            print(f"  {'='*80}")
            
            # 2.5. Quality Discrimination Analysis
            print(f"\n{'='*80}")
            print(f"🎯 QUALITY DISCRIMINATION - Can the model identify HIGH/MEDIUM/LOW quality?")
            print(f"{'='*80}")
            
            model_qual = comp.get('model_quality', {})
            baseline_qual = comp.get('baseline_quality', {})
            
            # Mean scores for each quality level
            print(f"\n  AVERAGE SCORE BY TRUE QUALITY LEVEL (Higher score for higher quality = Good)")
            print(f"  Model should score HIGH quality structures higher than LOW quality")
            print(f"  {'-'*80}")
            print(f"  {'Quality Level':<18} {'Model Score':<14} {'Baseline Score':<14} {'Difference':<12}")
            print(f"  {'-'*18} {'-'*14} {'-'*14} {'-'*12}")
            
            for level in ['high', 'medium', 'acceptable', 'poor']:
                m_score = model_qual.get(f'mean_score_{level}', np.nan)
                b_score = baseline_qual.get(f'mean_score_{level}', np.nan)
                if not np.isnan(m_score) and not np.isnan(b_score):
                    diff = m_score - b_score
                    level_name = f"{level.capitalize()} (>{0.80 if level=='high' else 0.49 if level=='medium' else 0.23 if level=='acceptable' else 0})"
                    print(f"  {level_name:<18} {m_score:>12.3f}  {b_score:>12.3f}  {diff:>+11.3f}")
            
            # Check monotonicity (good model should have increasing scores for better quality)
            print(f"\n  QUALITY SEPARATION (scores should increase with quality):")
            m_high = model_qual.get('mean_score_high', np.nan)
            m_poor = model_qual.get('mean_score_poor', np.nan)
            b_high = baseline_qual.get('mean_score_high', np.nan)
            b_poor = baseline_qual.get('mean_score_poor', np.nan)
            
            if not np.isnan(m_high) and not np.isnan(m_poor):
                m_separation = m_high - m_poor
                print(f"  Model separation (High - Poor):    {m_separation:+.3f}")
            if not np.isnan(b_high) and not np.isnan(b_poor):
                b_separation = b_high - b_poor
                print(f"  Baseline separation (High - Poor): {b_separation:+.3f}")
                if not np.isnan(m_separation):
                    if m_separation > b_separation:
                        print(f"  ✅ Model has BETTER quality discrimination!")
                    else:
                        print(f"  ❌ Baseline has BETTER quality discrimination")
            
            # AUC scores
            print(f"\n  AUC FOR BINARY CLASSIFICATION (Higher = Better discrimination)")
            print(f"  Can the ranking identify structures above each quality threshold?")
            print(f"  {'-'*80}")
            print(f"  {'Threshold':<18} {'Model AUC':<14} {'Baseline AUC':<14} {'Difference':<12}")
            print(f"  {'-'*18} {'-'*14} {'-'*14} {'-'*12}")
            
            for level in ['high', 'medium', 'acceptable']:
                m_auc = model_qual.get(f'auc_{level}', np.nan)
                b_auc = baseline_qual.get(f'auc_{level}', np.nan)
                if not np.isnan(m_auc) and not np.isnan(b_auc):
                    diff = m_auc - b_auc
                    threshold = 0.80 if level == 'high' else 0.49 if level == 'medium' else 0.23
                    status = "✅" if diff > 0.02 else ("❌" if diff < -0.02 else "⚖️")
                    print(f"  {level.capitalize()} (>{threshold:<4})    {m_auc:>12.3f}  {b_auc:>12.3f}  {diff:>+11.3f} {status}")
            
            # Precision for high-quality in top-K
            print(f"\n  PRECISION FOR HIGH-QUALITY STRUCTURES (DockQ > 0.80) in Top-K")
            print(f"  What % of top-K selections are actually high quality?")
            print(f"  {'-'*80}")
            print(f"  {'K':<6} {'Model':<12} {'Baseline':<12} {'Difference':<12} {'Status'}")
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*20}")
            
            for k in [1, 3, 5, 10]:
                m_prec = model_qual.get(f'precision_high_top{k}', np.nan)
                b_prec = baseline_qual.get(f'precision_high_top{k}', np.nan)
                if not np.isnan(m_prec) and not np.isnan(b_prec):
                    diff = m_prec - b_prec
                    status = "✅ Better" if diff > 0.05 else ("❌ Worse" if diff < -0.05 else "⚖️  Similar")
                    print(f"  Top-{k:<2} {m_prec*100:>10.1f}%  {b_prec*100:>10.1f}%  {diff*100:>+10.1f}%  {status}")
            
            print(f"  {'='*80}")
            
            # 3. Win/Tie/Loss
            print(f"\n🏆 PER-COMPLEX WIN/TIE/LOSS ANALYSIS")
            print("─" * 80)
            print(f"  Analysis level: STRICTLY PER-COMPLEX")
            print(f"  Compares: For EACH complex, Model's ranking correlation vs Baseline's ranking correlation")
            print(f"  How it works: Compute Spearman ρ between scores and true DockQ for each complex.")
            print(f"  Win = Model's ρ is >0.05 better | Loss = Baseline's ρ >0.05 better | Tie = within 0.05")
            print(f"  This shows consistency: does model beat baseline on MOST complexes?")
            print(f"  See the Win/Tie/Loss plot for visual breakdown!\n")
            wins = wtl['wins']
            ties = wtl['ties']
            losses = wtl['losses']
            total = wins + ties + losses
            win_rate = wtl['win_rate']
            
            if total > 0:
                print(f"  Total complexes analyzed: {total}")
                print(f"  Model Wins:   {wins:3d} ({wins/total*100:5.1f}%)")
                print(f"  Ties:         {ties:3d} ({ties/total*100:5.1f}%)")
                print(f"  Model Losses: {losses:3d} ({losses/total*100:5.1f}%)")
                print(f"\n  Win Rate: {win_rate*100:.1f}%")
                
                if win_rate > 0.55:
                    print(f"  ✅ MODEL WINS MORE OFTEN than baseline!")
                elif win_rate < 0.45:
                    print(f"  ❌ BASELINE WINS MORE OFTEN than model")
                else:
                    print(f"  ⚖️  MODEL AND BASELINE are evenly matched")
                
                # Show top wins and losses
                if wtl['win_complexes']:
                    print(f"\n  Top 5 Complexes Where Model Excels:")
                    for i, (cid, m_corr, b_corr, delta) in enumerate(wtl['win_complexes'][:5], 1):
                        print(f"    {i}. {cid}: Model ρ={m_corr:.3f}, Baseline ρ={b_corr:.3f}, Δ={delta:+.3f}")
                
                if wtl['loss_complexes']:
                    print(f"\n  Top 5 Complexes Where Baseline Excels:")
                    for i, (cid, m_corr, b_corr, delta) in enumerate(wtl['loss_complexes'][:5], 1):
                        print(f"    {i}. {cid}: Model ρ={m_corr:.3f}, Baseline ρ={b_corr:.3f}, Δ={delta:+.3f}")
            
            # Overall verdict - WEIGHTED BY IMPORTANCE
            print(f"\n{'='*80}")
            print(f"OVERALL VERDICT FOR {model_id} on {split}:")
            print(f"{'='*80}")
            
            evidence_better = 0
            evidence_worse = 0
            
            # TOP-K MEAN QUALITY (WEIGHTED x4 - MOST CRITICAL!)
            quality_improvements = []
            for k in [1, 3, 5]:
                m_qual = model_topk.get(f'Top{k}_mean_quality', np.nan)
                b_qual = baseline_topk.get(f'Top{k}_mean_quality', np.nan)
                if not np.isnan(m_qual) and not np.isnan(b_qual):
                    quality_improvements.append(m_qual - b_qual)
            
            if quality_improvements:
                avg_quality_improvement = np.mean(quality_improvements)
                if avg_quality_improvement > 0.01:  # >0.01 DockQ improvement
                    evidence_better += 4  # Quadruple weight!
                    print(f"  ⭐⭐ Top-K Quality: Model's top-K avg {avg_quality_improvement:+.3f} DockQ higher")
                elif avg_quality_improvement < -0.01:
                    evidence_worse += 4
                    print(f"  ⚠️⚠️  Top-K Quality: Baseline's top-K avg {-avg_quality_improvement:.3f} DockQ higher")
            
            # TOP-K OVERLAP (WEIGHTED x3 - VERY IMPORTANT!)
            overlap_improvements = []
            for k in [1, 3, 5]:
                m_overlap = model_topk.get(f'Top{k}_overlap', np.nan)
                b_overlap = baseline_topk.get(f'Top{k}_overlap', np.nan)
                if not np.isnan(m_overlap) and not np.isnan(b_overlap):
                    overlap_improvements.append(m_overlap - b_overlap)
            
            if overlap_improvements:
                avg_overlap_improvement = np.mean(overlap_improvements)
                if avg_overlap_improvement > 0.05:  # >5% better overlap
                    evidence_better += 3
                    print(f"  ⭐ Top-K Overlap: Model captures {avg_overlap_improvement*100:+.1f}% more true top-K")
                elif avg_overlap_improvement < -0.05:
                    evidence_worse += 3
                    print(f"  ⚠️  Top-K Overlap: Baseline captures {-avg_overlap_improvement*100:.1f}% more true top-K")
            
            # Mean rank of best structure (WEIGHTED x2 - VERY IMPORTANT)
            model_rank = model_topk.get('mean_rank_of_best_structure', np.nan)
            baseline_rank = baseline_topk.get('mean_rank_of_best_structure', np.nan)
            if not np.isnan(model_rank) and not np.isnan(baseline_rank):
                rank_diff = model_rank - baseline_rank
                if rank_diff < -0.5:  # Model ranks best structure higher
                    evidence_better += 2  # Double weight!
                    print(f"  ⭐ Best Structure Rank: Model places it {-rank_diff:.1f} positions higher")
                elif rank_diff > 0.5:
                    evidence_worse += 2
                    print(f"  ⚠️  Best Structure Rank: Baseline places it {rank_diff:.1f} positions higher")
            
            # Quality discrimination (WEIGHTED x2 - VERY IMPORTANT)
            model_qual = comp.get('model_quality', {})
            baseline_qual = comp.get('baseline_quality', {})
            m_auc_high = model_qual.get('auc_high', np.nan)
            b_auc_high = baseline_qual.get('auc_high', np.nan)
            if not np.isnan(m_auc_high) and not np.isnan(b_auc_high):
                auc_diff = m_auc_high - b_auc_high
                if auc_diff > 0.02:  # Better at identifying high-quality
                    evidence_better += 2
                    print(f"  ⭐ Quality ID: Model better at identifying high-quality (AUC {auc_diff:+.3f})")
                elif auc_diff < -0.02:
                    evidence_worse += 2
                    print(f"  ⚠️  Quality ID: Baseline better at identifying high-quality (AUC {-auc_diff:.3f})")
            
            # Statistical significance (WEIGHTED x1 - Supporting evidence)
            if not np.isnan(diff) and diff > 0 and p_val < 0.05:
                evidence_better += 1
                print(f"  ✓ Statistical: Significantly better (p={p_val:.4f})")
            elif not np.isnan(diff) and diff < 0 and p_val < 0.05:
                evidence_worse += 1
                print(f"  ✗ Statistical: Significantly worse (p={p_val:.4f})")
            
            # Win rate (WEIGHTED x1 - Supporting evidence)
            if total > 0 and win_rate > 0.55:
                evidence_better += 1
                print(f"  ✓ Win Rate: {win_rate*100:.1f}% (beats baseline on most complexes)")
            elif total > 0 and win_rate < 0.45:
                evidence_worse += 1
                print(f"  ✗ Win Rate: {win_rate*100:.1f}% (loses to baseline on most complexes)")
            
            print(f"\n  TOTAL EVIDENCE SCORE: Model {evidence_better} vs Baseline {evidence_worse}")
            print(f"  (Top-K Quality 4x, Top-K Overlap 3x, Rank 2x, Quality-ID 2x, Stats/WinRate 1x)")
            
            if evidence_better > evidence_worse + 2:
                print(f"\n  🎯 STRONG CONCLUSION: MODEL IS BETTER THAN BASELINE")
                print(f"     → Your model helps users find good structures more effectively!")
            elif evidence_better > evidence_worse:
                print(f"\n  ✅ CONCLUSION: MODEL IS BETTER THAN BASELINE")
                print(f"     → Evidence favors your model, especially for practical ranking")
            elif evidence_worse > evidence_better + 2:
                print(f"\n  ❌ STRONG CONCLUSION: BASELINE IS BETTER THAN MODEL")
                print(f"     → AlphaFold3's ranking is more effective")
            elif evidence_worse > evidence_better:
                print(f"\n  ⚠️  CONCLUSION: BASELINE IS BETTER THAN MODEL")
                print(f"     → Evidence favors baseline ranking")
            else:
                print(f"\n  ⚖️  CONCLUSION: MODEL AND BASELINE HAVE SIMILAR PERFORMANCE")
                print(f"     → No clear winner, performance is comparable")
            print("="*80)
    
    print(f"\nComparison finished. Reports and plots are in: {args.output_dir}")

if __name__ == "__main__":
    main() 
