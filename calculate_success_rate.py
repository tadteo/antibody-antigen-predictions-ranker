#!/usr/bin/env python
"""
calculate_success_rate.py

Script to calculate Top K success rate metrics as defined in the paper.

Success Rate Definition:
- Top K Success Rate: Percentage of antibody-antigen complexes where at least one 
  "near-native" model (DockQ > 0.23) appears in the top K ranked predictions.
- Formula: Success Rate = (Number of complexes with ≥1 near-native in Top K) / (Total complexes)

Core functionality:
- Load and evaluate models or use baseline rankings
- Calculate Top K success rates for specified K values
- Compute additional metrics (mean rank of best, mean DockQ of top K)
- Generate summary reports in CSV and console output
- Compare models against baseline with percentage improvements
"""

import os
import argparse
import json
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict, OrderedDict
from tqdm import tqdm

try:
    from src.models.deep_set import DeepSet
    from src.data.dataloader import get_eval_dataloader
except ImportError:
    print("Warning: Could not import from src. Ensure PYTHONPATH is set correctly.")
    raise


#DEFAULT PATHS

DEFAULT_MODEL_LIST_FILE = "/proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/configs/models_to_test.txt"
# DEFAULT_MANIFEST_FILE = "/proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/data/manifest_new_with_distance_filtered_pae_centered_density_with_clipping_500k_maxlen_esm.csv"
DEFAULT_MANIFEST_FILE = "/proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/data/manifest_new_with_filtered_test_density_with_clipping_500k_maxlen_esm.csv"
# ============================================================================
# HELPER FUNCTIONS (reused from compare_models_and_baseline.py)
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
        for batch in tqdm(dataloader, desc="Evaluating model"):
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


def compare_ranking_scores(model_scores: np.ndarray, complex_ids: list, 
                          manifest_df: pd.DataFrame, ranking_col: str = None) -> dict:
    """
    Compare model ranking scores with AlphaFold ranking scores from manifest.
    
    Returns dictionary with correlation metrics and statistics.
    """
    # Determine ranking column name
    if ranking_col is None:
        if 'ranking_confidence' in manifest_df.columns:
            ranking_col = 'ranking_confidence'
        elif 'ranking_score' in manifest_df.columns:
            ranking_col = 'ranking_score'
        else:
            return {'error': 'No ranking score column found in manifest'}
    
    # Create DataFrame for matching
    pred_df = pd.DataFrame({
        'complex_id': complex_ids,
        'model_score': model_scores
    })
    
    # Group by complex_id and add row numbers within each group
    pred_df['row_num'] = pred_df.groupby('complex_id').cumcount()
    
    # Get AF ranking scores from manifest
    manifest_subset = manifest_df[['complex_id', ranking_col]].copy()
    manifest_subset['row_num'] = manifest_subset.groupby('complex_id').cumcount()
    
    # Merge to match predictions with AF scores
    merged = pred_df.merge(manifest_subset, on=['complex_id', 'row_num'], how='inner')
    
    if len(merged) == 0:
        return {'error': 'No matching samples found between predictions and manifest'}
    
    af_scores = merged[ranking_col].values
    model_scores_matched = merged['model_score'].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(af_scores) | np.isnan(model_scores_matched))
    af_scores_clean = af_scores[valid_mask]
    model_scores_clean = model_scores_matched[valid_mask]
    
    if len(af_scores_clean) < 2:
        return {'error': 'Insufficient valid samples for comparison'}
    
    # Calculate correlations
    try:
        pearson_r, pearson_p = pearsonr(af_scores_clean, model_scores_clean)
    except:
        pearson_r, pearson_p = np.nan, np.nan
    
    try:
        spearman_r, spearman_p = spearmanr(af_scores_clean, model_scores_clean)
    except:
        spearman_r, spearman_p = np.nan, np.nan
    
    # Calculate statistics
    mean_af = np.mean(af_scores_clean)
    mean_model = np.mean(model_scores_clean)
    std_af = np.std(af_scores_clean)
    std_model = np.std(model_scores_clean)
    mean_diff = mean_model - mean_af
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mean_af_score': mean_af,
        'mean_model_score': mean_model,
        'std_af_score': std_af,
        'std_model_score': std_model,
        'mean_difference': mean_diff,
        'n_samples': len(af_scores_clean)
    }


def calculate_combined_scores(predictions: np.ndarray, complex_ids: list, 
                              manifest_df: pd.DataFrame, 
                              model_weight: float = 0.8, ptm_weight: float = 0.2) -> np.ndarray:
    """
    Calculate combined scores: model_weight * model_prediction + ptm_weight * ptm_value.
    
    Matches predictions with manifest rows by complex_id and order.
    """
    # Create a DataFrame for matching
    pred_df = pd.DataFrame({
        'complex_id': complex_ids,
        'model_pred': predictions
    })
    
    # Group by complex_id and add row numbers within each group
    pred_df['row_num'] = pred_df.groupby('complex_id').cumcount()
    
    # Get ptm values from manifest
    if 'ptm' not in manifest_df.columns:
        print("Warning: 'ptm' column not found in manifest. Using model predictions only.")
        return predictions
    
    # Create matching DataFrame from manifest
    manifest_subset = manifest_df[['complex_id', 'ptm']].copy()
    manifest_subset['row_num'] = manifest_subset.groupby('complex_id').cumcount()
    
    # Merge to match predictions with ptm values
    merged = pred_df.merge(manifest_subset, on=['complex_id', 'row_num'], how='left')
    
    # Fill missing ptm values with 0 (or could use mean/model_pred)
    merged['ptm'] = merged['ptm'].fillna(0.0)
    
    # Calculate combined score
    combined_scores = (model_weight * merged['model_pred'].values + 
                       ptm_weight * merged['ptm'].values)
    
    return combined_scores


# ============================================================================
# SUCCESS RATE CALCULATION
# ============================================================================

def calculate_topk_success_rate(predictions: np.ndarray, true_labels: np.ndarray, 
                                complex_ids: list, k_values: list = [1, 5, 10],
                                quality_threshold: float = 0.23) -> dict:
    """
    Calculate Top K success rate as defined in the paper.
    
    For each complex:
    1. Rank predictions by model score (descending)
    2. Check if any of top K have DockQ > threshold
    3. Count as success if yes
    
    Args:
        predictions: Model predictions/scores for ranking
        true_labels: True DockQ values
        complex_ids: List of complex IDs corresponding to each prediction
        k_values: List of K values to evaluate (e.g., [1, 5, 10])
        quality_threshold: DockQ threshold for "near-native" (default: 0.23)
    
    Returns:
        Dictionary with:
        - 'Top{K}_success_rate': Success rate for each K
        - 'Top{K}_success_count': Number of successful complexes for each K
        - 'Top{K}_total_complexes': Total number of complexes evaluated for each K
        - 'Top{K}_mean_dockq': Mean DockQ of top K structures for each K
        - 'mean_rank_of_best': Mean rank of the best structure (highest DockQ)
        - 'per_complex_results': Detailed results per complex (optional)
    """
    df = pd.DataFrame({
        'pred': predictions,
        'true': true_labels,
        'complex_id': complex_ids
    })
    
    results = {}
    per_complex_results = []
    
    # Calculate metrics for each K value
    for k in k_values:
        success_count = 0
        total_complexes = 0
        mean_topk_dockq_list = []
        success_complexes = []
        
        for cid, group in df.groupby('complex_id'):
            if len(group) < k:
                # Skip complexes with fewer than K structures
                continue
            
            total_complexes += 1
            
            # Rank by prediction score (descending)
            sorted_group = group.sort_values('pred', ascending=False)
            topk = sorted_group.head(k)
            
            # Check if any of top K have DockQ > threshold
            topk_dockq = topk['true'].values
            has_near_native = (topk_dockq > quality_threshold).any()
            
            if has_near_native:
                success_count += 1
                success_complexes.append(cid)
            
            # Calculate mean DockQ of top K
            mean_topk_dockq_list.append(np.mean(topk_dockq))
            
            # Store per-complex result
            per_complex_results.append({
                'complex_id': cid,
                'k': k,
                'success': has_near_native,
                'mean_topk_dockq': np.mean(topk_dockq),
                'max_dockq_in_topk': np.max(topk_dockq),
                'num_structures': len(group)
            })
        
        # Calculate success rate
        success_rate = (success_count / total_complexes * 100) if total_complexes > 0 else 0.0
        
        results[f'Top{k}_success_rate'] = success_rate
        results[f'Top{k}_success_count'] = success_count
        results[f'Top{k}_total_complexes'] = total_complexes
        results[f'Top{k}_mean_dockq'] = np.mean(mean_topk_dockq_list) if mean_topk_dockq_list else np.nan
    
    # Calculate mean rank of best structure
    mean_best_ranks = []
    for cid, group in df.groupby('complex_id'):
        if len(group) < 2:
            continue
        
        # Find the structure with highest true DockQ
        best_idx = group['true'].idxmax()
        
        # Rank all structures by prediction (descending)
        sorted_group = group.sort_values('pred', ascending=False)
        
        # Find rank of best structure (1-indexed)
        # Get the position of best_idx in the sorted order
        rank = sorted_group.index.get_loc(best_idx) + 1
        mean_best_ranks.append(rank)
    
    results['mean_rank_of_best'] = np.mean(mean_best_ranks) if mean_best_ranks else np.nan
    
    # Store per-complex results
    results['per_complex_results'] = per_complex_results
    
    return results


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_console_output(results: dict, model_name: str, split: str, k_values: list,
                          baseline_results: dict = None, random_results: dict = None,
                          ranking_comparison: dict = None, score_type: str = "pDockQ"):
    """Format results for console output with optional comparisons.
    
    Args:
        score_type: Description of what score is being used (e.g., "pDockQ" or "combined: 0.8*pDockQ + 0.2*ptm")
    """
    print("\n" + "="*80)
    print(f"Top K Success Rate Results")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Split: {split}")
    print(f"Ranking Score Used: {score_type}")
    print(f"Total Complexes: {results.get('Top1_total_complexes', 'N/A')}")
    print(f"Quality Threshold (near-native): DockQ > 0.23")
    print("-"*80)
    
    # Print ranking score comparison if available
    if ranking_comparison and 'error' not in ranking_comparison:
        print(f"\nRanking Score Comparison (Model vs AlphaFold):")
        print(f"  Samples compared: {ranking_comparison.get('n_samples', 'N/A')}")
        print(f"  Pearson correlation: {ranking_comparison.get('pearson_r', np.nan):.4f}")
        print(f"  Spearman correlation: {ranking_comparison.get('spearman_r', np.nan):.4f}")
        print(f"  Mean AF score: {ranking_comparison.get('mean_af_score', np.nan):.4f}")
        print(f"  Mean Model score: {ranking_comparison.get('mean_model_score', np.nan):.4f}")
        print(f"  Mean difference (Model - AF): {ranking_comparison.get('mean_difference', np.nan):.4f}")
        print("-"*80)
    
    for k in k_values:
        success_rate = results.get(f'Top{k}_success_rate', 0.0)
        success_count = results.get(f'Top{k}_success_count', 0)
        total = results.get(f'Top{k}_total_complexes', 0)
        mean_dockq = results.get(f'Top{k}_mean_dockq', np.nan)
        
        print(f"K={k:2d}: {success_rate:5.1f}% ({success_count}/{total} complexes)", end="")
        if not np.isnan(mean_dockq):
            print(f" | Mean DockQ of Top {k}: {mean_dockq:.3f}", end="")
        
        # Add comparison if baseline/random available
        if baseline_results and model_name not in ['Baseline', 'Random', 'Oracle']:
            baseline_rate = baseline_results.get(f'Top{k}_success_rate', 0.0)
            if baseline_rate > 0:
                improvement = ((success_rate - baseline_rate) / baseline_rate) * 100
                print(f" | {improvement:+.1f}% vs Baseline", end="")
        
        if random_results and model_name not in ['Baseline', 'Random', 'Oracle']:
            random_rate = random_results.get(f'Top{k}_success_rate', 0.0)
            if random_rate > 0:
                improvement = ((success_rate - random_rate) / random_rate) * 100
                print(f" | {improvement:+.1f}% vs Random", end="")
        
        print()
    
    mean_rank = results.get('mean_rank_of_best', np.nan)
    if not np.isnan(mean_rank):
        print(f"\nMean Rank of Best Structure: {mean_rank:.2f}")
    
    print("="*80 + "\n")


def save_csv_summary(all_results: dict, output_path: str):
    """Save summary results to CSV with improvement calculations."""
    summary_data = []
    
    # Get baseline and random results for comparison
    baseline_results = {}
    random_results = {}
    for model_name, split_data in all_results.items():
        for split, results in split_data.items():
            if model_name == 'Baseline':
                baseline_results[split] = results
            elif model_name == 'Random':
                random_results[split] = results
    
    for model_name, split_data in all_results.items():
        for split, results in split_data.items():
            row = {
                'Model': model_name,
                'Split': split,
                'Total_Complexes': results.get('Top1_total_complexes', 0),
                'Mean_Rank_of_Best': results.get('mean_rank_of_best', np.nan),
                'Score_Type': results.get('score_type', 'N/A'),
                'Mean_Raw_pDockQ': results.get('raw_pdockq_mean', np.nan)
            }
            
            # Add ranking comparison metrics if available
            if 'ranking_comparison' in results and 'error' not in results['ranking_comparison']:
                rc = results['ranking_comparison']
                row['Ranking_Pearson_Correlation'] = rc.get('pearson_r', np.nan)
                row['Ranking_Spearman_Correlation'] = rc.get('spearman_r', np.nan)
                row['Ranking_Mean_AF_Score'] = rc.get('mean_af_score', np.nan)
                row['Ranking_Mean_Model_Score'] = rc.get('mean_model_score', np.nan)
                row['Ranking_Mean_Difference'] = rc.get('mean_difference', np.nan)
                row['Ranking_N_Samples'] = rc.get('n_samples', 0)
            
            # Add Top K success rates
            for key, value in results.items():
                if key.startswith('Top') and key.endswith('_success_rate'):
                    k = key.replace('Top', '').replace('_success_rate', '')
                    row[f'Top{k}_Success_Rate'] = value
                    row[f'Top{k}_Success_Count'] = results.get(f'Top{k}_success_count', 0)
                    row[f'Top{k}_Mean_DockQ'] = results.get(f'Top{k}_mean_dockq', np.nan)
                    
                    # Add improvement columns if not baseline/random/oracle
                    if model_name not in ['Baseline', 'Random', 'Oracle']:
                        if split in baseline_results:
                            baseline_val = baseline_results[split].get(f'Top{k}_success_rate', 0.0)
                            if baseline_val > 0:
                                improvement = ((value - baseline_val) / baseline_val) * 100
                                row[f'Top{k}_Improvement_vs_Baseline_%'] = improvement
                        
                        if split in random_results:
                            random_val = random_results[split].get(f'Top{k}_success_rate', 0.0)
                            if random_val > 0:
                                improvement = ((value - random_val) / random_val) * 100
                                row[f'Top{k}_Improvement_vs_Random_%'] = improvement
            
            summary_data.append(row)
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        # Sort columns: Model, Split, Total_Complexes, then Top K metrics
        cols = ['Model', 'Split', 'Total_Complexes', 'Mean_Rank_of_Best']
        topk_cols = sorted([c for c in df.columns if c.startswith('Top') and 'Success_Rate' in c])
        improvement_cols = sorted([c for c in df.columns if 'Improvement' in c])
        other_cols = [c for c in df.columns if c not in cols + topk_cols + improvement_cols]
        df = df[cols + topk_cols + improvement_cols + other_cols]
        df.to_csv(output_path, index=False)
        print(f"Summary CSV saved to: {output_path}")


def save_json_detailed(all_results: dict, output_path: str):
    """Save detailed per-complex results to JSON."""
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Detailed JSON results saved to: {output_path}")


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


def print_comparison_summary(all_results: dict, split: str, k_values: list):
    """Print detailed comparison summary similar to compare_models_and_baseline.py."""
    print("\n" + "="*80)
    print(f"Performance Comparison Summary - Split: {split}")
    print("="*80)
    
    # Get baseline and random results
    baseline_results = None
    random_results = None
    oracle_results = None
    
    for model_name, split_data in all_results.items():
        if split in split_data:
            if model_name == 'Baseline':
                baseline_results = split_data[split]
            elif model_name == 'Random':
                random_results = split_data[split]
            elif model_name == 'Oracle':
                oracle_results = split_data[split]
    
    # Print comparison for each K value
    for k in k_values:
        metric_key = f'Top{k}_success_rate'
        print(f"\n  Top {k} Success Rate:")
        
        # Print baseline/random/oracle first
        if baseline_results:
            val = baseline_results.get(metric_key, 0.0)
            print(f"    Baseline: {val:.2f}%")
        
        if random_results:
            val = random_results.get(metric_key, 0.0)
            print(f"    Random: {val:.2f}%")
        
        if oracle_results:
            val = oracle_results.get(metric_key, 0.0)
            print(f"    Oracle: {val:.2f}%")
        
        # Print models with improvements
        for model_name, split_data in all_results.items():
            if model_name in ['Baseline', 'Random', 'Oracle']:
                continue
            if split not in split_data:
                continue
            
            model_val = split_data[split].get(metric_key, 0.0)
            parts = [f"    {model_name}: {model_val:.2f}% ("]
            
            if baseline_results:
                baseline_val = baseline_results.get(metric_key, 0.0)
                if baseline_val > 0:
                    improvement = ((model_val - baseline_val) / baseline_val) * 100
                    parts.append(f"{improvement:+.1f}% vs Baseline")
            
            if random_results and baseline_results:
                parts.append(", ")
            
            if random_results:
                random_val = random_results.get(metric_key, 0.0)
                if random_val > 0:
                    improvement = ((model_val - random_val) / random_val) * 100
                    parts.append(f"{improvement:+.1f}% vs Random")
            
            parts.append(")")
            print("".join(parts))
    
    print("="*80 + "\n")


# ============================================================================
# MAIN LOGIC
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate Top K success rate metrics for antibody-antigen predictions"
    )
    parser.add_argument(
        "--model_list_file", type=str,
        default=DEFAULT_MODEL_LIST_FILE,
        help="Text file with model paths, one per line (optional)"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Single model path to evaluate (optional)"
    )
    parser.add_argument(
        "--manifest_path", type=str,
        default=DEFAULT_MANIFEST_FILE,
        help="Path to manifest CSV"
    )
    parser.add_argument(
        "--output_dir", type=str, default="success_rate_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Data split to evaluate (e.g., test, val, train)"
    )
    parser.add_argument(
        "--k_values", type=str, default="1,5,10",
        help="Comma-separated K values (e.g., 1,5,10)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.23,
        help="DockQ threshold for near-native classification (default: 0.23)"
    )
    parser.add_argument(
        "--include_baseline", default=True,
        help="Include baseline (AlphaFold ranking confidence) in evaluation"
    )
    parser.add_argument(
        "--include_random", action="store_true",
        help="Include random baseline in evaluation"
    )
    parser.add_argument(
        "--include_oracle", action="store_true",
        help="Include oracle baseline (perfect ranking) in evaluation"
    )
    parser.add_argument(
        "--generate_plots", action="store_true",
        help="Generate bar chart plots for metrics"
    )
    parser.add_argument(
        "--save_json", action="store_true",
        help="Save detailed per-complex results to JSON"
    )
    parser.add_argument(
        "--use_combined_score", action="store_true",
        help="Use combined score (0.8 * model_prediction + 0.2 * ptm) for ranking"
    )
    parser.add_argument(
        "--model_weight", type=float, default=0.8,
        help="Weight for model prediction in combined score (default: 0.8)"
    )
    parser.add_argument(
        "--ptm_weight", type=float, default=0.2,
        help="Weight for PTM value in combined score (default: 0.2)"
    )
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_values = [int(k.strip()) for k in args.k_values.split(',') if k.strip()]
    
    # Default to including baseline if no models specified
    if not args.model_list_file and not args.model_path:
        args.include_baseline = True
    
    # Validate inputs
    if not args.model_list_file and not args.model_path and not args.include_baseline:
        print("Error: Must provide either --model_list_file, --model_path, or --include_baseline")
        return
    
    # Check if default model_list_file exists, if not, allow evaluation with just baseline
    if args.model_list_file and not os.path.exists(args.model_list_file):
        if not args.include_baseline:
            print(f"Warning: Model list file not found: {args.model_list_file}")
            print("Continuing with baseline evaluation only (use --include_baseline)")
        args.model_list_file = None
    
    if not os.path.exists(args.manifest_path):
        print(f"Error: Manifest not found: {args.manifest_path}")
        return
    
    # Load manifest
    try:
        main_df = pd.read_csv(args.manifest_path)
    except Exception as e:
        print(f"Error reading manifest: {e}")
        return
    
    ranking_col = "ranking_confidence" if "ranking_confidence" in main_df.columns else "ranking_score"
    required_cols = ['complex_id', 'label', ranking_col, 'split']
    if not all(col in main_df.columns for col in required_cols):
        print(f"Error: Missing required columns in manifest. Found: {main_df.columns.tolist()}")
        return
    
    # Filter by split
    split_df = main_df[main_df['split'] == args.split].reset_index(drop=True)
    if split_df.empty:
        print(f"Error: No data found for split '{args.split}'")
        return
    
    print(f"\n=== Processing split: {args.split} ===")
    print(f"Total samples: {len(split_df)}")
    print(f"Total complexes: {len(split_df['complex_id'].unique())}")
    print(f"K values: {k_values}")
    print(f"Quality threshold: DockQ > {args.threshold}")
    
    # Collect results
    all_results = defaultdict(dict)
    
    # Generate model colors (similar to compare_models_and_baseline.py)
    # Will be updated as models are added
    cmap = matplotlib.colormaps.get_cmap('tab10')
    model_colors = {}
    
    # Get baseline results for comparison
    baseline_results_dict = None
    random_results_dict = None
    
    # Baseline evaluation
    if args.include_baseline:
        print("\n--- Evaluating Baseline ---")
        baseline_predictions = split_df[ranking_col].values
        baseline_labels = split_df['label'].values
        baseline_cids = split_df['complex_id'].tolist()
        
        baseline_results = calculate_topk_success_rate(
            baseline_predictions, baseline_labels, baseline_cids,
            k_values=k_values, quality_threshold=args.threshold
        )
        all_results["Baseline"][args.split] = baseline_results
        baseline_results_dict = baseline_results
        model_colors["Baseline"] = cmap(1)
        format_console_output(baseline_results, "Baseline", args.split, k_values)
    
    # Random baseline evaluation
    if args.include_random:
        print("\n--- Evaluating Random Baseline ---")
        np.random.seed(42)
        random_predictions = np.random.uniform(0, 1, len(split_df))
        random_labels = split_df['label'].values
        random_cids = split_df['complex_id'].tolist()
        
        random_results = calculate_topk_success_rate(
            random_predictions, random_labels, random_cids,
            k_values=k_values, quality_threshold=args.threshold
        )
        all_results["Random"][args.split] = random_results
        random_results_dict = random_results
        model_colors["Random"] = cmap(0)
        format_console_output(random_results, "Random", args.split, k_values)
    
    # Oracle baseline evaluation (perfect ranking)
    if args.include_oracle:
        print("\n--- Evaluating Oracle Baseline ---")
        oracle_predictions = split_df['label'].values  # Perfect predictions
        oracle_labels = split_df['label'].values
        oracle_cids = split_df['complex_id'].tolist()
        
        oracle_results = calculate_topk_success_rate(
            oracle_predictions, oracle_labels, oracle_cids,
            k_values=k_values, quality_threshold=args.threshold
        )
        all_results["Oracle"][args.split] = oracle_results
        model_colors["Oracle"] = cmap(2)
        format_console_output(oracle_results, "Oracle", args.split, k_values)
    
    # Model evaluation
    model_paths = []
    if args.model_path:
        model_paths = [args.model_path]
    elif args.model_list_file and os.path.exists(args.model_list_file):
        with open(args.model_list_file, 'r') as f:
            model_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    for model_path in model_paths:
        model_name = os.path.basename(os.path.dirname(model_path))
        print(f"\n--- Evaluating Model: {model_name} ---")
        
        try:
            model, config = load_model_and_config(model_path, device)
            data_cfg = config.get('data', {})
            
            batch_size = data_cfg.get('batch_size_per_gpu', 16)
            num_workers = data_cfg.get('num_workers', 20)
            dataloader = get_eval_dataloader(
                manifest_csv=args.manifest_path,
                split=args.split,
                batch_size=batch_size,
                num_workers=num_workers,
                samples_per_complex=data_cfg.get('samples_per_complex', 10),
                feature_transform=data_cfg.get('feature_transform', True),
                feature_centering=data_cfg.get('feature_centering', False),
                use_interchain_ca_distances=data_cfg.get('use_interchain_ca_distances', True),
                use_interchain_pae=data_cfg.get('use_interchain_pae', True),
                use_esm_embeddings=data_cfg.get('use_esm_embeddings', False),
                use_distance_cutoff=data_cfg.get('use_distance_cutoff', True),
                use_file_cache=data_cfg.get('use_file_cache', True),
                cache_size_mb=data_cfg.get('cache_size_mb', 2048),
                max_cached_files=data_cfg.get('max_cached_files', 150),
                seed=data_cfg.get('seed', 42)
            )
            
            preds, labels, cids = evaluate_model(model, dataloader, device)
            
            if len(preds) > 0:
                # preds is the model's pDockQ output (sigmoid of logits)
                # This is the raw predicted DockQ value from the model
                
                # Calculate combined scores if requested
                if args.use_combined_score:
                    # Use combined score: model_weight * pDockQ + ptm_weight * ptm
                    combined_scores = calculate_combined_scores(
                        preds, cids, split_df,
                        model_weight=args.model_weight,
                        ptm_weight=args.ptm_weight
                    )
                    ranking_scores = combined_scores
                    model_name_display = model_name
                    score_type = f"Combined: {args.model_weight:.1f}*pDockQ + {args.ptm_weight:.1f}*ptm"
                else:
                    # Use raw model pDockQ output only
                    ranking_scores = preds
                    model_name_display = model_name
                    score_type = "pDockQ (model output only)"
                
                # Compare with AlphaFold ranking scores
                ranking_comparison = compare_ranking_scores(
                    ranking_scores, cids, split_df, ranking_col=ranking_col
                )
                
                model_results = calculate_topk_success_rate(
                    ranking_scores, labels, cids,
                    k_values=k_values, quality_threshold=args.threshold
                )
                
                # Store ranking comparison and score type in results
                if 'error' not in ranking_comparison:
                    model_results['ranking_comparison'] = ranking_comparison
                model_results['score_type'] = score_type
                model_results['raw_pdockq_mean'] = np.mean(preds)  # Store mean of raw pDockQ
                
                all_results[model_name][args.split] = model_results
                model_colors[model_name] = cmap(len(model_colors) % 10)
                format_console_output(
                    model_results, model_name_display, args.split, k_values,
                    baseline_results=baseline_results_dict,
                    random_results=random_results_dict,
                    ranking_comparison=ranking_comparison if 'error' not in ranking_comparison else None,
                    score_type=score_type
                )
            else:
                print(f"  Warning: No predictions generated for {model_name}")
        
        except Exception as e:
            print(f"  Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Continuing with other models/baselines...")
    
    # Generate bar charts
    if args.generate_plots and all_results:
        print("\n=== Generating Bar Charts ===")
        plots_dir = os.path.join(args.output_dir, f"plots_{args.split}")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate bar charts for each K value
        for k in k_values:
            metric_key = f'Top{k}_success_rate'
            metric_data = OrderedDict()
            
            # Add models in order: Random, Baseline, Oracle, then trained models
            for model_name in ['Random', 'Baseline', 'Oracle']:
                if model_name in all_results and args.split in all_results[model_name]:
                    val = all_results[model_name][args.split].get(metric_key, np.nan)
                    if not np.isnan(val):
                        metric_data[model_name] = val
            
            # Add trained models
            for model_name in sorted(all_results.keys()):
                if model_name not in ['Random', 'Baseline', 'Oracle']:
                    if args.split in all_results[model_name]:
                        val = all_results[model_name][args.split].get(metric_key, np.nan)
                        if not np.isnan(val):
                            metric_data[model_name] = val
            
            if metric_data:
                plot_metric_bars(
                    metric_data,
                    title=f"Top {k} Success Rate - Split: {args.split}",
                    ylabel=f"Top {k} Success Rate (%)",
                    outpath=os.path.join(plots_dir, f"bar_Top{k}_success_rate.png"),
                    model_colors=model_colors,
                    lower_is_better=False
                )
        
        # Generate bar chart for mean rank of best
        metric_data = OrderedDict()
        for model_name in ['Random', 'Baseline', 'Oracle']:
            if model_name in all_results and args.split in all_results[model_name]:
                val = all_results[model_name][args.split].get('mean_rank_of_best', np.nan)
                if not np.isnan(val):
                    metric_data[model_name] = val
        
        for model_name in sorted(all_results.keys()):
            if model_name not in ['Random', 'Baseline', 'Oracle']:
                if args.split in all_results[model_name]:
                    val = all_results[model_name][args.split].get('mean_rank_of_best', np.nan)
                    if not np.isnan(val):
                        metric_data[model_name] = val
        
        if metric_data:
            plot_metric_bars(
                metric_data,
                title=f"Mean Rank of Best Structure - Split: {args.split}",
                ylabel="Mean Rank (lower is better)",
                outpath=os.path.join(plots_dir, "bar_mean_rank_of_best.png"),
                model_colors=model_colors,
                lower_is_better=True
            )
        
        print(f"Bar charts saved to: {plots_dir}")
    
    # Print comparison summary
    if all_results:
        print_comparison_summary(all_results, args.split, k_values)
    
    # Save results
    if all_results:
        csv_path = os.path.join(args.output_dir, f"success_rate_summary_{args.split}.csv")
        save_csv_summary(all_results, csv_path)
        
        if args.save_json:
            json_path = os.path.join(args.output_dir, f"success_rate_detailed_{args.split}.json")
            save_json_detailed(all_results, json_path)
        
        print(f"\nAll results saved to: {args.output_dir}")
        
        # Print summary of what was evaluated
        evaluated_models = list(all_results.keys())
        print(f"\nSuccessfully evaluated: {', '.join(evaluated_models)}")
    else:
        print("\nNo results to save.")
        print("\nNote: If you're trying to evaluate models and getting 'ranking_score' errors,")
        print("      some H5 files may be missing the 'ranking_score' field.")
        print("      Try running with --include_baseline to evaluate baseline only,")
        print("      or use a manifest file that references H5 files with complete data.")


if __name__ == "__main__":
    main()
