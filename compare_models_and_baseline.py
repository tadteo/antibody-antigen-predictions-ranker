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
    # data_cfg = config.get('data', {}) # Not directly used here, but in main dataloader call

    # TODO: Extend this to support different model types if necessary
    model = DeepSet(
        input_dim=model_cfg.get('input_dim'), # Provide default if not in config
        phi_hidden_dims=model_cfg.get('phi_hidden_dims'),
        rho_hidden_dims=model_cfg.get('rho_hidden_dims'),
        aggregator=model_cfg.get('aggregator')
    )

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
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
    parser.add_argument("--output_dir", type=str, default="comparison_reports",
                        help="Directory to save plots and reports.")
    parser.add_argument("--splits", type=str, default="val,test,train",
                        help="Comma-separated list of splits to evaluate (e.g., val,test,train).")
    parser.add_argument("--primary_manifest_path", type=str, default=None,
                        help="Optional: Path to a specific manifest CSV to use for all evaluations. "
                             "If not provided, uses the manifest from the first model's config.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for dataloaders.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for dataloaders.")
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

    required_cols = ['complex_id', 'label', 'tm_normalized', 'ranking_score', 'split']
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
            required_cols = ['complex_id', 'label', 'tm_normalized', 'ranking_score', 'split']
            if not all(col in main_df_for_baseline_lookup.columns for col in required_cols):
                missing = [col for col in required_cols if col not in main_df_for_baseline_lookup.columns]
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
                    "predictions": split_df_for_baseline['ranking_score'].values,
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
                    feature_centering=data_cfg.get('feature_centering', True),
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


    print(f"Comparison finished. Reports and plots are in: {args.output_dir}")

if __name__ == "__main__":
    main() 
