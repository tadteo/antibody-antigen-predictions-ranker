#!/usr/bin/env python
"""
compare_models_and_baseline.py

A comprehensive tool to compare multiple trained DeepSet models (or other compatible models)
and a baseline ranking_confidence score against ground truth DockQ and TM scores.

Functionality:
- Loads models specified in a text file.
- Uses a primary manifest (from the first model's config) for consistent dataset splits
  and ground truth data (DockQ, TM scores, ranking_confidence baseline).
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
from collections import defaultdict

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
    """Run model inference on the dataloader, return predictions."""
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            feats = batch['features'].to(device)
            lengths = batch['lengths'].to(device)
            # Labels from dataloader are not used here for ground truth, primary manifest is used.
            # labels  = batch["label"].to(device) 
            
            logits = model(feats, lengths)
            preds = torch.sigmoid(logits) # Output in [0,1]
            all_preds.append(preds.cpu().numpy())
            
    if not all_preds:
        return np.array([])
    return np.concatenate(all_preds).flatten()

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
    """Extracts an informative string from the model path."""
    try:
        folder_name = os.path.basename(os.path.dirname(model_path))
        # A simple heuristic: often the folder name is descriptive enough
        # Or use basename of the model file itself if folder is generic (e.g. "checkpoints")
        if folder_name.lower() in ["checkpoints", "models", "saved_models"]:
            base_file_name = os.path.basename(model_path).replace('.pt','').replace('.pth','')
            info_str = base_file_name
        else:
            info_str = folder_name
        
        # Limit length
        return info_str[:75].replace(" ", "_").replace("/", "_").replace("\\", "_")

    except Exception:
        return f"model_{hash(model_path) % 10000:04d}"

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

    unique_cids = sorted(list(set(complex_ids)))
    cid_to_idx = {cid: i for i, cid in enumerate(unique_cids)}
    colors_mapped = [cid_to_idx.get(c, -1) for c in complex_ids] 

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(x_values, y_values, c=colors_mapped, cmap='tab20', alpha=0.6, s=20)
    
    if ref_line:
        # Calculate min/max ignoring NaNs
        all_valid_values = np.concatenate([x_values[~np.isnan(x_values)], y_values[~np.isnan(y_values)]])
        if len(all_valid_values) > 0:
            # min_val = np.min(all_valid_values)
            # max_val = np.max(all_valid_values)
            # plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
            plt.plot([0,1],[0,1], 'k--', alpha=0.3) # Explicitly plot 0-1 line
        else: # if all values are NaN
            plt.plot([0,1],[0,1], 'k--', alpha=0.3)


    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if 0 < len(unique_cids) <= 20:
        cbar = plt.colorbar(scatter, ticks=range(len(unique_cids)))
        try: # Set tick labels, handle potential errors if cid_to_idx mapping was incomplete
            cbar.ax.set_yticklabels(unique_cids)
        except Exception as e:
            print(f"Warning: Could not set colorbar labels for {title}: {e}")
            plt.colorbar(scatter, label='Complex ID Index (Labeling Error)')
    elif len(unique_cids) > 0 : # More than 20 unique cids
        plt.colorbar(scatter, label='Complex ID Index')
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_combined_scatter(
    all_plot_data: list, # List of tuples: (x, y, name, complex_ids)
    xlabel: str, ylabel: str, title: str, outpath: str,
    ref_line: bool = True
):
    """Plots multiple prediction sets on a single scatter plot."""
    plt.figure(figsize=(10, 8))
    
    colors = matplotlib.colormaps.get_cmap('tab10')
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    min_overall, max_overall = float('inf'), float('-inf')

    for i, (x_vals, y_vals, name, _) in enumerate(all_plot_data): 
        if len(x_vals) == 0: continue
        valid_mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        x_plot, y_plot = x_vals[valid_mask], y_vals[valid_mask]
        if len(x_plot) == 0: continue

        plt.scatter(x_plot, y_plot, alpha=0.5, s=30, label=name, 
                    color=colors(i % colors.N), 
                    marker=markers[i % len(markers)])
        current_min = np.min(np.concatenate([x_plot, y_plot]))
        current_max = np.max(np.concatenate([x_plot, y_plot]))
        min_overall = min(min_overall, current_min)
        max_overall = max(max_overall, current_max)

    if ref_line and min_overall != float('inf') and max_overall != float('-inf'):
        # plt.plot([min_overall, max_overall], [min_overall, max_overall], 'k--', alpha=0.4, label='Ideal')
        plt.plot([0,1],[0,1], 'k--', alpha=0.4, label='Ideal') # Explicitly plot 0-1 line
    elif ref_line: # Default if no data
         plt.plot([0,1],[0,1], 'k--', alpha=0.4, label='Ideal')


    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if len(all_plot_data) > 0: # Only show legend if there's data
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_metrics_bar_chart(
    metrics_dict: dict, 
    title: str, ylabel: str, outpath: str, lower_is_better: bool = False
):
    """
    Generates a bar chart for comparing a single metric across models/baseline.
    metrics_dict: {model_name: value} for a single metric type.
    """
    if not metrics_dict or all(np.isnan(v) for v in metrics_dict.values()):
        print(f"Skipping bar chart {title} due to no valid metrics data.")
        return

    # Filter out NaN values for sorting and plotting
    valid_metrics = {k: v for k, v in metrics_dict.items() if not np.isnan(v)}
    if not valid_metrics:
        print(f"Skipping bar chart {title} as all metric values are NaN.")
        return

    sorted_items = sorted(valid_metrics.items(), key=lambda item: item[1], reverse=not lower_is_better)
    
    model_names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    num_models = len(model_names)
    cmap = matplotlib.colormaps.get_cmap('viridis')


    plt.figure(figsize=(max(8, num_models * 0.7), 6))
    bars = plt.bar(model_names, values, color=[cmap(i/num_models if num_models > 1 else 0.5) for i in range(num_models)])
    plt.ylabel(ylabel)
    plt.title(title, fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=max(8, 10 - num_models // 3)) # Adjust font size
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', 
                 va='bottom' if yval >= 0 else 'top', ha='center', fontsize=8)

    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
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

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_to_eval = [s.strip() for s in args.splits.split(',') if s.strip()]

    # --- 1. Load Model Paths ---
    if not os.path.exists(args.model_list_file):
        print(f"Error: Model list file not found: {args.model_list_file}")
        return
    with open(args.model_list_file, 'r') as f:
        model_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not model_paths:
        print("No model paths found in the list file.")
        return

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
        main_df = pd.read_csv(primary_manifest_path)
    except Exception as e:
        print(f"Error reading primary manifest {primary_manifest_path}: {e}")
        return

    required_cols = ['complex_id', 'label', 'tm_normalized', 'ranking_confidence', 'split']
    if not all(col in main_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in main_df.columns]
        print(f"Error: Primary manifest {primary_manifest_path} is missing required columns: {missing}")
        return

    # --- 3. Data Collection for Models and Baseline ---
    all_results_data = defaultdict(dict) 

    for split in splits_to_eval:
        print(f"Processing split: {split}")
        split_df = main_df[main_df['split'] == split].reset_index(drop=True)
        if split_df.empty:
            print(f"Warning: No data found for split '{split}' in manifest. Skipping this split.")
            continue

        true_dockq_split = split_df['label'].values
        true_tm_split = split_df['tm_normalized'].values
        ranking_conf_split = split_df['ranking_confidence'].values
        complex_ids_split = split_df['complex_id'].tolist()

        baseline_id = "Baseline (Ranking Confidence)"
        all_results_data[baseline_id][split] = {
            "predictions": ranking_conf_split,
            "true_dockq": true_dockq_split,
            "true_tm": true_tm_split,
            "complex_ids": complex_ids_split,
        }
        print(f"  Added baseline data for split '{split}'. Samples: {len(ranking_conf_split)}")

        for model_idx, model_path in enumerate(model_paths):
            model_short_name = get_model_info_from_path(model_path)
            # Ensure unique names if multiple models have same derived short name
            # For example, if get_model_info_from_path is not very discriminative
            # This simple check might not be robust enough for all cases.
            temp_name = model_short_name
            count = 1
            while temp_name in all_results_data and all_results_data[temp_name].get(split, {}).get("model_path") != model_path :
                # if short name exists AND it's not from the exact same model path (can happen if looping splits)
                temp_name = f"{model_short_name}_{count}"
                count += 1
            model_short_name = temp_name


            print(f"  Evaluating model: {model_short_name} ({os.path.basename(model_path)})")
            try:
                model, model_config = load_model_and_config(model_path, device)
                
                data_cfg = model_config.get('data', {})
                dataloader = get_eval_dataloader(
                    manifest_csv=primary_manifest_path,
                    split=split,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    feature_transform=data_cfg.get('feature_transform', True),
                    feature_centering=data_cfg.get('feature_centering', True),
                )
                
                model_preds = evaluate_model_on_dataloader(model, dataloader, device)

                if len(model_preds) != len(true_dockq_split):
                    print(f"    Warning: Prediction length mismatch for model {model_short_name} on split {split}. "
                          f"Expected {len(true_dockq_split)}, got {len(model_preds)}. Skipping model for this split.")
                    continue
                
                all_results_data[model_short_name][split] = {
                    "predictions": model_preds,
                    "true_dockq": true_dockq_split,
                    "true_tm": true_tm_split,
                    "complex_ids": complex_ids_split,
                    "model_path": model_path 
                }
                print(f"    Finished evaluation. Samples: {len(model_preds)}")

            except FileNotFoundError as e:
                 print(f"    Skipping model {model_short_name} due to FileNotFoundError: {e}")
            except Exception as e:
                print(f"    Error evaluating model {model_short_name} on split {split}: {e}")
                import traceback
                traceback.print_exc()
    
    # --- 4. Metric Computation ---
    all_metrics_summary = defaultdict(lambda: defaultdict(dict)) 
    
    print("--- Computing Metrics ---")
    for model_id, model_split_data in all_results_data.items():
        for split, data in model_split_data.items():
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
    for split in splits_to_eval:
        # Directory for comparative plots for the current split
        comparative_split_plot_dir = os.path.join(args.output_dir, "comparative_reports", f"plots_split_{split}")
        os.makedirs(comparative_split_plot_dir, exist_ok=True)
        
        combined_scatter_data_dockq = []
        combined_scatter_data_tm = []

        # Check if any model has data for this split before proceeding
        has_data_for_split = any(split in all_results_data[mid] for mid in all_results_data)
        if not has_data_for_split:
            print(f"No data processed for split '{split}'. Skipping plots for this split.")
            continue

        for model_id in all_results_data.keys():
            if split not in all_results_data[model_id]: continue
            
            data = all_results_data[model_id][split]
            preds, true_dockq, true_tm, c_ids = data['predictions'], data['true_dockq'], data['true_tm'], data['complex_ids']
            
            if not isinstance(preds, np.ndarray) or preds.size == 0 : continue

            # model_id is already sanitized by get_model_info_from_path and suffixing logic
            # Directory for this specific model and split
            model_specific_split_plot_dir = os.path.join(args.output_dir, model_id, f"plots_split_{split}")
            os.makedirs(model_specific_split_plot_dir, exist_ok=True)

            plot_scatter_by_complex(
                true_dockq, preds, c_ids,
                xlabel="True DockQ", ylabel="Model Prediction",
                title=f"True DockQ vs. {model_id} Prediction - Split: {split}",
                outpath=os.path.join(model_specific_split_plot_dir, f"true_dockq_vs_prediction_by_complex.png")
            )
            plot_scatter_by_complex(
                preds, true_tm, c_ids,
                xlabel=f"Prediction", ylabel="True TM Score",
                title=f"{model_id} vs. True TM Score - Split: {split}",
                outpath=os.path.join(model_specific_split_plot_dir, f"scatter_vs_tm_by_complex.png")
            )
            
            combined_scatter_data_dockq.append((true_dockq, preds, model_id, c_ids))
            combined_scatter_data_tm.append((true_tm, preds, model_id, c_ids))

        if combined_scatter_data_dockq:
            plot_combined_scatter(
                combined_scatter_data_dockq, xlabel="True DockQ", ylabel="Predicted DockQ",
                title=f"All Models & Baseline vs. True DockQ - Split: {split}",
                outpath=os.path.join(comparative_split_plot_dir, f"all_models_vs_dockq_combined_{split}.png")
            )
        if combined_scatter_data_tm:
            plot_combined_scatter(
                combined_scatter_data_tm, xlabel="True TM Score", ylabel="Predicted (DockQ Model Output)",
                title=f"All Models & Baseline vs. True TM Score - Split: {split}",
                outpath=os.path.join(comparative_split_plot_dir, f"all_models_vs_tm_combined_{split}.png")
            )

        # Global metrics bar charts
        metric_types_for_bar = [
            ('Pearson_DockQ', False), ('Spearman_DockQ', False), ('MSE_DockQ', True),
            ('Pearson_TM', False), ('Spearman_TM', False), ('MSE_TM', True),
            ('Avg_PerComplex_Pearson_DockQ', False), ('Avg_PerComplex_Spearman_DockQ', False)
        ]
        for mt, lower_better in metric_types_for_bar:
            current_metric_data = {
                model_id: all_metrics_summary[model_id][split].get(mt, np.nan)
                for model_id in all_results_data.keys() if split in all_metrics_summary[model_id]
            }
            plot_metrics_bar_chart(
                current_metric_data,
                title=f"Global {mt.replace('_', ' ')} - Split: {split}",
                ylabel=mt.replace('_', ' '),
                outpath=os.path.join(comparative_split_plot_dir, f"bar_global_{mt.lower()}_{split}.png"),
                lower_is_better=lower_better
            )
            
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
            print(full_summary_df.to_string(index=False, na_rep="NaN"))
        except Exception as e:
            print(f"Error printing summary table: {e}. Printing raw DF: {full_summary_df}")

        for split_val in splits_to_eval: # Use split_val to avoid conflict with outer 'split'
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
        full_summary_df.to_csv(csv_path, index=False, na_rep="NaN")
        print(f"Summary report saved to CSV: {csv_path}")

    if "md" in report_formats:
        md_path = os.path.join(args.output_dir, "comparison_performance_summary.md")
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
