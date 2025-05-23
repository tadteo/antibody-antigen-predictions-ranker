"""
Test Multiple DeepSet Models on a Test Dataset

This script loads and evaluates one or more trained DeepSet models on a test dataset,
computes the Mean Squared Error (MSE) for each, and generates scatter plots of predictions
vs. true labels.

Usage:
    1. Prepare a manifest CSV (e.g., 'test_manifest.csv') with columns:
        complex_id,h5_file,sample,len_sample,label,bucket,weight_complex,weight_bucket,weight,split

    2. Prepare a text file (default: 'models_to_test.txt') listing the paths to the model .pt files,
       one per line.

    3. Adjust the model and dataloader configuration in the script as needed (input_dim, hidden dims, etc).

    4. Run the script:
        python test_models.py --config configs/config.yaml

    5. Results:
        - MSE for each model is printed to the console.
        - Scatter plots are saved in the 'test_reports' directory.

Arguments and configuration can be modified in the script as needed.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.deep_set import DeepSet
from src.data.dataloader import get_eval_dataloader  # or your test loader
import yaml
import pandas as pd
import argparse
from omegaconf import OmegaConf

def load_model_and_config(model_path, device):
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, 'config.yaml')
    # print(f"Loading config from: {config_path}")  # Debug print
    with open(config_path, 'r') as f:
        config = OmegaConf.load(config_path)
    model_cfg = config['model']
    # print(f"Model config: {model_cfg}")  # Debug print
    
    model = DeepSet(
        input_dim=model_cfg['input_dim'],
        output_dim=model_cfg['output_dim'],
        phi_hidden_dims=model_cfg['phi_hidden_dims'],
        rho_hidden_dims=model_cfg['rho_hidden_dims'],
        aggregator=model_cfg['aggregator']
    )

    checkpoint = torch.load(model_path, map_location=device)
    # If the checkpoint is a dict with 'model_state_dict', use it
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # fallback for plain state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, dataloader, device):
    all_preds, all_labels, all_complex_ids = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            feats = batch['features'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['label'].to(device)
            
            preds = model(feats, lengths)
            # convert back logits to 0-1 range
            preds = torch.sigmoid(preds)
            # also convert labels from logits to 0-1 range
            labels = torch.sigmoid(labels)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # If 'complex_id' is in the batch, add it to the list
            if 'complex_id' in batch:
                all_complex_ids.extend(batch['complex_id'])
    
    flat_preds = np.concatenate(all_preds)
    flat_labels = np.concatenate(all_labels)
    return flat_preds, flat_labels, all_complex_ids

def get_model_info_from_path(model_path):
    # Extract the folder name that contains model details
    folder_name = os.path.basename(os.path.dirname(model_path))
    # Extract key information from the folder name
    try:
        # Parse the model configuration from the folder name
        parts = folder_name.split('_')
        # Get basic model info
        model_type = parts[0]  # e.g., 'DeepSet'
        # Extract date and time information
        # Assuming format like 'DeepSet_2025-04-29_14-42-02'
        date_parts = []
        for part in parts[1:]:
            if part.replace('-', '').isdigit():  # Check if part contains only digits and hyphens
                date_parts.append(part)
            if len(date_parts) == 2:  # We have both date and time
                break
        
        # Get other key parameters if they exist
        # example:DeepSet_2025-04-30_11-24-04_encode_features_True_phi_hidden_dims_128_128_128_rho_hidden_dims_128_128_128_seed_42_samples_per_complex_20_aggregator_concat_stats_by_set_size_weighted_loss_False_lr_scheduler_ReduceLROnPlateau_factor_0.66_patience_10
        params = {}
        for index, part in enumerate(parts):
            #since parts split by underscores if parts have multiple underscores, we need to join them
            if 'seed' == part:
                params['seed'] = parts[index+1]
            elif 'aggregator' == part:
                # get all parts after the aggregator part
                #get index of part weighted and take all parts from aggegator to weighted index
                weighted_index = parts.index('weighted')
                # print(f"weighted_index: {weighted_index}")
                # print(f"parts[index+1:weighted_index]: {parts[index+1:weighted_index]}")
                # concatenate the parts from index+1 to weighted_index not inplace
                tmp_str = ''
                for s in parts[index+1:weighted_index]:
                    tmp_str += s + '_'

                params['aggregator'] = tmp_str

        # print(f"params: {params}")

        # Create a shortened but informative string
        # Format: ModelType_Date_Time_KeyParams
        date_time = '_'.join(date_parts) if date_parts else 'unknown'
        param_str = f"_s{params.get('seed', '')}_agg_{params.get('aggregator', '')}" if params else ''
        
        short_info = f"{model_type}_{date_time}{param_str}"
        return short_info
    except Exception as e:
        # Fallback to a hash-based name if parsing fails
        return f"model_{hash(folder_name) % 10000:04d}"

def plot_results(preds, labels, model_name, model_info, output_dir, split):
    plt.figure()
    plt.scatter(labels, preds, alpha=0.5)
    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    
    # Create a safe filename with limited length but including more info
    safe_model_info = model_info[:100]  # Allow longer names but still limit
    filename = f'{model_name}_{safe_model_info}_scatter_{split}.png'
    
    # Ensure filename is not too long
    if len(filename) > 200:  # Conservative limit
        # Create a shorter filename using hash but keep some readable info
        model_prefix = model_info[:50]  # Keep first 50 chars of model info
        short_hash = hash(filename) % 10000
        filename = f'{model_name}_{model_prefix}_{short_hash:04d}_scatter_{split}.png'
    
    plt.title(f'Model: {model_name}\n{safe_model_info}', fontsize=10)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_results_by_complex(preds, labels, complex_ids, model_name, model_info, output_dir, split):
    """Scatter plot of preds vs labels, colored by complex_id."""
    # Make sure all arrays are flattened
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()
    
    # If complex_ids is empty or None, generate dummy IDs
    if not complex_ids:
        print("No complex IDs provided, generating dummy IDs...")
        complex_ids = [f"complex_{i}" for i in range(len(preds))]
        
    # If number of complex IDs doesn't match predictions, repeat them to match
    if len(complex_ids) != len(preds):
        print(f"Warning: Number of complex IDs ({len(complex_ids)}) doesn't match number of predictions ({len(preds)})")
        print("This might be due to multiple predictions per complex.")
        
        # Option 1: Repeat complex IDs to match predictions (if predictions are ordered by complex)
        # Calculate how many predictions per complex on average
        if len(preds) > len(complex_ids):
            repeat_factor = len(preds) // len(complex_ids)
            remainder = len(preds) % len(complex_ids)
            
            # Repeat each complex ID repeat_factor times
            repeated_cids = []
            for cid in complex_ids:
                repeated_cids.extend([cid] * repeat_factor)
            
            # Add remainder elements if needed
            if remainder > 0:
                repeated_cids.extend([complex_ids[0]] * remainder)
            
            complex_ids = repeated_cids[:len(preds)]  # Truncate to match exactly
            print(f"Expanded complex IDs by repeating them ({repeat_factor} times each)")
        # If more complex IDs than predictions, truncate
        else:
            complex_ids = complex_ids[:len(preds)]
            print("Truncated complex IDs to match predictions")
    
    # Create a numeric mapping for each unique complex_id
    unique_cids = list(dict.fromkeys(complex_ids))
    cid2idx = {cid:i for i,cid in enumerate(unique_cids)}
    colors = np.array([cid2idx[c] for c in complex_ids])
    
    # Ensure all arrays have the same length for the final plot
    min_len = min(len(preds), len(labels), len(colors))
    preds = preds[:min_len]
    labels = labels[:min_len]
    colors = colors[:min_len]
    
    print(f"Final array shapes - preds: {preds.shape}, labels: {labels.shape}, colors: {colors.shape}")

    safe_info = model_info[:100]
    fn = f"{model_name}_{safe_info}_by_complex_{split}.png"
    if len(fn) > 200:
        prefix = safe_info[:50]
        fn = f"{model_name}_{prefix}_{hash(fn)%10000:04d}_by_complex_{split}.png"

    plt.figure(figsize=(6,6))
    sc = plt.scatter(labels, preds, c=colors, cmap='tab20', s=20, alpha=0.6)
    plt.xlabel('True Labels'); plt.ylabel('Predictions')
    plt.title(f"{model_name}\n{safe_info}", fontsize=10)

    # if few complexes, label them in the colorbar
    if len(unique_cids) <= 20:
        cbar = plt.colorbar(sc, ticks=range(len(unique_cids)))
        cbar.ax.set_yticklabels(unique_cids)
    else:
        plt.colorbar(sc, label='complex_id index')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fn), dpi=150)
    plt.close()

def plot_all_results(all_preds_labels, all_model_names, all_model_infos, output_dir, split):
    plt.figure(figsize=(12, 8))  # Larger figure size
    
    # Use a different colormap with more distinct colors
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_preds_labels)))
    
    # Plot diagonal line for reference
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect prediction')
    
    # Plot each model's predictions
    for idx, (preds, labels) in enumerate(all_preds_labels):
        # Use smaller points and less alpha for better visibility
        plt.scatter(labels, preds, alpha=0.3, s=20, 
                   label=f"Model {idx+1}", color=colors[idx])
    
    plt.xlabel('True Labels (DockQ score)', fontsize=12)
    plt.ylabel('Predicted Labels', fontsize=12)
    plt.title('Predictions vs True Labels (All Models)', fontsize=14, pad=20)
    
    # Improve grid
    plt.grid(True, alpha=0.2)
    
    # Set consistent axis limits
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    # Move legend outside of plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Modify the model details text to be more concise
    model_details = []
    for idx, (name, info) in enumerate(zip(all_model_names, all_model_infos)):
        # Limit the length of the info text
        short_info = info[:50] + '...' if len(info) > 50 else info
        model_details.append(f"Model {idx+1}: {name}\n{short_info}")
    
    # Use a smaller font size if there are many models
    fontsize = min(8, 12 - len(model_details) // 2)
    plt.figtext(0.95, 0.05, '\n\n'.join(model_details),
                fontsize=fontsize, ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Save with high DPI
    plt.savefig(os.path.join(output_dir, f'all_models_combined_scatter_{all_model_infos[0]}_{split}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # --- Setup argument parsing ---
    parser = argparse.ArgumentParser(
        description="Test DeepSet models and generate evaluation plots"
    )
    parser.add_argument(
        '--model_list', type=str, default=None,
        help='Path to file listing models to test. Overrides config if specified.'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save evaluation plots. Overrides config if specified.'
    )
    
    # Parse known arguments (config path, model_list, output_dir)
    # and collect unknown arguments for OmegaConf to parse as overrides
    args, unknown_cli_args = parser.parse_known_args()

    # Load base config from YAML
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)


    # Merge CLI-provided parameters into the config
    cli_provided_params = {}
    if args.model_list is not None:
        cli_provided_params['model_list'] = args.model_list
    else:
        model_list_file = 'configs/models_to_test.txt'
    if args.output_dir is not None:
        cli_provided_params['output_dir'] = args.output_dir
    else:
        output_dir = 'test_reports'
    
    if cli_provided_params:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_provided_params))

    # Parse and merge remaining command line arguments as OmegaConf overrides
    if unknown_cli_args:
        try:
            override_conf = OmegaConf.from_cli(unknown_cli_args)
            cfg = OmegaConf.merge(cfg, override_conf)
        except Exception as e:
            print(f"Warning: Could not parse all CLI overrides with OmegaConf: {e}")
            print(f"Ensure overrides are in 'key=value' format (e.g., data.manifest_file=path/to/manifest). Unknown args received: {unknown_cli_args}")

    # --- Process config values ---
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load model paths ---
    with open(model_list_file) as f:
        model_paths = [line.strip() for line in f if line.strip()]

    # Ensure unique model names
    name_count = {}
    all_preds_labels = []
    all_model_names = []
    all_model_infos = []

    for split in ['val', 'test', 'train']:
        for model_path in model_paths:
            base_name = os.path.basename(model_path).replace('.pt', '')
            model_info = get_model_info_from_path(model_path)
            name_count[base_name] = name_count.get(base_name, 0) + 1
            if name_count[base_name] > 1:
                model_name = f"{base_name}_{name_count[base_name]:02d}"
            else:
                model_name = base_name

            # --- DataLoader ---
            # Get config from model
            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, 'config.yaml')

            with open(config_path, 'r') as f:
                model_config = OmegaConf.load(config_path)

            manifest_file = model_config['data']['manifest_file']
        
            # --- Pull complex_ids straight from the manifest (no dataloader change) ---
            manifest_df = pd.read_csv(manifest_file)
            manifest_df = manifest_df[manifest_df['split'] == split].reset_index(drop=True)
            test_cids = manifest_df['complex_id'].tolist()
            
            # Print diagnostic information about the data
            print(f"Number of unique complexes in manifest for {split} split: {len(set(test_cids))}")
            
            test_loader = get_eval_dataloader(
                manifest_file,  # your test manifest
                split=split,
                batch_size=8,
                num_workers=2,
                feature_transform=model_config['data'].get('feature_transform', True),
                feature_centering=model_config['data'].get('feature_centering', True),
                number_of_samples_per_feature=model_config['data'].get('number_of_samples_per_feature')
            )

            print(f"Testing model: {model_name} with info: {model_info}")
            model = load_model_and_config(model_path, device)
            preds, labels, model_complex_ids = evaluate_model(model, test_loader, device)
            
            # Use the complex IDs from the model if available, otherwise use those from the manifest
            if model_complex_ids:
                print(f"Using {len(model_complex_ids)} complex IDs from model evaluation")
                complex_ids_to_use = model_complex_ids
            else:
                print(f"Using {len(test_cids)} complex IDs from manifest")
                complex_ids_to_use = test_cids
            
            print(f"Number of predictions: {len(preds)}, Number of complex IDs: {len(complex_ids_to_use)}")
            
            # Add some debugging prints
            print(f"Predictions stats - Min: {preds.min():.4f}, Max: {preds.max():.4f}, Mean: {preds.mean():.4f}")
            print(f"Labels stats - Min: {labels.min():.4f}, Max: {labels.max():.4f}, Mean: {labels.mean():.4f}")
            
            mse = ((preds - labels) ** 2).mean()
            print(f"{model_name} - Test MSE: {mse:.4f}")
            plot_results(preds, labels, model_name, model_info, output_dir, split)
            plot_results_by_complex(preds, labels, complex_ids_to_use, model_name, model_info, output_dir, split)
            all_preds_labels.append((preds, labels))
            all_model_names.append(model_name)
            all_model_infos.append(model_info)

        # Plot all models together
        plot_all_results(all_preds_labels, all_model_names, all_model_infos, output_dir, split)

if __name__ == '__main__':
    main()
