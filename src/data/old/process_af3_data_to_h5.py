#!/usr/bin/env python3

import os
import json
import h5py
import glob
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import concurrent.futures
import multiprocessing

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alphafold_processing.log'),
        logging.StreamHandler()
    ]
)

CSV_PATH = "/proj/elofssonlab/users/x_matta/abag_dataset/scores_alphafold3.csv"

logger = logging.getLogger(__name__)

# Add these lines after imports
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

def extract_chain_info(structure_data):
    """
    Extract chain information from the structure data.
    Returns a mapping of residue indices to chain indices.
    """
    try:
        chains = structure_data.get('chains', [])
        residue_chain_mapping = {}
        
        for chain_idx, chain in enumerate(chains):
            start = chain.get('start', 0)
            end = chain.get('end', 0)
            for res_idx in range(start, end + 1):
                residue_chain_mapping[res_idx] = chain_idx
                
        logger.debug(f"Extracted chain info: {len(residue_chain_mapping)} residues mapped")
        return residue_chain_mapping
    except Exception as e:
        logger.error(f"Error extracting chain info: {str(e)}")
        raise

def process_pae_matrix(pae_matrix, residue_chain_mapping):
    """
    Process the PAE matrix to extract inter-chain interactions.
    Returns arrays for inter-chain indices and corresponding PAE values.
    """
    try:
        n_residues = len(pae_matrix)
        inter_idx = []
        inter_jdx = []
        pae_vals = []
        
        # Iterate through the PAE matrix
        for i in range(n_residues):
            for j in range(n_residues):
                # Only process if residues belong to different chains
                if (i in residue_chain_mapping and 
                    j in residue_chain_mapping and 
                    residue_chain_mapping[i] != residue_chain_mapping[j]):
                    inter_idx.append(i)
                    inter_jdx.append(j)
                    pae_vals.append(pae_matrix[i][j])
        
        logger.debug(f"Processed PAE matrix: found {len(inter_idx)} inter-chain interactions")
        return np.array(inter_idx), np.array(inter_jdx), np.array(pae_vals)
    except Exception as e:
        logger.error(f"Error processing PAE matrix: {str(e)}")
        raise

def plot_pae_matrices(pae_matrix, token_chain_ids, output_prefix):
    """
    Plot the PAE matrix with color-coded interaction types
    """
    fig = plt.figure(figsize=(20, 8))
    
    # Plot 1: Original PAE matrix
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(pae_matrix, cmap='viridis')
    plt.colorbar(im1, label='PAE (Å)')
    plt.title('Original PAE Matrix')
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    
    # Plot 2: Interaction type matrix
    ax2 = plt.subplot(1, 2, 2)
    
    # Get chain boundaries
    unique_chains = np.unique(token_chain_ids)
    n_chains = len(unique_chains)
    chain_boundaries = [0]
    current_chain = token_chain_ids[0]
    for i, chain in enumerate(token_chain_ids):
        if chain != current_chain:
            chain_boundaries.append(i)
            current_chain = chain
    chain_boundaries.append(len(token_chain_ids))
    
    # Create interaction type matrix
    interaction_matrix = np.zeros_like(pae_matrix)
    
    # Create colormaps for intra and inter-chain interactions
    intra_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_chains))
    inter_colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_chains * (n_chains-1) // 2))
    
    # Create custom colormap
    interaction_types = []
    colors = []
    
    # Add intra-chain colors
    for i, chain in enumerate(unique_chains):
        interaction_types.append(f'Intra-{chain}')
        colors.append(intra_colors[i])
    
    # Add inter-chain colors
    inter_idx = 0
    for i, chain_i in enumerate(unique_chains):
        for j, chain_j in enumerate(unique_chains):
            if i < j:
                interaction_types.append(f'Inter-{chain_i}{chain_j}')
                colors.append(inter_colors[inter_idx])
                inter_idx += 1
    
    # Create custom colormap
    custom_cmap = plt.cm.colors.ListedColormap(colors)
    
    # Fill interaction matrix
    for i, chain_i in enumerate(unique_chains):
        start_i = chain_boundaries[i]
        end_i = chain_boundaries[i+1]
        
        for j, chain_j in enumerate(unique_chains):
            start_j = chain_boundaries[j]
            end_j = chain_boundaries[j+1]
            
            if chain_i == chain_j:
                # Intra-chain interactions
                idx = interaction_types.index(f'Intra-{chain_i}')
                interaction_matrix[start_i:end_i, start_j:end_j] = idx + 1
            else:
                # Inter-chain interactions
                if i < j:
                    idx = interaction_types.index(f'Inter-{chain_i}{chain_j}')
                    interaction_matrix[start_i:end_i, start_j:end_j] = idx + 1
                    interaction_matrix[start_j:end_j, start_i:end_i] = idx + 1
    
    # Plot interaction matrix without colorbar
    im2 = ax2.imshow(interaction_matrix, cmap=custom_cmap)
    
    # Add chain labels
    # Top labels
    for i, chain in enumerate(unique_chains):
        mid = (chain_boundaries[i] + chain_boundaries[i+1]) // 2
        ax2.text(mid, -10, f'Chain {chain}', 
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=10, fontweight='bold')
    
    # Left labels (y-axis) - rotated 90 degrees
    for i, chain in enumerate(unique_chains):
        mid = (chain_boundaries[i] + chain_boundaries[i+1]) // 2
        ax2.text(-30, mid, f'Chain {chain}', 
                horizontalalignment='right', verticalalignment='center',
                fontsize=10, fontweight='bold', rotation=90)
    
    ax2.set_title('Chain Interaction Types')
    ax2.set_xlabel('Residue Index')
    ax2.set_ylabel('Residue Index')
    
    # Create custom legend
    legend_elements = []
    for i, (interaction, color) in enumerate(zip(interaction_types, colors)):
        legend_elements.append(plt.Rectangle((0,0), 1, 1, fc=color, label=interaction))
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), 
              loc='upper left', title='Interaction Types')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    fig.savefig(f"{output_prefix}_pae_visualization.png", 
                bbox_inches='tight', dpi=300, 
                facecolor='white', edgecolor='none')
    plt.close(fig)

def plot_interchain_matrix(inter_idx, inter_jdx, pae_vals, matrix_size, token_chain_ids, output_prefix):
    """
    Plot the inter-chain interactions matrix
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Create empty matrix
    inter_matrix = np.zeros((matrix_size, matrix_size))
    
    # Fill matrix with PAE values only for inter-chain interactions
    for idx, jdx, val in zip(inter_idx, inter_jdx, pae_vals):
        inter_matrix[idx, jdx] = val
    
    # Plot matrix
    plt.imshow(inter_matrix, cmap='viridis')
    plt.colorbar(label='PAE (Å)')
    
    # Get chain boundaries
    unique_chains = np.unique(token_chain_ids)
    chain_boundaries = [0]
    current_chain = token_chain_ids[0]
    for i, chain in enumerate(token_chain_ids):
        if chain != current_chain:
            chain_boundaries.append(i)
            current_chain = chain
    chain_boundaries.append(len(token_chain_ids))
    
    # Add white lines between chains
    for boundary in chain_boundaries[1:-1]:
        plt.axhline(y=boundary, color='white', linestyle='-', linewidth=2)
        plt.axvline(x=boundary, color='white', linestyle='-', linewidth=2)
    
    # Add chain labels
    for i, chain in enumerate(unique_chains):
        mid = (chain_boundaries[i] + chain_boundaries[i+1]) // 2
        plt.text(mid, -10, f'Chain {chain}', 
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=10, fontweight='bold')
        plt.text(-10, mid, f'Chain {chain}', 
                horizontalalignment='right', verticalalignment='center',
                fontsize=10, fontweight='bold', rotation=90)
    
    plt.title('Inter-chain PAE Matrix')
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    
    plt.tight_layout()
    fig.savefig(f"{output_prefix}_interchain_matrix.png", 
                bbox_inches='tight', dpi=300, 
                facecolor='white', edgecolor='none')
    plt.close(fig)

def process_json_file(json_path, scores_file, h5_file, complex_name, vis_folder, debug_mode=False, plot_mode=False):
    """
    Modified process_json_file function to include visualization folder
    """
    try:
        logger.debug(f"Processing file: {json_path}")
        
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if debug_mode:
            logger.debug("=== DEBUG MODE ===")
            logger.debug(f"JSON keys available: {list(data.keys())}")
        
        # Extract data from JSON
        pae_matrix = np.array(data['pae'])
        token_chain_ids = np.array(data['token_chain_ids'])
        token_res_ids = np.array(data['token_res_ids'])
        
        if debug_mode:
            logger.debug(f"PAE matrix shape: {pae_matrix.shape}")
            logger.debug(f"Token chain IDs shape: {token_chain_ids.shape}")
            logger.debug(f"Token residue IDs shape: {token_res_ids.shape}")
            logger.debug(f"Unique chain IDs: {np.unique(token_chain_ids)}")
        
        # Extract filename info
        filepath = Path(json_path)
        filename = filepath.stem
        parts = filename.split('_')
        seed_idx = parts.index('seed')
        sample_idx = parts.index('sample')
        seed_num = parts[seed_idx + 1]
        sample_num = parts[sample_idx + 1]
        
        sample_group_name = f"seed_{seed_num}_sample_{sample_num}"
        
        # Create groups in H5 file
        if complex_name not in h5_file:
            complex_group = h5_file.create_group(complex_name)
        else:
            complex_group = h5_file[complex_name]
        
        if sample_group_name in complex_group:
            logger.warning(f"Sample {sample_group_name} already exists. Skipping...")
            return
            
        sample_group = complex_group.create_group(sample_group_name)
        
        # --- New: look up ABag scores in the CSV -----------------
        # must match exactly the sample_id column in your CSV
        sample_id = f"{complex_name}_{sample_group_name}_alphafold3"
        if sample_id in scores_file.index:
            row = scores_file.loc[sample_id]
            # these are floats
            abag_dockq = float(row["abag_dockq"])
            abag_lrmsd = float(row["abag_lrmsd"])
            abag_irmsd = float(row["abag_irmsd"])
            ranking_confidence = float(row["ranking_confidence"])
            ptm = float(row["ptm"])
            iptm = float(row["iptm"])
            tm_normalized_reference = float(row["TM_normalized_reference"])
            tm_normalized_query = float(row["TM_normalized_query"])
        else:
            logger.warning(f"No ABag scores found for {sample_id}, filling with NaN")
            abag_dockq = np.nan
            abag_lrmsd = np.nan
            abag_irmsd = np.nan
            ranking_confidence = np.nan
            ptm = np.nan
            iptm = np.nan
            tm_normalized_reference = np.nan
            tm_normalized_query = np.nan

        # store them in H5
        sample_group.create_dataset("abag_dockq", data=abag_dockq)
        sample_group.create_dataset("abag_lrmsd", data=abag_lrmsd)
        sample_group.create_dataset("abag_irmsd", data=abag_irmsd)
        sample_group.create_dataset("ranking_confidence", data=ranking_confidence)
        sample_group.create_dataset("ptm", data=ptm)
        sample_group.create_dataset("iptm", data=iptm)
        sample_group.create_dataset("tm_normalized_reference", data=tm_normalized_reference)
        sample_group.create_dataset("tm_normalized_query", data=tm_normalized_query)
        # -----------------------------------------------------------


        # Get the scores for the current sample
        # Fix the dtype issue for chain_ids
        token_chain_ids = np.array(data['token_chain_ids'])
        # Convert string chain IDs to integers for storage
        unique_chains = np.unique(token_chain_ids)
        chain_to_int = {chain: idx for idx, chain in enumerate(unique_chains)}
        token_chain_ids_int = np.array([chain_to_int[chain] for chain in token_chain_ids], dtype=np.int32)
        
        # Store the chain mapping in the H5 file
        if debug_mode:
            logger.debug(f"Chain mapping: {chain_to_int}")
        
        # Process inter-chain interactions
        inter_idx = []
        inter_jdx = []
        interchain_pae_vals = []
        
        # Find inter-chain interactions
        for i in range(len(token_chain_ids)):
            for j in range(len(token_chain_ids)):
                # only process if residues belong to different chains
                if token_chain_ids[i] != token_chain_ids[j]:
                    inter_idx.append(i)
                    inter_jdx.append(j)
                    interchain_pae_vals.append(pae_matrix[i][j])
        
        inter_idx = np.array(inter_idx)
        inter_jdx = np.array(inter_jdx)
        interchain_pae_vals = np.array(interchain_pae_vals)
        
        if debug_mode:
            logger.debug(f"Number of inter-chain interactions: {len(inter_idx)}")
            logger.debug("First 5 interactions:")
            for i in range(min(5, len(inter_idx))):
                logger.debug(f"Residue {token_res_ids[inter_idx[i]]} (chain {token_chain_ids[inter_idx[i]]}) -> "
                           f"Residue {token_res_ids[inter_jdx[i]]} (chain {token_chain_ids[inter_jdx[i]]}): "
                           f"PAE = {interchain_pae_vals[i]}")
        
        # Create datasets
        sample_group.create_dataset('chain_ids', data=token_chain_ids_int, dtype='int32')
        sample_group.create_dataset('chain_mapping', data=str(chain_to_int))
        sample_group.create_dataset('residue_ids', data=token_res_ids, dtype='int32')
        sample_group.create_dataset('pae_matrix', data=pae_matrix, dtype='float32')
        sample_group.create_dataset('inter_idx', data=inter_idx, dtype='int32')
        sample_group.create_dataset('inter_jdx', data=inter_jdx, dtype='int32')
        sample_group.create_dataset('interchain_pae_vals', data=interchain_pae_vals, dtype='float32')
        
        # Create visualization if in debug mode
        if plot_mode:
            # Create output prefix with visualization folder path
            output_prefix_pae = os.path.join(vis_folder, "pae", f"{complex_name}_{sample_group_name}")
            output_prefix_interchain = os.path.join(vis_folder, "interchain", f"{complex_name}_{sample_group_name}")
            
            # plot just one every 10
            if (int(seed_num)*4 + int(sample_num)) % 10 == 0:
                # Create regular PAE visualization
                plot_pae_matrices(pae_matrix, token_chain_ids, output_prefix_pae)
                
                # Create inter-chain matrix visualization
                plot_interchain_matrix(inter_idx, inter_jdx, interchain_pae_vals, 
                                    len(token_chain_ids), token_chain_ids, 
                                    output_prefix_interchain)
            
            logger.debug(f"Created visualizations in: {vis_folder}")
        
        logger.debug(f"Successfully processed {complex_name} - {sample_group_name}")
        
    except Exception as e:
        logger.error(f"Error processing file {json_path}: {str(e)}")
        raise

def get_user_confirmation(filepath):
    """
    Ask user for confirmation to override existing file
    """
    while True:
        response = input(f"\nFile {filepath} already exists. Do you want to override it? (y/n): ").lower()
        if response in ['y', 'n']:
            return response == 'y'
        print("Please answer 'y' or 'n'")

def process_complex_folder(scores_file, input_folder, pattern, output_folder, debug_mode=False, plot_mode=False):
    """
    Modified process_complex_folder function to create visualization folder
    """
    try:
        complex_name = Path(input_folder).name
        output_h5 = os.path.join(output_folder, f"{complex_name}_interchain_data.h5")
        
        # Create visualization folder
        vis_folder = os.path.join(output_folder, f"{complex_name}_vis")
        os.makedirs(vis_folder, exist_ok=True)
        os.makedirs(os.path.join(vis_folder, "pae"), exist_ok=True)
        os.makedirs(os.path.join(vis_folder, "interchain"), exist_ok=True)
        logger.info(f"Created visualization folder: {vis_folder}")
                
        if os.path.exists(output_h5):
            #delete the file
            os.remove(output_h5)

            # if debug_mode and get_user_confirmation(output_h5):
            #     logger.info(f"Skipping complex {complex_name} as per user request")
            #     return
        
        json_files = list(Path(input_folder).glob(pattern))
        logger.debug(f"Found {len(json_files)} JSON files to process")
        
        if not json_files:
            logger.warning(f"No files matching pattern '{pattern}' found in {input_folder}")
            return
        
        # Process all files in debug mode now
        with h5py.File(output_h5, 'w') as h5_file:
            for json_path in json_files:
                process_json_file(str(json_path), scores_file, h5_file, complex_name, vis_folder, debug_mode, plot_mode)
        
        logger.info(f"Successfully created {output_h5}")
        
    except Exception as e:
        logger.error(f"Error processing complex folder {input_folder}: {str(e)}")
        raise

def main():
    """
    Main function to process all JSON files and create H5 files per complex.
    """
    try:
        # Define input folders and pattern
        scores_file = pd.read_csv(CSV_PATH, index_col="sample_id")

        input_folder = "/proj/elofssonlab/users/x_matta/abag_dataset/alphafold3"
        pattern = "full_confidences_seed_*_sample_*_alphafold3.json"
        
        output_folder = "/proj/elofssonlab/users/x_matta/abag_dataset_processed_with_ptm"

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)   

        # Set debug mode
        debug_mode = False  # Set to False for normal operation
        plot_mode = False

        if debug_mode:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Starting processing of AlphaFold data {'(DEBUG MODE)' if debug_mode else ''} {'(PLOT MODE)' if plot_mode else ''}")
        
        # Get the number of available CPU cores
        num_cores = 64
        
        # Use ThreadPoolExecutor to process complexes in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            subfolders = [subfolder for subfolder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, subfolder))]
            logger.info(f"Found {len(subfolders)} complexes to process, processing in parallel with {num_cores} cores")
            for subfolder in subfolders:
                folder_path = os.path.join(input_folder, subfolder)
                logger.info(f"Scheduling processing for complex: {subfolder}")
                futures.append(executor.submit(process_complex_folder, scores_file, folder_path, pattern, output_folder, debug_mode, plot_mode))
            
            # Use tqdm to show progress
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing complexes"):
                try:
                    future.result()  # This will raise an exception if the thread raised one
                except Exception as e:
                    logger.error(f"Error in processing: {str(e)}")

        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
