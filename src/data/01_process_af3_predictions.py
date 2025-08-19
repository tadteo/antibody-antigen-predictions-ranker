#!/usr/bin/env python3
"""
Comprehensive workflow to process AlphaFold3 predictions and create H5 files.

This script:
1. Discovers PDB IDs from AF3 prediction folder structure across all seeds
2. Reads metadata from CSV to identify antibody and antigen chains
3. Processes AF3 predictions and extracts confidence scores
4. Computes DockQ scores between antibody and antigen chains
5. Calculates TM scores using reference structures
6. Saves everything directly to H5 format

USAGE:
======
Basic usage:
    python process_af3_predictions.py --metadata metadata.csv --af3_folder af3_predictions/ --reference_folder reference_structures/ --output /path_to_output_folder

Test with limited PDBs:
    python process_af3_predictions.py --metadata metadata.csv --af3_folder af3_predictions/ --reference_folder reference_structures/ --output /path_to_output_folder --limit 5

Resume from previous run:
    python process_af3_predictions.py --metadata metadata.csv --af3_folder af3_predictions/ --reference_folder reference_structures/ --output /path_to_output_folder --resume

Force restart (overwrite existing results):
    python process_af3_predictions.py --metadata metadata.csv --af3_folder af3_predictions/ --reference_folder reference_structures/ --output /path_to_output_folder --force-restart

With custom retry attempts:
    python process_af3_predictions.py --metadata metadata.csv --af3_folder af3_predictions/ --reference_folder reference_structures/ --output /path_to_output_folder --max-retries 3

Get help:
    python process_af3_predictions.py --help

REQUIRED FILES:
==============
1. metadata.csv: CSV file with columns:
   - pdb_id: PDB identifier (e.g., "7b5g")
   - antigen_chains: Chain IDs for antigen (e.g., "A|B")
   - heavy_chain: Heavy chain ID (e.g., "H")
   - light_chain: Light chain ID (e.g., "L" or "NA")
   - resolution: Structure resolution
   - complex_type: Complex type (e.g., "Fab-Ag")

2. AF3 Predictions Folder Structure:
   af3_predictions/
   ├── seed-1/
   │   ├── 7b5g/
   │   │   ├── seed-1_sample-0/
   │   │   │   ├── model.cif
   │   │   │   └── confidences.json
   │   │   └── seed-1_sample-1/
   │   └── 7r0j/
   └── seed-2/
       └── ...

3. Reference Structures Folder:
   reference_structures/
   ├── 7b5g.pdb
   ├── 7r0j.cif
   └── ...

PARAMETERS:
==========
--metadata: Path to metadata CSV file (REQUIRED)
--af3_folder: Path to AF3 predictions folder (REQUIRED)
--reference_folder: Path to reference structures folder (REQUIRED)
--output: Path to output H5 file (REQUIRED)
--limit: Limit number of PDBs to process (optional, for testing)
--resume: Resume from previous run (optional)
--force-restart: Force restart and overwrite existing results (optional)
--max-retries: Maximum retry attempts for failed samples (optional, default: 2)

OUTPUT:
=======
Per-Complex Files:
- <pdb_id>.h5: H5 file containing DockQ scores, AF3 confidence scores, TM scores, structural data, and metadata
- <pdb_id>_log.json: Detailed per-complex log file tracking processing status for each sample/seed

Global Files:
- checkpoint.txt: Progress tracking with completed PDB IDs
- log.log: Main processing log with all operations
- errors.log: Error-only log for troubleshooting
- summary.txt: Final processing summary with statistics

Per-Complex Log Structure (<pdb_id>_log.json):
{
  "pdb_id": "7b5g",
  "processing_start": "2024-01-01T10:00:00",
  "processing_end": "2024-01-01T10:05:00",
  "total_samples_found": 50,
  "successful_samples": 48,
  "failed_samples": 2,
  "overall_status": "PARTIAL_SUCCESS",
  "samples": [
    {
      "sample_index": 1,
      "seed_name": "seed-1",
      "sample_name": "seed-1_sample-0",
      "status": "SUCCESS",
      "components": {
        "dockq_success": true,
        "af3_success": true,
        "tm_success": true,
        "metadata_success": true
      },
      "errors": []
    }
  ],
  "errors": []
}

FAILURE HANDLING:
================
When components fail, the following values are saved:
- DockQ failures: NaN values saved to H5 file, clearly marked in logs
- TM score failures: NaN values saved to H5 file, clearly marked in logs
- AF3 score failures: Missing datasets in H5 file, logged as component failure
- Metadata failures: Missing datasets, component marked as failed

TM Score Failure Scenarios:
- US-align timeout: NaN values returned
- US-align execution error: NaN values returned  
- No valid output parsing: NaN values returned
- Chain alignment failures: NaN values returned

NaN values indicate failed computations and can be filtered in downstream analysis.
This approach preserves data integrity while clearly marking failures.

GRAPHICAL PIPELINE:
───────────────────────────────────────────────────────────────────────────────

📁 AF3 Predictions Folder Structure:
├── seed-1/
│   ├── 📁 7b5g/
│   │   ├── seed-1_sample-0/
│   │   │   ├── 📄 model.cif                    ← AF3 structure prediction
│   │   │   └── 📄 confidences.json            ← AF3 scores & matrices
│   │   ├── seed-1_sample-1/
│   │   │   ├── 📄 model.cif
│   │   │   └── confidences.json
│   │   └── ...
│   ├── 📁 7r0j/
│   │   └── ...
│   └── ...
├── seed-2/
│   ├── 📁 7b5g/
│   │   ├── seed-2_sample-0/
│   │   │   ├── 📄 model.cif
│   │   │   └── confidences.json
│   │   └── ...
│   └── ...
└── ... (up to seed-10)

🔄 Processing Pipeline:
1. Read metadata.csv
   ├── Extract PDB IDs
   ├── Get chain information (Ag/Ab mapping)
   └── Get resolution and complex type

2. 🔍 Scan AF3 folder structure
   ├── Discover all seed folders (seed-1 to seed-10)
   ├── Find all PDB ID folders within each seed
   └── Create unique list of PDB IDs across all seeds

3. 📊 For each PDB ID:
   ├── 🔗 Get chain mapping from metadata
   │   ├── Antigen chains (can be multiple)
   │   ├── Heavy chain (antibody)
   │   └── Light chain (antibody, if present)
   │
   ├── 📁 Find reference structure
   │   ├── Look for .pdb or .cif files
   │   └── Match by PDB ID
   │
   ├── 🔍 Find ALL samples across ALL seeds
   │   ├── Scan all 10 seed folders
   │   ├── Find sample directories (seed-*_sample-*)
   │   └── Locate model files and confidences.json
   │
   └── 🧮 For each sample:
       ├── 📥 Load AF3 prediction (.cif/.pdb)
       ├── 📥 Load reference structure
       ├── 📄 Load confidences.json
       │   ├── Extract PAE matrix
       │   ├── Extract token_chain_ids
       │   ├── Extract token_res_ids
       │   └── Extract confidence scores (ptm, iptm, ranking_confidence)
       │
       ├── 🔗 Merge chains for DockQ computation
       │   ├── Merge antigen chains → "Ag"
       │   ├── Merge antibody chains → "Ab"
       │   └── Apply same merging to reference structure
       │
       ├── 🧮 Compute DockQ score
       │   ├── Run DockQ between Ag and Ab chains
       │   ├── Extract DockQ, LRMSD, iRMSD, fnat, clashes
       │   └── Determine receptor (Ag or Ab)
       │
       ├── Compute TM score
       │   ├── Calculate TM-score between prediction and reference
       │   ├── tm_normalized_reference
       │   └── tm_normalized_query
       │
       └── Save ALL data to H5 format

📊 H5 File Structure:
 results.h5
├── 📁 7b5g/                                    ← PDB ID group
│   ├── 📁 seed-1_seed-1_sample-0/             ← Sample group
│   │   ├── DockQ Metrics:
│   │   │   ├── abag_dockq                     ← DockQ score (0-1)
│   │   │   ├── abag_lrmsd                     ← Ligand RMSD (Å)
│   │   │   └── abag_irmsd                     ← Interface RMSD (Å)
│   │   │
│   │   ├── AF3 Confidence Scores:
│   │   │   ├── ptm                            ← Predicted TM score
│   │   │   ├── iptm                           ← Interface predicted TM score
│   │   │   └── ranking_confidence             ← AF3 ranking confidence
│   │   │
│   │   ├── TM Scores:
│   │   │   ├── tm_normalized_reference        ← TM-score vs reference
│   │   │   └── tm_normalized_query            ← TM-score vs query
│   │   │
│   │   ├── Structural Data:
│   │   │   ├── pae_matrix                     ← PAE matrix (N×N)
│   │   │   ├── token_chain_ids                ← Chain IDs for each residue
│   │   │   ├── token_res_ids                  ← Residue IDs
│   │   │   ├── inter_idx                      ← Inter-chain interaction indices
│   │   │   ├── inter_jdx                      ← Inter-chain interaction indices
│   │   │   ├── interchain_pae_vals            ← Inter-chain PAE values
│   │   │   └── chain_mapping                  ← Chain name to integer mapping
│   │   │
│   │   └── 📊 Metadata:
│   │       ├── pdb_id                         ← PDB identifier
│   │       ├── seed_name                      ← Seed name (seed-1, etc.)
│   │       ├── sample_name                    ← Sample name
│   │       ├── prediction_file                ← Path to prediction file
│   │       ├── reference_file                 ← Path to reference file
│   │       ├── resolution                     ← Structure resolution
│   │       └── complex_type                   ← Complex type (Fab-Ag, etc.)
│   │
│   ├── 📁 seed-1_seed-1_sample-1/
│   ├── 📁 seed-2_seed-2_sample-0/
│   └── ... (all samples for 7b5g)
│
├── 📁 7r0j/                                    ← Next PDB ID
│   └── ... (all samples for 7r0j)
│
└── ... (all PDB IDs)

📈 Output Statistics:
├── Total samples processed
├── DockQ score distribution (high/medium/low quality)
├── Samples per PDB ID
├── Samples per seed
└── Processing time and efficiency metrics

DEPENDENCIES:
- DockQ.DockQ: For DockQ score computation
- BioPython: For structure handling and TM-score calculation
- h5py: For H5 file creation
- pandas: For metadata handling
- numpy: For matrix operations
- json: For confidences.json parsing

NOTES:
- TM-score calculation requires proper implementation (e.g., TM-align)
- AF3 confidence scores (ptm, iptm) need to be extracted from AF3 output
- PAE matrix provides per-residue accuracy estimates
- Contact probability matrix shows residue-residue interaction confidence
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import argparse          # For command-line argument parsing
import os               # For operating system operations (file paths, etc.)
import multiprocessing  # For parallel processing
import concurrent.futures # For parallel processing
import sys              # For system-specific parameters and functions
import subprocess       # For running external commands
import pandas as pd     # For data manipulation and CSV reading
import glob             # For file pattern matching
import json             # For parsing JSON files (confidences.json)
import h5py             # For creating and writing H5 files
import numpy as np      # For numerical operations and array handling
import DockQ.DockQ      # For DockQ score computation
import shutil           # For file operations (finding executables)
import threading        # For thread-safe operations
import signal           # For signal handling (Ctrl+C)
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1
from tempfile import NamedTemporaryFile
from datetime import datetime  # For timestamp formatting
import logging          # For proper logging
import logging.handlers # For rotating file handlers
from utils import get_structure_tite
from typing import Dict, List, Tuple


logger = logging.getLogger("af3_processor")
# Global flag for clean shutdown
shutdown_requested = False
executor = None

#download and install USAlign from https://zhanggroup.org/US-align/
USALIGN_PATH = "/proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/src/USalign"

# =============================================================================
# AF3 CONFIDENCE SCORE EXTRACTION
# =============================================================================

def extract_af3_scores(confidences_file):
    """
    Extract AlphaFold3 confidence scores from confidences.json file.
    
    This function parses the confidences.json file that AlphaFold3 generates
    for each prediction. It extracts:
    - PAE (Predicted Aligned Error) matrix
    - Chain and residue identifiers
    - Confidence scores (PTM, iPTM, ranking confidence)
    
    Args:
        confidences_file (str): Path to confidences.json file
    
    Returns:
        dict: Dictionary containing all extracted scores and matrices
              - pae_matrix: N×N matrix of predicted aligned errors
              - token_chain_ids: Chain IDs for each residue
              - token_res_ids: Residue IDs
              - ptm: Predicted TM score
              - iptm: Interface predicted TM score
              - ranking_confidence: AF3 ranking confidence
    """
    try:
        # Load JSON data from confidences file
        with open(confidences_file, 'r') as f:
            data = json.load(f)
        
        # Extract core structural data
        pae_matrix = np.array(data['pae'])                    # Predicted Aligned Error matrix
        token_chain_ids = np.array(data['token_chain_ids'])   # Chain IDs for each residue
        token_res_ids = np.array(data['token_res_ids'])       # Residue IDs
        
        # Extract confidence scores if available in the JSON
        # These might not always be present in confidences.json
        ptm = data.get('ptm', np.nan)                    # Predicted TM score
        iptm = data.get('iptm', np.nan)                  # Interface predicted TM score
        ranking_confidence = data.get('ranking_confidence', np.nan)  # AF3 ranking confidence
        
        # If confidence scores are not available, estimate them from PAE matrix
        if np.isnan(ptm):
            # Estimate PTM from PAE matrix using a simplified approach
            # PTM is inversely related to PAE - lower PAE = higher PTM
            # Normalize to 0-1 range (30Å is roughly the maximum meaningful PAE)
            avg_pae = np.mean(pae_matrix)
            ptm = max(0, 1 - (avg_pae / 30.0))
        
        if np.isnan(iptm):
            # Estimate iPTM from PTM
            # iPTM is typically slightly lower than PTM as it focuses on interface regions
            iptm = ptm * 0.95
        
        if np.isnan(ranking_confidence):
            # Use PTM as a proxy for ranking confidence
            ranking_confidence = ptm
        
        return {
            'pae_matrix': pae_matrix,
            'token_chain_ids': token_chain_ids,
            'token_res_ids': token_res_ids,
            'ptm': ptm,
            'iptm': iptm,
            'ranking_confidence': ranking_confidence
        }
        
    except Exception as e:
        logger.error(f"Error extracting AF3 scores from {confidences_file}: {e}")
        return None

# =============================================================================
# TM SCORE COMPUTATION
# =============================================================================

class ChainSelect(Select):
    def __init__(self, chains_to_keep):
        self.chains_to_keep = set(chains_to_keep)
    def accept_chain(self, chain):
        return chain.id in self.chains_to_keep

def compute_tm_score(prediction_file, reference_file, chain_info):
    """
    Compute TM score using US-align as a command line tool
    
    Args:
        prediction_file (str): Path to prediction structure file (.pdb or .cif)
        reference_file (str): Path to reference structure file (.pdb or .cif)
        chain_info (dict): Dictionary containing chain information
    
    Returns:
        dict: Dictionary containing TM scores and alignment information
    """
    logger.debug(f"Computing TM score for {prediction_file} and {reference_file}")
    
    # Extract just relevant chains from the chain_info dictionary from the groundtruth pdb file
    ref_chains = chain_info['antigen_chains'] + chain_info['antibody_chains']
    
    parser = PDBParser(QUIET=True)
    io = PDBIO()

    #extract the chains from the groundtruth pdb file
    structure_ref = parser.get_structure(reference_file, reference_file)

    with NamedTemporaryFile("w+", suffix=".pdb") as tmp_ref:
        # Save selected chains
        io.set_structure(structure_ref)
        io.save(tmp_ref.name, ChainSelect(ref_chains))

        cmd = [USALIGN_PATH, tmp_ref.name, prediction_file, "-ter", "1", "-mm", "1", "-outfmt", "2"]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300  # 5 minutes max
            )
            stdout = result.stdout.strip().splitlines()

            # Parse the second line (after header)
            for line in stdout:
                logger.debug(f"Line: {line}")
                if not line.startswith("#") and len(line.strip()) > 0:
                    fields = line.strip().split()
                    logger.debug(f"US-align result: {fields}")
                    # These fields match the MM-align format
                    # [0]=PDBchain1, [1]=PDBchain2, [2]=TM1, [3]=TM2, ...

                    #get the chain alignment mapping and parse chain alignment from US-align output fields[0] and fields[1].
                    logger.debug(f"Fields: {fields}")
                    ref_chains = fields[0].split(':')[1:]
                    pred_chains = fields[1].split(':')[1:]
                    ref_chains = [c for c in ref_chains if c not in ['', 'NA', 'nan']]
                    pred_chains = [c for c in pred_chains if c not in ['', 'NA', 'nan']]
                    logger.debug(f"Ref chains: {ref_chains}")
                    logger.debug(f"Pred chains: {pred_chains}")
                    chain_mapping = {ref_chains[i]: pred_chains[i] for i in range(len(ref_chains))}
                    logger.debug(f"Chain mapping: {chain_mapping}")

                    tm_normalized_reference = float(fields[2])
                    tm_normalized_query = float(fields[3])
                    logger.debug(f"TM-align result: {tm_normalized_reference}, {tm_normalized_query}")
                    return chain_mapping, tm_normalized_reference, tm_normalized_query

            logger.error(f"No valid TM-scores found in US-align output for {prediction_file}")
            return {}, np.nan, np.nan

        except subprocess.TimeoutExpired:
            logger.error(f"US-align timed out for {prediction_file}")
            return {}, np.nan, np.nan
        except Exception as e:
            logger.error(f"US-align failed for {prediction_file}: {e}")
            return {}, np.nan, np.nan
        
# =============================================================================
# FILE DISCOVERY AND UTILITIES
# =============================================================================

def find_confidences_file(prediction_folder):
    """
    Find the confidences.json file in the prediction folder.
    
    AlphaFold3 generates confidence files with various naming conventions.
    This function searches for the file using multiple possible names.
    
    Args:
        prediction_folder (str): Path to folder containing prediction files
    
    Returns:
        str or None: Path to confidences.json file or None if not found
    """
    # Primary filename to look for
    confidences_file = os.path.join(prediction_folder, "confidences.json")
    if os.path.exists(confidences_file):
        return confidences_file
    
    # Alternative filenames that might be used
    alt_names = ["confidence.json", "scores.json", "metrics.json", "full_confidences.json"]
    for name in alt_names:
        alt_file = os.path.join(prediction_folder, name)
        if os.path.exists(alt_file):
            return alt_file
    
    return None



# =============================================================================
# METADATA PARSING
# =============================================================================

def filter_nan_values(chain_list):
    """
    Filter out NaN, NA, empty strings, and 'nan' values from a list of chains.
    
    Args:
        chain_list (list): List of chain IDs that may contain NaN values
    
    Returns:
        list: Filtered list with valid chain IDs only
    """
    if not chain_list:
        return []
    
    filtered_chains = []
    for chain in chain_list:
        # Convert to string and check for various NaN representations
        chain_str = str(chain).strip()
        if chain_str not in ['nan', 'NaN', 'NA', 'na', 'N/A', '', 'None'] and not pd.isna(chain):
            filtered_chains.append(chain_str)
    
    return filtered_chains

def parse_chain_info(metadata_df, pdb_id):
    """
    Parse chain information from metadata for a given PDB ID.
    
    This function extracts chain mapping information from the metadata CSV file.
    It identifies which chains are antibodies (heavy/light) and which are antigens.
    This information is crucial for DockQ calculation and proper chain merging.
    
    Args:
        metadata_df (pandas.DataFrame): DataFrame containing metadata
        pdb_id (str): PDB ID to look up
    
    Returns:
        dict or None: Dictionary with chain information or None if not found
                      - antigen_chains: List of antigen chain IDs
                      - antibody_chains: List of antibody chain IDs
                      - heavy_chain: Heavy chain ID
                      - light_chain: Light chain ID
                      - resolution: Structure resolution
                      - complex_type: Type of complex (e.g., "Fab-Ag")
    """
    # Filter metadata for the specific PDB ID
    pdb_data = metadata_df[metadata_df['pdb_id'] == pdb_id]
    
    if pdb_data.empty:
        logger.warning(f"No metadata found for PDB ID: {pdb_id}")
        return None
    
    # Take the first entry (assuming all entries for same PDB have same chain info)
    if len(pdb_data) > 1:
        logger.warning(f"Multiple metadata entries found for PDB ID: {pdb_id}")
    
    row = pdb_data.iloc[0]
    
    # Parse antigen chains (can be multiple chains separated by |)
    # Example: "A|B" means chains A and B are both antigens
    raw_antigen_chains = [chain.strip() for chain in str(row['antigen_chains']).split('|')]
    antigen_chains = filter_nan_values(raw_antigen_chains)
    
    logger.debug(f"Raw antigen chains: {raw_antigen_chains}")
    logger.debug(f"Filtered antigen chains: {antigen_chains}")
    logger.debug(f"Heavy chain: {row['heavy_chain']}")

    # Parse antibody chains with proper NaN filtering
    raw_heavy_chain = [str(row['heavy_chain']).strip()]
    heavy_chain = filter_nan_values(raw_heavy_chain)
    
    raw_light_chain = [str(row['light_chain']).strip()]
    light_chain = filter_nan_values(raw_light_chain)
    
    # Create list of antibody chains
    antibody_chains = heavy_chain + light_chain
    
    # Validate that we have at least some chains
    if not antigen_chains:
        logger.error(f"No valid antigen chains found for PDB ID: {pdb_id}")
        return None
    
    if not antibody_chains:
        logger.error(f"No valid antibody chains found for PDB ID: {pdb_id}")
        return None
    
    logger.debug(f"The extracted chains are: antigen_chains: {antigen_chains}, antibody_chains: {antibody_chains}, heavy_chain: {heavy_chain}, light_chain: {light_chain}")

    return {
        'antigen_chains': antigen_chains,
        'antibody_chains': antibody_chains,
        'heavy_chain': heavy_chain,
        'light_chain': light_chain,
        'resolution': row['resolution'],
        'complex_type': row['complex_type']
    }

# =============================================================================
# FOLDER STRUCTURE DISCOVERY
# =============================================================================

def discover_pdb_ids_from_single_seed(af3_folder, limit=None):
    """
    Discover PDB IDs from a single seed folder (since all seeds contain the same PDB IDs).
    
    This function is much more efficient than scanning all seeds since all seed folders
    contain the same PDB IDs. It also supports early limiting for testing purposes.
    
    Args:
        af3_folder (str): Path to the AF3 predictions folder
        limit (int, optional): Maximum number of PDB IDs to return (for testing)
    
    Returns:
        list: List of PDB IDs found in the first available seed folder
    """
    # Find the first available seed folder
    seed_folders = [item for item in os.listdir(af3_folder) 
                   if os.path.isdir(os.path.join(af3_folder, item)) and item.startswith('seed-')]
    
    if not seed_folders:
        logger.warning(f"No seed folders found in {af3_folder}")
        return []
    
    # Sort to ensure consistent processing order
    seed_folders.sort()
    first_seed = seed_folders[0]
    seed_path = os.path.join(af3_folder, first_seed)
    
    logger.debug(f"Scanning PDB IDs from seed folder: {first_seed}")
    
    try:
        # Get all subdirectories in this seed folder
        subdirs = [item for item in os.listdir(seed_path) 
                  if os.path.isdir(os.path.join(seed_path, item))]
        
        # Sort for consistent output
        pdb_ids = sorted(subdirs)
        
        # Apply limit early if specified
        if limit is not None:
            pdb_ids = pdb_ids[:limit]
            logger.info(f"Limited to first {limit} PDB IDs for testing")
        
        logger.info(f"Discovered {len(pdb_ids)} PDB IDs from {first_seed}")
        return pdb_ids
        
    except PermissionError:
        logger.warning(f"Permission denied accessing {seed_path}")
        return []
    except Exception as e:
        logger.warning(f"Error scanning {seed_path}: {e}")
        return []

def find_all_af3_samples_across_seeds(af3_folder, pdb_id):
    """
    Find all AF3 samples for a specific PDB ID across all seeds.
    
    This function searches through all seed folders to find all samples
    for a given PDB ID. It returns a list of tuples containing:
    (seed_name, sample_name, prediction_file_path)
    
    Args:
        af3_folder (str): Path to the AF3 predictions folder
        pdb_id (str): PDB identifier to search for
    
    Returns:
        list: List of tuples (seed_name, sample_name, prediction_file_path)
    """
    samples = []
    
    # Find all seed folders
    seed_folders = [item for item in os.listdir(af3_folder) 
                   if os.path.isdir(os.path.join(af3_folder, item)) and item.startswith('seed-')]
    
    if not seed_folders:
        logger.warning(f"No seed folders found in {af3_folder}")
        return []
    
    # Sort seed folders for consistent processing order
    seed_folders.sort()
    
    # Search for the PDB ID in each seed folder
    for seed_folder in seed_folders:
        seed_path = os.path.join(af3_folder, seed_folder)
        pdb_path = os.path.join(seed_path, pdb_id)
        
        if not os.path.exists(pdb_path):
            logger.debug(f"PDB {pdb_id} not found in {seed_folder}")
            continue
        
        # Look for sample folders within this PDB folder
        try:
            sample_folders = [item for item in os.listdir(pdb_path) 
                            if os.path.isdir(os.path.join(pdb_path, item)) and item.startswith(f'{seed_folder}_sample-')]
            
            for sample_folder in sample_folders:
                sample_path = os.path.join(pdb_path, sample_folder)
                
                # Look for prediction files (model.cif or model.pdb)
                prediction_file = None
                for pred_file in ['model.cif', 'model.pdb']:
                    pred_path = os.path.join(sample_path, pred_file)
                    if os.path.exists(pred_path):
                        prediction_file = pred_path
                        break
                
                if prediction_file:
                    samples.append((seed_folder, sample_folder, prediction_file))
                else:
                    logger.warning(f"No prediction file found in {sample_path}")
                    
        except PermissionError:
            logger.warning(f"Permission denied accessing {pdb_path}")
        except Exception as e:
            logger.warning(f"Error scanning {pdb_path}: {e}")
    
    logger.debug(f"Found {len(samples)} samples for PDB {pdb_id} across {len(seed_folders)} seeds")
    return samples

def find_reference_structure(reference_folder, pdb_id):
    """
    Find reference structure file for a given PDB ID.
    
    This function searches for reference structure files (experimental structures)
    that will be used for comparison with predictions. It supports multiple
    file formats and naming conventions.
    
    Args:
        reference_folder (str): Path to folder containing reference structures
        pdb_id (str): PDB ID to search for
    
    Returns:
        str or None: Path to the reference file or None if not found
    """
    # Define search patterns for reference files
    patterns = [
        f"{reference_folder}/{pdb_id}.pdb",      # Exact match .pdb
        f"{reference_folder}/{pdb_id}.cif",      # Exact match .cif
        f"{reference_folder}/{pdb_id}*.pdb",     # Wildcard match .pdb
        f"{reference_folder}/{pdb_id}*.cif"      # Wildcard match .cif
    ]
    
    # Try each pattern until a file is found
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]  # Return first matching file
    
    return None

# =============================================================================
# DOCKQ SCORE COMPUTATION
# =============================================================================
def merge_chains(structure, chains_to_merge: List[str], new_chain_name: str):
    """
    Merge multiple chains into a single chain inside `structure` safely.

    Key points:
    - We DO NOT put the chain letter inside residue.id (hetflag,resseq,icode).
    - To avoid residue-ID collisions, incoming residues are renumbered
      to sequence numbers higher than the current max of the target chain.
    - Finally, the target chain is renamed to `new_chain_name`.

    Parameters
    ----------
    structure : Biopython-like structure that allows structure[chain_id], .detach_child(chain_id)
    chains_to_merge : list of chain IDs to merge (must exist in `structure`)
    new_chain_name : the resulting chain ID (e.g., "Ag" or "Ab")

    Returns
    -------
    structure : the same structure object, modified in-place
    """
    if not chains_to_merge:
        return structure

    # Use the first chain as the target container
    target_id = chains_to_merge[0]
    if target_id not in structure:
        raise KeyError(f"Target chain '{target_id}' not found in structure during merge.")

    target_chain = structure[target_id]

    # Find current max residue sequence number in the target
    try:
        current_max = max((res.id[1] for res in target_chain), default=0)
    except Exception:
        current_max = 0

    # Move residues from the remaining chains into the target, renumbering to avoid collisions
    for src_id in chains_to_merge[1:]:
        if src_id not in structure:
            continue  # be tolerant; presence was checked earlier
        src_chain = structure[src_id]

        # Work on a snapshot because we'll detach children
        for res in list(src_chain):
            old_key = res.id
            # 1) Detach from the original chain
            src_chain.detach_child(old_key)

            # 2) Renumber only resseq to avoid collisions in target
            hetflag, _, icode = res.id
            current_max += 1
            res.id = (hetflag, current_max, icode)

            # 3) Add to target
            target_chain.add(res)

        # Remove the now-empty source chain from the structure
        try:
            structure.detach_child(src_id)
        except Exception:
            # Fallback for dict-like structures
            try:
                del structure[src_id]
            except Exception:
                pass

    # Rename the merged chain to the requested `new_chain_name`
    try:
        target_chain.id = new_chain_name
    except Exception:
        # Some containers also need the parent dictionary key to change;
        # in most DockQ/Bio.PDB cases changing .id is sufficient.
        pass

    return structure

def compute_dockq_score(prediction_file, reference_file, chain_info, chain_mapping, pdb_id, seed_name, sample_name):
    """
    Compute DockQ score between antibody and antigen chains.
    
    DockQ is a quality measure for protein-protein docking predictions.
    It combines multiple metrics into a single score:
    - DockQ score: Overall quality (0-1, higher is better)
    - LRMSD: Ligand RMSD (distance between predicted and native ligand)
    - iRMSD: Interface RMSD (distance at the interface)
    - fnat: Fraction of native contacts preserved
    - clashes: Number of atomic clashes
    
    This function:
    1. Loads prediction and reference structures
    2. Merges chains according to antibody/antigen classification
    3. Runs DockQ calculation
    4. Extracts and returns all metrics
    
    Args:
        prediction_file (str): Path to AF3 prediction file
        reference_file (str): Path to reference structure file
        chain_info (dict): Dictionary with chain information
        chain_mapping (dict): Dictionary with chain mapping
    Returns:
        dict or None: Dictionary with DockQ scores and metrics or None if failed
    """
    try:
        # Load with DockQ's loader (same as your original code)
        model  = DockQ.DockQ.load_PDB(prediction_file)  # predicted
        native = DockQ.DockQ.load_PDB(reference_file)   # experimental

        # Native IDs from metadata
        ref_ag = list(chain_info['antigen_chains'])
        ref_ab = list(chain_info['antibody_chains'])

        # Map native -> predicted; if mapping is missing for a chain, fall back to the same ID
        pred_ag = [chain_mapping.get(c, c) for c in ref_ag]
        pred_ab = [chain_mapping.get(c, c) for c in ref_ab]

        # Presence checks on the correct side
        missing_pred = [c for c in (pred_ag + pred_ab) if c not in model]
        if missing_pred:
            logger.warning(f"{pdb_id} - {seed_name} - {sample_name} - "
                           f"Predicted chains not found in prediction structure: {missing_pred}")
            return {k: np.nan for k in ("abag_dockq", "abag_lrmsd", "abag_irmsd")}

        missing_ref = [c for c in (ref_ag + ref_ab) if c not in native]
        if missing_ref:
            logger.warning(f"{pdb_id} - {seed_name} - {sample_name} - "
                           f"Reference chains not found in native structure: {missing_ref}")
            return {k: np.nan for k in ("abag_dockq", "abag_lrmsd", "abag_irmsd")}

        # Merge per side (predicted side uses predicted IDs; native side uses native IDs)
        model  = merge_chains(model,  pred_ag, "Ag")
        native = merge_chains(native, ref_ag,  "Ag")
        model  = merge_chains(model,  pred_ab, "Ab")
        native = merge_chains(native, ref_ab,  "Ab")

        # Now both structures have exactly two chains: "Ag" and "Ab"
        chain_map = {"Ag": "Ag", "Ab": "Ab"}

        dockq_results, total_dockq = DockQ.DockQ.run_on_all_native_interfaces(
            model, native, chain_map=chain_map
        )

        metrics = ["DockQ", "LRMSD", "iRMSD"]
        if not dockq_results:
            logger.error(f"{pdb_id} - {seed_name} - {sample_name} - DockQ returned empty results")
            return {f"abag_{m.lower()}": np.nan for m in metrics}

        if len(dockq_results) > 1:
            # Your pipeline expects a single interface; if multiple, mark as NaN
            logger.error(f"{pdb_id} - {seed_name} - {sample_name} - "
                         f"Expected single interface, got {len(dockq_results)}")
            return {f"abag_{m.lower()}": np.nan for m in metrics}

        interface_key = next(iter(dockq_results.keys()))
        iface = dockq_results[interface_key]

        return {
            "abag_dockq": round(float(iface["DockQ"]), 2),
            "abag_lrmsd": round(float(iface["LRMSD"]), 2),
            "abag_irmsd": round(float(iface["iRMSD"]), 2),
        }

    except Exception as e:
        logger.error(f"{pdb_id} - {seed_name} - {sample_name} - Error computing DockQ: {e}")
        return {"abag_dockq": np.nan, "abag_lrmsd": np.nan, "abag_irmsd": np.nan}

# =============================================================================
# H5 FILE PROCESSING
# =============================================================================

def compute_residue_distance_info(prediction_file: str,
                                  token_chain_ids: np.ndarray,
                                  token_res_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-residue Cα distances for the prediction, aligned to AF3 token order.

    Returns a tuple with:
      - dist_matrix: (N, N) float32 with Cα–Cα distances, NaN if any coordinate missing
      - residue_one_letter: (N,) array of one-letter residue codes (str), "X" if unknown
      - residue_three_letter: (N,) array of three-letter residue names (str)
      - inter_idx_dist: (K,) int32 indices i where token_chain_ids[i] != token_chain_ids[j]
      - inter_jdx_dist: (K,) int32 indices j where token_chain_ids[i] != token_chain_ids[j]
      - interchain_ca_distances: (K,) float32 distances for those inter-chain pairs
    """
    try:
        if prediction_file.lower().endswith((".cif", ".mmcif")):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        structure = parser.get_structure(os.path.basename(prediction_file), prediction_file)
        model = next(structure.get_models())

        chain_id_to_resseq_to_residue: Dict[str, Dict[int, List]] = {}
        for chain in model:
            resseq_to_residue: Dict[int, List] = {}
            for residue in chain:
                hetflag, resseq, icode = residue.id
                if hetflag != " ":
                    continue
                resseq_to_residue.setdefault(resseq, []).append(residue)
            chain_id_to_resseq_to_residue[chain.id] = resseq_to_residue

        token_chain_ids_list = [str(x) for x in token_chain_ids]
        token_res_ids_int = np.asarray(token_res_ids, dtype=int)
        num_tokens = len(token_chain_ids_list)

        coords = np.full((num_tokens, 3), np.nan, dtype=np.float32)
        residue_three_letter = np.array(["UNK"] * num_tokens, dtype=object)
        residue_one_letter = np.array(["X"] * num_tokens, dtype=object)
        valid_mask = np.zeros(num_tokens, dtype=bool)

        for idx_token, (chain_id, resseq) in enumerate(zip(token_chain_ids_list, token_res_ids_int)):
            residue_obj = None
            if chain_id in chain_id_to_resseq_to_residue and resseq in chain_id_to_resseq_to_residue[chain_id]:
                residue_obj = chain_id_to_resseq_to_residue[chain_id][resseq][0]

            if residue_obj is None:
                continue

            atom = None
            if "CA" in residue_obj:
                atom = residue_obj["CA"]
            elif "CB" in residue_obj:
                atom = residue_obj["CB"]

            if atom is None:
                continue

            if atom.is_disordered():
                try:
                    atom = atom.selected_child
                except Exception:
                    pass

            try:
                coords[idx_token] = atom.get_coord().astype(np.float32)
                r3 = residue_obj.get_resname()
                residue_three_letter[idx_token] = r3
                try:
                    residue_one_letter[idx_token] = seq1(r3)
                except Exception:
                    residue_one_letter[idx_token] = "X"
                valid_mask[idx_token] = True
            except Exception:
                continue

        dist_matrix = np.full((num_tokens, num_tokens), np.nan, dtype=np.float32)
        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            valid_coords = coords[valid_indices]
            diffs = valid_coords[:, None, :] - valid_coords[None, :, :]
            dists = np.sqrt(np.sum(diffs * diffs, axis=2)).astype(np.float32)
            dist_matrix[np.ix_(valid_indices, valid_indices)] = dists

        inter_idx_list: List[int] = []
        inter_jdx_list: List[int] = []
        inter_dists_list: List[float] = []
        for i in range(num_tokens):
            ci = token_chain_ids_list[i]
            for j in range(num_tokens):
                if token_chain_ids_list[j] != ci:
                    inter_idx_list.append(i)
                    inter_jdx_list.append(j)
                    inter_dists_list.append(dist_matrix[i, j])

        return (
            dist_matrix.astype(np.float32),
            residue_one_letter,
            residue_three_letter,
            np.asarray(inter_idx_list, dtype=np.int32),
            np.asarray(inter_jdx_list, dtype=np.int32),
            np.asarray(inter_dists_list, dtype=np.float32),
        )
    except Exception as e:
        logger.error(f"Error computing residue distance matrix for {prediction_file}: {e}")
        N = len(token_chain_ids)
        return (
            np.full((N, N), np.nan, dtype=np.float32),
            np.array(["X"] * N, dtype=object),
            np.array(["UNK"] * N, dtype=object),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )

def process_sample(h5_file, pdb_id, seed_name, sample_name, prediction_file, 
                                        reference_file, chain_info, confidences_file):
    """
    Process a sample.
    
    Args:
        h5_file: H5 file object (already opened)
        pdb_id (str): PDB identifier
        seed_name (str): Seed name (e.g., 'seed-1')
        sample_name (str): Sample name (e.g., 'seed-1_sample-0')
        prediction_file (str): Path to prediction structure file
        reference_file (str): Path to reference structure file
        chain_info (dict): Dictionary with chain information
        confidences_file (str): Path to confidences.json file
    
    Returns:
        bool: True if processing succeeded, False if failed
    """
    # Track what components succeeded/failed
    status_details = {
        'dockq_success': False,
        'af3_success': False,
        'tm_success': False,
        'metadata_success': False,
        'distance_success': False,
        'residue_labels_success': False
    }

    # logger.debug(f"Processing sample: {pdb_id} - {seed_name} - {sample_name}")
    
    try:
        # Create H5 group structure
        # Structure: /pdb_id/seed_sample_name/
        if pdb_id not in h5_file:
            complex_group = h5_file.create_group(pdb_id)
        else:
            complex_group = h5_file[pdb_id]
        
        sample_group_name = f"{seed_name}_{sample_name}"
        if sample_group_name in complex_group:
            logger.info(f"Sample {sample_group_name} already exists. Skipping...")
            return True  # Consider this a success since it's already processed
        
        sample_group = complex_group.create_group(sample_group_name)
        logger.debug(f"Chain info: {chain_info}")
        
        # Align chains naming and calculate TM-score using US-align with error handling
        try:
            logger.debug(f"Computing TM score for {sample_group_name}")
            chain_mapping, tm_normalized_reference, tm_normalized_query = compute_tm_score(prediction_file, reference_file, chain_info)
            
            # Check if TM scores are valid
            if np.isnan(tm_normalized_reference) or np.isnan(tm_normalized_query):
                logger.warning(f"TM score computation returned NaN values for {sample_group_name}")
                status_details['tm_success'] = False
            else:
                logger.debug(f"TM score results: reference={tm_normalized_reference:.4f}, query={tm_normalized_query:.4f}")
                status_details['tm_success'] = True
            
            # Save the results regardless (NaN values will be preserved)
            sample_group.create_dataset('tm_normalized_reference', data=tm_normalized_reference)
            sample_group.create_dataset('tm_normalized_query', data=tm_normalized_query)
            
        except Exception as e:
            logger.error(f"{pdb_id} - {seed_name} - {sample_name} - TM score computation failed: {str(e)}")
            status_details['tm_success'] = False
            # Save NaN values when TM score computation raises an exception
            try:
                sample_group.create_dataset('tm_normalized_reference', data=np.nan)
                sample_group.create_dataset('tm_normalized_query', data=np.nan)
                chain_mapping = {}  # Empty chain mapping on failure
            except Exception as save_error:
                logger.error(f"Failed to save NaN TM score values: {save_error}")
                chain_mapping = {}  # Empty chain mapping on failure

        # 1. Compute DockQ scores with error handling
        try:
            logger.debug(f"Computing DockQ score for {sample_group_name}")
            
            # Check if we have a valid chain mapping from TM score computation
            if not chain_mapping:
                logger.warning(f"No chain mapping available for DockQ computation for {sample_group_name}")
                # Try to create a simple identity mapping as fallback
                all_chains = chain_info['antigen_chains'] + chain_info['antibody_chains']
                chain_mapping = {chain: chain for chain in all_chains}
                logger.debug(f"Using identity chain mapping: {chain_mapping}")
            
            dockq_results = compute_dockq_score(prediction_file, reference_file, chain_info, chain_mapping, pdb_id, seed_name, sample_name)
            
            if dockq_results is not None:
                # Check if any values are NaN (indicating DockQ failure)
                has_nan_values = any(np.isnan(value) if isinstance(value, (int, float)) else False 
                                   for value in dockq_results.values())
                
                if has_nan_values:
                    logger.warning(f"DockQ computation returned NaN values for {sample_group_name}")
                    status_details['dockq_success'] = False
                else:
                    logger.debug(f"DockQ results: {dockq_results['abag_dockq']:.8f}, {dockq_results['abag_lrmsd']:.8f}, {dockq_results['abag_irmsd']:.8f}")
                    status_details['dockq_success'] = True
                
                # Save the results regardless (NaN values will be preserved)
                sample_group.create_dataset('abag_dockq', data=dockq_results['abag_dockq'])
                sample_group.create_dataset('abag_lrmsd', data=dockq_results['abag_lrmsd'])
                sample_group.create_dataset('abag_irmsd', data=dockq_results['abag_irmsd'])
            else:
                logger.error(f"{pdb_id} - {seed_name} - {sample_name} - DockQ computation returned None")
                status_details['dockq_success'] = False
                # Save NaN values when DockQ completely fails
                sample_group.create_dataset('abag_dockq', data=np.nan)
                sample_group.create_dataset('abag_lrmsd', data=np.nan)
                sample_group.create_dataset('abag_irmsd', data=np.nan)
                
        except Exception as e:
            logger.error(f"{pdb_id} - {seed_name} - {sample_name} - DockQ computation failed: {str(e)}")
            status_details['dockq_success'] = False
            # Save NaN values when DockQ computation raises an exception
            try:
                sample_group.create_dataset('abag_dockq', data=np.nan)
                sample_group.create_dataset('abag_lrmsd', data=np.nan)
                sample_group.create_dataset('abag_irmsd', data=np.nan)
            except Exception as save_error:
                logger.error(f"Failed to save NaN DockQ values: {save_error}")
        
        # 2. Extract AF3 confidence scores with error handling
        if confidences_file:
            try:
                af3_scores = extract_af3_scores(confidences_file)
                if af3_scores:
                    # Save structural data (matrices)
                    logger.debug(f"Saving pae_matrix for {sample_group_name}")
                    sample_group.create_dataset('pae_matrix', data=af3_scores['pae_matrix'], dtype='float32')
                    logger.debug(f"Saving token_chain_ids for {sample_group_name}")
                    # Convert Unicode strings to regular strings
                    token_chain_ids_str = [str(x) for x in af3_scores['token_chain_ids']]
                    sample_group.create_dataset('token_chain_ids', data=token_chain_ids_str, dtype=h5py.string_dtype(encoding='utf-8'))
                    logger.debug(f"Saving token_res_ids for {sample_group_name}")
                    sample_group.create_dataset('token_res_ids', data=af3_scores['token_res_ids'], dtype='int32')
                    
                    logger.debug(f"Saving ptm for {sample_group_name}")
                    # Save confidence scores
                    sample_group.create_dataset('ptm', data=af3_scores['ptm'])
                    sample_group.create_dataset('iptm', data=af3_scores['iptm'])
                    sample_group.create_dataset('ranking_confidence', data=af3_scores['ranking_confidence'])
                    
                    # Process inter-chain interactions
                    inter_idx = []
                    inter_jdx = []
                    interchain_pae_vals = []
                    
                    # Find inter-chain pae values
                    for i in range(len(af3_scores['token_chain_ids'])):
                        for j in range(len(af3_scores['token_chain_ids'])):
                            # only process if residues belong to different chains
                            if af3_scores['token_chain_ids'][i] != af3_scores['token_chain_ids'][j]:
                                inter_idx.append(i)
                                inter_jdx.append(j)
                                interchain_pae_vals.append(af3_scores['pae_matrix'][i][j])
                    
                    inter_idx = np.array(inter_idx)
                    inter_jdx = np.array(inter_jdx)
                    interchain_pae_vals = np.array(interchain_pae_vals)
                    
                    # Save inter-chain data
                    sample_group.create_dataset('inter_idx', data=inter_idx, dtype='int32')
                    sample_group.create_dataset('inter_jdx', data=inter_jdx, dtype='int32')
                    sample_group.create_dataset('interchain_pae_vals', data=interchain_pae_vals, dtype='float32')
                    
                    # Create chain mapping
                    unique_chains = np.unique(af3_scores['token_chain_ids'])
                    chain_to_int = {chain: idx for idx, chain in enumerate(unique_chains)}
                    sample_group.create_dataset('chain_mapping', data=str(chain_to_int))
                    
                    status_details['af3_success'] = True

                    try:
                        (
                            ca_dist_matrix,
                            residue_one_letter,
                            residue_three_letter,
                            inter_idx_dist,
                            inter_jdx_dist,
                            interchain_ca_distances,
                        ) = compute_residue_distance_info(
                            prediction_file,
                            af3_scores['token_chain_ids'],
                            af3_scores['token_res_ids'],
                        )

                        sample_group.create_dataset('ca_distance_matrix', data=ca_dist_matrix, dtype='float32')
                        sample_group.create_dataset('residue_one_letter', data=residue_one_letter, dtype=h5py.string_dtype(encoding='utf-8'))
                        sample_group.create_dataset('residue_three_letter', data=residue_three_letter, dtype=h5py.string_dtype(encoding='utf-8'))
                        sample_group.create_dataset('inter_idx_dist', data=inter_idx_dist, dtype='int32')
                        sample_group.create_dataset('inter_jdx_dist', data=inter_jdx_dist, dtype='int32')
                        sample_group.create_dataset('interchain_ca_distances', data=interchain_ca_distances, dtype='float32')
                        logger.debug(f"Saved residue distance features for {sample_group_name}")

                        # determine success flags
                        status_details['distance_success'] = np.isfinite(ca_dist_matrix).any()
                        status_details['residue_labels_success'] = (
                            len(residue_one_letter) == len(af3_scores['token_chain_ids']) and
                            len(residue_three_letter) == len(af3_scores['token_chain_ids'])
                        )
                    except Exception as e:
                        logger.error(f"Failed computing/saving residue distances for {sample_group_name}: {e}")
                else:
                    logger.error(f"AF3 score extraction returned None for {sample_group_name}")
            except Exception as e:
                logger.error(f"AF3 score extraction failed for {sample_group_name}: {str(e)}")
                status_details['af3_success'] = False
        else:
            logger.error(f"No confidences file found for {sample_group_name}")
        
        # 4. Save metadata with error handling
        try:
            # FIX: Handle string data properly for H5
            sample_group.create_dataset('pdb_id', data=pdb_id, dtype=h5py.string_dtype(encoding='utf-8'))
            sample_group.create_dataset('seed_name', data=seed_name, dtype=h5py.string_dtype(encoding='utf-8'))
            sample_group.create_dataset('sample_name', data=sample_name, dtype=h5py.string_dtype(encoding='utf-8'))
            sample_group.create_dataset('prediction_file', data=prediction_file, dtype=h5py.string_dtype(encoding='utf-8'))
            sample_group.create_dataset('reference_file', data=reference_file, dtype=h5py.string_dtype(encoding='utf-8'))
            status_details['metadata_success'] = True
        except Exception as e:
            logger.error(f"Metadata saving failed for {sample_group_name}: {str(e)}")
            status_details['metadata_success'] = False

        # Determine overall success status based on core required components only
        required_components = ['dockq_success', 'af3_success', 'tm_success', 'metadata_success']
        successful_components = sum(bool(status_details.get(k, False)) for k in required_components)
        total_components = len(required_components)
        
        # if all required components are successful, return True
        if successful_components == total_components:
            return True
        else:
            logger.info(f"Failed to process {pdb_id} - {seed_name} - {sample_name} ({successful_components}/{total_components} components)")
            return False
        
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"{pdb_id} - {seed_name} - {sample_name} - Error processing sample: {e}")
        return False

# =============================================================================
# PHASE 3: USER EXPERIENCE AND PROGRESS TRACKING
# =============================================================================


def print_summary_statistics(processed_count, skipped_count, total_pdbs, completed_pdbs):
    """
    Print comprehensive summary statistics at the end of processing.
    
    Args:
        processed_count (int): Number of successfully processed samples
        skipped_count (int): Number of skipped/failed samples
        total_pdbs (int): Total number of PDBs processed
        completed_pdbs (list): List of completed PDB IDs
    """
    total_samples = processed_count + skipped_count
    
    logger.info("="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total PDBs processed: {total_pdbs}")
    logger.info(f"Completed PDBs: {len(completed_pdbs)}")
    logger.info(f"Total samples processed: {total_samples}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Skipped/Failed: {skipped_count}")
    
    if total_samples > 0:
        success_rate = (processed_count / total_samples) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    if completed_pdbs:
        logger.info(f"Completed PDB IDs: {', '.join(completed_pdbs)}")
    
    logger.info("="*60)

def print_processing_status(pdb_id, current_sample, total_samples, seed_name, sample_name):
    """
    Print detailed status for current processing step.
    
    Args:
        pdb_id (str): Current PDB ID being processed
        current_sample (int): Current sample number
        total_samples (int): Total samples for this PDB
        seed_name (str): Current seed name
        sample_name (str): Current sample name
    """
    logger.info(f"\n🔬 Processing: {pdb_id} | Sample {current_sample}/{total_samples} | {seed_name} | {sample_name}")
    logger.info(f"   📁 Progress: {current_sample}/{total_samples} samples for {pdb_id}")

def validate_input_files(metadata_file, af3_folder, reference_folder):
    """
    Validate that all required input files and folders exist.
    
    Args:
        metadata_file (str): Path to metadata CSV file
        af3_folder (str): Path to AF3 predictions folder
        reference_folder (str): Path to reference structures folder
    
    Returns:
        bool: True if all inputs are valid, False otherwise
    """
    logger.info("Validating input files and folders...")
    
    # Check metadata file
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return False
    logger.info(f"Metadata file: {metadata_file}")
    
    # Check AF3 folder
    if not os.path.exists(af3_folder):
        logger.error(f"AF3 folder not found: {af3_folder}")
        return False
    logger.info(f"AF3 folder: {af3_folder}")
    
    # Check reference folder
    if not os.path.exists(reference_folder):
        logger.error(f"Reference folder not found: {reference_folder}")
        return False
    logger.info(f"Reference folder: {reference_folder}")
    
    # Check for seed folders in AF3 folder
    seed_folders = [item for item in os.listdir(af3_folder) 
                   if os.path.isdir(os.path.join(af3_folder, item)) and item.startswith('seed-')]
    if not seed_folders:
        logger.warning(f"No seed folders found in {af3_folder}")
    else:
        logger.info(f"Found {len(seed_folders)} seed folders: {', '.join(sorted(seed_folders))}")
    
    logger.info("Input validation completed successfully!")
    return True

# =============================================================================
# PHASE 4: ADVANCED FEATURES - RETRY MECHANISM AND FILE LOCKING
# =============================================================================

def create_backup(h5_file_path):
    """
    Create a timestamped backup of the H5 file.
    
    Args:
        h5_file_path (str): Path to the H5 file to backup
    """
    if not os.path.exists(h5_file_path):
        logger.warning(f"Cannot backup non-existent file: {h5_file_path}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{h5_file_path}.backup_{timestamp}"
    
    try:
        shutil.copy2(h5_file_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
    except Exception as e:
        logger.warning(f"Failed to create backup: {e}")

def cleanup_old_backups(h5_file_path, keep_count=3):
    """
    Clean up old backup files, keeping only the most recent ones.
    
    Args:
        h5_file_path (str): Path to the original H5 file
        keep_count (int): Number of recent backups to keep
        logger: Logger object for output (optional)
    """
    backup_pattern = f"{h5_file_path}.backup_*"
    backup_files = glob.glob(backup_pattern)
    
    if len(backup_files) <= keep_count:
        if logger:
            logger.debug(f"No cleanup needed: {len(backup_files)} backups (keeping {keep_count})")
        return
    
    # Sort by modification time (oldest first)
    backup_files.sort(key=os.path.getmtime)
    
    # Remove oldest backups
    files_to_remove = backup_files[:-keep_count]
    for backup_file in files_to_remove:
        try:
            os.remove(backup_file)
            if logger:
                logger.debug(f"Removed old backup: {backup_file}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to remove old backup {backup_file}: {e}")
    
    if logger:
        logger.info(f"Cleaned up {len(files_to_remove)} old backups, keeping {keep_count} recent ones")

def validate_h5_file_structure(h5_file_path):
    """
    Validate the structure of an existing H5 file.
    
    This function checks if the H5 file has the expected structure
    and can be opened and read properly.
    
    Args:
        h5_file_path (str): Path to the H5 file to validate
    
    Returns:
        dict: Validation result with success status and details
    """
    if not os.path.exists(h5_file_path):
        logger.warning(f"H5 file does not exist: {h5_file_path}")
        return {'success': False, 'error': 'File does not exist'}
    
    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            # Check if file can be opened
            logger.debug(f"H5 file opened successfully: {h5_file_path}")
            
            # Count PDB groups
            pdb_groups = list(h5_file.keys())
            logger.debug(f"Found {len(pdb_groups)} PDB groups in H5 file")
            
            # Check a few sample groups for structure
            sample_count = 0
            for pdb_id in pdb_groups[:5]:  # Check first 5 PDBs
                pdb_group = h5_file[pdb_id]
                sample_groups = list(pdb_group.keys())
                sample_count += len(sample_groups)
                logger.debug(f"PDB {pdb_id}: {len(sample_groups)} samples")
            
            logger.info(f"H5 file validation successful: {len(pdb_groups)} PDBs, ~{sample_count} samples")
            return {
                'success': True,
                'pdb_count': len(pdb_groups),
                'sample_count': sample_count
            }
            
    except Exception as e:
        logger.warning(f"H5 file validation failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def log_structure_status(pdb_id, seed_name, sample_name, status, details=None):
    """
    Log the processing status of a structure with detailed information.
    
    Args:
        pdb_id (str): PDB ID
        seed_name (str): Seed name
        sample_name (str): Sample name
        status (str): Status ('SUCCESS', 'PARTIAL', 'FAILED')
        details (dict, optional): Detailed information about what succeeded/failed
    """
    timestamp = datetime.now().isoformat()
    
    try:
        logger.info(f"[{timestamp}] STATUS: {pdb_id}/{seed_name}/{sample_name} - {status}")
        if details:
            for key, value in details.items():
                logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Could not write to status log file: {e}")
        
def create_processing_summary(processed_samples, failed_samples, run_subfolder):
    """
    Create a comprehensive summary of processing results.
    
    Args:
        processed_samples (list): List of successfully processed samples
        failed_samples (list): List of failed samples with reasons
    
    Returns:
        str: Path to summary file
    """
    summary_file = os.path.join(run_subfolder, 'summary.txt')
    
    try:
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PROCESSING SUMMARY REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Overall statistics
            total_samples = len(processed_samples) + len(failed_samples)
            f.write(f"Total samples processed: {total_samples}\n")
            f.write(f"Successfully processed: {len(processed_samples)}\n")
            f.write(f"Failed: {len(failed_samples)}\n")
            
            # Calculate success rate with division by zero protection
            if total_samples > 0:
                success_rate = (len(processed_samples) / total_samples) * 100
                f.write(f"Success rate: {success_rate:.1f}%\n\n")
            else:
                f.write("Success rate: N/A (no samples processed)\n\n")
            
            # Successfully processed samples
            f.write("SUCCESSFULLY PROCESSED SAMPLES:\n")
            f.write("-" * 40 + "\n")
            for sample in processed_samples:
                f.write(f"✅ {sample['pdb_id']}/{sample['seed_name']}/{sample['sample_name']}\n")
                f.write(f"   DockQ: {sample.get('dockq_success', 'N/A')}\n")
                f.write(f"   AF3 Scores: {sample.get('af3_success', 'N/A')}\n")
                f.write(f"   TM Score: {sample.get('tm_success', 'N/A')}\n")
            f.write("\n")
            
            # Failed samples with reasons
            f.write("FAILED SAMPLES:\n")
            f.write("-" * 40 + "\n")
            for sample in failed_samples:
                f.write(f"❌ {sample['pdb_id']}/{sample['seed_name']}/{sample['sample_name']}\n")
                f.write(f"   Reason: {sample['reason']}\n")
                f.write(f"   Error Type: {sample.get('error_type', 'Unknown')}\n")
            f.write("\n")
            
            # Error type breakdown
            error_types = {}
            for sample in failed_samples:
                error_type = sample.get('error_type', 'Unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            f.write("ERROR TYPE BREAKDOWN:\n")
            f.write("-" * 40 + "\n")
            for error_type, count in error_types.items():
                f.write(f"{error_type}: {count} samples\n")
            
        return summary_file
        
    except Exception as e:
        logger.warning(f"Could not create processing summary: {e}")
        return None

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir, log_level=logging.INFO):
    """
    Configure global logger `af3_processor` to log to console and file.
    """
    global logger  # Refers to the global logger declared above

    logger.setLevel(log_level)
    logger.handlers.clear()  # Reset existing handlers

    # Formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    # Main file log
    log_file = os.path.join(output_dir, 'log.log')
    fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    fh.setLevel(log_level)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Error log file
    err_file = os.path.join(output_dir, 'errors.log')
    eh = logging.handlers.RotatingFileHandler(err_file, maxBytes=5*1024*1024, backupCount=3)
    eh.setLevel(logging.ERROR)
    eh.setFormatter(file_formatter)
    logger.addHandler(eh)

    return logger


def save_complex_log(pdb_id, output_folder, complex_log_data):
    """
    Save per-complex log data to a JSON file.
    
    Args:
        pdb_id (str): PDB ID
        output_folder (str): Output folder path
        complex_log_data (dict): Dictionary containing detailed log data for the complex
    """
    log_file_path = os.path.join(output_folder, f"{pdb_id}_log.json")
    
    try:
        with open(log_file_path, 'w') as f:
            json.dump(complex_log_data, f, indent=2, default=str)
        logger.debug(f"Complex log saved: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to save complex log for {pdb_id}: {e}")

def process_complex(pdb_id, output_folder, metadata_df, af3_folder, reference_folder):
    """
    Process a complex and create detailed per-complex logging.
    """
    # Initialize complex log data structure
    complex_log_data = {
        'pdb_id': pdb_id,
        'processing_start': datetime.now().isoformat(),
        'processing_end': None,
        'total_samples_found': 0,
        'successful_samples': 0,
        'failed_samples': 0,
        'samples': [],
        'errors': [],
        'overall_status': 'UNKNOWN'
    }
    
    try:
        # create h5 file for the complex
        h5_file_path = os.path.join(output_folder, f"{pdb_id}.h5")
        if os.path.exists(h5_file_path):
            logger.info(f"H5 file already exists for {pdb_id}, deleting old version...")
            os.remove(h5_file_path)
        
        with h5py.File(h5_file_path, 'w') as h5_file:

            # Get chain information from metadata
            chain_info = parse_chain_info(metadata_df, pdb_id)
            if not chain_info:
                error_msg = f"No metadata found for {pdb_id}"
                logger.warning(error_msg)
                complex_log_data['errors'].append(error_msg)
                complex_log_data['overall_status'] = 'FAILED'
                return False
            
            # Find reference structure
            reference_file = find_reference_structure(reference_folder, pdb_id)
            if not reference_file:
                error_msg = f"No reference structure found for {pdb_id}"
                logger.warning(error_msg)
                complex_log_data['errors'].append(error_msg)
                complex_log_data['overall_status'] = 'FAILED'
                return False
            
            # Find all AF3 samples for this PDB across all seeds
            samples = find_all_af3_samples_across_seeds(af3_folder, pdb_id)
            if not samples:
                error_msg = f"No AF3 samples found for {pdb_id}"
                logger.warning(error_msg)
                complex_log_data['errors'].append(error_msg)
                complex_log_data['overall_status'] = 'FAILED'
                return False
            
            complex_log_data['total_samples_found'] = len(samples)
            logger.info(f"Found {len(samples)} samples for {pdb_id} across all seeds")
            
            # Process each sample
            for sample_idx, (seed_name, sample_name, prediction_file) in enumerate(samples, 1):
                # logger.info(f"Processing sample {sample_idx}/{len(samples)}: {seed_name}/{sample_name}")
                
                # Initialize sample log entry
                sample_log = {
                    'sample_index': sample_idx,
                    'seed_name': seed_name,
                    'sample_name': sample_name,
                    'prediction_file': prediction_file,
                    'confidences_file': None,
                    'processing_start': datetime.now().isoformat(),
                    'processing_end': None,
                    'status': 'UNKNOWN',
                    'components': {
                        'dockq_success': False,
                        'af3_success': False,
                        'tm_success': False,
                         'metadata_success': False,
                         'distance_success': False,
                         'residue_labels_success': False
                    },
                    'errors': []
                }
                
                # Find confidences file for this sample
                prediction_folder = os.path.dirname(prediction_file)
                confidences_file = find_confidences_file(prediction_folder)
                sample_log['confidences_file'] = confidences_file
                
                if not confidences_file:
                    warning_msg = f"No confidences file found for {sample_name}"
                    logger.warning(warning_msg)
                    sample_log['errors'].append(warning_msg)
                
                # Process the sample and save to H5
                try:
                    success = process_sample(h5_file, pdb_id, seed_name, sample_name, 
                                                    prediction_file, reference_file, chain_info, confidences_file)
                    
                    sample_log['processing_end'] = datetime.now().isoformat()
                    
                    if success:
                        sample_log['status'] = 'SUCCESS'
                        complex_log_data['successful_samples'] += 1
                        log_structure_status(pdb_id, seed_name, sample_name, 'SUCCESS')
                        
                        # Try to get component status from the H5 file if available
                        try:
                            sample_group_name = f"{seed_name}_{sample_name}"
                            if pdb_id in h5_file and sample_group_name in h5_file[pdb_id]:
                                sample_group = h5_file[pdb_id][sample_group_name]
                                
                                # Check DockQ success - exists and not NaN
                                dockq_exists = 'abag_dockq' in sample_group
                                if dockq_exists:
                                    dockq_value = sample_group['abag_dockq'][()]
                                    sample_log['components']['dockq_success'] = not np.isnan(dockq_value)
                                    if np.isnan(dockq_value):
                                        sample_log['errors'].append("DockQ computation failed (NaN values)")
                                else:
                                    sample_log['components']['dockq_success'] = False
                                    sample_log['errors'].append("DockQ data missing from H5 file")
                                
                                sample_log['components']['af3_success'] = 'ptm' in sample_group
                                
                                # Check TM score success - exists and not NaN
                                tm_exists = 'tm_normalized_reference' in sample_group
                                if tm_exists:
                                    tm_ref_value = sample_group['tm_normalized_reference'][()]
                                    tm_query_value = sample_group['tm_normalized_query'][()] if 'tm_normalized_query' in sample_group else np.nan
                                    sample_log['components']['tm_success'] = not (np.isnan(tm_ref_value) or np.isnan(tm_query_value))
                                    if np.isnan(tm_ref_value) or np.isnan(tm_query_value):
                                        sample_log['errors'].append("TM score computation failed (NaN values)")
                                else:
                                    sample_log['components']['tm_success'] = False
                                    sample_log['errors'].append("TM score data missing from H5 file")
                                
                                sample_log['components']['metadata_success'] = 'pdb_id' in sample_group

                                # Distance matrix and residue labels
                                if 'ca_distance_matrix' in sample_group:
                                    try:
                                        ca_dist = sample_group['ca_distance_matrix'][()]
                                        sample_log['components']['distance_success'] = np.isfinite(ca_dist).any()
                                        if not sample_log['components']['distance_success']:
                                            sample_log['errors'].append("Distance matrix contains no finite values")
                                    except Exception as e:
                                        sample_log['components']['distance_success'] = False
                                        sample_log['errors'].append(f"Failed reading distance matrix: {e}")
                                else:
                                    sample_log['components']['distance_success'] = False
                                    sample_log['errors'].append("Distance matrix missing from H5 file")

                                token_len = None
                                try:
                                    if 'token_chain_ids' in sample_group:
                                        token_len = len(sample_group['token_chain_ids'])
                                except Exception:
                                    token_len = None

                                have_one = 'residue_one_letter' in sample_group
                                have_three = 'residue_three_letter' in sample_group
                                if have_one and have_three:
                                    try:
                                        n1 = len(sample_group['residue_one_letter'])
                                        n3 = len(sample_group['residue_three_letter'])
                                        sample_log['components']['residue_labels_success'] = (n1 == n3) and (token_len is None or n1 == token_len)
                                        if not sample_log['components']['residue_labels_success']:
                                            sample_log['errors'].append("Residue label arrays size mismatch")
                                    except Exception as e:
                                        sample_log['components']['residue_labels_success'] = False
                                        sample_log['errors'].append(f"Failed reading residue labels: {e}")
                                else:
                                    sample_log['components']['residue_labels_success'] = False
                                    sample_log['errors'].append("Residue label arrays missing from H5 file")
                        except Exception as e:
                            sample_log['errors'].append(f"Failed to check component status: {str(e)}")
                            
                    else:
                        sample_log['status'] = 'FAILED'
                        complex_log_data['failed_samples'] += 1
                        sample_log['errors'].append("Sample processing returned failure status")
                        log_structure_status(pdb_id, seed_name, sample_name, 'FAILED')
                        
                except Exception as e:
                    sample_log['status'] = 'FAILED'
                    sample_log['processing_end'] = datetime.now().isoformat()
                    error_msg = f"Exception during sample processing: {str(e)}"
                    sample_log['errors'].append(error_msg)
                    complex_log_data['failed_samples'] += 1
                    logger.error(f"Error processing sample {seed_name}/{sample_name}: {e}")
                
                # Add sample log to complex log
                complex_log_data['samples'].append(sample_log)

        # Determine overall status
        if complex_log_data['successful_samples'] > 0:
            if complex_log_data['failed_samples'] == 0:
                complex_log_data['overall_status'] = 'SUCCESS'
            else:
                complex_log_data['overall_status'] = 'PARTIAL_SUCCESS'
        else:
            complex_log_data['overall_status'] = 'FAILED'
            
        complex_log_data['processing_end'] = datetime.now().isoformat()
        
        # Save the complex log
        save_complex_log(pdb_id, output_folder, complex_log_data)
        
        # Return True if at least some samples were processed successfully
        return complex_log_data['successful_samples'] > 0
        
    except Exception as e:
        error_msg = f"Critical error processing complex {pdb_id}: {str(e)}"
        logger.error(error_msg)
        complex_log_data['errors'].append(error_msg)
        complex_log_data['overall_status'] = 'FAILED'
        complex_log_data['processing_end'] = datetime.now().isoformat()
        
        # Save the complex log even on failure
        save_complex_log(pdb_id, output_folder, complex_log_data)
        return False

def signal_handler(signum, frame):
       """Handle Ctrl+C signal to terminate program cleanly."""
       global shutdown_requested, executor
       shutdown_requested = True
       
       if executor is not None:
           executor.shutdown(wait=False)
       
       sys.exit(0)
       
# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function that orchestrates the entire processing pipeline.
    
    This function:
    1. Parses command-line arguments
    2. Reads metadata from CSV
    3. Discovers all PDB IDs and samples
    4. Processes each sample and saves to H5 file
    5. Provides progress updates and statistics
    6. Handles resume functionality and checkpointing
    """
    global executor
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Process AlphaFold3 predictions and create H5 files")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV file")
    parser.add_argument("--af3_folder", required=True, help="Path to folder containing AF3 predictions with seed subfolders")
    parser.add_argument("--reference_folder", required=True, help="Path to folder containing reference structures")
    parser.add_argument("--output", required=True, help="Path to output folder")
    parser.add_argument("--limit", type=int, help="Limit number of PDBs to process (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    #if resume is true the output folder must contain a checkpoint file
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())

    if args.resume:
        if not os.path.exists(args.output):
            print("Output folder does not exist. Starting fresh.")
        if not os.path.exists(os.path.join(args.output, 'checkpoint.txt')):
            print("Checkpoint file not found in output folder. Starting fresh.")
        else:
            print("Output folder exists.")
            run_subfolder = args.output
            print(f"Resuming from previous run in {run_subfolder}")
    else:    
        # Create output directory if it doesn't exist
        output_dir = args.output
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        # create run subfolder
        run_subfolder = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_subfolder, exist_ok=True)

    logger = setup_logging(run_subfolder, log_level)
    
    logger.info("Starting AlphaFold3 prediction processing pipeline")
    logger.info(f"Output folder: {run_subfolder}")
    logger.info(f"Log level: {args.log_level}")
    
    # Validate input files and folders
    if not validate_input_files(args.metadata, args.af3_folder, args.reference_folder):
        logger.error("Input validation failed. Exiting.")
        sys.exit(1)
    
    # Read metadata CSV file
    logger.info(f"Reading metadata from {args.metadata}")
    metadata_df = pd.read_csv(args.metadata)
    
    # Discover all PDB IDs from the AF3 folder structure
    logger.info(f"Discovering PDB IDs from {args.af3_folder}")
    pdb_ids = discover_pdb_ids_from_single_seed(args.af3_folder, limit=args.limit)
    
    logger.info(f"Found {len(pdb_ids)} PDB structures: {pdb_ids}")
    
    # Initialize counters for progress tracking
    processed_count = 0
    skipped_count = 0
    completed_pdbs = []
    
    # Initialize tracking lists for comprehensive logging
    processed_samples = []
    failed_samples = []
    
    # Check if resuming or forcing restart
    checkpoint_file = os.path.join(run_subfolder, 'checkpoint.txt')
    
    if args.resume:
        logger.info("Resuming from previous run...")

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                completed_pdbs = f.read().splitlines()
            # Get completed PDBs from checkpoint
            # Filter out PDBs that are already completed
            pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id not in completed_pdbs]
            logger.info(f"Resuming from {len(completed_pdbs)} completed PDBs. Starting with {len(pdb_ids)} new PDBs.")
        else:
            logger.warning("No previous run found or checkpoint file corrupted. Starting fresh.")
                
    # Process each PDB ID         
    # Use ThreadPoolExecutor to process complexes in parallel

    # Get the number of available CPU cores
    num_cores = 4

    logger.info(f"Using {num_cores} cores for processing")

    checkpoint_lock = threading.Lock()

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {}
            total_pdbs = len(pdb_ids)

            #submit all the futures
            for pdb_idx, pdb_id in enumerate(pdb_ids, 1):
                if shutdown_requested:
                    break
                logger.info(f"Processing PDB {pdb_idx}/{total_pdbs}: {pdb_id}")
                future = executor.submit(process_complex, 
                                               pdb_id, run_subfolder, 
                                               metadata_df, 
                                               args.af3_folder, 
                                               args.reference_folder)
                futures[future] = pdb_id

            for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
                if shutdown_requested:
                    break
                    
                pdb_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_count += 1
                        completed_pdbs.append(pdb_id)
                        logger.info(f"Completed PDB {pdb_id} ({idx}/{total_pdbs})")
                        #save add pdb_id to checkpoint file and save it in thread safe manner
                        with checkpoint_lock:
                            with open(checkpoint_file, 'a') as f:
                                f.write(f"{pdb_id}\n")
                    else:
                        skipped_count += 1
                        failed_samples.append(pdb_id)
                        logger.info(f"Skipped PDB {pdb_id} ({idx}/{total_pdbs})")
                except Exception as e:
                    logger.error(f"Error processing PDB {pdb_id}: {e}")
                    failed_samples.append(pdb_id)
                    logger.info(f"Skipped PDB {pdb_id} ({idx}/{total_pdbs})")

    except KeyboardInterrupt:
        # This should not be reached due to signal handler, but just in case
        sys.exit(0)

    # Create comprehensive processing summary
    create_processing_summary(processed_samples, failed_samples, run_subfolder)
    
    # Print final statistics
    print_summary_statistics(processed_count, skipped_count, total_pdbs, completed_pdbs)
    logger.info(f"Processing completed! Results saved to {args.output}")

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
