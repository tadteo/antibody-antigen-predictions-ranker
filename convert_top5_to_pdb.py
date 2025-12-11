#!/usr/bin/env python3
"""
Script to convert top 5 ranked CIF files to PDB format and create submission zip files.
"""

import os
import pandas as pd
import glob
import zipfile
import shutil
from pathlib import Path

# Directories
INFERENCE_DIR = "/proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/inference_results"
CAPRI_DIR = "/proj/berzelius-2021-29/users/x_matta/CAPRI_2025_October"
OUTPUT_DIR = "/proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/capri_submissions"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_cif_file(capri_dir, target_id, sample_name):
    """
    Find the CIF file for a given target and sample.
    Sample name format: seed-xxx_sample-y
    """
    # Search in all RUN directories
    search_pattern = f"{capri_dir}/{target_id}/INIT_MODELS/AF3_MODELS/RUN*/{sample_name}/model.cif"
    cif_files = glob.glob(search_pattern)
    
    if cif_files:
        return cif_files[0]
    else:
        print(f"WARNING: CIF file not found for {target_id}/{sample_name}")
        return None

def add_pdb_headers_and_ter(pdb_file, target_id):
    """
    Add required HEADER, AUTHOR, and COMPND records to PDB file.
    Also ensure TER records are present after each chain.
    """
    # Map target IDs to descriptions
    target_descriptions = {
        "T312_61": "CAPRI ROUND 61 TARGET 312",
        "T314_61": "CAPRI ROUND 61 TARGET 314",
        "T315_61": "CAPRI ROUND 61 TARGET 315",
        "T316_61": "CAPRI ROUND 61 TARGET 316",
    }
    
    target_desc = target_descriptions.get(target_id, f"CAPRI ROUND 61 {target_id}")
    
    # Read existing PDB content
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    # Prepare headers
    headers = [
        f"HEADER    {target_desc}\n",
        "AUTHOR    MATTEO TADIELLO\n",
        "COMPND    MOL_ID: 1;\n",
        "COMPND   2 MOLECULE: ANTIGEN;\n",
        "COMPND   3 CHAIN: A;\n",
        "COMPND   4 MOL_ID: 2;\n",
        "COMPND   5 MOLECULE: ANTIBODY HEAVY CHAIN;\n",
        "COMPND   6 CHAIN: H;\n",
        "COMPND   7 MOL_ID: 3;\n",
        "COMPND   8 MOLECULE: ANTIBODY LIGHT CHAIN;\n",
        "COMPND   9 CHAIN: L;\n",
        "REMARK 300\n",
        "REMARK 300 PREDICTED MODEL\n",
        "REMARK 300\n",
    ]
    
    # Process lines to add TER records after each chain
    new_lines = []
    current_chain = None
    last_atom_line = None
    has_model = False
    header_added = False
    
    for line in lines:
        # Skip existing headers that might interfere
        if line.startswith(('HEADER', 'AUTHOR', 'COMPND', 'REMARK')):
            continue
            
        if line.startswith('MODEL'):
            if not header_added:
                new_lines.extend(headers)
                header_added = True
            has_model = True
            new_lines.append(line)
        elif line.startswith('ATOM') or line.startswith('HETATM'):
            # Add headers before first ATOM if no MODEL found
            if not header_added:
                new_lines.extend(headers)
                header_added = True
                
            chain = line[21:22]
            if current_chain is not None and chain != current_chain and last_atom_line:
                # Add TER record for previous chain
                atom_num = int(last_atom_line[6:11].strip()) + 1
                res_name = last_atom_line[17:20]
                res_num = last_atom_line[22:26]
                ter_line = f"TER   {atom_num:5d}      {res_name} {current_chain}{res_num}\n"
                new_lines.append(ter_line)
            current_chain = chain
            last_atom_line = line
            new_lines.append(line)
        elif line.startswith('ENDMDL'):
            # Add final TER record before ENDMDL
            if last_atom_line:
                atom_num = int(last_atom_line[6:11].strip()) + 1
                res_name = last_atom_line[17:20]
                res_num = last_atom_line[22:26]
                ter_line = f"TER   {atom_num:5d}      {res_name} {current_chain}{res_num}\n"
                new_lines.append(ter_line)
            new_lines.append(line)
            current_chain = None
            last_atom_line = None
        elif line.startswith('END'):
            # Add TER before END if needed
            if last_atom_line and current_chain:
                atom_num = int(last_atom_line[6:11].strip()) + 1
                res_name = last_atom_line[17:20]
                res_num = last_atom_line[22:26]
                ter_line = f"TER   {atom_num:5d}      {res_name} {current_chain}{res_num}\n"
                new_lines.append(ter_line)
                current_chain = None
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Add final TER and END if not present
    if last_atom_line and current_chain:
        atom_num = int(last_atom_line[6:11].strip()) + 1
        res_name = last_atom_line[17:20]
        res_num = last_atom_line[22:26]
        ter_line = f"TER   {atom_num:5d}      {res_name} {current_chain}{res_num}\n"
        new_lines.append(ter_line)
    
    # Add END if not present
    if not any(line.startswith('END') for line in new_lines):
        new_lines.append("END\n")
    
    # Write back
    with open(pdb_file, 'w') as f:
        f.writelines(new_lines)

def convert_cif_to_pdb(cif_file, output_pdb, target_id):
    """
    Convert CIF file to PDB format using BioPython with proper CAPRI headers.
    """
    try:
        from Bio.PDB import MMCIFParser, PDBIO
        
        # Parse CIF file
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', cif_file)
        
        # Write to PDB
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_pdb)
        
        # Add required headers and TER records
        add_pdb_headers_and_ter(output_pdb, target_id)
        
        return True
    except ImportError:
        print("ERROR: BioPython not installed.")
        print("Please install: pip install biopython")
        return False
    except Exception as e:
        print(f"ERROR converting {cif_file}: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_submission_zip(target_id, pdb_files):
    """
    Create submission-ready zip files for a CAPRI target.
    Creates one zip file containing all 5 models named model_1.pdb through model_5.pdb.
    """
    if not pdb_files:
        print(f"  WARNING: No PDB files to zip for {target_id}")
        return []
    
    # Create main zip file with all 5 models
    zip_filename = f"{target_id.lower()}.zip"
    zip_path = os.path.join(OUTPUT_DIR, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, pdb_file in enumerate(pdb_files, start=1):
            if os.path.exists(pdb_file):
                # Add file as model_1.pdb, model_2.pdb, etc.
                arcname = f"model_{i}.pdb"
                zipf.write(pdb_file, arcname)
    
    return [zip_path]

def process_target(target_dir):
    """
    Process a single target directory.
    Returns list of created PDB files.
    """
    target_id = os.path.basename(target_dir)
    
    # Find the ranked CSV file
    csv_pattern = f"{target_dir}/*ranked_by_model.csv"
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"WARNING: No ranked CSV file found in {target_dir}")
        return []
    
    csv_file = csv_files[0]
    print(f"\nProcessing target: {target_id}")
    print(f"  CSV file: {csv_file}")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Get top 5 samples
    top5 = df.head(5)
    
    pdb_files = []
    
    print(f"  Top 5 samples:")
    for idx, row in top5.iterrows():
        sample_name = row['sample_name']
        rank = row['rank']
        
        print(f"    Rank {rank}: {sample_name}")
        
        # Find CIF file
        cif_file = find_cif_file(CAPRI_DIR, target_id, sample_name)
        
        if cif_file:
            # Parse sample name to extract seed and sample number
            # Format: seed-xxx_sample-y
            parts = sample_name.split('_')
            seed = parts[0].replace('seed-', '')
            sample = parts[1].replace('sample-', '')
            
            # Create output filename: target_x_seed_x_sample_x_rank_x.pdb
            output_filename = f"{target_id}_seed_{seed}_sample_{sample}_rank_{rank}.pdb"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Convert CIF to PDB
            print(f"      Converting to: {output_filename}")
            success = convert_cif_to_pdb(cif_file, output_path, target_id)
            
            if success:
                print(f"      ✓ Successfully converted")
                pdb_files.append(output_path)
            else:
                print(f"      ✗ Conversion failed")
        else:
            print(f"      ✗ CIF file not found")
    
    return pdb_files

def main():
    """
    Main function to process all targets.
    """
    print("=" * 70)
    print("Converting top 5 ranked CIF files to PDB and creating submission zips")
    print("=" * 70)
    
    # Find all target directories
    target_dirs = sorted(glob.glob(f"{INFERENCE_DIR}/T*"))
    
    if not target_dirs:
        print("ERROR: No target directories found in inference_results")
        return
    
    print(f"\nFound {len(target_dirs)} target directories")
    
    submission_zips = []
    
    # Process each target
    for target_dir in target_dirs:
        target_id = os.path.basename(target_dir)
        pdb_files = process_target(target_dir)
        
        if pdb_files:
            # Create submission zip file
            print(f"\n  Creating submission zip for {target_id}...")
            zip_paths = create_submission_zip(target_id, pdb_files)
            
            if zip_paths:
                for zip_path in zip_paths:
                    print(f"  ✓ Zip created: {os.path.basename(zip_path)}")
                    submission_zips.append((target_id, zip_path))
            else:
                print(f"  ✗ Failed to create zip")
    
    print("\n" + "=" * 70)
    print("CONVERSION AND ZIP CREATION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    if submission_zips:
        print(f"\n✓ Created {len(submission_zips)} submission zip files:")
        for target_id, zip_path in submission_zips:
            zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
            print(f"  - {os.path.basename(zip_path)} ({zip_size:.2f} MB)")
    else:
        print("\n✗ No submission zips were created")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

