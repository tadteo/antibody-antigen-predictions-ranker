#!/bin/bash

# Script to download missing PDB files identified in the SAbDab database analysis
# Run this script from the /proj/berzelius-2021-29/Database/sabdab/raw directory

echo "Downloading missing PDB files..."
echo "================================"

# List of missing PDB IDs
missing_pdbs=(
    "8z8l" "8zbi" "8zbj" "8zca" "8zh7" "8zym" "9b7g" "9bnk" "9bnl" "9c0u"
    "9c0x" "9c2h" "9cb5" "9cp5" "9d0a" "9d4z" "9d8y" "9d90" "9d98" "9eci"
    "9ego" "9fns" "9fnt" "9fq3" "9ggp" "9gix" "9gu0" "9gu1" "9gu2" "9gu3"
    "9hy1" "9hy2" "9hy3" "9hy5" "9ixv" "9nox" "9o38" "9qub" "9quw" "9r2e"
)

# Base URL for PDB files
pdb_base_url="https://files.rcsb.org/download"

# Counter for successful downloads
success_count=0
fail_count=0

echo "Total missing PDB files: ${#missing_pdbs[@]}"
echo ""

# Download each missing PDB file
for pdb_id in "${missing_pdbs[@]}"; do
    echo "Downloading ${pdb_id}.pdb..."
    
    # Try to download the file
    if wget -q "${pdb_base_url}/${pdb_id}.pdb" -O "${pdb_id}.pdb"; then
        echo "✓ Successfully downloaded ${pdb_id}.pdb"
        ((success_count++))
    else
        echo "✗ Failed to download ${pdb_id}.pdb"
        ((fail_count++))
        # Remove the empty file if download failed
        rm -f "${pdb_id}.pdb"
    fi
done

echo ""
echo "Download Summary:"
echo "=================="
echo "Successfully downloaded: $success_count"
echo "Failed downloads: $fail_count"
echo "Total processed: ${#missing_pdbs[@]}"

if [ $success_count -gt 0 ]; then
    echo ""
    echo "Verifying downloaded files..."
    new_count=$(ls *.pdb | wc -l)
    echo "Total PDB files now: $new_count"
fi

echo ""
echo "Download process completed!" 
