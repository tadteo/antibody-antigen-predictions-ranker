import gzip
import numpy as np
import biotite as bt
import biotite.structure.io as pdb_io
from biotite.structure.io import load_structure

def get_structure_tite(pdb_loc: str,
                       compressed=False,
                       backbone: bool = True,
                       ensemble: bool = False,
                       clean_hetero: bool = True) -> tuple:
    '''
    Load a protein structure from a PDB or CIF file using Biotite.

    Args:
        pdb_loc (str): Path to the structure file.
        compressed (bool): Whether the file is gzipped.
        backbone (bool): Return only backbone atoms if True.
        ensemble (bool): Load full ensemble if True.
        clean_hetero (bool): Remove heteroatoms if True.

    Returns:
        protein (AtomArray): Filtered protein structure.
        chain_id (str | list): Chain ID(s).
    '''
    EXCLUDE_HETERO = {
        "HOH", "SO4", "PO4", "CL", "NA", "K", "MG", "CA", "ZN", "NO3", "ACT", "DTT", "TRS", "BME",
        "GOL", "EDO", "MN", "CO", "CU", "FE", "NI", "CO3", "FES", "FAD", "FMN", "PCA", "PLM",
        "PQQ", "PQQH2", "PDS", "PPT", "PQQH", "FADH2", "FADH"
    }

    if compressed:
        with gzip.open(pdb_loc, "rt") as file_handle:
            pdb_file = pdb_io.PDBFile.read(file_handle)
            protein = pdb_io.get_structure(pdb_file, extra_fields=['b_factor'])
    else:
        protein = load_structure(pdb_loc, extra_fields=["b_factor"])

    if isinstance(protein, bt.structure.AtomArrayStack):
        print(f"Multiple models found in {pdb_loc}. Taking first model.")
        protein = protein[0]

    if ensemble:
        mask = (~protein.hetero) & (protein.ins_code == '')
        if backbone:
            mask &= np.isin(protein.atom_name, ['N', 'CA', 'C', 'O'])
        protein = protein[:, mask]
        return protein, list(np.unique(protein.chain_id))

    # Single-structure mode
    chain_ids = np.unique(protein.chain_id)
    if clean_hetero:
        protein = protein[
            (~protein.hetero) &
            (protein.ins_code == '')
        ]
    else:
        protein = protein[
            (protein.ins_code == '') &
            (~np.isin(protein.res_name, list(EXCLUDE_HETERO)))
        ]

    if backbone:
        protein = protein[np.isin(protein.atom_name, ['N', 'CA', 'C', 'O'])]

    if len(protein) == 0:
        print(f"No atoms left after filtering in {pdb_loc}")
        return None, None

    return protein, chain_ids.tolist() if len(chain_ids) > 1 else chain_ids[0]
