#!/usr/bin/env python3
import os
import math
import yaml
import random
import pandas as pd
import numpy as np
import h5py
import torch
from collections import OrderedDict
from functools import partial
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
import torch.nn.functional as F
from src.data.triangular_positional_encoding import triangular_encode_features
from tqdm import tqdm
import gc

# ==============================================================================
# Helper functions for distributed training
# ==============================================================================
def create_distributed_collate_fn(local_batch_size: int, samples_per_complex: int):
    """Create collate function with correct local batch size for distributed training"""
    return partial(pad_collate_fn, batch_size=local_batch_size, samples_per_complex=samples_per_complex)

# ==============================================================================
# Load our user‐config (e.g. number of models to draw per complex per epoch)
# ==============================================================================
# We try to load config, but if it fails (e.g. file not found), we continue as it might not be needed
try:
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except Exception:
    config = {}

# ==============================================================================
#  Collate function
# ==============================================================================

def pad_collate_fn(batch, batch_size, samples_per_complex):
    features   = [b["features"] for b in batch]            # (F, N_i)
    labels  = torch.tensor([b["label"]  for b in batch])   # (B·K,)
    weights = torch.tensor([b["weight"] for b in batch])   # (B·K,)
    lengths = torch.tensor([b["length"] for b in batch])   # (B·K,)
    complex_ids = [b["complex_id"] for b in batch]         # list of str
    ranking_score = torch.tensor([b["ranking_score"] for b in batch])   # (B·K,)
    ranking_score = ranking_score.reshape(batch_size, samples_per_complex)

    max_n   = max(lengths)
    
    # Prepare the features tensor
    padded = torch.zeros(batch_size * samples_per_complex, features[0].shape[0], max_n)  # (B*K, F, max_n)
    
    # Insert features into padded tensor
    for i, f in enumerate(features):
        N_i = f.shape[1]  # Number of residues in this sample
        padded[i, :, :N_i] = f  # Pad with zeros to match max_n

    # Transpose to (B*K, N, F)
    padded = padded.transpose(1, 2)
    # Reshape to (B, K, N, F)
    padded = padded.reshape(batch_size, samples_per_complex, max_n, features[0].shape[0])
    # print(f"padded shape: {padded.shape}")

    #prepare the labels, weights and complex_ids tensors
    labels = labels.reshape(batch_size, samples_per_complex)
    lengths = lengths.reshape(batch_size, samples_per_complex)
    weights = weights.reshape(batch_size, samples_per_complex)
    complex_ids = np.array(complex_ids).reshape(batch_size, samples_per_complex)

    # Ensure all K decoys per B row belong to the same complex
    rows_equal = np.all([len(set(row)) == 1 for row in complex_ids.tolist()])
    if not rows_equal:
        raise ValueError("All decoys in a row must share the same complex_id")

    return {
      "features": padded,        # [B, K, N, F]
      "lengths":  lengths,       # [B, K]
      "label":    labels,        # [B, K]
      "weight":   weights,       # [B, K]
      "complex_id": complex_ids,  # [B, K]
      "ranking_score": ranking_score  # [B, K]
    }

# ==============================================================================
# DistributedBatchSampler - Wrapper for DDP compatibility
# ==============================================================================
class DistributedBatchSampler(torch.utils.data.Sampler):
    """
    Wraps a *batch_sampler* that yields a flat list of indices of length B*K.
    We shard each full batch into per-rank chunks of length local_B*K.
    If needed, we PAD by repeating head indices so every rank gets the same size.
    
    This preserves your WeightedComplexSampler logic while making it DDP-safe.
    """
    def __init__(self, base_batch_sampler, *, world_size=None, rank=None,
                 local_batch_size: int, samples_per_complex: int,
                 drop_last: bool = False, pad: bool = True):
        super().__init__(None)
        
        # Get distributed parameters
        try:
            import torch.distributed as dist
            if world_size is None:
                world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            if rank is None:
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except ImportError:
            world_size = 1
            rank = 0
            
        self.base = base_batch_sampler
        self.world_size = world_size
        self.rank = rank
        self.local_B = local_batch_size           # per-rank complexes
        self.K = samples_per_complex              # samples per complex
        self.per_rank_need = self.local_B * self.K
        self.drop_last = drop_last
        self.pad = pad

        # Optional: forward set_epoch if the base supports it
        self._has_set_epoch = hasattr(self.base, "set_epoch")

    def set_epoch(self, epoch: int):
        """Forward epoch setting to base sampler if supported (for reproducibility)"""
        if self._has_set_epoch:
            self.base.set_epoch(epoch)

    def __iter__(self):
        for full_batch in self.base:  # full_batch: List[int], len ≈ B*K
            want_total = self.per_rank_need * self.world_size

            if len(full_batch) < want_total:
                if self.pad:
                    # repeat from the head deterministically to reach want_total
                    need = want_total - len(full_batch)
                    cycle_indices = []
                    for i in range(need):
                        cycle_indices.append(full_batch[i % len(full_batch)])
                    full_batch = list(full_batch) + cycle_indices
                elif self.drop_last:
                    # drop the tail so it's divisible
                    keep = (len(full_batch) // self.per_rank_need) * self.per_rank_need
                    full_batch = full_batch[:keep]
                else:
                    # last batch might be smaller → shard unevenly but still yield something
                    pass

            # ensure length is multiple of per_rank_need for even sharding
            if len(full_batch) >= want_total:
                full_batch = full_batch[:want_total]

            # Shard across ranks
            start = self.rank * self.per_rank_need
            end = start + self.per_rank_need
            shard = full_batch[start:end] if start < len(full_batch) else []
            
            if shard or not self.drop_last:
                yield shard

    def __len__(self):
        """Return number of batches this rank will see"""
        # Each full batch from base becomes one per-rank batch here
        return len(self.base)

# ==============================================================================
# WeightedComplexSampler
# ==============================================================================
class WeightedComplexSampler(Sampler):
    """
    This sampler provide a B,K batch of complexes, where B is the batch size and K is the number of complexes in the batch.
    The final dimension will be B,K,N,F, where N is the number of residues in the complex and F is the number of features.
    """
    def __init__(self,
                 dataset,
                 batch_size: int,
                 samples_per_complex: int,
                 weighted: bool = True,
                 replacement: bool = True,
                 seed: int = None
                 ):
        """
        dataset: an instance of AntibodyAntigenPAEDataset
        batch_size: how many complexes to draw per batch used for bucket balancing
        samples_per_complex: how many samples to draw per complex each epoch
        weighted: whether to weight the samples within a complex by their weight
        replacement: whether to draw with replacement when a complex has fewer
                     samples than samples_per_complex
        seed: seed for the RNG
        """
        self.df = dataset.df  # the manifest filtered to one split
        self.batch_size = batch_size # B
        self.samples_per_complex  = samples_per_complex # K
        self.weighted = weighted
        self.replacement = replacement
        self.rng = np.random.RandomState(seed)

        #create a cache dictionary of the type:
        # (1)  sort rows by length – ascending or descending
        df_sorted = self.df.sort_values("len_sample", ascending=True)   # or False for longest-first
        
        # (2)  build an ordered mapping  {cid: {...}}  in that length order
        self.by_cid = OrderedDict(
            (
                cid,
                {
                    "length":  int(g["len_sample"].iloc[0]),
                    "idx_list": g.index.to_numpy(),         # or g["row_idx"].to_numpy()
                    "weights":  g["weight"].to_numpy(),
                },
            )
            for cid, g in df_sorted.groupby("complex_id", sort=False)
        )
        
        self.chunks = []
        for cid, info in self.by_cid.items():
            rows = info["idx_list"].tolist()
            for i in range(0, len(rows), self.samples_per_complex):
                chunk = rows[i:i+self.samples_per_complex]
                if len(chunk) < self.samples_per_complex:
                    # If the chunk is shorter, fill it with random samples from the same complex
                    remaining = self.samples_per_complex - len(chunk)
                    chunk.extend(self.rng.choice(rows, size=remaining, replace=True).tolist())
                self.chunks.append(chunk)


    def __iter__(self):
        """
        Yields a list of indices whose length is exactly
            batch_size * samples_per_complex   ==  B · K
        except for the very last batch, which may be smaller.
        """

        if self.weighted:
            current_batch = []          # list of row-indices that we'll yield
            complexes_in_batch = 0      # how many distinct complexes already in list

            for complexid, info in self.by_cid.items():

                rows  = info["idx_list"]         # 1-D numpy array of row indices
                w     = info["weights"]          # 1-D numpy array of per-row weights

                # It samples just K elemnts per complex
                p = w / w.sum()              # normalise *inside* the complex
                picked = self.rng.choice(
                    rows,
                    size=self.samples_per_complex,
                    replace=self.replacement or len(rows) < self.samples_per_complex,
                    p=p,
                )                  

                current_batch.extend(picked.tolist())
                complexes_in_batch += 1

                # ---- yield once we have B complexes --------------------------
                if complexes_in_batch == self.batch_size:
                    yield current_batch           # length == B·K
                    current_batch = []
                    complexes_in_batch = 0

            # last incomplete batch
            if current_batch:
                yield current_batch
        else:
            # Take all the chunks from each complex (not ordering for padding but never mind)
            # Shuffle the chunks
            self.rng.shuffle(self.chunks)
            # Yield the chunks
            for i in range(0, len(self.chunks), self.batch_size):
                tmp = self.chunks[i:i+self.batch_size]  # List[List[int]]
                # Flatten the list of lists
                
                if len(tmp) < self.batch_size:
                    # If the last chunk is shorter, fill it with random samples from the same complex
                    remaining = self.batch_size - len(tmp)
                    # print(f"remaining: {remaining}")
                    # Get random chunk indices
                    random_chunk_indices = self.rng.choice(len(self.chunks), size=remaining, replace=True)
                    # Get the actual random chunks using the indices
                    random_selected_chunks = [self.chunks[idx] for idx in random_chunk_indices]
                    # print(f"random_selected_chunks shape: {len(random_selected_chunks)}, remaining: {remaining}")
                    tmp.extend(random_selected_chunks)
                    # print(f"tmp shape: {len(tmp)}")
                flat_indices = [idx for chunk in tmp for idx in chunk]
                # Yield the flat indices    
                yield flat_indices
    
    # -------------------------------------------------------
    def __len__(self):
        if self.weighted:
            total_samples = self.number_of_complexes * self.samples_per_complex
        else:
            # If samples_per_complex specified, return total number of samples per epoch
            total_samples = len(self.df)/self.samples_per_complex
        
        return math.ceil(total_samples / self.batch_size)


class PerComplexSequentialSampler(torch.utils.data.Sampler):
    """
    Deterministic, no-shuffle sampler for evaluation.
    Yields flat lists of indices of length B*K (B complexes × K decoys),
    in manifest order. Pads the last batch by repeating chunks to ensure all data is evaluated.
    """
    def __init__(self, dataset, batch_size: int, samples_per_complex: int, pad_last_batch: bool = True):
        self.df = dataset.df                      # already split='val'
        self.batch_size = batch_size              # B
        self.K = samples_per_complex              # K
        self.pad_last_batch = pad_last_batch

        # Complexes in manifest order (no shuffle)
        from collections import OrderedDict
        self.by_cid = OrderedDict(
            (cid, g.index.to_list())
            for cid, g in self.df.groupby("complex_id", sort=False)
        )

        # Build per-complex chunks of exactly K indices (drop the remainder)
        self.chunks = []
        for _, rows in self.by_cid.items():
            rows = list(rows)
             # only keep full K-sized windows; drop the remainder
            full = (len(rows) // self.K) * self.K
            for i in range(0, full, self.K):
                chunk = rows[i:i+self.K]
                self.chunks.append(chunk)

        # Batch chunks deterministically into size B
        self.batches = []
        print(f"the batch size for the validation sampler is {self.batch_size}")
        for i in range(0, len(self.chunks), self.batch_size):
            batch_chunks = self.chunks[i:i+self.batch_size]
            
            # Pad incomplete batches if requested
            if len(batch_chunks) < self.batch_size:
                if self.pad_last_batch:
                    # Pad by repeating chunks from the beginning
                    needed = self.batch_size - len(batch_chunks)
                    padding_chunks = self.chunks[:needed]
                    batch_chunks.extend(padding_chunks)
                    print(f"Padding last validation batch: added {needed} chunks to reach {self.batch_size} complexes")
                else:
                    print(f"Dropping incomplete validation batch with {len(batch_chunks)} complexes (expected {self.batch_size})")
                    continue
            
            # Flatten to a single list of indices
            flat = [idx for ch in batch_chunks for idx in ch]
            self.batches.append(flat)

    def __iter__(self):
        for flat in self.batches:
            yield flat

    def __len__(self):
        return len(self.batches)


# ==============================================================================
#  Dataset: In-Memory Implementation
# ==============================================================================
class AntibodyAntigenPAEDataset(Dataset):
    """
    In-memory dataset that pre-loads all samples from H5 files during initialization.
    Performs filtering, distance cutoff, and feature transformation once at start-up.
    
    This avoids repetitive H5 file opening/closing and repeated preprocessing during training.
    """
    def __init__(self,
                 manifest_csv: str,
                 split: str = "train",
                 feature_transform=None,
                 feature_centering=False,
                 use_interchain_pae: bool = True,
                 use_interchain_ca_distances: bool = False,
                 use_esm_embeddings: bool = False,
                 use_distance_cutoff: bool = False,
                 distance_cutoff: float = 12.0,
                 # Arguments kept for API compatibility but ignored/not used as before
                 use_file_cache: bool = False, 
                 cache_size_mb: int = 512,
                 max_cached_files: int = 20):
        
        print(f"Initializing In-Memory Dataset for split: {split}")
        self.df = pd.read_csv(manifest_csv)
        # Filter to only this split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        
        self.feature_transform = feature_transform
        self.feature_centering = feature_centering
        self.use_interchain_pae = use_interchain_pae
        self.use_interchain_ca_distances = use_interchain_ca_distances
        self.use_esm_embeddings = use_esm_embeddings
        self.use_distance_cutoff = use_distance_cutoff
        self.distance_cutoff = distance_cutoff
        
        # Pre-load all data into memory
        self.samples = self._preload_data()
        
        # After pre-loading, we can force a GC to clean up temporary buffers
        gc.collect()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Direct memory access - O(1)
        return self.samples[idx]
    
    def _preload_data(self):
        """
        Iterates over the manifest grouped by H5 file, opens each file once, 
        and processes all relevant samples. Returns a list of processed samples
        aligned with self.df.
        """
        # Create a placeholder list of the correct size to fill in order
        all_samples = [None] * len(self.df)
        
        # Group indices by H5 file path to minimize file opens
        # We need to preserve the original DataFrame index to place processed samples correctly
        self.df['orig_index'] = self.df.index
        file_groups = self.df.groupby("h5_file")
        
        print(f"Loading {len(self.df)} samples from {len(file_groups)} unique H5 files...")
        
        for h5_path, group in tqdm(file_groups, desc="Loading H5 Files"):
            try:
                with h5py.File(h5_path, 'r') as hf:
                    # Sort group by complex_id to allow caching the complex group
                    # We convert to a list of namedtuples or similar for faster iteration than iterrows
                    # But sorting the dataframe first is easiest
                    group_sorted = group.sort_values("complex_id")
                    
                    current_complex_id = None
                    current_complex_group = None
                    
                    # itertuples is significantly faster than iterrows
                    for row in group_sorted.itertuples(index=False):
                        idx = row.orig_index
                        complex_id = row.complex_id
                        sample_id = row.sample
                        
                        # Cache the complex group handle if we are still on the same complex
                        if complex_id != current_complex_id:
                            current_complex_id = complex_id
                            current_complex_group = hf[complex_id]
                        
                        # Pass the complex group directly to avoid repeated dictionary lookups in h5py
                        processed_sample = self._process_single_sample_optimized(
                            current_complex_group, row, sample_id, complex_id
                        )
                        all_samples[idx] = processed_sample
                        
            except Exception as e:
                print(f"Error reading file {h5_path}: {e}")
                # We might want to raise here or continue, depending on desired robustness.
                # For now, let's raise to be safe.
                raise e
                
        # Remove the helper column
        self.df.drop(columns=['orig_index'], inplace=True)
        
        # Verify no samples were missed
        if any(s is None for s in all_samples):
             raise RuntimeError("Some samples failed to load properly.")
             
        return all_samples

    def _process_single_sample_optimized(self, complex_group, row, sample_id, complex_id):
        """
        Optimized helper method to load and process a single sample.
        Args:
            complex_group: h5py.Group object for the current complex
            row: namedtuple from itertuples() containing metadata
            sample_id: str
            complex_id: str (passed for error reporting/structure)
        """
        try:
            # Data for the primary sample
            original_sample_group = complex_group[sample_id]
            
            # Read datasets
            # [()] reads the whole dataset into a numpy array
            interchain_pae_vals_raw = original_sample_group["interchain_pae_vals"][()]
            interchain_indexes_i = original_sample_group["inter_idx"][()]
            interchain_indexes_j = original_sample_group["inter_jdx"][()]
            ranking_score = original_sample_group["ranking_score"][()]
            residue_one_letter = original_sample_group["residue_one_letter"][()]
            
            # Convert bytes to strings if needed
            if len(residue_one_letter) > 0 and isinstance(residue_one_letter[0], bytes):
                residue_one_letter = [res.decode('utf-8') for res in residue_one_letter]

            # Create mask for non-X residues
            # Vectorized comparison if residue_one_letter is a numpy array of strings, 
            # but usually it's a list or array of bytes. List comp is fast enough for small L.
            non_x_mask = np.array([res != 'X' for res in residue_one_letter])
            
            # Filter interchain pairs
            valid_pair_mask = non_x_mask[interchain_indexes_i] & non_x_mask[interchain_indexes_j]

            interchain_pae_vals_raw = interchain_pae_vals_raw[valid_pair_mask]
            interchain_indexes_i = interchain_indexes_i[valid_pair_mask]
            interchain_indexes_j = interchain_indexes_j[valid_pair_mask]

            if not self.use_interchain_pae:
                interchain_pae_vals_raw = None

            # Optional: interchain Cα distances
            interchain_ca_distances_raw = None
            if self.use_interchain_ca_distances and 'interchain_ca_distances' in original_sample_group:
                interchain_ca_distances_raw = original_sample_group['interchain_ca_distances'][()]
                interchain_ca_distances_raw = interchain_ca_distances_raw[valid_pair_mask]

            # Remap indices to account for removed X residues
            # If no X residues, we can skip remapping? 
            # Usually non_x_mask is all True.
            if not np.all(non_x_mask):
                non_x_indices = np.where(non_x_mask)[0]
                old_to_new_idx = np.full(len(residue_one_letter), -1, dtype=int)
                old_to_new_idx[non_x_indices] = np.arange(len(non_x_indices))
                
                interchain_indexes_i = old_to_new_idx[interchain_indexes_i]
                interchain_indexes_j = old_to_new_idx[interchain_indexes_j]
            else:
                # No X residues, indices remain valid (assuming 0-indexed contiguous)
                pass

            # Distance cutoff
            distance_mask = None
            if self.use_distance_cutoff and interchain_ca_distances_raw is not None:
                distance_mask = interchain_ca_distances_raw <= self.distance_cutoff
                n_pairs_after = distance_mask.sum()
                
                if n_pairs_after == 0:
                    closest_idx = np.argmin(interchain_ca_distances_raw)
                    distance_mask = np.zeros(len(interchain_ca_distances_raw), dtype=bool)
                    distance_mask[closest_idx] = True
                
                interchain_indexes_i = interchain_indexes_i[distance_mask]
                interchain_indexes_j = interchain_indexes_j[distance_mask]
                if interchain_pae_vals_raw is not None:
                    interchain_pae_vals_raw = interchain_pae_vals_raw[distance_mask]
                interchain_ca_distances_raw = interchain_ca_distances_raw[distance_mask]

            # ESM embeddings
            esm_i = None
            esm_j = None
            if self.use_esm_embeddings and 'esm_embeddings' in original_sample_group:
                esm_embeddings_raw = original_sample_group['esm_embeddings'][()]
                esm_embeddings = esm_embeddings_raw[non_x_mask]
                esm_i = esm_embeddings[interchain_indexes_i]
                esm_j = esm_embeddings[interchain_indexes_j]

            # Construct features
            # Use lists for stacking if possible, but numpy array construction is standard
            feature_list = [interchain_indexes_i, interchain_indexes_j]
            
            if interchain_pae_vals_raw is not None:
                if self.feature_centering:
                    pae_col_mean = complex_group["pae_col_mean"][()]
                    pae_col_mean = pae_col_mean[valid_pair_mask]
                    if distance_mask is not None:
                        pae_col_mean = pae_col_mean[distance_mask]
                    
                    interchain_pae_vals_centered = interchain_pae_vals_raw - pae_col_mean
                    feature_list.append(pae_col_mean)
                    feature_list.append(interchain_pae_vals_centered)
                else:
                    feature_list.append(interchain_pae_vals_raw)
            
            if interchain_ca_distances_raw is not None:
                feature_list.append(interchain_ca_distances_raw)
                
            feats = np.array(feature_list, dtype=np.float32)

            # Add ESM embeddings
            if esm_i is not None and esm_j is not None:
                # Transpose: (N, D) -> (D, N)
                esm_i_T = esm_i.T.astype(np.float32)
                esm_j_T = esm_j.T.astype(np.float32)
                feats = np.vstack([feats, esm_i_T, esm_j_T])
        
            # Transform
            if self.feature_transform:
                feats = self.feature_transform(feats)
            
            # Label
            raw_label = row.label
            epsilon = 1e-6
            # Single scalar ops are fast
            label_val = np.clip(raw_label, epsilon, 1 - epsilon)
            label_val = np.log(label_val / (1 - label_val))
            
            # Pre-compute weight tensor
            weight_val = row.weight if hasattr(row, "weight") and pd.notna(row.weight) else 1.0

            return {
                "features": torch.from_numpy(feats),
                "label":    torch.tensor(label_val, dtype=torch.float32),
                "weight":   torch.tensor(weight_val, dtype=torch.float32),
                "length":   torch.tensor(feats.shape[1], dtype=torch.int64),
                "complex_id": complex_id,
                "ranking_score": torch.tensor(ranking_score)
            }
        except Exception as e:
            print(f"Error loading sample {complex_id}/{sample_id}: {e}")
            raise


# ==============================================================================
#  Factory: get_eval_dataloader
# ==============================================================================
def get_eval_dataloader(manifest_csv: str,
                        split: str,
                        batch_size: int = 32,
                        num_workers: int = 4,
                        samples_per_complex: int = None,
                        feature_transform: bool = False,
                        feature_centering: bool = False,
                        use_interchain_ca_distances: bool = False,
                        use_interchain_pae: bool = True,
                        use_esm_embeddings: bool = False,
                        use_distance_cutoff: bool = False,
                        distance_cutoff: float = 12.0,
                        use_file_cache: bool = True,
                        cache_size_mb: int = 512,
                        max_cached_files: int = 100,
                        seed: int = None,
                        distributed: bool = False,
                        world_size: int = 1,
                        rank: int = 0):
    """
    DataLoader for evaluation: sequential (no shuffle), no special sampling.
    For distributed training, wraps with DistributedBatchSampler.
    """
    # Note: feature_transform arg is a boolean flag in factory, but passed as function to Dataset
    transform_fn = triangular_encode_features if feature_transform else None
    
    if feature_transform:
        print("Val dataloader: Using triangular positional encoding")
    
    dataset = AntibodyAntigenPAEDataset(
        manifest_csv, 
        split=split, 
        feature_transform=transform_fn, 
        feature_centering=feature_centering, 
        use_interchain_pae=use_interchain_pae, 
        use_interchain_ca_distances=use_interchain_ca_distances, 
        use_esm_embeddings=use_esm_embeddings, 
        use_distance_cutoff=use_distance_cutoff, 
        distance_cutoff=distance_cutoff,
        use_file_cache=use_file_cache, # Ignored but kept for interface
        cache_size_mb=cache_size_mb,
        max_cached_files=max_cached_files
    )
    
    local_batch_size = math.ceil(batch_size / world_size)

    # Calculate local batch size for distributed training
    collate = create_distributed_collate_fn(local_batch_size, samples_per_complex)

    # --- Deterministic, sequential sampler for validation ---
    base_sampler = PerComplexSequentialSampler(
        dataset,
        batch_size=batch_size,
        samples_per_complex=samples_per_complex
    )
        
    # Wrap with distributed sampler if needed (validation typically doesn't drop_last)
    if distributed:
        sampler = DistributedBatchSampler(
            base_sampler,
            world_size=world_size,
            rank=rank,
            local_batch_size=local_batch_size,
            samples_per_complex=samples_per_complex,
            drop_last=False,  # Keep all validation data
            pad=True  # Ensure same number of steps across ranks
        )
    else:
        sampler = base_sampler
        
    return DataLoader(dataset,
                      batch_sampler=sampler,
                      num_workers=0, # Force 0 workers for in-memory dataset to avoid pickling overhead
                      pin_memory=True,
                      persistent_workers=False,
                      collate_fn=collate)


# ==============================================================================
#  Factory: get_dataloader
# ==============================================================================
def get_dataloader(manifest_csv: str,
                   split: str,
                   batch_size: int = 32,
                   num_workers: int = 4,
                   samples_per_complex: int = None,
                   bucket_balance: bool = False,
                   feature_transform: bool = False,
                   feature_centering: bool = False,
                   use_interchain_ca_distances: bool = False,
                   use_interchain_pae: bool = True,
                   use_esm_embeddings: bool = False,
                   use_distance_cutoff: bool = False,
                   distance_cutoff: float = 12.0,
                   use_file_cache: bool = True,
                   cache_size_mb: int = 512,
                   max_cached_files: int = 100,
                   seed: int = None,
                   distributed: bool = False,
                   world_size: int = 1,
                   rank: int = 0):
    """
    Factory for creating the training dataloader with the in-memory dataset.
    """
    # Note: feature_transform arg is a boolean flag in factory, but passed as function to Dataset
    transform_fn = triangular_encode_features if feature_transform else None

    if feature_transform:
        print("Train dataloader: Using triangular positional encoding")

    dataset = AntibodyAntigenPAEDataset(
        manifest_csv, 
        split=split, 
        feature_transform=transform_fn, 
        feature_centering=feature_centering, 
        use_interchain_pae=use_interchain_pae, 
        use_interchain_ca_distances=use_interchain_ca_distances, 
        use_esm_embeddings=use_esm_embeddings, 
        use_distance_cutoff=use_distance_cutoff, 
        distance_cutoff=distance_cutoff,
        use_file_cache=use_file_cache, # Ignored but kept for interface
        cache_size_mb=cache_size_mb,
        max_cached_files=max_cached_files
    )

    # Calculate local batch size for distributed training
    local_batch_size = math.ceil(batch_size / world_size)

    collate = create_distributed_collate_fn(local_batch_size, samples_per_complex)

    # Case 1: both constraints
    if samples_per_complex is not None and bucket_balance:
        print(f"Using WeightedPerComplexSampler with samples_per_complex={samples_per_complex}, batch_size={batch_size}")
        base_sampler = WeightedComplexSampler(
            dataset,
            samples_per_complex=samples_per_complex,
            batch_size=batch_size,
            weighted=True,
            replacement=True,
            seed=seed
        )
        
        # Wrap with distributed sampler if needed
        if distributed:
            sampler = DistributedBatchSampler(
                base_sampler,
                world_size=world_size,
                rank=rank,
                local_batch_size=local_batch_size,
                samples_per_complex=samples_per_complex,
                drop_last=True,
                pad=True
            )
        else:
            sampler = base_sampler
            
        return DataLoader(dataset,
                          batch_sampler=sampler,
                          num_workers=0, # Force 0 workers for in-memory dataset
                          pin_memory=True,
                          persistent_workers=False,
                          collate_fn=collate)

    # Case 2: uniform per complex
    if samples_per_complex is not None and bucket_balance is False:
        print(f"Using PerComplexSampler with samples_per_complex={samples_per_complex}, batch_size={batch_size}")
        base_sampler = WeightedComplexSampler(
            dataset,
            samples_per_complex=samples_per_complex,
            batch_size=batch_size,
            weighted=False,
            replacement=True,
            seed=seed
        )
        
        # Wrap with distributed sampler if needed
        if distributed:
            sampler = DistributedBatchSampler(
                base_sampler,
                world_size=world_size,
                rank=rank,
                local_batch_size=local_batch_size,
                samples_per_complex=samples_per_complex,
                drop_last=True,
                pad=True
            )
        else:
            sampler = base_sampler
            
        return DataLoader(dataset,
                          batch_sampler=sampler,
                          num_workers=0, # Force 0 workers for in-memory dataset
                          pin_memory=True,
                          persistent_workers=False,
                          collate_fn=collate)

    if samples_per_complex is None:
        print(f"Using WeightedRandomSampler with block length {batch_size}")
        base_sampler = WeightedComplexSampler(
            dataset,
            samples_per_complex=1,  # Default to 1 if not specified
            batch_size=batch_size,
            weighted=False,
            replacement=True,
            seed=seed
        )
        
        # Wrap with distributed sampler if needed
        if distributed:
            sampler = DistributedBatchSampler(
                base_sampler,
                world_size=world_size,
                rank=rank,
                local_batch_size=local_batch_size,
                samples_per_complex=1,
                drop_last=True,
                pad=True
            )
        else:
            sampler = base_sampler
            
        return DataLoader(dataset,
                          batch_sampler=sampler,
                          num_workers=0, # Force 0 workers for in-memory dataset
                          pin_memory=True,
                          persistent_workers=False,
                          collate_fn=collate)

