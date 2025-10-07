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
from threading import Lock
import gc
import atexit
import weakref

# ==============================================================================
# HDF5 File Handle Cache (per-worker)
# ==============================================================================
class H5FileCache:
    """
    Thread-safe cache for HDF5 file handles. Keeps files open for faster access.
    Each DataLoader worker process will have its own instance.
    """
    def __init__(self, chunk_cache_size_mb=512, max_open_files=10):
        self.cache = {}
        self.lock = Lock()
        self.chunk_cache_size = chunk_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.max_open_files = max_open_files
        self.access_count = {}  # Track access frequency for LRU eviction
        self._closed = False
        
        # Register cleanup on exit
        atexit.register(self.close_all)
        
    def get_file(self, h5_path):
        """Get an open HDF5 file handle, opening if necessary"""
        if self._closed:
            raise RuntimeError("Attempting to use closed H5FileCache")
            
        with self.lock:
            if h5_path not in self.cache:
                # If cache is full, remove least recently used file
                if len(self.cache) >= self.max_open_files:
                    lru_path = min(self.access_count, key=self.access_count.get)
                    try:
                        self.cache[lru_path].close()
                    except Exception as e:
                        print(f"Warning: Error closing HDF5 file {lru_path}: {e}")
                    del self.cache[lru_path]
                    del self.access_count[lru_path]
                
                # Open file with optimized settings
                try:
                    self.cache[h5_path] = h5py.File(
                        h5_path, 'r',
                        rdcc_nbytes=self.chunk_cache_size,  # Chunk cache size
                        rdcc_nslots=10007,  # Prime number for hash table
                        rdcc_w0=0.75  # Preemption policy (0.75 = balanced)
                    )
                    self.access_count[h5_path] = 0
                except Exception as e:
                    print(f"Error opening HDF5 file {h5_path}: {e}")
                    raise
            
            self.access_count[h5_path] += 1
            return self.cache[h5_path]
    
    def close_all(self):
        """Close all cached file handles"""
        if self._closed:
            return
            
        with self.lock:
            for path, f in list(self.cache.items()):
                try:
                    if f is not None and hasattr(f, 'close'):
                        f.close()
                except Exception as e:
                    print(f"Warning: Error closing HDF5 file {path}: {e}")
            self.cache.clear()
            self.access_count.clear()
            self._closed = True
    
    def __del__(self):
        try:
            self.close_all()
        except:
            pass

# Global cache - will be separate per worker process due to fork/spawn
_worker_file_cache = None

def get_worker_file_cache(chunk_cache_size_mb=512, max_open_files=20):
    """Get or create the file cache for this worker"""
    global _worker_file_cache
    if _worker_file_cache is None:
        _worker_file_cache = H5FileCache(chunk_cache_size_mb, max_open_files)
    return _worker_file_cache

def cleanup_worker():
    """Cleanup function to be called when DataLoader worker shuts down"""
    global _worker_file_cache
    if _worker_file_cache is not None:
        try:
            _worker_file_cache.close_all()
        except Exception as e:
            print(f"Warning: Error during worker cleanup: {e}")
        finally:
            _worker_file_cache = None
    
    # Force garbage collection to clean up any remaining references
    gc.collect()

def worker_init_fn(worker_id):
    """Initialize worker and register cleanup on exit"""
    # Register cleanup function to be called when worker exits
    atexit.register(cleanup_worker)

# ==============================================================================
# Helper functions for distributed training
# ==============================================================================
def create_distributed_collate_fn(local_batch_size: int, samples_per_complex: int):
    """Create collate function with correct local batch size for distributed training"""
    return partial(pad_collate_fn, batch_size=local_batch_size, samples_per_complex=samples_per_complex)

# ==============================================================================
# Load our user‐config (e.g. number of models to draw per complex per epoch)
# ==============================================================================
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


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

    # print(f"labels shape: {labels.shape}")
    # print(f"weights shape: {weights.shape}")
    # print(f"lengths shape: {lengths.shape}")
    # print(f"complex_ids shape: {complex_ids.shape}")

    # Check that all complex_ids in the last dimension are the same
    # if not np.all(complex_ids[:, 0] == complex_ids[:, 1]):
    #     raise ValueError("All complex_ids in the last dimension must be the same")
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

        ── notation used below ───────────────────────────────────────────────
        B : self.batch_size
        K : self.samples_per_complex
        rng : self.rng   (numpy RandomState)
        by_cid : {cid: [row-indices]}
        row_w  : 1-D numpy array with a weight per *row*  (only if self.wtd)
        ──────────────────────────────────────────────────────────────────────
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
#  Dataset: load features on‐the‐fly from your H5 files
# ==============================================================================
class AntibodyAntigenPAEDataset(Dataset):
    """
    Reads the manifest to know which H5-file + group corresponds to each sample.
    On __getitem__:
      - Opens the H5 (context‐managed)
      - Pulls out the PAE interface indexes and values
      - Returns a dict:
          {
            "features": Tensor([3 or 4, n]),
            "label":    Tensor(scalar DockQ),
            "weight":   Tensor(scalar importance weight)
          }
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
                 use_file_cache: bool = True,
                 cache_size_mb: int = 512,
                 max_cached_files: int = 20):
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
        self.use_file_cache = use_file_cache
        self.cache_size_mb = cache_size_mb
        self.max_cached_files = max_cached_files

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        h5_path    = row["h5_file"]
        complex_id = row["complex_id"] # Original complex_id for this item
        sample_id  = row["sample"]   # Original sample_id for this item
        
        # Load raw PAE data and create a 3xn with the interchain indexes and the interchain pae values
        # Use cached file handle if enabled, otherwise use context manager for proper cleanup
        if self.use_file_cache:
            file_cache = get_worker_file_cache(self.cache_size_mb, self.max_cached_files)
            hf = file_cache.get_file(h5_path)
            # Use cached file - don't close it
            return self._load_sample_data(hf, row, complex_id, sample_id)
        else:
            # Use context manager for automatic cleanup
            with h5py.File(h5_path, 'r') as hf:
                return self._load_sample_data(hf, row, complex_id, sample_id)
    
    def _load_sample_data(self, hf, row, complex_id, sample_id):
        """Helper method to load sample data from an open HDF5 file handle"""
        try:
            # Data for the primary sample
            original_sample_group = hf[complex_id][sample_id]
            interchain_pae_vals_raw = original_sample_group["interchain_pae_vals"][()]
            interchain_indexes_i = original_sample_group["inter_idx"][()]
            interchain_indexes_j = original_sample_group["inter_jdx"][()]
            ranking_score = original_sample_group["ranking_score"][()]
            # Load residue one-letter codes to filter out X residues
            residue_one_letter = original_sample_group["residue_one_letter"][()]
            # Convert bytes to strings if needed (HDF5 sometimes stores as bytes)
            if isinstance(residue_one_letter[0], bytes):
                residue_one_letter = [res.decode('utf-8') for res in residue_one_letter]

            # Create mask for non-X residues
            non_x_mask = np.array([res != 'X' for res in residue_one_letter])
            
            # Filter interchain pairs: keep only pairs where both residues are not X
            valid_pair_mask = non_x_mask[interchain_indexes_i] & non_x_mask[interchain_indexes_j]

            interchain_pae_vals_raw = interchain_pae_vals_raw[valid_pair_mask]
            interchain_indexes_i = interchain_indexes_i[valid_pair_mask]
            interchain_indexes_j = interchain_indexes_j[valid_pair_mask]

            if not self.use_interchain_pae:
                interchain_pae_vals_raw = None

            # Optional: interchain Cα distances aligned by same indices
            if self.use_interchain_ca_distances and 'interchain_ca_distances' in original_sample_group:
                interchain_ca_distances_raw = original_sample_group['interchain_ca_distances'][()]
                interchain_ca_distances_raw = interchain_ca_distances_raw[valid_pair_mask]
            else:
                interchain_ca_distances_raw = None

            # Remap indices to account for removed X residues
            non_x_indices = np.where(non_x_mask)[0]
            old_to_new_idx = np.full(len(residue_one_letter), -1, dtype=int)
            old_to_new_idx[non_x_indices] = np.arange(len(non_x_indices))
            
            # Remap the indices
            interchain_indexes_i = old_to_new_idx[interchain_indexes_i]
            interchain_indexes_j = old_to_new_idx[interchain_indexes_j]

            # Store distance mask before distance cutoff is applied
            distance_mask = None
            
            # Optional: Apply distance cutoff to filter out far-away residue pairs
            if self.use_distance_cutoff and interchain_ca_distances_raw is not None:
                # Count pairs before filtering
                n_pairs_before = len(interchain_ca_distances_raw)
                
                # Keep only pairs where CA distance is below the cutoff
                distance_mask = interchain_ca_distances_raw <= self.distance_cutoff
                n_pairs_after = distance_mask.sum()
                n_filtered = n_pairs_before - n_pairs_after
                pct_filtered = (n_filtered / n_pairs_before * 100) if n_pairs_before > 0 else 0.0
                
                # Debug output (first 10 samples only to avoid spam)
                # print(f"[Distance Cutoff Debug] Sample {complex_id}/{sample_id}:")
                # print(f"  Cutoff: {self.distance_cutoff} Å")
                # print(f"  Pairs before: {n_pairs_before}")
                # print(f"  Pairs after:  {n_pairs_after}")
                # print(f"  Filtered out: {n_filtered} ({pct_filtered:.1f}%)")
                # if n_pairs_after > 0:
                #     print(f"  Distance range (kept): [{interchain_ca_distances_raw[distance_mask].min():.2f}, {interchain_ca_distances_raw[distance_mask].max():.2f}] Å")
                
                # IMPORTANT: If all pairs are filtered out, keep the closest pair to avoid empty samples
                if n_pairs_after == 0:
                    # Find the closest pair and keep it
                    closest_idx = np.argmin(interchain_ca_distances_raw)
                    distance_mask = np.zeros(len(interchain_ca_distances_raw), dtype=bool)
                    distance_mask[closest_idx] = True
                    # print(f"WARNING: Sample {complex_id}/{sample_id} - all pairs beyond cutoff {self.distance_cutoff}Å. "
                    #       f"Keeping closest pair at {interchain_ca_distances_raw[closest_idx]:.2f}Å to avoid empty sample.")
                
                # Apply mask to all arrays
                interchain_indexes_i = interchain_indexes_i[distance_mask]
                interchain_indexes_j = interchain_indexes_j[distance_mask]
                if interchain_pae_vals_raw is not None:
                    interchain_pae_vals_raw = interchain_pae_vals_raw[distance_mask]
                interchain_ca_distances_raw = interchain_ca_distances_raw[distance_mask]

            # Optional: ESM embeddings
            # print(f"use_esm_embeddings: {self.use_esm_embeddings}")
            # print(f"esm_embeddings in original_sample_group: {'esm_embeddings' in original_sample_group}")
            if self.use_esm_embeddings and 'esm_embeddings' in original_sample_group:
                esm_embeddings_raw = original_sample_group['esm_embeddings'][()]  # [L, d_esm]
                # print(f"esm_embeddings_raw shape: {esm_embeddings_raw.shape}")
                # Filter to remove X residues
                esm_embeddings = esm_embeddings_raw[non_x_mask]  # [L', d_esm]
                # print(f"esm_embeddings shape: {esm_embeddings.shape}")
                # Extract embeddings for interchain pairs (after distance filtering)
                esm_i = esm_embeddings[interchain_indexes_i]  # [n, d_esm]
                esm_j = esm_embeddings[interchain_indexes_j]  # [n, d_esm]
            else:
                esm_i = None
                esm_j = None

            label = row["label"] # Scalar label for the single sample case
            if interchain_pae_vals_raw is not None:
                if not self.feature_centering:
                    # Base features: indices and interchain PAE
                    if interchain_ca_distances_raw is not None:
                        feats = np.array([
                            interchain_indexes_i,
                            interchain_indexes_j,
                            interchain_pae_vals_raw,
                            interchain_ca_distances_raw,
                        ], dtype=np.float32)
                    else:
                        feats = np.array([
                            interchain_indexes_i,
                            interchain_indexes_j,
                            interchain_pae_vals_raw
                        ], dtype=np.float32)
                else:
                    pae_col_mean = hf[complex_id]["pae_col_mean"][()]
                    pae_col_mean = pae_col_mean[valid_pair_mask]
                    # Apply distance mask if used
                    if distance_mask is not None:
                        pae_col_mean = pae_col_mean[distance_mask]

                    # pae_col_std = hf[complex_id]["pae_col_std"][()] # Not currently used for centering
                    interchain_pae_vals_centered = interchain_pae_vals_raw - pae_col_mean
                    # If interchain ca distances are available, add them to the features
                    if interchain_ca_distances_raw is not None:
                        feats = np.array([
                            interchain_indexes_i,
                            interchain_indexes_j,
                            pae_col_mean,
                            interchain_pae_vals_centered,
                            interchain_ca_distances_raw,
                        ], dtype=np.float32)
                    else:
                        feats = np.array([
                            interchain_indexes_i,
                            interchain_indexes_j,
                            pae_col_mean,
                            interchain_pae_vals_centered,
                        ], dtype=np.float32)
            else:
                feats = np.array([
                    interchain_indexes_i,
                    interchain_indexes_j,
                    interchain_ca_distances_raw,
                ], dtype=np.float32)
            

            # Add ESM embeddings if enabled
            if esm_i is not None and esm_j is not None:
                # Transpose ESM embeddings to match feature format [d_esm, n]
                esm_i_T = esm_i.T.astype(np.float32)  # [d_esm, n]
                esm_j_T = esm_j.T.astype(np.float32)  # [d_esm, n]
                # Concatenate along feature dimension
                feats = np.vstack([feats, esm_i_T, esm_j_T])  # [F + 2*d_esm, n]
                # print(f"feats shape after adding ESM embeddings: {feats.shape}")
        
            # Optional user‐provided transform (applies to feats)
            if self.feature_transform:
                feats = self.feature_transform(feats)
            
            # print(f"feats shape: {feats.shape}")

            # Transform label(s) using clipped logit - works element-wise if label is an array
            # print(f"label before: {label}")
            epsilon = 1e-6
            label = np.clip(label, epsilon, 1 - epsilon)
            label = np.log(label / (1 - label))

            return {
                "features": torch.from_numpy(feats),            # [3..5, n]
                "label":    torch.tensor(label, dtype=torch.float32), # DockQ (logit transformed)
                "weight":   torch.tensor(row["weight"]),         # importance weight
                "length":   torch.tensor(len(feats[0])),         # length of the sample
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
                        max_cached_files: int = 20,
                        seed: int = None,
                        distributed: bool = False,
                        world_size: int = 1,
                        rank: int = 0):
    """
    DataLoader for evaluation: sequential (no shuffle), no special sampling.
    For distributed training, wraps with DistributedBatchSampler.
    """
    if feature_transform:
        print("Val dataloader: Using triangular positional encoding")
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_transform=triangular_encode_features, feature_centering=feature_centering, use_interchain_pae=use_interchain_pae, use_interchain_ca_distances=use_interchain_ca_distances, use_esm_embeddings=use_esm_embeddings, use_distance_cutoff=use_distance_cutoff, distance_cutoff=distance_cutoff, use_file_cache=use_file_cache, cache_size_mb=cache_size_mb, max_cached_files=max_cached_files)
    else:
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_centering=feature_centering, use_interchain_pae=use_interchain_pae, use_interchain_ca_distances=use_interchain_ca_distances, use_esm_embeddings=use_esm_embeddings, use_distance_cutoff=use_distance_cutoff, distance_cutoff=distance_cutoff, use_file_cache=use_file_cache, cache_size_mb=cache_size_mb, max_cached_files=max_cached_files)
    
    local_batch_size = math.ceil(batch_size / world_size)

    # Calculate local batch size for distributed training
    collate = create_distributed_collate_fn(local_batch_size, samples_per_complex)

    # base_sampler = WeightedComplexSampler(
    #         dataset,
    #         samples_per_complex=samples_per_complex,
    #         batch_size=batch_size,
    #         weighted=False,
    #         replacement=True,
    #         seed=seed
    #     )

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
                      num_workers=num_workers,
                      pin_memory=True,
                      persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
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
                   max_cached_files: int = 20,
                   seed: int = None,
                   distributed: bool = False,
                   world_size: int = 1,
                   rank: int = 0):
    """
    Priority of sampling strategies:
      1) If both samples_per_complex and bucket_balance → use WeightedPerComplexSampler
      2) If samples_per_complex only     → use PerComplexSampler (uniform within complex)
      3) If bucket_balance only          → use WeightedRandomSampler (global)
      4) Else                            → plain shuffle
      
    For distributed training:
      - Wraps the base sampler with DistributedBatchSampler
      - Calculates local_batch_size per rank
      - Uses appropriate collate function
    """
    if feature_transform:
        print("Train dataloader: Using triangular positional encoding")
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_transform=triangular_encode_features, feature_centering=feature_centering, use_interchain_pae=use_interchain_pae, use_interchain_ca_distances=use_interchain_ca_distances, use_esm_embeddings=use_esm_embeddings, use_distance_cutoff=use_distance_cutoff, distance_cutoff=distance_cutoff, use_file_cache=use_file_cache, cache_size_mb=cache_size_mb, max_cached_files=max_cached_files)
    else:
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_centering=feature_centering, use_interchain_pae=use_interchain_pae, use_interchain_ca_distances=use_interchain_ca_distances, use_esm_embeddings=use_esm_embeddings, use_distance_cutoff=use_distance_cutoff, distance_cutoff=distance_cutoff, use_file_cache=use_file_cache, cache_size_mb=cache_size_mb, max_cached_files=max_cached_files)

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
                          num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
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
                          num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
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
                          num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
                          collate_fn=collate)


