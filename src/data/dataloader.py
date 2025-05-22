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
    if not np.all(complex_ids[:, 0] == complex_ids[:, 1]):
        raise ValueError("All complex_ids in the last dimension must be the same")

    return {
      "features": padded,        # [B, K, N, F]
      "lengths":  lengths,       # [B, K]
      "label":    labels,        # [B, K]
      "weight":   weights,       # [B, K]
      "complex_id": complex_ids  # [B, K]
    }

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
                 feature_centering=False):
        self.df = pd.read_csv(manifest_csv)
        # Filter to only this split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.feature_transform = feature_transform
        self.feature_centering = feature_centering

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        h5_path    = row["h5_file"]
        complex_id = row["complex_id"] # Original complex_id for this item
        sample_id  = row["sample"]   # Original sample_id for this item
        
        # Load raw PAE data and create a 3xn with the interchain indexes and the interchain pae values
        with h5py.File(h5_path, 'r') as hf:
            # Data for the primary sample
            original_sample_group = hf[complex_id][sample_id]
            interchain_pae_vals_raw = original_sample_group["interchain_pae_vals"][()]
            interchain_indexes_i = original_sample_group["inter_idx"][()]
            interchain_indexes_j = original_sample_group["inter_jdx"][()]

            label = row["label"] # Scalar label for the single sample case
            if not self.feature_centering:
                # Create a 3xn with the interchain indexes and the interchain pae values
                feats = np.array([
                    interchain_indexes_i,
                    interchain_indexes_j,
                    interchain_pae_vals_raw
                ], dtype=np.float32)
            else:
                pae_col_mean = hf[complex_id]["pae_col_mean"][()]
                # pae_col_std = hf[complex_id]["pae_col_std"][()] # Not currently used for centering
                interchain_pae_vals_centered = interchain_pae_vals_raw - pae_col_mean
                feats = np.array([
                    interchain_indexes_i,
                    interchain_indexes_j,
                    pae_col_mean,
                    interchain_pae_vals_centered,
                ], dtype=np.float32)
            
            
        # Optional user‐provided transform (applies to feats)
        if self.feature_transform:
            feats = self.feature_transform(feats)
        
        # Transform label(s) using clipped logit - works element-wise if label is an array
        # print(f"label before: {label}")
        epsilon = 1e-8
        label = np.clip(label, epsilon, 1 - epsilon)
        label = np.log(label / (1 - label))
        

        return {
            "features": torch.from_numpy(feats),            # [3 or 4, n]
            "label":    torch.tensor(label, dtype=torch.float32), # DockQ (logit transformed)
            "weight":   torch.tensor(row["weight"]),         # importance weight
            "length":   torch.tensor(len(feats[0])),         # length of the sample
            "complex_id": complex_id
        }


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
                        seed: int = None):
    """
    DataLoader for evaluation: sequential (no shuffle), no special sampling.
    """
    if feature_transform:
        print("Val dataloader: Using triangular positional encoding")
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_transform=triangular_encode_features, feature_centering=feature_centering)
    else:
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_centering=feature_centering)
    
    collate = partial(pad_collate_fn, batch_size=batch_size, samples_per_complex=samples_per_complex)

    sampler = WeightedComplexSampler(
            dataset,
            samples_per_complex=samples_per_complex,
            batch_size=batch_size,
            weighted=False,
            replacement=True,
            seed=seed
        )
    return DataLoader(dataset,
                      batch_sampler=sampler,
                      num_workers=num_workers,
                      pin_memory=True,
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
                   seed: int = None):
    """
    Priority of sampling strategies:
      1) If both samples_per_complex and bucket_balance → use WeightedPerComplexSampler
      2) If samples_per_complex only     → use PerComplexSampler (uniform within complex)
      3) If bucket_balance only          → use WeightedRandomSampler (global)
      4) Else                            → plain shuffle
    """
    if feature_transform:
        print("Train dataloader: Using triangular positional encoding")
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_transform=triangular_encode_features, feature_centering=feature_centering)
    else:
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_centering=feature_centering)

    collate = partial(pad_collate_fn, batch_size=batch_size, samples_per_complex=samples_per_complex)

    # Case 1: both constraints
    if samples_per_complex is not None and bucket_balance:
        print(f"Using WeightedPerComplexSampler with samples_per_complex={samples_per_complex}, batch_size={batch_size}")
        sampler = WeightedComplexSampler(
            dataset,
            samples_per_complex=samples_per_complex,
            batch_size=batch_size,
            weighted=True,
            replacement=True,
            seed=seed
        )
        return DataLoader(dataset,
                          batch_sampler=sampler,
                          num_workers=num_workers,
                          pin_memory=True,
                          collate_fn=collate)

    # Case 2: uniform per complex
    if samples_per_complex is not None and bucket_balance is False:
        print(f"Using PerComplexSampler with samples_per_complex={samples_per_complex}, batch_size={batch_size}")
        sampler = WeightedComplexSampler(
            dataset,
            samples_per_complex=samples_per_complex,
            batch_size=batch_size,
            weighted=False,
            replacement=True,
            seed=seed
        )
        return DataLoader(dataset,
                          batch_sampler=sampler,
                          num_workers=num_workers,
                          pin_memory=True,
                          collate_fn=collate)

    if samples_per_complex is None:
        print(f"Using WeightedRandomSampler with block length {batch_size}")
        sampler = WeightedPerComplexSampler(
            dataset,
            samples_per_complex=samples_per_complex,
            batch_size=batch_size,
            weighted=False,
            replacement=True,
            seed=seed
        )
        return DataLoader(dataset,
                          batch_sampler=sampler,
                          num_workers=num_workers,
                          pin_memory=True,
                          collate_fn=collate)


