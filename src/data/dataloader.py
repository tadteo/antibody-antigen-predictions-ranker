#!/usr/bin/env python3
import os
import yaml
import random
import pandas as pd
import numpy as np
import h5py
import torch
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

def pad_collate_fn(batch):
    features = [item["features"] for item in batch]  # each is [3, n_i]
    labels   = torch.stack([item["label"]  for item in batch])
    weights  = torch.stack([item["weight"] for item in batch])

    lengths = [feat.size(1) for feat in features]    # record each n_i
    max_n   = max(lengths)

    padded_feats = []
    for feat, L in zip(features, lengths):
        pad = max_n - L
        # pad on the *right* of dim=1 (the varying-n axis)
        padded = F.pad(feat, (0, pad, 0, 0), value=0.0)
        padded_feats.append(padded)

    # stack into (batch_size, 3, max_n) then permute to (B, max_n, 3)
    features = torch.stack(padded_feats, dim=0).permute(0, 2, 1)

    return {
      "features": features,        # [B, set_size, input_dim]
      "lengths":  torch.tensor(lengths, dtype=torch.long),
      "label":    labels,
      "weight":   weights
    }

# ==============================================================================
# WeightedPerComplexSampler
# ==============================================================================
class WeightedPerComplexSampler(Sampler):
    """
    For each complex, draws exactly `samples_per_complex` indices per epoch,
    sampling *within* that complex with probability proportional to `weight`.
    Finally shuffles across complexes.
    Per epoch, the sampler will draw `samples_per_complex` * `number_of_complexes` samples.
    """
    def __init__(self,
                 dataset,
                 samples_per_complex: int,
                 batch_size: int,
                 weighted: bool = True,
                 replacement: bool = True,
                 seed: int = None
                 ):
        """
        dataset: an instance of AntibodyAntigenPAEDataset
        samples_per_complex: how many samples to draw per complex each epoch
        batch_size: how many complexes to draw per batch used for bucket balancing
        weighted: whether to weight the samples within a complex by their weight
        replacement: whether to draw with replacement when a complex has fewer
                     samples than samples_per_complex
        seed: seed for the RNG
        """
        self.df = dataset.df  # the manifest filtered to one split
        self.samples_per_complex  = samples_per_complex
        self.batch_size = batch_size
        self.weighted = weighted
        self.replacement = replacement
        self.rng = np.random.RandomState(seed)

        # number of complexes in the df
        self.number_of_complexes = len(self.df["complex_id"].unique())

    def __iter__(self):
        # Get the indices to sample from
        indices = np.arange(len(self.df))
        # Sample from indices using the weights

        if self.samples_per_complex is None:
            #act as weighted random sampler
            if self.weighted:
                #normalize weights so they sum exactly to 1
                self.df["weight"] = self.df["weight"] / self.df["weight"].sum()
                
                sampled_indices = self.rng.choice(indices, size=len(indices), replace=False, p=self.df["weight"].values)
            else:
                sampled_indices = self.rng.choice(indices, size=len(indices), replace=False)
        else:
            if self.weighted:
                sampled_indices = self.rng.choice(indices, size=(self.samples_per_complex*self.batch_size), replace=False, p=self.df["weight"].values)
            else:
                sampled_indices = self.rng.choice(indices, size=(self.samples_per_complex*self.batch_size), replace=False)

        # Get the samples and their lengths
        samples = self.df.iloc[sampled_indices]
        samples = samples.sort_values(by="len_sample", ascending=False)
        
        # Create blocks of indices (not DataFrame slices)
        blocks = [
            sampled_indices[i : i + self.batch_size]
            for i in range(0, len(sampled_indices), self.batch_size)
        ]
        #shuffle the blocks
        self.rng.shuffle(blocks)
        # Yield integer indices, not DataFrame slices
        for block in blocks:
            yield block.tolist()  # Convert numpy array to list of integers

    def __len__(self):
        if self.samples_per_complex is None:
            # If no samples_per_complex specified, return total number of samples
            return len(self.df)
        else:
            # If samples_per_complex specified, return total number of samples per epoch
            return self.number_of_complexes * self.samples_per_complex


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
            "features": Tensor([3, n]),
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
        complex_id = row["complex_id"]
        sample_id  = row["sample"]

        # Load raw PAE data and create a 3xn with the interchain indexes and the interchain pae values
        with h5py.File(h5_path, 'r') as hf:
            grp        = hf[complex_id][sample_id]
            interchain_pae_vals = grp["interchain_pae_vals"][()]
            interchain_indexes_i = grp["inter_idx"][()]
            interchain_indexes_j = grp["inter_jdx"][()]

            if self.feature_centering:
                pae_col_mean = hf[complex_id]["pae_col_mean"][()]
                pae_col_std = hf[complex_id]["pae_col_std"][()]

                interchain_pae_vals = (interchain_pae_vals - pae_col_mean) / (pae_col_std + 1e-6)


        # Create a 3xn with the interchain indexes and the interchain pae values
        feats = np.array([
            interchain_pae_vals,
            interchain_indexes_i,
            interchain_indexes_j
        ], dtype=np.float32)

        # Optional user‐provided transform
        if self.feature_transform:
            # print("Applying feature transform")
            feats = self.feature_transform(feats)


        #print first 10 features or indexes
        # print(feats[:10])
        # print(feats[1][:10])
        # print(feats[2][:10])

        return {
            "features": torch.from_numpy(feats),            # [3, n]
            "label":    torch.tensor(row["label"]),         # DockQ
            "weight":   torch.tensor(row["weight"])         # importance weight
        }


# ==============================================================================
#  Factory: get_eval_dataloader
# ==============================================================================
def get_eval_dataloader(manifest_csv: str,
                        split: str,
                        batch_size: int = 32,
                        num_workers: int = 4,
                        feature_transform: bool = False,
                        feature_centering: bool = False,
                        seed: int = None):
    """
    DataLoader for evaluation: sequential (no shuffle), no special sampling.
    """
    if feature_transform:
        print("Using triangular positional encoding")
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_transform=triangular_encode_features, feature_centering=feature_centering)
    else:
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_centering=feature_centering)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers,
                      pin_memory=True,
                      collate_fn=pad_collate_fn)


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
        print("Using triangular positional encoding")
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_transform=triangular_encode_features, feature_centering=feature_centering)
    else:
        dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split, feature_centering=feature_centering)

    
    # Case 1: both constraints
    if samples_per_complex is not None and bucket_balance:
        print(f"Using WeightedPerComplexSampler with samples_per_complex={samples_per_complex}, batch_size={batch_size}")
        sampler = WeightedPerComplexSampler(
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
                          collate_fn=pad_collate_fn)

    # Case 2: uniform per complex
    if samples_per_complex is not None and bucket_balance is False:
        print(f"Using PerComplexSampler with samples_per_complex={samples_per_complex}, batch_size={batch_size}")
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
                          collate_fn=pad_collate_fn)

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
                          collate_fn=pad_collate_fn)


