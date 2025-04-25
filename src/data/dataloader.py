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

# ==============================================================================
# Load our user‐config (e.g. number of models to draw per complex per epoch)
# ==============================================================================
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


# ==============================================================================
#  Collate function
# ==============================================================================

def pad_collate_fn(batch):
    """
    Collate function that pads 3×n_i feature tensors across the batch
    to shape (batch_size, 3, max_n), stacking labels and weights.
    """
    # Extract lists
    features = [item["features"] for item in batch]
    labels   = torch.stack([item["label"]  for item in batch])
    weights  = torch.stack([item["weight"] for item in batch])

    # Find max sequence length
    max_n = max(feat.size(1) for feat in features)
    padded_feats = []
    for feat in features:
        pad = max_n - feat.size(1)
        # pad on right of dim=1 (the varying n‐axis), no pad on dim=0
        padded = F.pad(feat, (0, pad, 0, 0), value=0)
        padded_feats.append(padded)

    # Stack into a single tensor of shape (batch_size, 3, max_n)
    features = torch.stack(padded_feats, dim=0)
    # swap to (batch_size, max_n, 3) so that set_size=max_n and input_dim=3
    features = features.permute(0, 2, 1)

    return {"features": features, "label": labels, "weight": weights}


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
                 complexes_per_batch: int,
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
        self.M  = samples_per_complex
        self.complexes_per_batch = complexes_per_batch
        self.weighted = weighted
        self.replacement = replacement
        self.rng = np.random.RandomState(seed)

        #order the df by the len_sample
        self.df = self.df.sort_values(by="len_sample", ascending=False)

        # Build mapping: complex_id -> list of (index, weight, len_sample)
        self.by_complex = {}
        for idx, (cid, w, len_sample) in enumerate(zip(self.df["complex_id"],
                                           self.df["weight"],
                                           self.df["len_sample"])):
            self.by_complex.setdefault(cid, []).append((idx, w, len_sample))

        self.sorted_complex_ids = list(self.by_complex.keys())
        #print the complex_id and the length of samples of all sorted complexes
        # for cid in self.complex_ids:
        #     print(f"{cid}: {self.by_complex[cid][0][2]}")

    def __iter__(self):
        
        all_idxs = []

        # create complexes blocks of batch_size given the len_sample
        blocks = [
            self.sorted_complex_ids[i : i + self.complexes_per_batch]
            for i in range(0, len(self.sorted_complex_ids), self.complexes_per_batch)
        ]
        
        
        for block in blocks:

            samples_per_complexes_in_block = []
            for cid in block:
                idx_w_triples = self.by_complex[cid]
                
                idxs, weights, len_sample = zip(*idx_w_triples)
                if self.weighted:
                    probs = np.array(weights, dtype=np.float64)
                    probs /= probs.sum()  # normalize
                else:
                    probs = np.ones(len(idxs)) / len(idxs)

                # draw M times
                chosen = self.rng.choice(
                    a=idxs,
                    size=self.M,
                    replace=self.replacement,
                    p=probs
                )
                samples_per_complexes_in_block.extend(chosen.tolist())
            
            #print the length of the chosen
            # print(len(samples_per_complexes_in_block) == self.M * len(block))

            # Now we have a list of `len(block)` lists, each of length M.
            # We'll build M batches of size len(block).
            for i in range(self.M):
                #first we shuffle the samples_per_complexes_in_block
                self.rng.shuffle(samples_per_complexes_in_block)
                #then we build the batch
                batch = []
                for j in range(len(block)):
                    batch.append(samples_per_complexes_in_block[i*len(block) + j])
                # print(batch)
                yield batch
            

    def __len__(self):
        # total draws per epoch
        return len(self.sorted_complex_ids) * self.M


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
                 feature_transform=None):
        self.df = pd.read_csv(manifest_csv)
        # Filter to only this split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.feature_transform = feature_transform

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

        # Create a 3xn with the interchain indexes and the interchain pae values
        feats = np.array([
            interchain_pae_vals,
            interchain_indexes_i,
            interchain_indexes_j
        ], dtype=np.float32)

        # Optional user‐provided transform
        if self.feature_transform:
            feats = self.feature_transform(feats)

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
                        num_workers: int = 4):
    """
    DataLoader for evaluation: sequential (no shuffle), no special sampling.
    """
    dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split)
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
                   seed: int = None):
    """
    Priority of sampling strategies:
      1) If both samples_per_complex and bucket_balance → use WeightedPerComplexSampler
      2) If samples_per_complex only     → use PerComplexSampler (uniform within complex)
      3) If bucket_balance only          → use WeightedRandomSampler (global)
      4) Else                            → plain shuffle
    """
    dataset = AntibodyAntigenPAEDataset(manifest_csv, split=split)
    df      = dataset.df

    # Case 1: both constraints
    if samples_per_complex is not None and bucket_balance:
        sampler = WeightedPerComplexSampler(
            dataset,
            samples_per_complex=samples_per_complex,
            complexes_per_batch=batch_size,
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
    if samples_per_complex is not None:
        sampler = WeightedPerComplexSampler(
            dataset,
            samples_per_complex=samples_per_complex,
            complexes_per_batch=batch_size,
            weighted=False,
            replacement=True,
            seed=seed
        )
        return DataLoader(dataset,
                          batch_sampler=sampler,
                          sampler=sampler,
                          num_workers=num_workers,
                          pin_memory=True,
                          collate_fn=pad_collate_fn)

    # Case 3: bucket‐balance globally
    if bucket_balance:
        weights = df["weight"].values
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=sampler,
                          num_workers=num_workers,
                          pin_memory=True,
                          collate_fn=pad_collate_fn)

    # Case 4: plain shuffle
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=True,
                      collate_fn=pad_collate_fn)

