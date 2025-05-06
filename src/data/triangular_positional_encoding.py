import numpy as np

def cantor_pairing(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """
    Cantor pairing function: maps two non-negative integers (i, j)
    to a single non-negative integer uniquely.
    k = (i + j) * (i + j + 1) // 2 + j
    """
    # ensure integer arithmetic
    ij = i + j
    k = (ij * (ij + 1)) // 2 + j
    return k


def triangular_encode_features(feats: np.ndarray) -> np.ndarray:
    """
    Given a raw feature matrix feats of shape [3, n] containing:
      feats[0] = PAE values
      feats[1] = normalized i indices in [0,1]
      feats[2] = normalized j indices in [0,1]
    and the original sequence length `length`,
    returns a transformed feature matrix of the same shape [3, n] where:
      - channel 0 is still PAE
      - channel 1 is the Cantor-pairing code k, normalized to [0,1]
      - channel 2 is the relative distance |i - j| (already in [0,1])

    This gives the model a single compact code for (i,j) plus their absolute separation.
    """
    # unpack
    pae = feats[0]           # [n]
    i = feats[1]        # [n]
    j = feats[2]        # [n]

    length = len(pae)

    # Getting the max of the two indexes
    max = np.max(np.array([i, j]))
    # print(f"the max of the two indexes: {max}")

    # Cantor pairing
    k = cantor_pairing(i, j).astype(np.float32)
    # print(f"the cantor pairing: {k}")
    # maximum possible k occurs when i=j=length-1
    max_k = ((2*(max-1)) * (2*(max-1) + 1)) / 2 + (max-1)
    # print(f"the maximum possible k: {max_k}")
    k_norm = k / max_k
    # print(f"the normalized cantor pairing: {k_norm}")

    # Normalizing the distance
    dist = np.abs(i - j)
    # print(f"the distance: {dist}")
    dist_norm = dist / dist.max()
    # print(f"the normalized distance: {dist_norm}")
    # check that the distance is normalized
    # assert dist_norm.max() <= 1
    # assert dist_norm.min() >= 0
    # assert k_norm.max() <= 1
    # assert k_norm.min() >= 0

    # re-stack into same shape [3, n]
    return np.stack([pae, k_norm, dist_norm], axis=0).astype(np.float32)
