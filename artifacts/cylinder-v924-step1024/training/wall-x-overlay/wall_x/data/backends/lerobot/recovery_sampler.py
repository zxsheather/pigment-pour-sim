"""Optional, seeded distributed frame sampling for recovery demonstrations.

Weights refer to the full LeRobot frame index. Validation frames are removed
before drawing; each rank receives a distinct slice of the same seeded draw.
This module is copied into the pinned Wall-X loader by the patch script.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DistributedSampler


class RecoveryDistributedSampler(DistributedSampler):
    def __init__(self, dataset, *, weights_path, **kwargs):
        super().__init__(dataset, **kwargs)
        weights = np.load(Path(weights_path), allow_pickle=False)
        if weights.shape != (len(dataset.hf_dataset),):
            raise ValueError("Recovery weights must match the full LeRobot dataset")
        selected = weights[np.asarray(dataset.train_dataset.indices)]
        if not np.isfinite(selected).all() or (selected < 0).any() or not (selected > 0).any():
            raise ValueError("Sampling weights must be finite and nonnegative with positive total mass")
        self.weights = torch.as_tensor(selected, dtype=torch.double)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size,
                                    replacement=True, generator=generator).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])
