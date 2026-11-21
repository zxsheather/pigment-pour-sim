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
    def __init__(self, dataset, *, weights_path, groups_path=None, block_size=None, **kwargs):
        super().__init__(dataset, **kwargs)
        weights = np.load(Path(weights_path), allow_pickle=False)
        if weights.shape != (len(dataset.hf_dataset),):
            raise ValueError("Recovery weights must match the full LeRobot dataset")
        train_indices = np.asarray(dataset.train_dataset.indices)
        selected = weights[train_indices]
        if not np.isfinite(selected).all() or (selected <= 0).any():
            raise ValueError("Sampling weights must be finite and positive")
        self.weights = torch.as_tensor(selected, dtype=torch.double)
        self.group_members = None
        if (groups_path is None) != (block_size is None):
            raise ValueError("Stratified sampling requires both groups and a block size")
        if groups_path is not None:
            if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
                raise ValueError("Stratified sampling block size must be a positive integer")
            groups = np.load(Path(groups_path), allow_pickle=False)
            if groups.shape != weights.shape or not np.issubdtype(groups.dtype, np.integer):
                raise ValueError("Sampling groups must label each full LeRobot frame with an integer")
            train_groups = groups[train_indices]
            self.group_members = tuple(torch.from_numpy(np.flatnonzero(train_groups == tag))
                                       for tag in np.unique(train_groups))
            self.block_size = block_size
            mass = torch.stack([self.weights[members].sum() for members in self.group_members])
            self.group_cdf = (mass / mass.sum()).cumsum(dim=0)
            self.group_cdf[-1] = 1.

    def _stratified_groups(self, blocks, width, generator):
        offset = torch.rand((blocks, 1), dtype=torch.double, generator=generator)
        points = (torch.arange(width, dtype=torch.double) + offset) / width
        labels = torch.searchsorted(self.group_cdf, points, right=True)
        order = torch.rand((blocks, width), generator=generator).argsort(dim=1)
        return labels.gather(1, order).flatten()

    def _stratified_indices(self, generator):
        blocks, tail = divmod(self.total_size, self.block_size)
        labels = self._stratified_groups(blocks, self.block_size, generator)
        if tail:
            labels = torch.cat((labels, self._stratified_groups(1, tail, generator)))
        indices = torch.empty_like(labels)
        for group, members in enumerate(self.group_members):
            positions = (labels == group).nonzero(as_tuple=True)[0]
            if len(positions):
                picked = torch.multinomial(self.weights[members], len(positions),
                                           replacement=True, generator=generator)
                indices[positions] = members[picked]
        return indices

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.group_members is None:
            indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator)
        else:
            indices = self._stratified_indices(generator)
        return iter(indices[self.rank:self.total_size:self.num_replicas].tolist())
