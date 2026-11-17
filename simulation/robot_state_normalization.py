"""Opt-in state-normalization ablation using the existing affine scale.

The pinned official Normalizer uses this formula followed by clamp(-1,1).
This instance-only override preserves information outside the training bounds.
It must only be installed on the robot-state normalizer, never the action one.
"""
from types import MethodType

import torch


def normalize_robot_state_without_clipping(self,xs,dataset_names):
    outputs=[]
    names=[name for name in dataset_names if name!='x2_multimodal']
    for x,name in zip(xs,names):
        delta=self.delta[name]
        delta=torch.where(delta==0,torch.ones_like(delta),delta)
        outputs.append(((x-self.min[name])/delta)*2-1)
    return torch.stack(outputs)


def enable_unclipped_robot_state(normalizer):
    normalizer.normalize_data=MethodType(normalize_robot_state_without_clipping,normalizer)
