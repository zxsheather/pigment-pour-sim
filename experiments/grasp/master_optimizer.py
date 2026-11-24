"""AdamW with FP32 master weights for the single-GPU BF16 experiment.

The model and DDP buckets retain their original dtypes. Clipping still acts
on model gradients; step() copies those gradients to FP32 master parameters,
updates them, then rounds only the forward-model copies back to BF16.
Evaluation snapshots remain non-resumable, as in the original experiment.
"""
import inspect

import torch


class FP32MasterAdamW(torch.optim.AdamW):
    def __init__(self, original):
        if original.state:
            raise ValueError("Create FP32 masters before the first optimizer update")
        pairs = []
        groups = []
        for group in original.param_groups:
            masters = []
            for parameter in group["params"]:
                if parameter.dtype not in (torch.bfloat16,torch.float32):
                    raise ValueError("This experiment supports BF16 and FP32 model parameters only")
                master = (parameter if parameter.dtype == torch.float32 else
                          torch.nn.Parameter(parameter.detach().float().clone()))
                pairs.append((parameter, master))
                masters.append(master)
            groups.append({**group, "params": masters})
        keys = inspect.signature(torch.optim.AdamW.__init__).parameters
        defaults = {key:value for key,value in original.defaults.items() if key in keys}
        super().__init__(groups, **defaults)
        self.model_master_pairs = pairs

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("The experiment uses AdamW without closures")
        for parameter, master in self.model_master_pairs:
            if master is not parameter:
                master.grad = None if parameter.grad is None else parameter.grad.float()
        result = super().step()
        for parameter, master in self.model_master_pairs:
            if master is not parameter:
                parameter.copy_(master)
        return result

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for parameter, master in self.model_master_pairs:
            if master is not parameter and parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()
