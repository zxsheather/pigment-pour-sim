"""Shared native-SDPA compatibility for the pinned public Wall-X model."""
from functools import wraps


def configure_runtime():
    from wall_x.model.qact.qwen2_5 import modeling_qwen2_5_vl as vision

    vision.Qwen2_5_VLMLP._GEMM_ALIGN = 1
    original = vision.Qwen2_5_VLVisionSdpaAttention.forward
    if getattr(original, "_accepts_wallx_max_seqlen", False):
        return

    @wraps(original)
    def forward(self, hidden_states, cu_seqlens, max_seqlen=None, **kwargs):
        return original(self, hidden_states, cu_seqlens, **kwargs)

    forward._accepts_wallx_max_seqlen = True
    vision.Qwen2_5_VLVisionSdpaAttention.forward = forward
