"""Make FlashAttention optional for an otherwise native SDPA inference path.

The upstream action/vision models implement PyTorch SDPA, but joint.py imports
the optional FlashAttention package unconditionally. Do not change any model
weights, layer definitions, or attention calculations.
"""
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
args = parser.parse_args()
path = args.source / "wall_x/model/core/attention/joint.py"
original = "from flash_attn import flash_attn_func\n"
replacement = "try:\n    from flash_attn import flash_attn_func\nexcept ImportError:\n    flash_attn_func = None  # Native SDPA does not use FlashAttention.\n"
text = path.read_text()
if replacement not in text:
    if text.count(original) != 1:
        raise RuntimeError("Upstream import changed; inspect before applying patch")
    path.write_text(text.replace(original, replacement, 1))
print(f"Optional FlashAttention import ready: {path}")
