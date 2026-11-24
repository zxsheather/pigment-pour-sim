"""Reconstruct the exact V924 safetensors file with NumPy on a CPU."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from zipfile import ZipFile

import numpy as np


HERE = Path(__file__).resolve().parent
ITEM_BYTES = {"BF16": 2, "F32": 4}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class TensorFile:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            self.header = stream.read(length)
        self.start = length + 8
        self.entries = json.loads(self.header)
        self.mapped = np.memmap(self.path, dtype=np.uint8, mode="r")

    def raw(self, name):
        start, end = self.entries[name]["data_offsets"]
        return self.mapped[self.start + start:self.start + end]


def reference_bytes(base, name, target):
    # The official file stores tied token/output weights only once.
    key = "model.embed_tokens.weight" if name == "lm_head.weight" else name
    if key not in base.entries:
        return None
    entry = base.entries[key]
    if entry["dtype"] != target["dtype"]:
        raise ValueError(f"Unexpected dtype change for {key}")
    raw = base.raw(key)
    if entry["shape"] == target["shape"]:
        return raw
    if not (key.startswith("visual.blocks.") and ".mlp." in key):
        raise ValueError(f"Unexpected shape change for {key}")
    # The training runtime pads the vision feed-forward width for its kernels.
    dtype = np.dtype(f"V{ITEM_BYTES[entry['dtype']]}")
    source = raw.view(dtype).reshape(entry["shape"])
    padded = np.zeros(target["shape"], dtype=dtype)
    if key.endswith("down_proj.weight"):
        padded[:, :source.shape[1]] = source
    elif key.endswith(("gate_up_proj.weight", "gate_up_proj.bias")):
        half, half_out = source.shape[0] // 2, padded.shape[0] // 2
        padded[:half] = source[:half]
        padded[half_out:half_out + half] = source[half:]
    else:
        raise ValueError(f"Unknown padding pattern for {key}")
    return padded.view(np.uint8).reshape(-1)


def restore(base_path, destination, folder=HERE):
    manifest = json.loads((folder / "weights.json").read_text())
    if sha256(base_path) != manifest["official_base_sha256"]:
        raise ValueError("The supplied official base has the wrong SHA-256")
    if destination.exists():
        raise FileExistsError(destination)
    base = TensorFile(base_path)
    header = (folder / "checkpoint-header.json").read_bytes()
    entries = json.loads(header)
    order = sorted((k for k in entries if k != "__metadata__"),
                   key=lambda k: entries[k]["data_offsets"][0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging, ExitStack() as stack:
        output = Path(staging) / "model.safetensors"
        archives = {}
        with output.open("wb") as out:
            out.write(struct.pack("<Q", len(header)))
            out.write(header)
            for key in order:
                entry = entries[key]
                start, end = entry["data_offsets"]
                if out.tell() != 8 + len(header) + start:
                    raise ValueError(f"Non-contiguous tensor offsets for {key}")
                value = reference_bytes(base, key, entry)
                asset = manifest["changed"].get(key)
                if asset:
                    filename, member = asset
                    if filename not in archives:
                        archives[filename] = stack.enter_context(ZipFile(folder / "layers" / filename))
                    change = np.frombuffer(archives[filename].read(member), dtype=np.uint8)
                    value = change if value is None else np.bitwise_xor(value, change)
                if value is None or value.nbytes != end - start:
                    raise ValueError(f"Missing bytes for {key}")
                out.write(value)
        actual = sha256(output)
        if actual != manifest["checkpoint_sha256"]:
            raise ValueError(f"Restored checkpoint SHA-256 did not match: {actual}")
        # Publish a fully verified file and never overwrite another writer's result.
        os.link(output, destination)
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Official model.safetensors")
    parser.add_argument("--output", type=Path, required=True, help="New output model.safetensors path")
    options = parser.parse_args()
    print(restore(options.base, options.output))
