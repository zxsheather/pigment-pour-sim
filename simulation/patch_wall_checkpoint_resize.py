"""Skip unused covariance initialization when loading full checkpoint embeddings.

The pinned loader resizes the vocabulary, then overwrites the entire embedding
tensor from the checkpoint. mean_resizing=False changes only that discarded
initialization, avoiding a large temporary allocation on this Thor environment.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--wall-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    path=a.wall_root/"wall_x/trainer/fsdp_trainer/checkpoint_io.py"
    old=path.read_text();needle="resize_token_embeddings(ckpt_vocab)"
    if old.count(needle)!=2:raise ValueError("Expected pinned unpatched checkpoint loader")
    new=old.replace(needle,"resize_token_embeddings(ckpt_vocab, mean_resizing=False)")
    path.write_text(new)
    report=dict(file=str(path.resolve()),before_sha256=hashlib.sha256(old.encode()).hexdigest(),
                after_sha256=hashlib.sha256(new.encode()).hexdigest(),
                change="Disable discarded covariance initialization before loading full checkpoint embedding weights.")
    a.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))


if __name__=="__main__":main()
