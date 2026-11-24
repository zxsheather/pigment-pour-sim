"""Allow a train-only episode export in the pinned Wall-X config validator.

The loader already handles an empty episode-test slice and makes its separate
95/5 frame split inside selected training episodes. This changes validation
only; independent full-task evaluation remains outside the training dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--wall-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    path=a.wall_root/"wall_x/data/backends/lerobot/config.py"
    old=path.read_text();new=old.replace("if not 0 < self.train_test_split < 1:",
                                       "if not 0 < self.train_test_split <= 1:")
    new=new.replace("train_test_split must be between 0 and 1, got",
                    "train_test_split must be in (0, 1], got")
    if new==old:raise ValueError("Expected unpatched pinned Wall-X validator")
    path.write_text(new)
    report=dict(file=str(path.resolve()),before_sha256=hashlib.sha256(old.encode()).hexdigest(),
                after_sha256=hashlib.sha256(new.encode()).hexdigest(),
                change="Allow train_test_split == 1.0; no change to dataset conversion or model.")
    a.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))


if __name__=="__main__":main()
