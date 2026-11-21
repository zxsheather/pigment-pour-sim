"""Add opt-in recovery sampling to the pinned Wall-X training loader only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wall-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    directory = a.wall_root / "wall_x/data/backends/lerobot"
    path = directory / "loader.py"
    old = path.read_text()
    marker = "        # Create distributed sampler\n        sampler = DistributedSampler("
    start = old.index(marker, old.index("    def get_train_dataloader(self):"))
    end = old.index("\n\n        dataloader =", start)
    original = old[start:end]
    replacement = '''        # Opt-in training sampling; default and validation paths are unchanged.
        weights_path = self.lerobot_config.get("frame_sampling_weights_path")
        if weights_path:
            from wall_x.data.backends.lerobot.recovery_sampler import RecoveryDistributedSampler
            sampler = RecoveryDistributedSampler(
                self, weights_path=weights_path, num_replicas=self.world_size,
                rank=self.rank, shuffle=True, seed=self.seed, drop_last=True,
                groups_path=self.lerobot_config.get("frame_sampling_groups_path"),
                block_size=self.lerobot_config.get("frame_sampling_block_size"))
        else:
'''+"\n".join("    "+line for line in original.splitlines())
    assert "RecoveryDistributedSampler" not in old, "Already patched"
    new = old[:start] + replacement + old[end:]
    helper = Path(__file__).with_name("recovery_sampler.py")
    target = directory / helper.name
    assert not target.exists(), "Refuse to overwrite an existing loader helper"
    compile(new, str(path), "exec")
    shutil.copyfile(helper, target)
    path.write_text(new)
    report = dict(file=str(path.resolve()),
        before_sha256=hashlib.sha256(old.encode()).hexdigest(),
        after_sha256=hashlib.sha256(new.encode()).hexdigest(),
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        option="data.lerobot_config.frame_sampling_weights_path",
        stratification_options=["frame_sampling_groups_path", "frame_sampling_block_size"],
        change="Weighted seeded training-frame sampling only when explicitly configured; no validation/inference/model changes")
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
