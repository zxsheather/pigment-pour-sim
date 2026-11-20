"""Keep Wall's initial proprioception projection when widening state bounds."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def projection_scales(original, expanded):
    old = original['norm_stats']['observation.state']
    new = expanded['norm_stats']['observation.state']
    allowed = original | {'norm_stats': original['norm_stats'] | {
        'observation.state': old | {key: new[key] for key in ('q01', 'q99')},
    }}
    if expanded != allowed:
        raise ValueError('Only proprioception quantile bounds may change')

    scales = {}
    for i, (lo, hi, low, high) in enumerate(zip(old['q01'], old['q99'], new['q01'], new['q99'], strict=True)):
        if (lo, hi) == (low, high):
            continue
        if not (low <= lo < hi <= high and math.isclose(lo + hi, low + high, abs_tol=1e-10, rel_tol=0.)):
            raise ValueError(f'Expected a symmetric expansion for state dimension {i}')
        scales[i] = (high - low) / (hi - lo)
    if not scales:
        raise ValueError('No state bounds widened')
    return len(old['q01']), scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-root', type=Path, required=True)
    parser.add_argument('--old-stats', type=Path, required=True)
    parser.add_argument('--new-stats', type=Path, required=True)
    parser.add_argument('--training-stats-path', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.wall_root/'wall_x/trainer/fsdp_trainer/fsdp_trainer.py'
    old_stats = args.old_stats.read_bytes()
    new_stats = args.new_stats.read_bytes()
    dimensions, scales = projection_scales(json.loads(old_stats), json.loads(new_stats))
    expected_stats = hashlib.sha256(new_stats).hexdigest()

    before = source.read_text()
    marker = '''                {"ckpt": self.cfg.checkpoint.resume_from},
            )

        self._wrap_model(self.model)'''
    if before.count(marker) != 1 or 'def _resume_from_single_file' not in before:
        raise ValueError('Expected the pinned single-file checkpoint initialization')
    amended = marker.replace('\n\n        self._wrap_model', f'''
            from hashlib import sha256
            from pathlib import Path
            stats = Path(self.cfg.data.norm_stats_path)
            if (stats.resolve() != Path({str(args.training_stats_path.resolve())!r})
                    or Path(self.cfg.checkpoint.resume_from).resolve() != Path({str(args.parent.resolve())!r})
                    or sha256(stats.read_bytes()).hexdigest() != {expected_stats!r}):
                raise RuntimeError("Proprioception rescaling requires the verified stats and parent checkpoint")
            projection = self.model.action_preprocessor.propri_proj
            if projection.bias is not None or projection.in_features != {2 * dimensions}:
                raise RuntimeError("Expected the pinned bias-free proprioception projection with mask")
            columns = {list(scales)!r}
            with torch.no_grad():
                projection.weight[:, columns] *= projection.weight.new_tensor({list(scales.values())!r})
            self.logger.info("Rescaled initial proprioception projection columns %s", columns)

        self._wrap_model''')
    after = before.replace(marker, amended)
    compile(after, str(source), 'exec')
    source.write_text(after)
    record = dict(kind='wall_proprioception_projection_rescaling', source=str(source), state_width=dimensions,
                  scales=scales, old_stats_sha256=hashlib.sha256(old_stats).hexdigest(),
                  new_stats_sha256=expected_stats,
                  before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                  after_sha256=hashlib.sha256(after.encode()).hexdigest(),
                  parent=str(args.parent.resolve()), training_stats=str(args.training_stats_path.resolve()))
    with args.output.open('x') as out:
        out.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
