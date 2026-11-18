#!/usr/bin/env bash
# Stage and verify the drift recovery demos without changing the active run.
set -euo pipefail
cd /data/hlou/pigment-pour-study
root=$PWD
sim=$root/pigment-pour-sim/simulation
old=$root/datasets/cylinder-joint-reopen-recovery-v4
data=$root/datasets/cylinder-joint-drift-recovery-v6
stage=$root/runs/cylinder-joint-drift-recovery-data-check-v6
raw=$sim/generated/cylinder-joint-v4-drift-recovery-v1
checked=$root/runs/cylinder-joint-v4-drift-recovery-v1-validation
live=$root/runs/cylinder-joint-grasp-sampling-bc-v5
v5_weights=$root/runs/cylinder-joint-grasp-sampling-data-check-v5/grasp-weights.npy
python=$root/.venv-wall/bin/python
export PYTHONPATH="$sim:$root/wall-x" HF_HUB_OFFLINE=1 MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
test ! -e "$data"
mkdir "$stage"

names=(
  volume_15_seed_9021_snapshot_01
  volume_25_seed_9023_snapshot_04
  volume_35_seed_9025_snapshot_05
  volume_35_seed_9025_snapshot_06_fine_alignment
)
episodes=()
reconstructions=()
for name in "${names[@]}"; do
  "$python" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["data_integrity_passed"] and d["normalization_covers_all_labels"]' "$checked/$name/validation.json"
  episodes+=("$raw/$name")
  reconstructions+=("$checked/$name/joint-reconstruction")
done
find "$old" -type f \( -name '*.parquet' -o -name '*.mp4' -o -name '*.npy' -o -name '*.json' \) -print0 |
  sort -z | xargs -0 sha256sum > "$stage/protected-before.sha256"
sha256sum "$live/train.yml" "$v5_weights" >> "$stage/protected-before.sha256"

"$python" -c "import sys; import torch; import runpy; runpy.run_path(sys.argv.pop(1), run_name='__main__')" \
  "$sim/export_carry_dataset.py" --episodes "${episodes[@]}" \
  --joint-reconstruction "${reconstructions[@]}" --base-dataset "$old" --output "$data" \
  --repo-id "$data" --reuse-source-videos > "$stage/export.log" 2>&1
"$python" "$sim/prepare_recovery_weights.py" --dataset "$data" --output "$stage/recovery-weights.npy" \
  --target-recovery-fraction .5 --full-start-boost 2 --full-start-seconds 8 > "$stage/recovery-weights.log"
"$python" "$sim/prepare_cylinder_grasp_weights.py" --dataset "$data" --base-weights "$stage/recovery-weights.npy" \
  --output "$stage/grasp-weights.npy" --seed 48 --reopen-share .08 --start-close-share .06 > "$stage/grasp-weights.log"
"$python" - "$live/train.yml" "$data" "$stage" <<'PY'
from pathlib import Path
import sys
import yaml

source, data, stage = map(Path, sys.argv[1:])
cfg = yaml.safe_load(source.read_text())
cfg['data']['lerobot_config'].update(root=str(data), repo_id=str(data),
                                      frame_sampling_weights_path=str(stage / 'grasp-weights.npy'))
cfg['data']['norm_stats_path'] = str(data / 'joint_norm_stats.json')
cfg['checkpoint']['save_path'] = str(stage / 'checkpoints')
cfg['logging']['log_name'] = stage.name
(stage / 'train.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
"$python" -m torch.distributed.run --nproc_per_node=1 --master_port=29969 "$sim/validate_wall_cylinder_joints.py" \
  --config "$stage/train.yml" --check-episode 59 --check-episode 60 --check-episode 61 --check-episode 62 \
  --output "$stage/data-validation.json" > "$stage/loader.log" 2>&1

"$python" - "$old" "$data" "$stage" "${episodes[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

import pyarrow.dataset as arrow

from validate_cylinder_joint_execution import joint_replay

old, data, stage, *episodes = map(Path, sys.argv[1:])
def load_table(folder):
    return arrow.dataset(folder / 'data', format='parquet').to_table().sort_by('index')

base, full = load_table(old), load_table(data)
assert len(base) == 36082 and len(full) == 38531
assert base.equals(full.slice(0, len(base)))
base_provenance = json.loads((old / 'demonstration_provenance.json').read_text())
new_provenance = json.loads((data / 'demonstration_provenance.json').read_text())
assert len(base_provenance) == 59 and len(new_provenance) == 63
assert base_provenance == new_provenance[:59]
assert [Path(row['source']) for row in new_provenance[59:]] == episodes
assert (old / 'joint_norm_stats.json').read_bytes() == (data / 'joint_norm_stats.json').read_bytes()
assert all(os.path.samefile(f, data / 'videos' / f.relative_to(old / 'videos'))
           for f in (old / 'videos').rglob('*.mp4'))
new_videos = [f for episode in episodes for f in episode.glob('training_*.mp4')]
assert len(new_videos) == 12
exported_videos = list((data / 'videos').rglob('*.mp4'))
assert all(sum(os.path.samefile(f, other) for other in exported_videos) == 1 for f in new_videos)
loader = json.loads((stage / 'data-validation.json').read_text())
assert loader['passed'] and loader['sampling_weights_match_source_indices']
replays = []
for index in range(59, 63):
    replay = joint_replay(data, index, stage / f'joint_{index:03d}')
    replays.append({k: replay[k] for k in ('episode', 'demonstration_frames', 'executed_frames',
        'demonstration_replay_success', 'max_slip_m', 'max_lifted_m', 'geometry_audit_passed')})
assert all(r['demonstration_replay_success'] and r['geometry_audit_passed'] for r in replays)
report = dict(kind='staged_joint_drift_corrections', dataset=str(data), base_dataset=str(old),
    episodes=63, frames=len(full), new_frames=len(full)-len(base), exact_unchanged_base_rows=len(base),
    norm_sha256=hashlib.sha256((data / 'joint_norm_stats.json').read_bytes()).hexdigest(),
    all_video_payloads_shared=True, new_duplicate_video_bytes=0,
    data_loader_passed=True, teacher_decoder_replays=replays, neural_training_started=False,
    task_success_measured=False, independent_cases_read=False)
(stage / 'audit-pending-hash-check.json').write_text(json.dumps(report, indent=2) + '\n')
PY
sha256sum --status -c "$stage/protected-before.sha256"
mv "$stage/audit-pending-hash-check.json" "$stage/summary.json"
cat "$stage/summary.json"
