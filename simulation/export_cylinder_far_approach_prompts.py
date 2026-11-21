"""Create a language-only dataset view with a far approach prompt."""

import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.dataset as arrow
import pyarrow.parquet as pq

from cylinder_far_approach_prompt import (
    FAR_APPROACH_DISTANCE_M, FAR_APPROACH_PROMPT_PROTOCOL, RANGE_CONDITIONED_PROMPTS,
    range_conditioned_prompt_indices,
)
from cylinder_joint_actions import require
from cylinder_progress_prompts import PROGRESS_PROMPTS
from cylinder_tray_vision import WRIST_TARGET_CUE_PROTOCOL
from export_cylinder_joint_dataset import digest
from export_cylinder_progress_prompts import replace_column, task_statistics


def export(source, output):
    require(not output.exists(), 'Output already exists')
    info = json.loads((source / 'meta/info.json').read_text())
    cue = json.loads((source / 'wrist_target_view_audit.json').read_text())
    require(cue['observations_protocol'] == WRIST_TARGET_CUE_PROTOCOL,
            'Expected the verified RGB wrist-target tensor')

    template = pq.read_table(source / 'meta/tasks.parquet')
    require(template.column_names == ['task_index', '__index_level_0__']
            and template['task_index'].to_pylist() == list(range(len(PROGRESS_PROMPTS)))
            and template['__index_level_0__'].to_pylist() == list(PROGRESS_PROMPTS),
            'Expected the three current progress prompts')
    require(info['total_tasks'] == len(PROGRESS_PROMPTS), 'Unexpected task count')
    frames = arrow.dataset(source / 'data', format='parquet').to_table(
        columns=['index', 'episode_index', 'task_index', 'observation.state']).sort_by('index')
    indices = np.asarray(frames['index'])
    require(np.array_equal(indices, np.arange(info['total_frames'])), 'Expected contiguous frame indices')
    episode_ids = np.asarray(frames['episode_index'])
    require(np.array_equal(np.unique(episode_ids), np.arange(info['total_episodes']))
            and np.all(np.diff(episode_ids) >= 0), 'Episode ordering changed')
    phase = np.asarray(frames['task_index'])
    state = np.asarray(frames['observation.state'].to_pylist(), dtype=np.float32)
    task_ids = range_conditioned_prompt_indices(state, phase)
    num_tasks = len(RANGE_CONDITIONED_PROMPTS)
    counts = np.bincount(task_ids, minlength=num_tasks)
    require((counts > 0).all(), 'The input corpus must contain observations for each prompt')
    global_stats = task_statistics(task_ids)
    blocks = np.split(task_ids, np.flatnonzero(np.diff(episode_ids)) + 1)
    descriptions = [[RANGE_CONDITIONED_PROMPTS[i] for i in np.unique(block)] for block in blocks]
    per_episode_stats = [task_statistics(block) for block in blocks]

    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        dest = Path(temporary) / 'dataset'
        shutil.copytree(source / 'meta', dest / 'meta')
        shutil.copytree(source / 'videos', dest / 'videos', copy_function=os.link)
        videos = list((source / 'videos').rglob('*.mp4'))
        require(videos and all(os.path.samefile(f, dest / f.relative_to(source)) for f in videos),
                'Source videos must be shared unchanged')
        for file in sorted((source / 'data').rglob('*.parquet')):
            original = pq.read_table(file)
            table = replace_column(original, 'task_index', task_ids[np.asarray(original['index'])])
            require(table.drop(['task_index']).equals(original.drop(['task_index'])), 'Non-language frame data changed')
            output_file = dest / file.relative_to(source)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output_file)
        for file in sorted((source / 'meta/episodes').rglob('*.parquet')):
            table = pq.read_table(file)
            ids = np.asarray(table['episode_index'])
            table = replace_column(table, 'tasks', [descriptions[i] for i in ids])
            for key in global_stats:
                table = replace_column(table, f'stats/task_index/{key}', [per_episode_stats[i][key] for i in ids])
            pq.write_table(table, dest / file.relative_to(source))
        tasks = pa.Table.from_arrays(
            [pa.array(range(num_tasks)), pa.array(RANGE_CONDITIONED_PROMPTS)], schema=template.schema)
        pq.write_table(tasks, dest / 'meta/tasks.parquet')
        info['total_tasks'] = num_tasks
        (dest / 'meta/info.json').write_text(json.dumps(info, indent=2) + '\n')
        stats = json.loads((source / 'meta/stats.json').read_text())
        stats['task_index'] = global_stats
        (dest / 'meta/stats.json').write_text(json.dumps(stats, indent=2) + '\n')
        for name in ('demonstration_provenance.json', 'joint_norm_stats.json', 'wrist_target_view_audit.json'):
            shutil.copyfile(source / name, dest / name)
        report = dict(kind='far_approach_prompt_view', source=str(source), dataset=str(output),
                      protocol=FAR_APPROACH_PROMPT_PROTOCOL, distance_m=FAR_APPROACH_DISTANCE_M,
                      prompts=RANGE_CONDITIONED_PROMPTS, frames_per_prompt=counts.tolist(),
                      frames=len(task_ids), episodes=len(blocks), shared_videos=len(videos),
                      non_language_frame_data_exact=True, measured_rgb_target_and_progress_only=True,
                      normalizer_sha256=digest(dest / 'joint_norm_stats.json'), independent_cases_read=False)
        (dest / 'far_approach_prompt_view.json').write_text(json.dumps(report, indent=2) + '\n')
        dest.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.base_dataset.resolve(), args.output.resolve())), flush=True)


if __name__ == '__main__':
    main()
