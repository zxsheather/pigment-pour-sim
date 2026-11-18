"""Assign language prompts to each frame using the recorded robot's progress."""
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

from cylinder_joint_actions import JOINT_SERVO_STATE_NAMES, require
from cylinder_progress_prompts import CylinderProgressPrompts, PROGRESS_PROMPTS, PROGRESS_PROTOCOL
from export_cylinder_joint_dataset import digest, statistics


def task_statistics(actions):
    stats = statistics(actions[:, None])
    for key in ('min', 'max'):
        stats[key] = [int(stats[key][0])]
    return stats


def replace_column(table, name, values):
    field = table.schema.field(name)
    return table.set_column(table.schema.get_field_index(name), field, pa.array(values, type=field.type))


def export(source, output):
    require(not output.exists(), 'Output already exists')
    info = json.loads((source/'meta/info.json').read_text())
    demos = json.loads((source/'demonstration_provenance.json').read_text())
    require(info['features']['observation.state']['names'] == JOINT_SERVO_STATE_NAMES
            and info['total_tasks'] == 1, 'Expected the single-instruction joint-servo dataset')
    frames = arrow.dataset(source/'data', format='parquet').to_table(columns=[
        'index', 'episode_index', 'frame_index', 'observation.state', 'task_index']).sort_by('index')
    indices = np.asarray(frames['index'])
    episode_ids = np.asarray(frames['episode_index'])
    frame_ids = np.asarray(frames['frame_index'])
    state = np.asarray(frames['observation.state'].to_pylist(), dtype=np.float32)
    require(np.array_equal(indices, np.arange(info['total_frames']))
            and np.array_equal(np.unique(episode_ids), np.arange(len(demos)))
            and not np.any(np.asarray(frames['task_index'])), 'Expected contiguous single-task indices')
    resets = [i for i, demo in enumerate(demos) if not demo['metadata']['result'].get('initial_state_provenance')]
    require(bool(resets), 'A recorded scene reset is required to establish the base reference')
    home = state[episode_ids == resets[0]][0, 15:18]
    task_ids = np.empty(len(state), dtype=np.int64)
    episodes = []
    for index, demo in enumerate(demos):
        select = episode_ids == index
        measured = state[select]
        meta = demo['metadata']
        require(np.array_equal(frame_ids[select], np.arange(meta['frames'])), 'Invalid episode frame indices')
        provenance = meta['result'].get('initial_state_provenance')
        if not provenance or provenance['source_simulated_seconds'] == 0:
            np.testing.assert_allclose(measured[0, 15:18], home, atol=1e-6, rtol=0.)
        router = CylinderProgressPrompts(home)
        ids = np.array([router.observe(row) for row in measured], dtype=np.int64)
        task_ids[select] = ids
        changes = np.flatnonzero(np.diff(ids, prepend=-1))
        episodes.append(dict(episode_index=index, prompt_changes=[dict(frame=int(k), task_index=int(ids[k]))
                                                                    for k in changes],
                             tasks=[PROGRESS_PROMPTS[i] for i in np.unique(ids)], stats=task_statistics(ids)))
    counts = np.bincount(task_ids, minlength=len(PROGRESS_PROMPTS))
    require((counts > 0).all(), 'All progress prompts must occur in the verified dataset')
    global_stats = task_statistics(task_ids)

    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        staged = Path(temporary)/'dataset'
        staged.mkdir()
        shutil.copytree(source/'meta', staged/'meta')
        shutil.copytree(source/'videos', staged/'videos', copy_function=os.link)
        videos = list((source/'videos').rglob('*.mp4'))
        require(videos and all(os.path.samefile(p, staged/p.relative_to(source)) for p in videos),
                'Expected shared source video files')
        for path in sorted((source/'data').rglob('*.parquet')):
            original = pq.read_table(path)
            table = replace_column(original, 'task_index', task_ids[np.asarray(original['index'])])
            require(table.drop(['task_index']).equals(original.drop(['task_index'])), 'Non-language data changed')
            target = staged/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
        for path in sorted((source/'meta/episodes').rglob('*.parquet')):
            table = pq.read_table(path)
            subset = [episodes[i] for i in np.asarray(table['episode_index'])]
            table = replace_column(table, 'tasks', [row['tasks'] for row in subset])
            for key in global_stats:
                table = replace_column(table, f'stats/task_index/{key}', [row['stats'][key] for row in subset])
            pq.write_table(table, staged/path.relative_to(source))
        template = pq.read_table(source/'meta/tasks.parquet')
        require(template.column_names == ['task_index', '__index_level_0__'], 'Unexpected LeRobot tasks schema')
        tasks = pa.Table.from_arrays([pa.array(range(len(PROGRESS_PROMPTS))), pa.array(PROGRESS_PROMPTS)],
                                     schema=template.schema)
        pq.write_table(tasks, staged/'meta/tasks.parquet')
        info['total_tasks'] = len(PROGRESS_PROMPTS)
        (staged/'meta/info.json').write_text(json.dumps(info, indent=2)+'\n')
        stats = json.loads((source/'meta/stats.json').read_text())
        stats['task_index'] = global_stats
        (staged/'meta/stats.json').write_text(json.dumps(stats, indent=2)+'\n')
        for name in ('demonstration_provenance.json', 'joint_norm_stats.json'):
            shutil.copyfile(source/name, staged/name)
        report = dict(kind='joint_progress_prompt_view', protocol=PROGRESS_PROTOCOL, prompts=PROGRESS_PROMPTS,
                      dataset=str(output), source=str(source), episodes=len(demos), frames=len(state),
                      frames_per_prompt=counts.tolist(), initial_robot_base=home.tolist(),
                      frame_task_index_dtype=str(task_ids.dtype),
                      normalizer_sha256=digest(staged/'joint_norm_stats.json'),
                      non_language_data_exact=True, shared_videos=len(videos),
                      uses_only_current_and_previous_robot_observations=True,
                      independent_cases_read=False, episode_prompt_changes=[
                          dict(episode_index=row['episode_index'], changes=row['prompt_changes']) for row in episodes])
        (staged/'progress_prompt_view.json').write_text(json.dumps(report, indent=2)+'\n')
        staged.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.base_dataset.resolve(), args.output.resolve())
    print(json.dumps({k: v for k, v in result.items() if k != 'episode_prompt_changes'}), flush=True)


if __name__ == '__main__':
    main()
