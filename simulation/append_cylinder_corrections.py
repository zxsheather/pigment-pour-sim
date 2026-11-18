"""Append verified model-visited teacher corrections to the official wrist-target data."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import av
import numpy as np
import pyarrow.dataset as arrow

from carry_env import CAMERAS, CONTROL_DT
from cylinder_joint_actions import JOINT_ACTION_WIDTH, JOINT_SERVO_STATE_WIDTH, require
from cylinder_progress_prompts import PROGRESS_PROMPTS
from cylinder_tray_vision import WRIST_TARGET_CUE_PROTOCOL
from export_carry_dataset import LeRobotDataset, SourceVideoDataset, file_digest, share_immutable_video


def append(base, qualifications, base_weights, output):
    require(not output.exists(), 'Output exists')
    view = json.loads((base/'wrist_target_view_audit.json').read_text())
    bundle = json.loads(qualifications.read_text())
    require(view['observations_protocol'] == WRIST_TARGET_CUE_PROTOCOL
            and Path(bundle['base_dataset']).resolve() == base and not bundle['independent_cases_read'],
            'Expected correction proofs for this wrist-target dataset')
    manifest = json.loads((base/'demonstration_provenance.json').read_text())
    rows = [case for case in bundle['results'] if case['qualified']]
    require(rows and len(rows) == bundle['qualified_count'], 'No verified teacher correction data')
    norm = json.loads((base/'joint_norm_stats.json').read_text())['norm_stats']
    old = arrow.dataset(base/'data',format='parquet').to_table().sort_by('index')
    old_frames = len(old)
    weights = np.load(base_weights, allow_pickle=False)
    require(weights.shape == (old_frames,) and np.isfinite(weights).all() and np.all(weights > 0),
            'Existing sampling weights do not align with the base dataset')
    video_hashes = {p.relative_to(base):file_digest(p) for p in (base/'videos').rglob('*.mp4')}
    new_rows, new_sources, task_blocks = [], [], []
    episode_count = len(manifest)
    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        path = Path(temporary)/'dataset'
        shutil.copytree(base, path)
        dataset = SourceVideoDataset(str(output), root=path, video_backend='pyav', vcodec='h264')
        dataset.keep_source_files_separate = True
        dataset.start_image_writer(num_threads=3)
        require(dataset.num_episodes == episode_count and dataset.num_frames == old_frames and dataset.fps == 20,
                'Copied LeRobot dataset differs from the certified base')
        try:
            for row in rows:
                proof = json.loads(Path(row['proof']).read_text())
                source = Path(proof['source'])
                metadata = json.loads((source/'metadata.json').read_text())
                require(proof == row['certification'] and proof['teacher_full_task_succeeded']
                        and proof['geometry_audit_passed'] and not proof['model_task_success_measured']
                        and not proof['independent_cases_read']
                        and proof['trajectory_sha256'] == file_digest(source/'trajectory.npz')
                        and not any(x['source'] == str(source) or x['metadata']['result']['seed'] == row['seed']
                                    for x in manifest), 'Correction provenance is not unique and verified')
                frames = proof['frames']
                with np.load(row['training_arrays'], allow_pickle=False) as raw:
                    state, action, tasks = (raw[k] for k in ('state','action','task_index'))
                require(state.shape == (frames, JOINT_SERVO_STATE_WIDTH) and action.shape == (frames, JOINT_ACTION_WIDTH)
                        and state.dtype == action.dtype == np.float32 and tasks.shape == (frames,)
                        and np.array_equal(np.unique(tasks),np.arange(len(PROGRESS_PROMPTS))),
                        'Correction frame statistics or phase metadata differs')
                for name, values in [('observation.state',state),('action',action)]:
                    require(np.isfinite(values).all() and np.all(values >= np.asarray(norm[name]['q01']) - 1e-7)
                            and np.all(values <= np.asarray(norm[name]['q99']) + 1e-7),
                            'Prepared correction does not fit the fixed normalizer')
                keys = {camera:f'observation.images.{camera}' for camera in CAMERAS}
                media = {keys[camera]:Path(video['path']) for camera,video in proof['alignment_video'].items()}
                shas = {keys[camera]:video['sha256'] for camera,video in proof['alignment_video'].items()}
                require(set(media) == set(keys.values()) and all(file_digest(media[k]) == shas[k] for k in media),
                        'Teacher camera content changed after qualification')
                dataset.source_videos = media
                with ExitStack() as stack:
                    decoders = {key:iter(stack.enter_context(av.open(str(p))).decode(video=0)) for key,p in media.items()}
                    for i,(obs,act,phase) in enumerate(zip(state,action,tasks,strict=True)):
                        frame = {'observation.state':obs,'action':act,'task':PROGRESS_PROMPTS[phase]}
                        for key, decoder in decoders.items():
                            rgb=next(decoder)
                            require(abs(float(rgb.time)-i*CONTROL_DT)<1e-4, 'Teacher camera timestamps differ')
                            frame[key]=rgb.to_ndarray(format='rgb24')
                        dataset.add_frame(frame)
                    require(all(next(decoder,None) is None for decoder in decoders.values()), 'Extra camera frames')
                dataset.save_episode(parallel_encoding=False)
                task_blocks.append(tasks)
                new_rows.append((state,action,tasks))
                new_sources.append((media,shas))
                manifest.append(dict(source=str(source),metadata=metadata,trajectory_sha256=proof['trajectory_sha256'],
                                     source_video_sha256=shas,observation_state_fields=dataset.features['observation.state']['names'],
                                     video_export='source_h264_stream_copy',causal_correction_certificate=row['proof'],
                                     prepared_frame_sha256=file_digest(Path(row['training_arrays']))))
        finally:
            dataset.finalize()

        after = arrow.dataset(path/'data',format='parquet').to_table().sort_by('index')
        require(all(old[k].equals(after[k][:old_frames]) for k in old.column_names), 'Existing training rows changed')
        all_new = after.slice(old_frames)
        for name, values in zip(('observation.state','action','task_index'),map(np.concatenate,zip(*new_rows)),strict=True):
            np.testing.assert_array_equal(np.asarray(all_new[name].to_pylist()),values)
        require(file_digest(path/'joint_norm_stats.json') == file_digest(base/'joint_norm_stats.json'),
                'Original training normalization changed')
        for relative, digest in video_hashes.items():
            share_immutable_video(base/relative,path/relative,digest)
        reopened = LeRobotDataset(str(output),root=path,video_backend='pyav')
        seen = set(video_hashes)
        for i,(assets,shas) in enumerate(new_sources,start=episode_count):
            for name,video in assets.items():
                rel = reopened.meta.get_video_file_path(i,name)
                require(rel not in seen,'Expected a separate video container for the new episode')
                seen.add(rel)
                share_immutable_video(video,path/rel,shas[name])
        require(reopened.num_episodes == len(manifest) and reopened.num_frames == len(after)
                and len(reopened.hf_dataset) == len(after), 'Appended dataset does not reopen')
        (path/'demonstration_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
        phases = np.concatenate(task_blocks)
        # Keep the relative original weights and assign 20% of total mass to
        # corrections, of which 75% is used for approaching and closing.
        new_mass = weights.sum() * .2 / .8
        approach = phases == 0
        new_weights = np.where(approach, new_mass*.75/approach.sum(), new_mass*.25/(~approach).sum())
        np.save(path/'frame-sampling-weights.npy',np.concatenate((weights,new_weights)))
        report = dict(kind='verified_wrist_target_correction_append',base=str(base),dataset=str(output),
                      qualifications=str(qualifications),new_episodes=len(rows),new_frames=len(all_new),
                      episodes=len(manifest),frames=len(after),base_frames_exact=True,normalization_exact=True,
                      original_video_hardlinks=len(video_hashes),appended_video_hardlinks=len(seen)-len(video_hashes),
                      new_sampling_fraction=.2,new_approach_fraction=.75,
                      reader_reopen_passed=True,independent_cases_read=False)
        (path/'correction_append_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        path.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dataset', required=True, type=Path)
    parser.add_argument('--qualification-report', required=True, type=Path)
    parser.add_argument('--base-weights', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(append(args.base_dataset.resolve(),args.qualification_report.resolve(),
                            args.base_weights.resolve(),args.output.resolve())))


if __name__ == '__main__':
    main()
