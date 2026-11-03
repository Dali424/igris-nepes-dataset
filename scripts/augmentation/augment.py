#!/usr/bin/env python3
"""Augment a completed collection. Each preset/mode creates K children.

Default K is the source episode count. Video defaults to one darker preset;
--mode both creates K video + K native render children. Selecting two video
presets creates2K video children. Every child retains its parent's split group.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import render_collect as process_tools
from render_scene import read, require, sha, write
from source_selection import complete_source, seeded_module, selected_video_source, source_quota_arguments

TOOLS = Path(__file__).resolve().parent
DEFAULT_BUNDLE = TOOLS.parents[1]
PRESETS = {'darker': 'visible_darker_v2', 'brighter': 'visible_brighter_v2'}


def now():
    return datetime.now(timezone.utc).isoformat()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-batch', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE,
                        help='Exact immutable source bundle named by the source batch; default this release')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('video', 'render', 'both'), default='both')
    parser.add_argument('--episodes', type=int, help='Parent count K per selected mode/preset; default all source episodes')
    parser.add_argument('--video-presets', nargs='+', choices=PRESETS, default=['darker'])
    parser.add_argument('--workers', type=int, default=2, help='CPU video encoders,1 to4')
    parser.add_argument('--max-attempts-per-parent', type=int, default=5, help='Finite native render attempt budget per parent')
    parser.add_argument('--isaac-python', type=Path, default=os.environ.get('NEPES_SIM_PYTHON'))
    parser.add_argument('--dataset-python', type=Path, default=os.environ.get('NEPES_DATASET_PYTHON', sys.executable))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = arguments()
    bundle = args.bundle.expanduser().resolve(strict=True)
    batch, output = args.source_batch.expanduser().resolve(strict=True), args.output.expanduser().resolve()
    require(all(not output.is_relative_to(protected) and not protected.is_relative_to(output)
                for protected in (bundle, DEFAULT_BUNDLE, batch)), 'Augmentation output must be separate from source batch/bundles')
    require(1 <= args.workers <= 4 and args.max_attempts_per_parent > 0, 'Invalid worker/attempt budget')
    require(len(set(args.video_presets)) == len(args.video_presets), 'Duplicate video preset')
    source = complete_source(batch, bundle, args.episodes)
    source_state = read(batch / 'batch.json'); source_plan = source_state['plan']
    isaac = args.isaac_python or Path(source_plan['isaac_python'])
    isaac, dataset = str(isaac.expanduser().absolute()), str(args.dataset_python.expanduser().absolute())
    require(Path(dataset).is_file(), 'Dataset Python is missing')
    if args.mode in ('render', 'both'):
        require(Path(isaac).is_file(), 'Isaac Python is missing')
    count = source['selected_episodes']
    plan = {'schema': 'nepes_portable_augmentation_plan_v1', 'source': source,
            'source_plan': source_plan, 'bundle': str(bundle), 'output': str(output),
            'mode': args.mode, 'episodes_per_mode_preset': count, 'video_presets': args.video_presets,
            'workers': args.workers, 'max_attempts_per_parent': args.max_attempts_per_parent,
            'isaac_python': isaac, 'dataset_python': dataset,
            'tools_sha256': {p.name: sha(p) for p in sorted(TOOLS.glob('*.py'))},
            'source_tools_sha256': source_plan['tools_sha256'],
            'target_video_episodes': count * len(args.video_presets) if args.mode in ('video', 'both') else 0,
            'target_render_episodes': count if args.mode in ('render', 'both') else 0,
            'split_policy': 'Original and every visual derivative share seeded_motion_<parent exact motion SHA>. Split by this group.'}
    if args.dry_run:
        print(json.dumps({'writes_performed': False, 'plan': plan,
                          'note': 'Execution fully verifies source native archives and merged data before creating derivatives.'}, indent=2))
        return 0
    if args.resume:
        require((output / 'state.json').is_file(), 'No augmentation state to resume')
    else:
        require(not output.exists(), 'Output exists; use --resume with the same plan')
    with ExitStack() as locks:
        global_lock = locks.enter_context((Path(tempfile.gettempdir()) / f'igris-nepes-{os.getuid()}.lock').open('a'))
        fcntl.flock(global_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch_lock = locks.enter_context((batch / '.batch.lock').open('r'))
        fcntl.flock(batch_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not process_tools.native_pids(), 'Another Isaac simulation is running')
        require(sha(batch / 'batch.json') == source['parent_batch_sha256'], 'Source batch changed before lock acquisition')
        source_process = read(batch / 'process.json')
        require(not process_tools.alive(source_process.get('owned_processes', {})), 'Source still owns active children')
        if args.resume:
            state = read(output / 'state.json'); require(state['plan'] == plan, 'Source, code or augmentation plan changed')
            require(not process_tools.alive(state.get('owned_processes', {})), 'Prior augmentation child remains alive')
        else:
            output.mkdir(parents=True)
            state = {'schema': 'nepes_portable_augmentation_batch_v1', 'plan': plan, 'created_utc': now(), 'status': 'prepared'}
        output_lock = locks.enter_context((output / '.augmentation.lock').open('a'))
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume and (output / 'STOP').exists():
            (output / 'STOP').rename(output / f'STOP_resumed_{time.time_ns()}')
        stop, owned = [False], {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.__setitem__(0, True))
        def save():
            state.update(updated_utc=now(), pid=os.getpid(), owned_processes=process_tools.alive(owned))
            write(output / 'state.json', state)
        def run(label, command, fds=()):
            if stop[0] or (output / 'STOP').exists(): raise InterruptedError('STOP requested before next job')
            require(sha(batch / 'batch.json') == source['parent_batch_sha256'], 'Source batch changed')
            require(all(sha(TOOLS / name) == expected for name, expected in plan['tools_sha256'].items()), 'Augmentation tools changed')
            state.update(status=label, active={'label': label, 'command': command}); save()
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            env.pop('PYTHONPATH', None)
            with (output / (label + '.log')).open('a') as log:
                child = subprocess.Popen(command, cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=fds)
                owned.update(process_tools.descendants(child.pid))
                state['active']['pid'] = child.pid; save()
                started = time.monotonic(); stopping = None
                while child.poll() is None:
                    owned.update(process_tools.descendants(child.pid))
                    if (output / 'STOP').exists(): stop[0] = True
                    if stop[0]:
                        if stopping is None:
                            stopping = time.monotonic(); process_tools.send(owned, signal.SIGINT)
                        elapsed = time.monotonic() - stopping
                        if elapsed > 170: process_tools.send(owned, signal.SIGKILL)
                        elif elapsed > 150: process_tools.send(owned, signal.SIGTERM)
                    if time.monotonic() - started > 30:
                        save(); started = time.monotonic()
                    try: child.wait(timeout=.5)
                    except subprocess.TimeoutExpired: pass
                require(not process_tools.alive(owned), 'Augmentation descendants outlived launcher')
                state['active'] = None; save()
                if stop[0]: raise InterruptedError('Cooperative augmentation stop')
                require(child.returncode == 0, f'{label} failed; inspect {output / (label + ".log")}')
        try:
            save()
            merger = bundle / 'scripts/seeded/merge_seeded.py'
            run('verify_source', [dataset, '-B', str(merger), 'merge', '--batch', str(batch),
                                 '--output', source['parent_merged_dataset'],
                                 *source_quota_arguments(source_plan),
                                 '--verify-existing', '--batch-lock-fd', str(batch_lock.fileno())], (batch_lock.fileno(),))
            results = {}
            if args.mode in ('video', 'both'):
                selected = selected_video_source(bundle, source, output / 'selected_parents')
                video_output = output / 'video'
                argv = [dataset, '-B', str(TOOLS / 'video_collect.py'), '--source', str(selected), '--output', str(video_output),
                        '--expected-episodes', str(count), '--workers', str(args.workers), '--execute',
                        '--presets', *[PRESETS[name] for name in args.video_presets]]
                if (video_output / 'state.json').exists(): argv.append('--resume')
                run('video', argv)
                result = read(video_output / 'state.json')
                require(result['status'] == 'complete' and result['total_augmented_episodes'] == plan['target_video_episodes'], 'Video quota incomplete')
                results['video'] = result['results']
            if args.mode in ('render', 'both'):
                render_output = output / 'render'
                argv = [dataset, '-B', str(TOOLS / 'render_collect.py'), '--batch', str(batch), '--bundle', str(bundle),
                        '--output', str(render_output), '--episodes', str(count), '--max-attempts-per-parent', str(args.max_attempts_per_parent),
                        '--isaac-python', isaac, '--dataset-python', dataset, '--execute', '--lock-fd', str(global_lock.fileno())]
                if (render_output / 'state.json').exists(): argv.append('--resume')
                run('render', argv, (global_lock.fileno(),))
                result = read(render_output / 'state.json')
                require(result['status'] == 'complete' and result['accepted'] == count, 'Render quota incomplete')
                results['render'] = {'dataset': result['dataset'], 'episodes': count, 'new_physics_execution': True}
            state.update(status='complete', results=results, finished_utc=now(), active=None); save()
            print(json.dumps({'status': 'complete', 'results': results,
                              'total_augmented_episodes': plan['target_video_episodes'] + plan['target_render_episodes']}, indent=2))
            return 0
        except BaseException as exc:
            for pid in list(owned):
                if process_tools.process(pid) == owned[pid]:
                    owned.update(process_tools.descendants(int(pid)))
            process_tools.send(owned, signal.SIGINT)
            started = time.monotonic()
            while process_tools.alive(owned) and time.monotonic() - started < 175:
                elapsed = time.monotonic() - started
                if elapsed > 170: process_tools.send(owned, signal.SIGKILL)
                elif elapsed > 150: process_tools.send(owned, signal.SIGTERM)
                time.sleep(.5)
            state.update(status='stopped' if isinstance(exc, InterruptedError) else 'needs_review',
                         error=f'{type(exc).__name__}: {exc}', active=None); save()
            if isinstance(exc, InterruptedError): return 130
            raise


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f'Augmentation stopped: {exc}') from exc
