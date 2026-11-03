#!/usr/bin/env python3
"""Validate or visually replay recorded measured states (no physics execution).

Examples (after sourcing .venv/env.sh):
  python3 scripts/replay.py --attempt /data/attempt_000000_seed_000101 --validate
  python3 scripts/replay.py --attempt /data/attempt_000000_seed_000101 \
      --render --views all --duration-seconds 5 --output /data/replay_preview

Keep collection/runs/*/{run.json,config.json,robot_model/,<episode>/} when
copying an attempt. trace.jsonl.gz is streamed directly. Source machine paths
are ignored: geometry is resolved from --bundle (default: this package).
For native lighting/material augmentation, also retain the trial's profile/
folder; its verified visual overrides are restored on the packaged scene.
This is measured-state playback, not action replay or a new successful rollout.
Original MP4s remain the only pixel-exact replay of recorded camera images.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def dispatch(args):
    """Select the installed environment before importing NumPy or Isaac Lab.

    Keep the invoked venv path, not its resolved interpreter symlink: several
    distinct venvs can share the same underlying Python executable.
    """
    bundle = args.bundle.expanduser().resolve()
    args.dataset_python = (args.dataset_python or bundle / '.venv/dataset/bin/python').expanduser().absolute()
    args.isaac_python = (args.isaac_python or bundle / '.venv/sim/bin/python').expanduser().absolute()
    executable = args.isaac_python if args.render else args.dataset_python
    if not executable.is_file() or not os.access(executable, os.X_OK):
        name = 'NEPES_SIM_PYTHON / --isaac-python' if args.render else 'NEPES_DATASET_PYTHON / --dataset-python'
        raise ValueError('Set ' + name + ' to the installed environment Python: ' + str(executable))
    marker = json.dumps([str(Path(__file__).resolve()), str(executable), 'render' if args.render else 'validate'])
    if os.environ.get('_NEPES_REPLAY_DISPATCH') == marker:
        if not os.path.samefile(sys.executable, executable):
            raise ValueError('Replay environment dispatch did not reach the requested Python')
        return
    if Path(sys.executable).absolute() == executable:
        return
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                       _NEPES_REPLAY_DISPATCH=marker)
    os.execve(str(executable), [str(executable), '-B', str(Path(__file__).resolve()), *sys.argv[1:]], environment)


def write(path, document):
    with Path(path).open('x') as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--attempt', type=Path, required=True, help='One outer collection attempt or one native run directory')
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--episode', help='Recorded episode ID; optional for single-episode runs')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate', action='store_true', help='CPU only: check every recorded pose/joint row and FK')
    mode.add_argument('--render', action='store_true', help='Isaac visual replay while physics timeline remains stopped')
    parser.add_argument('--output', type=Path, help='New output folder outside the input and package')
    parser.add_argument('--report', type=Path, help='Optional new validation JSON outside the source/package')
    parser.add_argument('--views', choices=('third_person', 'all'), default='third_person')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--start-seconds', type=float, default=0., help='Offset from the first retained episode trace row')
    parser.add_argument('--duration-seconds', type=float, help='Default: replay the full retained trace')
    parser.add_argument('--gui', action='store_true', help='Show an Isaac viewport during visual playback')
    parser.add_argument('--warmup-frames', type=int, default=40)
    parser.add_argument('--render-updates', type=int, default=4, help='Render updates per restored state; never physics steps')
    parser.add_argument('--dataset-python', type=Path,
                        default=os.environ.get('NEPES_DATASET_PYTHON'), help='Validation/PyAV environment; default from env.sh')
    parser.add_argument('--isaac-python', type=Path,
                        default=os.environ.get('NEPES_SIM_PYTHON'), help='Visual replay environment; default from env.sh')
    args = parser.parse_args()
    dispatch(args)
    from replay_support.trace import discover, read_states, schedule, validate, require
    require(args.warmup_frames >= 1 and args.render_updates >= 1, 'Render update counts must be positive')
    if args.render:
        require(args.output is not None, '--render requires a new --output directory')
    started = time.monotonic()
    source = discover(args.attempt, args.bundle, args.episode)
    for output in (args.output, args.report):
        if output is not None:
            destination = output.expanduser().resolve()
            require(not destination.exists(), 'Refusing to overwrite existing output: ' + str(destination))
            require(not destination.is_relative_to(source['attempt']) and not source['attempt'].is_relative_to(destination),
                    'Replay output and original attempt must not overlap')
            require(not destination.is_relative_to(source['bundle']), 'Write replay output outside the immutable package')
    print('Reading retained measured states; no simulation is started.', flush=True)
    states, decoded_digest = read_states(source)
    selected = schedule(states, args.fps, args.start_seconds, args.duration_seconds)
    checks = validate(source, states, selected)
    plan = {'schema': 'nepes_measured_state_visual_replay_v1', 'profile': source['profile'],
            'source_attempt': str(source['attempt']), 'source_run': str(source['run']),
            'source_episode_id': source['episode']['episode_id'],
            'source_reported_success': source['episode'].get('success'),
            'source_files_sha256': source['sources'], 'trace_decoded_sha256': decoded_digest,
            'storage_trace_binding_verified': source['storage_trace_binding_verified'],
            'bundle': str(source['bundle']), 'old_absolute_paths_used': False,
            'execution_python': sys.executable,
            'visual_overlay': source.get('visual_overlay'),
            'checks': checks, 'fps': args.fps, 'views': args.views,
            'validation_wall_seconds': time.monotonic() - started,
            'native_physics_reexecuted': False, 'action_replay': False, 'new_success_claim': False,
            'original_rgb_reproduced_exactly': False,
            'limitations': [
                'Measured 50Hz trace states are selected by nearest sample at video timestamps; no state interpolation.',
                'Trace states are not synchronized to every original 30Hz image; original MP4s are the exact recorded images.',
                'Kinematic link poses are reconstructed from recorded base/joints and checked against both measured TCPs.',
                'Contact forces and controller history are not simulated; this cannot establish a new physical success.',
                'Only recorded dynamic object poses are restored; unrecorded dynamic state cannot be recovered.',
                'Native visual overlays restore their recorded lighting/material values; offline video tone transforms require their saved MP4.',
            ]}
    if args.report:
        args.report = args.report.expanduser().resolve()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write(args.report, plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.validate:
        return 0
    args.output = args.output.expanduser().resolve()
    require(args.dataset_python.is_file() and os.access(args.dataset_python, os.X_OK),
            'Set NEPES_DATASET_PYTHON or --dataset-python to the installed PyAV dataset Python')
    # Native render takes the same host-wide lock as collection before loading Kit.
    from replay_support.render import render
    return render(args, source, states, selected, plan)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit('Replay stopped: ' + str(error)) from error
