#!/usr/bin/env python3
"""Collect independent LEFT 2→2 and RIGHT 4→4 episodes with seeded robot starts."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

BUNDLE = Path(__file__).resolve().parents[1]
# Resource gate of scripts/seeded/collect_parallel.py; forwarded only with --parallel 2 or 3.
GATE_OPTIONS = (('--min-free-vram-gib', float), ('--kit-vram-gib', float), ('--min-available-ram-gib', float),
                ('--kit-ram-gib', float), ('--disk-reserve-gib-per-attempt', float),
                ('--launch-spacing-seconds', float), ('--warmup-seconds', float), ('--gpu-index', int),
                ('--max-idle-wait-seconds', float), ('--foreign-kit', str))


def relative_ik_python():
    """Error text when NEPES_IK_PYTHON is relative, else None.

    The batch preflight resolves it against the current directory, but every Isaac
    attempt starts in the release folder, where the same relative path can name
    another interpreter than the one checked and frozen in the batch plan.
    """
    value = os.environ.get('NEPES_IK_PYTHON', '').strip()
    if value and not os.path.isabs(os.path.expanduser(value)):
        return (f'NEPES_IK_PYTHON must be an absolute path (env.sh exports one), not {value!r}: Isaac attempts start '
                'in the release folder, where a relative path can name another interpreter than the one checked and frozen')
    return None


def frozen_parallel(output):
    """Concurrency frozen in an existing batch: 1 when serial, None without a readable batch."""
    try:
        plan = json.loads((output.expanduser() / 'batch.json').read_text())['plan']
        return plan['parallel']['slots'] if 'parallel' in plan else 1
    except (OSError, ValueError, KeyError, TypeError):
        return None


def parallel_status(output, state):
    """Extra --status fields for a parallel batch (slots, gate and per-status record counts)."""
    records = {}
    for record in state.get('attempts', []):
        profile = records.setdefault(record['profile'], {})
        profile[record['status']] = profile.get(record['status'], 0) + 1
    summary = {'parallel': state['plan']['parallel'].get('slots'), 'records': records,
               'stop_mode': state.get('stop_mode'), 'draining': state.get('draining')}
    process = output.expanduser() / 'process.json'
    if process.exists():
        document = json.loads(process.read_text())
        summary.update(slots=document.get('slots'), gate=document.get('gate'), process_updated_utc=document.get('updated_utc'))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-pair', type=int, help='Equal accepted quota per pair; default 25; exclusive with left/right-successes')
    parser.add_argument('--left-successes', type=int, help='Exact LEFT quota; also provide --right-successes')
    parser.add_argument('--right-successes', type=int, help='Exact RIGHT quota; also provide --left-successes')
    parser.add_argument('--first-seed', type=int, default=0, help='First seed on this PC; allocate disjoint attempt ranges across PCs')
    parser.add_argument('--output', type=Path, help='New batch outside the bundle; required with --resume')
    parser.add_argument('--max-attempts-per-pair', type=int, help='Finite attempt budget for each pair; default 3 × per-pair')
    parser.add_argument('--max-wall-seconds', type=float, default=5400, help='Maximum wall time for a native attempt')
    parser.add_argument('--stop-after-total-successes', type=int, help='Optional resumable checkpoint; does not change final quota')
    parser.add_argument('--isaac-python', type=Path, default=os.environ.get('NEPES_SIM_PYTHON', str(BUNDLE / '.venv/sim/bin/python')))
    parser.add_argument('--dataset-python', type=Path, default=os.environ.get('NEPES_DATASET_PYTHON', str(BUNDLE / '.venv/dataset/bin/python')))
    parser.add_argument('--resume', action='store_true', help='Resume exactly the same batch plan after a checkpoint or STOP')
    parser.add_argument('--dry-run', action='store_true', help='Validate immutable sources and print the plan without launching Isaac')
    parser.add_argument('--status', action='store_true', help='Read saved progress without starting collection')
    parser.add_argument('--parallel', type=int, help='Concurrent Isaac attempts on this PC: 1 (default, serial), 2 or 3; '
                        'frozen in the batch, so --resume may omit it')
    for flag, kind in GATE_OPTIONS:
        if flag == '--foreign-kit':
            parser.add_argument(flag, choices=('wait', 'fail'), help='--parallel only: wait (default) or stop when another Isaac runs')
        else:
            parser.add_argument(flag, type=kind, help='--parallel only; see scripts/seeded/collect_parallel.py --help')
    parser.add_argument('--interrupt-unfinished', action='store_true',
                        help='--parallel --resume only: record attempts whose native never reported an exit '
                             '(it ended with the collection manager, e.g. a PC restart) as interrupted')
    args = parser.parse_args(argv)
    if (args.resume or args.status) and args.output is None:
        parser.error('--output is required with --resume/--status')
    if (args.left_successes is None) != (args.right_successes is None):
        parser.error('--left-successes and --right-successes must be supplied together')
    if args.left_successes is not None and args.per_pair is not None:
        parser.error('--per-pair cannot be combined with per-profile quotas')
    if args.parallel is not None and not 1 <= args.parallel <= 3:
        parser.error('--parallel must be 1, 2 or 3')
    output = args.output or BUNDLE.parent / 'runs' / datetime.now().strftime('collection_%Y%m%d_%H%M%S')
    if args.status:
        state = json.loads((output.expanduser() / 'batch.json').read_text())
        summary = {k: state.get(k) for k in ('status', 'counts', 'active', 'dataset', 'error', 'updated_utc')}
        if isinstance(state.get('plan', {}).get('parallel'), dict):
            summary.update(parallel_status(output, state))
        print(json.dumps(summary, indent=2))
        return 0
    if problem := relative_ik_python():
        parser.error(problem)
    frozen = frozen_parallel(output) if args.resume else None
    if frozen is not None and args.parallel is not None and args.parallel != frozen:
        parser.error(f'This batch was started with --parallel {frozen}; resume with that value or omit --parallel')
    parallel = args.parallel or frozen or 1
    gate = [(flag, getattr(args, flag[2:].replace('-', '_'))) for flag, _ in GATE_OPTIONS
            if getattr(args, flag[2:].replace('-', '_')) is not None]
    if gate and parallel == 1:
        parser.error('Resource gate options apply only to --parallel 2 or 3')
    if args.interrupt_unfinished and (parallel == 1 or not args.resume):
        parser.error('--interrupt-unfinished applies only to --resume of a --parallel batch')
    # N=1 keeps the unchanged serial engine and exactly the previous command.
    script = BUNDLE / 'scripts/seeded' / ('collect_seeded.py' if parallel == 1 else 'collect_parallel.py')
    command = [sys.executable, '-B', str(script), '--bundle', str(BUNDLE), '--output', str(output),
               '--isaac-python', str(args.isaac_python), '--dataset-python', str(args.dataset_python),
               '--first-seed', str(args.first_seed),
               '--max-wall-seconds', str(args.max_wall_seconds)]
    if args.left_successes is None:
        command.extend(['--max-successes-per-pair', str(args.per_pair if args.per_pair is not None else 25)])
    else:
        command.extend(['--left-successes', str(args.left_successes), '--right-successes', str(args.right_successes)])
    for key in ('max_attempts_per_pair', 'stop_after_total_successes'):
        if getattr(args, key) is not None:
            command.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
    for key in ('resume', 'dry_run'):
        if getattr(args, key):
            command.append('--' + key.replace('_', '-'))
    if parallel > 1:
        command.extend(['--parallel', str(parallel)])
        for flag, value in gate:
            command.extend([flag, str(value)])
        if args.interrupt_unfinished:
            command.append('--interrupt-unfinished')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    os.execve(sys.executable, command, environment)


if __name__ == '__main__':
    raise SystemExit(main())
