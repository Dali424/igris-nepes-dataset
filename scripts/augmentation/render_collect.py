#!/usr/bin/env python3
"""After a verified source merge, collect separate native visual variations.

Default is a read-only plan. --execute requires a complete source and selected parents, their completed
merge, an exclusive global simulator lock and no live Kit. First four children
cover both arms/conditions and must pass automated physical/media/visual gates.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from render_scene import CASES, read, require, sha, write

TOOLS = Path(__file__).resolve().parent
RECEIPT = Path('meta/nepes_storage_verification.json')

def now():
    return datetime.now(timezone.utc).isoformat()

def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None

def descendants(pid):
    result, pending = {}, [int(pid)]
    while pending:
        current = pending.pop(); started = process(current)
        if started is None:
            continue
        result[str(current)] = started
        try:
            pending.extend(map(int, Path(f'/proc/{current}/task/{current}/children').read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result

def alive(owned):
    return {pid: started for pid, started in owned.items() if process(pid) == started}

def send(owned, sig):
    for pid in reversed(list(alive(owned))):
        try:
            os.kill(int(pid), sig)
        except ProcessLookupError:
            pass

def native_pids():
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            maps = (proc / 'maps').read_text()
            if 'libomni.kit.app.plugin.so' in maps or 'libcarb.app.plugin.so' in maps:
                result.append(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return result

def check_inventory(root, receipt):
    require(receipt.get('verified') is True, 'Unverified dataset')
    expected = {item['path']: item for item in receipt['files']}
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() and path.relative_to(root) != RECEIPT}
    require(actual == set(expected), 'Dataset file inventory differs')
    for relative, item in expected.items():
        path = root / relative
        require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Unsafe dataset file')
        require(path.stat().st_size == item['size_bytes'] and sha(path) == item['sha256'], 'Dataset file changed: ' + relative)

def parent_plan(batch, bundle, execute, episodes=None):
    from source_selection import complete_source
    sources = complete_source(batch, bundle, episodes)
    for parent in sources['parents']:
        parent['case'] = tuple(CASES)[(parent['index'] // 2) % len(CASES)]
    return sources

def command(plan, parent, child):
    runtime = Path(plan['bundle']) / 'runtime' / parent['profile']
    return [plan['isaac_python'], '-B', '-u', str(runtime / 'scripts/nepes/collect_stationary.py'),
            '--catalog', 'v2', '--task', '1', '--execute', '--config', str(child / 'profile/config.json'),
            '--successes', '1', '--max-attempts', '1', '--seed', str(parent['seed']),
            '--output', str(child / 'collection'), '--export', str(child / 'dataset'),
            '--isaac-python', plan['isaac_python'], '--dataset-python', plan['dataset_python'],
            '--max-wall-seconds', '5400', '--storage-mode', 'compact', '--physics-trace', 'gzip',
            '--failed-physics-trace', 'gzip', '--min-free-gib', '20',
            '--capture-render-mode', 'capture_only', '--image-writer-workers', '3', '--image-writer-queue', '8', '--profile-throughput']

def upper_body_ik_plan(bundle, sim_python):
    """Pink release preflight before any render output or Isaac start; None if the bundle declares no IK backend.

    The returned IK backend/worker identity is frozen in the render plan (scripts/check_release.py batch_preflight).
    """
    manifest = bundle / 'manifest.json'
    if not manifest.is_file() or 'ik_backend' not in read(manifest):
        return None
    spec = importlib.util.spec_from_file_location('nepes_release_preflight', bundle / 'scripts/check_release.py')
    checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
    return checker.batch_preflight(sim_python=sim_python)

def accepted_native(child):
    path = child / 'collection/collection.json'
    if not path.exists():
        return None
    trials = read(path).get('attempts', [])
    if len(trials) != 1:
        return None
    trial = trials[0]
    if trial.get('accepted') is True and trial.get('storage_complete') is True and trial.get('returncode') == 0:
        return True
    if trial.get('accepted') is False and trial.get('returncode') is not None and not trial.get('stop_requested'):
        return False
    return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('batch', 'bundle', 'output', 'isaac-python', 'dataset-python'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--max-attempts-per-parent', type=int, default=5)
    parser.add_argument('--lock-fd', type=int, help='Inherited exclusive global simulator lock from postcompletion orchestrator')
    args = parser.parse_args()
    require(not args.resume or args.execute, '--resume requires --execute')
    batch, bundle, output = args.batch.resolve(strict=True), args.bundle.resolve(strict=True), args.output.resolve()
    require(not output.is_relative_to(bundle) and not bundle.is_relative_to(output) and not output.is_relative_to(batch) and not batch.is_relative_to(output), 'Outputs must be separate from immutable parent/bundle')
    require(args.max_attempts_per_parent > 0, 'Attempt budget must be positive')
    sources = parent_plan(batch, bundle, args.execute, args.episodes)
    source_tools = read(batch / 'batch.json')['plan']['tools_sha256']
    plan = {'schema': 'nepes_native_render_plan_v1', **sources, 'bundle': str(bundle), 'output': str(output),
            'isaac_python': str(args.isaac_python.absolute()), 'dataset_python': str(args.dataset_python.absolute()),
            'conditions': CASES, 'required_successes': sources['selected_episodes'], 'max_attempts_per_parent': args.max_attempts_per_parent,
            'parent_collection_tools_sha256': source_tools,
            'pilot_first_children': min(4, sources['selected_episodes']), 'source_files_sha256': {path.name: sha(path) for path in sorted(TOOLS.glob('*.py'))},
            'semantics': 'Fresh native physics with child actions/states; only initial condition linked to parent.Visual variants reuse parent starting conditions; they are not new sampled placements.'}
    ik_plan = upper_body_ik_plan(bundle, plan['isaac_python'])
    if ik_plan is not None:
        plan['upper_body_ik'] = ik_plan
    if not args.execute:
        import json
        print(json.dumps(plan, indent=2)); return 0
    global_path = Path(tempfile.gettempdir()) / f'igris-nepes-{os.getuid()}.lock'
    if args.lock_fd is not None:
        a, b = os.fstat(args.lock_fd), global_path.stat()
        require((a.st_dev, a.st_ino) == (b.st_dev, b.st_ino), 'Inherited fd is not the simulator global lock')
        fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        global_stream = None
    else:
        global_stream = global_path.open('a'); fcntl.flock(global_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not native_pids(), 'Another native Kit is running; rendering collection must wait')
    if args.resume:
        require((output / 'state.json').is_file(), 'Missing render batch to resume')
    else:
        require(not output.exists(), 'Output exists; explicitly --resume or choose another directory')
        output.mkdir(parents=True)
    lock = (output / '.render.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = output / 'state.json'
    if args.resume:
        state = read(path)
        require(state['plan'].get('upper_body_ik') == plan.get('upper_body_ik'), 'Resume upper-body IK backend/worker differs from the frozen render plan; use the same NEPES_IK_BACKEND/NEPES_IK_PYTHON and IK environment')
        require(state['plan'] == plan, 'Resume source/config/tools changed')
        require(state['status'] != 'needs_review', 'Failed automated gate requires investigation; no automatic bypass')
        require(not alive(state.get('owned_processes', {})), 'Prior owned child process remains alive')
        if (output / 'STOP').exists():
            (output / 'STOP').rename(output / f'STOP_resumed_{time.time_ns()}')
    else:
        state = {'schema': 'nepes_native_render_batch_v1', 'plan': plan, 'status': 'prepared', 'created_utc': now(),
                 'jobs': [{**parent, 'status': 'pending', 'attempts': []} for parent in sources['parents']]}
    stop, owned = [False], {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    environment.pop('PYTHONPATH', None)

    def save():
        state.update(updated_utc=now(), pid=os.getpid(), accepted=sum(job['status'] == 'accepted' for job in state['jobs']),
                     counts={profile: sum(job['status'] == 'accepted' and job['profile'] == profile for job in state['jobs']) for profile in ('left2', 'right4')},
                     owned_processes=alive(owned))
        write(path, state)

    def frozen():
        require(sha(batch / 'batch.json') == plan['parent_batch_sha256'], 'Parent batch changed')
        require(sha(Path(plan['parent_merged_dataset']) / RECEIPT) == plan['parent_merge_receipt_sha256'], 'Parent merge receipt changed')
        require(all(sha(TOOLS / name) == value for name, value in plan['source_files_sha256'].items()), 'Rendering worker code changed')
        require(all(sha(bundle / 'scripts/seeded' / name) == value for name, value in plan['parent_collection_tools_sha256'].items()), 'Parent collection verifier changed')
        for parent in plan['parents']:
            require(sha(parent['parent_verification']) == parent['parent_verification_sha256'], 'Parent evidence changed')

    def run(label, argv, child=None, profile=None):
        frozen(); require(not native_pids(), 'Unexpected concurrent Kit')
        env = dict(environment)
        if profile:
            runtime = bundle / 'runtime' / profile
            env['PYTHONPATH'] = os.pathsep.join(str(runtime / 'source' / name) for name in ('robros_motion_planning', 'robros_lab'))
        log = (child / (label + '.log')) if child else output / (label + '.log')
        with log.open('a') as stream:
            proc = subprocess.Popen(argv, cwd=bundle, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            owned.update(descendants(proc.pid)); state.update(active={'label': label, 'pid': proc.pid, 'command': argv}); save()
            last = time.monotonic(); stopping = None
            while proc.poll() is None:
                owned.update(descendants(proc.pid))
                if (output / 'STOP').exists():
                    stop[0] = True
                if stop[0]:
                    if stopping is None:
                        stopping = time.monotonic()
                        if child and (child / 'collection').is_dir():
                            (child / 'collection/STOP').touch(exist_ok=True)
                        send(owned, signal.SIGINT)
                    elapsed = time.monotonic() - stopping
                    if elapsed > 170:
                        send(owned, signal.SIGKILL)
                    elif elapsed > 150:
                        send(owned, signal.SIGTERM)
                if time.monotonic() - last >= 30:
                    save(); last = time.monotonic()
                try:
                    proc.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    pass
            require(not alive(owned), 'Owned native/export descendant outlived launcher')
            state['active'] = None; save()
            return 130 if stop[0] else proc.returncode

    def finalize(job, trial):
        child = Path(trial['directory']); derivative = child / 'augmented_dataset'
        if not derivative.exists():
            code = run('verify', [plan['dataset_python'], '-B', '-u', str(TOOLS / 'render_verify.py'), '--bundle', str(bundle),
                       '--child', str(child), '--parent-verification', job['parent_verification'], '--output', str(derivative)], child)
            if stop[0]:
                return False
            require(code == 0, 'Native rendering verification/appearance gate failed; inspect verify.log')
        receipt = read(derivative / RECEIPT); check_inventory(derivative, receipt)
        audit = read(derivative / 'meta/render_augmentation.json')
        require(audit['verified'] and audit['physical_success_revalidated'] and audit['parent_verification_sha256'] == job['parent_verification_sha256'] and audit['case'] == job['case'], 'Published augmentation lineage differs')
        trial.update(status='accepted', derivative=str(derivative), receipt_sha256=sha(derivative / RECEIPT))
        job.update(status='accepted', derivative=str(derivative), receipt_sha256=sha(derivative / RECEIPT),
                   split_group=audit['split_group'], parent_motion_exact_sha256=audit['parent_motion_exact_sha256']); save(); return True

    def classify_failure(job, trial):
        child = Path(trial['directory'])
        code = run('verify_failure', [plan['dataset_python'], '-B', str(TOOLS / 'render_verify.py'),
                   '--bundle', str(bundle), '--child', str(child), '--parent-verification', job['parent_verification'],
                   '--output', str(child / 'failure_verification.json'), '--failure-returncode', str(trial['returncode'])], child)
        if stop[0]: return False
        require(code == 0, 'Unclassified native crash/storage/integrity error; automatic retry is forbidden')
        evidence = read(child / 'failure_verification.json')
        require(evidence.get('retryable_physical_failure') is True and evidence['seed'] == job['seed']
                and evidence['profile'] == job['profile'] and evidence['case'] == job['case']
                and evidence['parent_verification_sha256'] == job['parent_verification_sha256'], 'Failure identity differs')
        trial.update(status='failed', failure_verification_sha256=sha(child / 'failure_verification.json'),
                     reason=evidence['reason']); save(); return True

    try:
        save()
        for job in state['jobs']:
            if job['status'] == 'accepted':
                continue
            require(sha(Path(job['parent_attempt']) / 'profile/config.json') == job['parent_config_sha256'], 'Parent config changed')
            for trial in job['attempts']:
                if trial['status'] == 'failed':
                    if not classify_failure(job, trial): break
                if trial['status'] == 'running':
                    if accepted_native(Path(trial['directory'])) is True:
                        if not finalize(job, trial):
                            break
                    elif accepted_native(Path(trial['directory'])) is False and 'returncode' in trial:
                        if not classify_failure(job, trial): break
                    else:
                        require(False, 'Missing normal outcome from interrupted/crashed child; review required before retry')
            while job['status'] != 'accepted' and not stop[0]:
                require(len(job['attempts']) < plan['max_attempts_per_parent'], 'Finite attempt budget exhausted for this parent; accepted derivatives remain preserved')
                child = output / 'children' / f'episode_{job["index"]:06d}' / f'trial_{len(job["attempts"]):02d}'
                child.mkdir(parents=True)
                trial = {'directory': str(child), 'status': 'running', 'started_utc': now()}
                job['attempts'].append(trial); state['status'] = 'pilot' if job['index'] < plan['pilot_first_children'] else 'collecting'; save()
                code = run('prepare', [plan['isaac_python'], '-B', str(TOOLS / 'render_scene.py'), '--parent-config',
                           str(Path(job['parent_attempt']) / 'profile/config.json'), '--bundle', str(bundle), '--output', str(child / 'profile'), '--case', job['case']], child)
                if stop[0]:
                    trial['status'] = 'interrupted'; save()
                    break
                require(code == 0, 'Visual scene/config preparation failed')
                code = run('native', command(plan, job, child), child, job['profile'])
                trial.update(returncode=code, finished_utc=now()); save()
                if stop[0]:
                    trial['status'] = 'interrupted'; save()
                    break
                accepted = accepted_native(child)
                if accepted is True:
                    require(code == 0, 'Native success did not complete storage/export normally')
                    finalize(job, trial)
                else:
                    require(accepted is False, 'Missing native outcome; not eligible for automatic retry')
                    classify_failure(job, trial)
            if stop[0]:
                state['status'] = 'stopped'; save(); return 130
            if job['index'] == plan['pilot_first_children'] - 1:
                require(all(j['status'] == 'accepted' for j in state['jobs'][:plan['pilot_first_children']]), 'Initial physical/media/visual gates did not pass')
                state['pilot_passed_utc'] = now(); save()
            print(f'Render condition {job["index"] + 1}/{plan["required_successes"]} verified: {job["profile"]} {job["case"]}', flush=True)
        require(all(j['status'] == 'accepted' for j in state['jobs']), 'Exact rendering quota not met')
        manifest = output / 'render_sources.json'
        write(manifest, {'schema': 'nepes_native_render_sources_v1', 'expected_episodes': plan['required_successes'], 'expected_profile_counts': plan['expected_profile_counts'], 'parent_batch_sha256': plan['parent_batch_sha256'],
                         'semantics': plan['semantics'], 'episodes': state['jobs']})
        target = output / f"dataset_{plan['required_successes']}"; state['status'] = 'merging'; save()
        argv = [plan['dataset_python'], '-B', str(TOOLS / 'render_merge.py'), '--bundle', str(bundle), '--manifest', str(manifest), '--output', str(target)]
        code = run('merge', argv)
        if stop[0]:
            state['status'] = 'stopped'; save(); return 130
        require(code == 0, 'Render merge failed; source derivatives preserved')
        state.update(status='complete', dataset=str(target), finished_utc=now()); save(); return 0
    except BaseException as exc:
        for pid in list(owned):
            if process(pid) == owned[pid]:
                owned.update(descendants(int(pid)))
        send(owned, signal.SIGINT)
        started = time.monotonic()
        while alive(owned) and time.monotonic() - started < 175:
            elapsed = time.monotonic() - started
            if elapsed > 170:
                send(owned, signal.SIGKILL)
            elif elapsed > 150:
                send(owned, signal.SIGTERM)
            time.sleep(.5)
        recoverable_merge = state.get('status') == 'merging'
        state.update(status='stopped' if stop[0] else 'merge_failed' if recoverable_merge else 'needs_review',
                     error=f'{type(exc).__name__}: {exc}', active=None); save()
        if stop[0]:
            return 130
        raise

if __name__ == '__main__':
    raise SystemExit(main())
