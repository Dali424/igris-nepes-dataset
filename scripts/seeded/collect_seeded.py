#!/usr/bin/env python3
"""Serial, resumable collection of distinct seeded robot XY starts.

The source bundle stays immutable. Each attempt records its reviewed private
config revision, sampled start, verified archive and actual reset evidence.
--stop-after-total-successes 4 pauses the plan at a checkpoint; --resume without
that checkpoint continues it. Verified physical failures consume the finite
per-profile attempt budget; unknown failures and integrity errors require review.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

class MergeFailure(RuntimeError):
    pass

PROFILES = ((2, 'left2'), (4, 'right4'))
TOOLS = Path(__file__).resolve().parent
PHYSICAL_FAILURE_CONTINUATION = {
    "enabled": True,
    "eligible_outcomes": "strictly_verified_physical_failures",
    "bound": "per_profile_attempt_budget",
    "reason": "user_requested_continue_until_quota_or_attempt_budget",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)

def module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result

def upper_body_ik_plan(bundle, sim_python=None):
    """Pink release preflight, before any output or Isaac start (also --dry-run).

    Returns the upper-body IK backend/worker identity frozen in the plan, so a
    resume with another backend or IK environment is refused; None when the
    bundle declares no IK backend. See scripts/check_release.py batch_preflight.
    """
    if 'ik_backend' not in read(bundle / 'manifest.json'):
        return None
    spec = importlib.util.spec_from_file_location('nepes_release_preflight', bundle / 'scripts/check_release.py')
    checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
    return checker.batch_preflight(sim_python=sim_python)

def verify_bundle(bundle):
    manifest = read(bundle / 'manifest.json')
    require(manifest.get('schema') == 'igris_nepes_portable_release_v1', 'Wrong bundle manifest')
    for relative, expected in manifest['files'].items():
        path = bundle / relative
        require(not path.is_symlink() and path.resolve().is_relative_to(bundle), 'Bundle path escaped or symlinked: ' + relative)
        require(path.is_file() and path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], 'Immutable bundle file changed: ' + relative)
    return sha(bundle / 'manifest.json')

def process(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return stat[19] if stat[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None

def descendants(pid):
    result, pending = {}, [pid]
    while pending:
        pid = pending.pop(); started = process(pid)
        if started is not None:
            result[str(pid)] = started
            try:
                pending.extend(map(int, Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
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
    """Detect an already loaded native Kit; an outer launcher may not hold flock."""
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            maps = (proc / 'maps').read_text()
            if 'libomni.kit.app.plugin.so' in maps or 'libcarb.app.plugin.so' in maps:
                result.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result

def command_for(plan, profile, seed, attempt):
    runtime = Path(plan['bundle']) / 'runtime' / profile
    return [plan['isaac_python'], '-B', '-u', str(runtime / 'scripts/nepes/collect_stationary.py'),
            '--catalog', 'v2', '--task', '1', '--execute', '--config', str(attempt / 'profile/config.json'),
            '--successes', '1', '--max-attempts', '1', '--seed', str(seed),
            '--output', str(attempt / 'collection'), '--export', str(attempt / 'dataset'),
            '--isaac-python', plan['isaac_python'], '--dataset-python', plan['dataset_python'],
            '--max-wall-seconds', str(plan['max_wall_seconds']), '--storage-mode', 'compact',
            '--physics-trace', 'gzip', '--failed-physics-trace', 'gzip', '--min-free-gib', '20',
            '--capture-render-mode', 'capture_only',
            '--image-writer-workers', '3', '--image-writer-queue', '8', '--profile-throughput']

def counts(state):
    return {profile: sum(a['profile'] == profile and a['status'] == 'accepted' for a in state['attempts'])
            for _, profile in PROFILES}

def quota_for(plan, profile):
    return plan['quota_by_profile'][profile] if 'quota_by_profile' in plan else plan['quota_each']

def attempts_for(plan, profile):
    return plan['max_attempts_by_profile'][profile] if 'max_attempts_by_profile' in plan else plan.get('max_attempts_each', float('inf'))

def next_profile(state):
    """Strict alternating attempts, skipping a profile after its quota is filled."""
    totals = counts(state)
    order = list(PROFILES)
    if state['attempts'] and state['attempts'][-1]['profile'] == 'left2':
        order.reverse()
    return next(((pair, name) for pair, name in order if totals[name] < quota_for(state['plan'], name)
                 and sum(a['profile'] == name for a in state['attempts']) < attempts_for(state['plan'], name)), None)

def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'output', 'isaac-python', 'dataset-python'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--max-successes-per-pair', type=int, help='Equal exact quota; default 25')
    p.add_argument('--left-successes', type=int, help='Exact LEFT quota; requires --right-successes')
    p.add_argument('--right-successes', type=int, help='Exact RIGHT quota; requires --left-successes')
    p.add_argument('--max-attempts-per-pair', type=int, default=None)
    p.add_argument('--first-seed', type=int, default=0)
    p.add_argument('--max-wall-seconds', type=float, default=5400)
    p.add_argument('--stop-after-total-successes', type=int, help='Optional checkpoint; excluded from frozen plan; remove when resuming')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p.parse_args(argv)

def main(argv=None):
    args = arguments(argv)
    bundle = args.bundle.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    require(not output.is_relative_to(bundle) and not bundle.is_relative_to(output), 'Collection output cannot overlap immutable bundle')
    require((args.left_successes is None) == (args.right_successes is None), 'Supply both LEFT and RIGHT quotas')
    require(args.left_successes is None or args.max_successes_per_pair is None, 'Equal and per-profile quota options cannot be mixed')
    quotas = ({'left2': args.left_successes, 'right4': args.right_successes} if args.left_successes is not None
              else {profile: args.max_successes_per_pair if args.max_successes_per_pair is not None else 25 for _, profile in PROFILES})
    require(all(type(q) is int and q > 0 for q in quotas.values()) and 0 <= args.first_seed <= 2_147_483_647, 'Invalid quota or first seed')
    require(args.max_wall_seconds > 0, 'Wall limit must be positive')
    require(args.stop_after_total_successes is None or args.stop_after_total_successes > 0, 'Checkpoint must be positive')
    maxima = {profile: args.max_attempts_per_pair if args.max_attempts_per_pair is not None else 3 * quota for profile, quota in quotas.items()}
    require(all(maxima[name] >= quotas[name] for name in quotas), 'Attempt budget is below quota')
    require(args.first_seed + max(maxima.values()) - 1 <= 2_147_483_647, 'Seed range exceeds native integer limit')
    quota_fields = ({'quota_each': quotas['left2'], 'max_attempts_each': maxima['left2']} if quotas['left2'] == quotas['right4']
                    else {'quota_by_profile': quotas, 'max_attempts_by_profile': maxima})
    sim, dataset = (str(p.expanduser().absolute()) for p in (args.isaac_python, args.dataset_python))
    require(all(Path(p).is_file() and os.access(p, os.X_OK) for p in (sim, dataset)), 'Missing Python executable')
    sampler, merger = module('seeded_config'), module('merge_seeded')
    frozen = {p.name: sha(p) for p in (Path(__file__).resolve(), TOOLS / 'merge_seeded.py', TOOLS / 'seeded_config.py')}
    plan = {'bundle': str(bundle), 'output': str(output), 'isaac_python': sim, 'dataset_python': dataset,
            **quota_fields,
            'first_seed': args.first_seed, 'xy_half_range_by_profile_m': {profile: sampler.effective_xy_range(pair, args.first_seed) for pair, profile in PROFILES},
            'max_wall_seconds': args.max_wall_seconds,
            'physical_failure_continuation': dict(PHYSICAL_FAILURE_CONTINUATION),
            'source_revisions': sampler.revision_plan(),
            'ordering': 'round_robin_serial', 'bundle_manifest_sha256': verify_bundle(bundle),
            'tools_sha256': frozen, 'cameras': 'four_existing_cameras_30fps_8spp',
            'motion_quantization_step': 1e-4, 'initial_xy_quantization_m': 1e-5}
    ik_plan = upper_body_ik_plan(bundle, sim)  # after verify_bundle; before any output or Isaac start
    if ik_plan is not None:
        plan['upper_body_ik'] = ik_plan
    if args.dry_run:
        examples = []
        for pair, profile in PROFILES:
            _, sample = sampler.build_config(bundle, pair, args.first_seed)
            attempt = output / profile / f'attempt_000000_seed_{args.first_seed:06d}'
            examples.append({'profile': profile, 'sample': sample,
                             'command': command_for(plan, profile, args.first_seed, attempt)})
        print(json.dumps({'plan': plan, 'checkpoint_total_successes': args.stop_after_total_successes, 'examples': examples}, indent=2)); return 0
    global_lock = (Path(tempfile.gettempdir()) / f'igris-nepes-{os.getuid()}.lock').open('a')
    fcntl.flock(global_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        require((output / 'batch.json').is_file(), 'Resume batch does not exist')
    else:
        require(not output.exists(), 'Output exists; use --resume with unchanged plan or choose a fresh directory')
        output.mkdir(parents=True)
    batch_lock = (output / '.batch.lock').open('a')
    fcntl.flock(batch_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = output / 'batch.json'
    process_path = output / 'process.json'
    if args.resume:
        state = read(path)
        require(state['plan'].get('upper_body_ik') == plan.get('upper_body_ik'), 'Resume upper-body IK backend/worker differs from the frozen batch plan; use the same NEPES_IK_BACKEND/NEPES_IK_PYTHON and IK environment')
        require(state['plan'] == plan, 'Resume plan/source/scripts differ; never mix incompatible batches')
        require(not alive(read(process_path).get('owned_processes', {})) if process_path.exists() else True, 'Previous owned processes remain alive')
        require(state.get('status') not in ('rejected', 'needs_review'), 'Batch requires investigation; automatic resume is blocked')
        if (output / 'STOP').exists():
            (output / 'STOP').rename(output / f'STOP_resumed_{time.time_ns()}')
    else:
        state = {'schema': 'nepes_seeded_batch_v1', 'plan': plan, 'created_utc': now(), 'attempts': [], 'status': 'prepared'}
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    owned = {}
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    environment.pop('PYTHONPATH', None)

    def save():
        state.update(updated_utc=now(), counts=counts(state), pid=os.getpid())
        write(path, state)

    def check_frozen():
        require(verify_bundle(bundle) == plan['bundle_manifest_sha256'], 'Bundle manifest changed')
        require({name: sha(TOOLS / name) for name in frozen} == frozen, 'Frozen batch tool changed')

    def run(label, command, profile=None, attempt=None, inherited=()):
        nonlocal owned
        check_frozen()
        if profile:
            busy = native_pids()
            require(not busy, f'Another native Kit process is active: {busy}')
        env = environment.copy()
        if profile:
            runtime = bundle / 'runtime' / profile
            env['PYTHONPATH'] = os.pathsep.join(str(runtime / 'source' / part) for part in ('robros_motion_planning', 'robros_lab'))
        log_path = (attempt if attempt else output) / (label + '.log')
        with log_path.open('a', buffering=1) as log:
            log.write(f'\n[{now()}] {json.dumps(command)}\n')
            child = subprocess.Popen(command, cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, pass_fds=inherited)
            owned = descendants(child.pid); stopping = None; last_save = 0.
            while True:
                owned.update(descendants(child.pid))
                requested = stop[0] or (output / 'STOP').exists()
                if requested and stopping is None:
                    stop[0] = True; stopping = time.monotonic()
                    send({str(child.pid): owned[str(child.pid)]} if str(child.pid) in owned else {}, signal.SIGINT)
                    if attempt and (attempt / 'collection').exists():
                        (attempt / 'collection/STOP').touch(exist_ok=True)
                code = child.poll(); remaining = alive(owned)
                if code is not None and not remaining:
                    owned = {}; write(process_path, {'owned_processes': {}, 'updated_utc': now(), 'returncode': code, 'label': label})
                    return code
                if code is not None and stopping is None:
                    stopping = time.monotonic(); send(owned, signal.SIGINT)
                if stopping is not None and time.monotonic() - stopping > 150:
                    send(owned, signal.SIGTERM if time.monotonic() - stopping < 160 else signal.SIGKILL)
                if stopping is not None and time.monotonic() - stopping > 180 and remaining:
                    raise RuntimeError('Owned processes survived cleanup; collection blocked')
                if time.monotonic() - last_save >= 30:
                    write(process_path, {'owned_processes': owned, 'updated_utc': now(), 'label': label, 'log': str(log_path)})
                    print(f'[{now()}] {label}: accepted {counts(state)}; log {log_path}', flush=True)
                    last_save = time.monotonic()
                time.sleep(.25)

    accepted = {name: [] for _, name in PROFILES}
    def verify(record):
        directory = Path(record['directory'])
        command = [dataset, '-B', str(TOOLS / 'merge_seeded.py'), 'verify-attempt', '--bundle', str(bundle), '--attempt', str(directory), '--output', str(directory / 'verification.json')]
        code = run('verify', command, attempt=directory)
        require(code == 0, f'Attempt integrity/reset verification failed; inspect {directory / "verify.log"}')
        evidence = read(directory / 'verification.json')
        require(evidence['profile'] == record['profile'] and evidence['pair'] == record['pair'] and evidence['seed'] == record['seed'] and evidence['xy_half_range_m'] == sampler.effective_xy_range(record['pair'], record['seed'], plan['xy_half_range_by_profile_m'][record['profile']]), 'Verification profile/pair/seed/effective range differs from batch record')
        merger.validate_record_revision(plan, record, read(directory / 'profile/sample.json'))
        merger.reject_duplicates(accepted[record['profile']], evidence)
        accepted[record['profile']].append(evidence)
        record.update(status='accepted', verification_sha256=sha(directory / 'verification.json'),
                      config_sha256=evidence['config_sha256'], config_file_sha256=evidence['config_file_sha256'], sampled_offset_xy_m=evidence['sampled_offset_xy_m'],
                      measured_reset_pose_w=evidence['startup_reset']['measured_pose_w'],
                      measured_initial_world_poses=evidence['initial_measurement']['poses'],
                      motion_exact_sha256=evidence['motion_exact_sha256'], motion_quantized_sha256=evidence['motion_quantized_sha256'])

    def verify_failure(record):
        directory = Path(record['directory'])
        command = [dataset, '-B', str(TOOLS / 'merge_seeded.py'), 'verify-failure', '--bundle', str(bundle),
                   '--attempt', str(directory), '--returncode', str(record['returncode']),
                   '--output', str(directory / 'failure_verification.json')]
        code = run('verify_failure', command, attempt=directory)
        require(code == 0, f'Unclassified failure/crash/storage error; inspect {directory / "verify_failure.log"}')
        evidence = read(directory / 'failure_verification.json')
        require(evidence.get('retryable_physical_failure') is True and evidence.get('success') is False
                and evidence['profile'] == record['profile'] and evidence['pair'] == record['pair']
                and evidence['seed'] == record['seed'], 'Failure verification identity differs')
        merger.validate_record_revision(plan, record, read(directory / 'profile/sample.json'))
        return evidence

    try:
        save()
        # Revalidate old accepted records before resuming, including measured evidence and payloads.
        for record in state['attempts']:
            if record['status'] == 'accepted':
                previous_hash = record['verification_sha256']; verify(record)
                require(previous_hash == record['verification_sha256'], 'Accepted verification changed since checkpoint')
            elif record['status'] in ('prepared', 'running'):
                document = Path(record['directory']) / 'collection/collection.json'
                trials = read(document).get('attempts', []) if document.exists() else []
                if len(trials) == 1 and trials[0].get('accepted') and trials[0].get('storage_complete') and (Path(record['directory']) / 'dataset/meta/info.json').exists():
                    verify(record)
                elif len(trials) == 1 and trials[0].get('accepted') is False and 'returncode' in record:
                    failure = verify_failure(record)
                    record.update(status='failed', reason=failure['reason'],
                                  failure_verification_sha256=sha(Path(record['directory']) / 'failure_verification.json'))
                else:
                    raise ValueError('Previous pending attempt lacks a complete classified outcome; preserved for review')
            elif record['status'] == 'failed':
                # Previous failures also need an explicit physical report before continuing.
                verify_failure(record)
        save()
        while not all(value == quota_for(plan, profile) for profile, value in counts(state).items()):
            if stop[0] or (output / 'STOP').exists():
                stop[0] = True; break
            # Verified physical failures consume the finite per-profile budget.
            # Reset, success, storage, duplicate, disk and per-attempt wall gates remain in force.
            if args.stop_after_total_successes is not None and sum(counts(state).values()) >= args.stop_after_total_successes:
                state.update(status='checkpoint', active=None); save(); print(f'Checkpoint reached: {counts(state)}', flush=True); return 0
            selected = next_profile(state)
            require(selected is not None, 'Profiles exhausted attempt budget before exact quotas: ' + str(counts(state)))
            pair, profile = selected
            index = sum(a['profile'] == profile for a in state['attempts'])
            require(index < attempts_for(plan, profile), f'{profile} exhausted attempt budget')
            seed = plan['first_seed'] + index
            directory = output / profile / f'attempt_{index:06d}_seed_{seed:06d}'
            require(not directory.exists(), 'Attempt directory already exists; refusing overwrite')
            directory.mkdir(parents=True)
            config, sample = sampler.build_config(bundle, pair, seed)
            sampler.materialize(directory / 'profile', config, sample)
            record = {'profile': profile, 'pair': pair, 'seed': seed, 'index': index, 'directory': str(directory),
                      'sampled_offset_xy_m': sample['applied_offset_xy_m'], 'config_sha256': sample['config_sha256'],
                      'sample_sha256': sha(directory / 'profile/sample.json'), 'source_revision': sampler.lineage(sample), 'status': 'prepared', 'started_utc': now()}
            state['attempts'].append(record); state.update(status='running', active=str(directory)); save()
            record['status'] = 'running'; save(); started = time.monotonic()
            code = run('collect', command_for(plan, profile, seed, directory), profile, directory)
            record.update(returncode=code, wall_seconds=time.monotonic() - started, finished_utc=now())
            document = directory / 'collection/collection.json'
            trials = read(document).get('attempts', []) if document.exists() else []
            if stop[0]:
                record.update(status='interrupted', reason='Stop requested; retained all partial artifacts'); save(); break
            if len(trials) == 1 and trials[0].get('accepted') is True:
                require(code == 0 and trials[0].get('storage_complete') is True, 'Accepted physics episode did not finish export; investigate storage before continuing')
                verify(record)
                print(f'{profile} seed {seed}: verified distinct success; totals {counts(state)}', flush=True)
            else:
                require(len(trials) == 1 and trials[0].get('accepted') is False, 'Missing/ambiguous native outcome; refusing to continue')
                failure = verify_failure(record)
                record.update(status='failed', reason=failure['reason'],
                              failure_verification_sha256=sha(directory / 'failure_verification.json'))
                print(f'{profile} seed {seed}: physical failure {record["reason"]}; totals {counts(state)}', flush=True)
            save()
        if stop[0]:
            state.update(status='stopped', active=None); save(); return 130
        state.update(status='collected', active=None); save()
        target = output / f'dataset_{sum(quota_for(plan, profile) for _, profile in PROFILES)}'
        state.update(status='merging'); save()
        expected_args = (['--expected-per-pair', str(plan['quota_each'])] if 'quota_each' in plan
                         else ['--expected-left', str(plan['quota_by_profile']['left2']), '--expected-right', str(plan['quota_by_profile']['right4'])])
        command = [dataset, '-B', '-u', str(TOOLS / 'merge_seeded.py'), 'merge', '--batch', str(output), '--output', str(target), *expected_args, '--batch-lock-fd', str(batch_lock.fileno())]
        if target.exists():
            command.append('--verify-existing')
        code = run('merge', command, inherited=(batch_lock.fileno(),))
        if code != 0:
            raise MergeFailure('Merge did not complete; inspect merge.log; source episodes are preserved')
        state.update(status='complete', dataset=str(target), finished_utc=now()); save()
        print(f'Verified dataset ready: {target}', flush=True); return 0
    except BaseException as exc:
        for pid in list(owned):
            if process(pid) == owned[pid]:
                owned.update(descendants(int(pid)))
        send(owned, signal.SIGINT); started = time.monotonic()
        while alive(owned) and time.monotonic() - started < 170:
            elapsed = time.monotonic() - started
            if elapsed > 160:
                send(owned, signal.SIGKILL)
            elif elapsed > 150:
                send(owned, signal.SIGTERM)
            time.sleep(.25)
        state.update(status='stopped' if stop[0] else 'merge_failed' if isinstance(exc, MergeFailure) else 'needs_review', active=None, error=f'{type(exc).__name__}: {exc}')
        write(process_path, {'owned_processes': alive(owned), 'updated_utc': now()}); save()
        if stop[0]:
            return 130
        raise
    finally:
        global_lock.close(); batch_lock.close()

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f'Seeded collection stopped: {exc}') from exc
