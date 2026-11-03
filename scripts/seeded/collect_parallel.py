#!/usr/bin/env python3
"""Concurrent, resumable seeded collection: up to N (2-3) native attempts per PC.

collect_seeded.py stays the serial engine (and the only engine for N=1). This
engine reuses its plan, seeds, native command, verification, admission and
merge, and adds one orchestrator that keeps the global simulator lock while it
runs up to N attempts. A profile receives a new attempt only while
accepted + in-flight < quota and attempts < budget, so the accepted episodes of
each profile are the serial ones for the same per-seed outcomes. Records are
created at dispatch in per-profile index order; merge and all downstream readers
use that order, never completion order. A resource gate (free VRAM, available
RAM, disk, launch spacing, other Isaac processes) is checked before an attempt
directory or record exists; it waits instead of failing. <batch>/STOP (or
SIGINT/SIGTERM) interrupts in-flight attempts; <batch>/DRAIN (or a terminal
hang-up, SIGHUP, unless it is ignored as under nohup) starts nothing new and lets
them finish. Each native runs under a small wrapper in this file that forwards
stop signals and atomically records the native exit in <attempt>/native_exit.json,
so --resume classifies attempts that ended while the orchestrator was gone; an
attempt that never reported an exit (PC restart) needs --resume --interrupt-unfinished.
Kit persistent physics settings are snapshotted per session and re-checked before
each launch. --resume inherits N from the frozen plan.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid

TOOLS = Path(__file__).resolve().parent


def _sibling(name):
    spec = importlib.util.spec_from_file_location('nepes_parallel_' + name, TOOLS / (name + '.py'))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


helpers = _sibling('collect_seeded')  # the unchanged serial engine
require, now, read, sha, write = helpers.require, helpers.now, helpers.read, helpers.sha, helpers.write
PROFILES = helpers.PROFILES

ORDERING = 'round_robin_parallel_v1'
OWNER_ENV = 'NEPES_PARALLEL_OWNER'
MAX_PARALLEL = 3
# Timers mirror collect_seeded.run(); module level so tests can shrink them.
POLL = .25
GATE_RETRY = 5.
HEARTBEAT = 10.
PROGRESS_EVERY = 30.
TOKEN_SCAN_EVERY = 5.
TERM_AFTER, KILL_AFTER, FATAL_AFTER, CLEANUP_GIVE_UP = 150., 160., 180., 170.
# Kit rewrites shared SDK files (user.config.json, material and mesh caches) up to ~60 s after its start.
MIN_LAUNCH_SPACING = 60.
GPU_SAMPLE_MAX_AGE = 2.     # a launch decision uses an nvidia-smi sample at most this old (s)
GPU_QUERY_STALL = 30.       # an nvidia-smi call running longer than this is reported as stalled (s)
REQUEST_CLOCK_SLACK = .05   # STOP/DRAIN touched later than (start - slack) belong to the starting session
NATIVE_EXIT = 'native_exit.json'
NATIVE_EXIT_SCHEMA = 'nepes_parallel_native_exit_v1'
WRAPPER_FLAG = '--native-exit-wrapper'
FORWARDED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
MIRRORED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGKILL, signal.SIGPIPE, signal.SIGALRM,
                    signal.SIGUSR1, signal.SIGUSR2)
# Persistent Kit settings that change the simulation; every Kit of this PC reads them from one shared file.
KIT_SETTINGS_KEYS = ('/persistent/physics/numThreads', '/persistent/physics/useLocalMeshCache',
                     '/persistent/physics/localMeshCacheSizeMB')

# (flag, type, default, help); runtime options, never part of the frozen plan.
RESOURCE_OPTIONS = (
    ('--min-free-vram-gib', float, 4.5, 'Free GPU memory (nvidia-smi) required before each Isaac launch, after warm-up reservations'),
    ('--kit-vram-gib', float, 4.5, 'Expected GPU memory of one collecting Isaac process; reserved while it warms up'),
    ('--min-available-ram-gib', float, 10., 'MemAvailable required before each Isaac launch, after warm-up reservations'),
    ('--kit-ram-gib', float, 6.5, 'Expected resident memory of one attempt process tree; reserved while it warms up'),
    ('--disk-reserve-gib-per-attempt', float, 10., 'Free disk per running Isaac attempt above the native --min-free-gib floor'),
    ('--launch-spacing-seconds', float, 90., 'Minimum time between two Isaac launches, at least 60 (Kit rewrites shared SDK files and names its log by the second)'),
    ('--warmup-seconds', float, 300., 'Time after a launch during which the expected memory of that attempt stays reserved'),
    ('--gpu-index', int, 0, 'nvidia-smi index of the GPU that Isaac renders on'),
    ('--max-idle-wait-seconds', float, None, 'Stop (resumable) when nothing runs and the gate stayed closed this long; default: wait'),
)
FOREIGN_KIT = ('wait', 'fail')


def say(message):
    try:
        print(f'[{now()}] {message}', flush=True)
    except OSError:  # A closed terminal must never break process ownership.
        pass


# ----------------------------------------------------------------------------- stub points
def native_pids():
    """Every loaded native Kit except this process; own children are removed by the caller."""
    return helpers.native_pids()


def nvidia_smi(*arguments):
    try:
        result = subprocess.run(['nvidia-smi', *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'nvidia-smi unavailable: {exc}') from exc
    require(result.returncode == 0, f'nvidia-smi failed ({result.returncode}): {(result.stderr or result.stdout).strip()[-300:]}')
    return result.stdout


def parse_mib(text):
    text = text.strip()
    if not text or text.startswith('[') or text.upper() in ('N/A', 'NA'):
        return None
    value = float(text)
    require(math.isfinite(value) and value >= 0, f'Invalid nvidia-smi memory value: {text}')
    return value


def parse_gpu_rows(text):
    rows = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(',')]
        require(len(fields) == 5 and fields[0].isdigit(), f'Unexpected nvidia-smi GPU row: {line!r}')
        rows[int(fields[0])] = {'index': int(fields[0]), 'uuid': fields[1], 'total_mib': parse_mib(fields[2]),
                                'used_mib': parse_mib(fields[3]), 'free_mib': parse_mib(fields[4])}
    return rows


def parse_compute_apps(text, gpu_uuid):
    """{pid: MiB} on one GPU; unreported per-process memory counts as 0 (the warm-up reservation stays)."""
    apps = {}
    for line in text.splitlines():
        fields = [part.strip() for part in line.split(',')]
        if len(fields) != 3 or fields[0] != gpu_uuid or not fields[1].isdigit():
            continue
        apps[int(fields[1])] = apps.get(int(fields[1]), 0.) + (parse_mib(fields[2]) or 0.)
    return apps


def query_gpu(gpu_index):
    rows = parse_gpu_rows(nvidia_smi('--query-gpu=index,uuid,memory.total,memory.used,memory.free',
                                     '--format=csv,noheader,nounits'))
    require(gpu_index in rows, f'nvidia-smi has no GPU index {gpu_index}: {sorted(rows)}')
    gpu = dict(rows[gpu_index])
    gpu['apps'] = parse_compute_apps(nvidia_smi('--query-compute-apps=gpu_uuid,pid,used_memory',
                                                '--format=csv,noheader,nounits'), gpu['uuid'])
    return gpu


def available_ram_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return float(line.split()[1]) / 1024.
    raise RuntimeError('MemAvailable is missing from /proc/meminfo')


def rss_mib(pids):
    total = 0.
    for pid in pids:
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total += float(line.split()[1]) / 1024.
                    break
        except OSError:
            pass
    return total


def disk_free_gib(path):
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 2 ** 30


def owner_marks(tokens):
    """{pid: (token, slot, starttime)} for live processes started with NEPES_PARALLEL_OWNER=<token>:<slot>."""
    tokens = {tokens} if isinstance(tokens, str) else set(tokens)
    needle = (OWNER_ENV + '=').encode()
    result = {}
    if not tokens:
        return result
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            data = (proc / 'environ').read_bytes()
        except OSError:
            continue
        if needle not in data:
            continue
        for entry in data.split(b'\0'):
            if entry.startswith(needle):
                token, _, slot = entry[len(needle):].decode(errors='replace').partition(':')
                started = helpers.process(int(proc.name)) if token in tokens else None
                if started is not None:
                    result[proc.name] = (token, slot, started)
                break
    return result


def token_pids(tokens, slot=None):
    return {pid: started for pid, (_, mark, started) in owner_marks(tokens).items()
            if slot is None or mark == str(slot)}


def descendants_all(pid):
    """Like collect_seeded.descendants, but follows children of every thread (Kit spawns from threads)."""
    result, pending = {}, [int(pid)]
    while pending:
        current = pending.pop()
        if str(current) in result:
            continue
        started = helpers.process(current)
        if started is None:
            continue
        result[str(current)] = started
        try:
            tasks = os.listdir(f'/proc/{current}/task')
        except OSError:
            continue
        for task in tasks:
            try:
                pending.extend(map(int, Path(f'/proc/{current}/task/{task}/children').read_text().split()))
            except OSError:
                pass
    return result


# ----------------------------------------------------------------------------- native exit records
def command_digest(command):
    return hashlib.sha256(json.dumps([str(part) for part in command]).encode()).hexdigest()


def wrapped_command(command, exit_path):
    """The native command run through this file's exit wrapper (same process group as the native)."""
    return [sys.executable, '-I', '-B', str(Path(__file__).resolve()), WRAPPER_FLAG, str(exit_path), '--', *command]


def mirror_exit(code):
    """Exit like the native: the same status, or the same plain stop/kill signal (128+n for core-dump signals)."""
    if code < 0 and -code in MIRRORED_SIGNALS:
        if -code != signal.SIGKILL:
            signal.signal(-code, signal.SIG_DFL)
        os.kill(os.getpid(), -code)
        time.sleep(5)
    return code if code >= 0 else 128 - code


def native_wrapper(argv):
    """WRAPPER_FLAG EXIT_JSON -- COMMAND...: run the native in this process group, forward stop signals and
    atomically record its exit before exiting like it, so the outcome survives an orchestrator crash."""
    if len(argv) < 3 or argv[1] != '--':
        raise SystemExit(f'Usage: {WRAPPER_FLAG} EXIT_JSON -- COMMAND...')
    target, command = Path(argv[0]), argv[2:]
    child, received = None, []

    def forward(sig, _frame):
        name = signal.Signals(sig).name
        if child is None:
            received.append([name, 'pending'])
        elif child.returncode is None:
            received.append([name, 'sent'])
            try:
                child.send_signal(sig)
            except OSError:
                pass
        else:
            received.append([name, 'late'])

    for sig in FORWARDED_SIGNALS:
        signal.signal(sig, forward)
    started, started_utc = time.monotonic(), now()
    child = subprocess.Popen(command)
    for entry in list(received):  # stop requested before the native existed
        if entry[1] == 'pending':
            entry[1] = 'sent'
            child.send_signal(getattr(signal, entry[0]))
    code = child.wait()
    document = {'schema': NATIVE_EXIT_SCHEMA, 'returncode': code, 'native_pid': child.pid, 'wrapper_pid': os.getpid(),
                'command_sha256': command_digest(command), 'started_utc': started_utc, 'finished_utc': now(),
                'wall_seconds': time.monotonic() - started,
                'forwarded_signals': [name for name, state in received if state == 'sent']}
    try:
        write(target, document)
    except OSError as exc:  # the orchestrator, if alive, still sees the exit status
        print(f'[{now()}] native exit record not written: {exc}', file=sys.stderr, flush=True)
    return mirror_exit(code)


def native_exit(plan, record):
    """The wrapper's exit record of this attempt's native command; None when absent, ValueError when foreign/corrupt."""
    path = Path(record['directory']) / NATIVE_EXIT
    if not path.exists():
        return None
    try:
        document = read(path)
    except ValueError as exc:
        raise ValueError(f'Unreadable native exit record {path}: {exc}') from exc
    command = helpers.command_for(plan, record['profile'], record['seed'], Path(record['directory']))
    require(isinstance(document, dict) and document.get('schema') == NATIVE_EXIT_SCHEMA
            and type(document.get('returncode')) is int and isinstance(document.get('forwarded_signals'), list)
            and document.get('command_sha256') == command_digest(command),
            f'Native exit record does not belong to this attempt: {path}')
    return document


def pending_outcome(plan, record):
    """(outcome, exit record) of a prepared/running record from its files: 'interrupted' (the wrapper forwarded a
    stop), 'success', 'failure', 'unfinished' (no exit was ever observed or recorded) or 'unclassified'."""
    directory = Path(record['directory'])
    trials, exit_record = trials_of(directory), native_exit(plan, record)
    if exit_record is not None and exit_record['forwarded_signals']:
        return 'interrupted', exit_record  # the orchestrator was stopping it: the live rule for a stop
    if (len(trials) == 1 and trials[0].get('accepted') and trials[0].get('storage_complete')
            and (directory / 'dataset/meta/info.json').exists()):
        return 'success', exit_record
    known = 'returncode' in record or exit_record is not None
    if len(trials) == 1 and trials[0].get('accepted') is False and known:
        return 'failure', exit_record
    return ('unclassified' if known else 'unfinished'), exit_record


# ----------------------------------------------------------------------------- GPU sampler and Kit settings
class GpuSampler:
    """nvidia-smi in a daemon thread: a slow or wedged driver delays launches, never supervision or STOP."""

    def __init__(self, gpu_index):
        self.gpu_index, self.lock, self.wanted = gpu_index, threading.Lock(), threading.Event()
        self.sample = self.started = None  # sample = (monotonic answer time, gpu dict or problem text)
        threading.Thread(target=self.loop, name='nvidia-smi sampler', daemon=True).start()

    def loop(self):
        while True:
            self.wanted.wait()
            self.wanted.clear()
            with self.lock:
                self.started = time.monotonic()
            try:
                value = query_gpu(self.gpu_index)
                if value['free_mib'] is None:
                    value = 'nvidia-smi does not report free memory'
            except Exception as exc:  # reported through the gate; the thread keeps serving
                value = f'GPU query failed: {exc}'
            with self.lock:
                self.sample, self.started = (time.monotonic(), value), None

    def latest(self, max_age):
        """(gpu, None) for a fresh sample; (None, None) while one is requested; (None, problem) on error or stall."""
        moment = time.monotonic()
        with self.lock:
            sample, started = self.sample, self.started
        if sample is not None and moment - sample[0] <= max_age:
            return (sample[1], None) if isinstance(sample[1], dict) else (None, sample[1])
        self.wanted.set()
        if started is not None and moment - started > GPU_QUERY_STALL:
            return None, f'nvidia-smi has not answered for {moment - started:.0f} s'
        return None, None


def kit_settings_files(isaac_python):
    """user.config.json of the Kit installed in the Isaac Python environment; every Kit of the PC shares it."""
    prefix = Path(isaac_python).parent.parent  # the interpreter is never resolved: a venv interpreter is a symlink
    found = prefix.glob('lib/python3*/site-packages/isaacsim/kit/data/Kit/*/*/user.config.json')
    return sorted({path.resolve() for path in found})  # conda also links lib/python3.1 -> python3.11


def kit_settings(path):
    document, values = json.loads(Path(path).read_text()), {}
    for key in KIT_SETTINGS_KEYS:
        value = document
        for part in key.strip('/').split('/'):
            value = value.get(part) if isinstance(value, dict) else None
        values[key] = value
    return values


def snapshot_kit_settings(isaac_python, retries=3):
    """({path: values}, {path: problem}) at session start; an unreadable file is retried, then reported unguarded."""
    values, problems = {}, {}
    for path in kit_settings_files(isaac_python):
        for attempt in range(retries):
            try:
                values[str(path)] = kit_settings(path)
                break
            except (OSError, ValueError) as exc:
                problems[str(path)] = f'{type(exc).__name__}: {exc}'
                if attempt + 1 < retries:
                    time.sleep(1.)
        if str(path) in values:
            problems.pop(str(path), None)
    return values, problems


def kit_settings_problem(snapshot):
    """Why a new Kit must not start now: the shared settings file is torn or its physics values changed."""
    for path, expected in snapshot.items():
        try:
            actual = kit_settings(path)
        except (OSError, ValueError) as exc:
            return f'Kit settings file unreadable (being rewritten?): {path}: {type(exc).__name__}: {exc}'
        changed = {key: [expected[key], actual[key]] for key in expected if actual.get(key) != expected[key]}
        if changed:
            return (f'Kit persistent physics settings changed since this session started {changed} in {path}; '
                    'check the Kit logs, then restore the file, or resume to accept the new values')
    return None


# ----------------------------------------------------------------------------- plan and dispatch
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
    p.add_argument('--parallel', type=int, help=f'Concurrent native attempts, 2..{MAX_PARALLEL}; frozen in the plan (resume default)')
    for flag, kind, default, text in RESOURCE_OPTIONS:
        p.add_argument(flag, type=kind, default=default, help=text)
    p.add_argument('--foreign-kit', choices=FOREIGN_KIT, default='wait',
                   help='Isaac process not owned by this batch: wait for it (default) or stop resumably before any new attempt')
    p.add_argument('--interrupt-unfinished', action='store_true',
                   help='With --resume: record attempts whose native never reported an exit (the native died with the '
                        'orchestrator, e.g. a PC restart) as interrupted; their seeds are consumed, partial files kept')
    return p.parse_args(argv)


def parallel_plan(slots):
    return {'schema': 'nepes_parallel_dispatch_v1', 'slots': slots, 'engine': Path(__file__).name,
            'dispatch': 'alternate on the last created record; per-profile index order; only while accepted+in_flight<quota and attempts<budget',
            'record_order': 'records are created at dispatch; merge and downstream readers use per-profile record order',
            'surplus_policy': 'a verified success beyond the frozen quota is recorded as surplus and excluded from counts and merge',
            'stop_policy': 'STOP/SIGINT/SIGTERM interrupt in-flight attempts; DRAIN or SIGHUP (unless ignored) finishes them and starts nothing new',
            'exit_policy': 'natives run under an exit wrapper that forwards stop signals and records native_exit.json; resume classifies '
                           'from it; attempts without any recorded exit need --interrupt-unfinished'}


def frozen_tools():
    paths = (Path(helpers.__file__).resolve(), TOOLS / 'merge_seeded.py', TOOLS / 'seeded_config.py', Path(__file__).resolve())
    return {path.name: sha(path) for path in paths}


def stored_slots(output):
    """Slots frozen in an existing batch plan; None when there is no readable batch."""
    try:
        plan = read(output / 'batch.json')['plan']
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(plan, dict):
        return None
    require('parallel' in plan, 'This batch was collected serially; resume it with collect_seeded.py (collect.py without --parallel)')
    return plan['parallel'].get('slots') if isinstance(plan['parallel'], dict) else None


def resolve_slots(args, output):
    stored = stored_slots(output) if args.resume else None
    slots = args.parallel if args.parallel is not None else stored
    require(slots is not None, f'Choose --parallel 2..{MAX_PARALLEL}; collect_seeded.py runs serial batches')
    require(type(slots) is int and 2 <= slots <= MAX_PARALLEL,
            f'--parallel must be 2..{MAX_PARALLEL} for this engine; collect_seeded.py runs serial batches')
    require(stored is None or slots == stored, f'This batch is frozen at --parallel {stored}; resume with that value or omit --parallel')
    return slots


def next_dispatch(state, in_flight=None, checkpoint=None):
    """collect_seeded.next_profile with in-flight attempts counted against quota and checkpoint."""
    in_flight = {name: (in_flight or {}).get(name, 0) for _, name in PROFILES}
    totals = helpers.counts(state)
    if checkpoint is not None and sum(totals.values()) + sum(in_flight.values()) >= checkpoint:
        return None
    order = list(PROFILES)
    if state['attempts'] and state['attempts'][-1]['profile'] == 'left2':
        order.reverse()
    return next(((pair, name) for pair, name in order
                 if totals[name] + in_flight[name] < helpers.quota_for(state['plan'], name)
                 and sum(a['profile'] == name for a in state['attempts']) < helpers.attempts_for(state['plan'], name)), None)


class ResourcePolicy:
    """Runtime resource-gate options (recorded per session, never frozen)."""

    def __init__(self, args):
        for flag, *_ in RESOURCE_OPTIONS:
            setattr(self, flag[2:].replace('-', '_'), getattr(args, flag[2:].replace('-', '_')))
        self.foreign_kit = args.foreign_kit
        self.native_floor_gib = 0.
        numbers = [self.min_free_vram_gib, self.kit_vram_gib, self.min_available_ram_gib, self.kit_ram_gib,
                   self.disk_reserve_gib_per_attempt, self.launch_spacing_seconds, self.warmup_seconds]
        require(all(math.isfinite(value) and value >= 0 for value in numbers), 'Resource gate values must be finite and nonnegative')
        require(self.launch_spacing_seconds >= MIN_LAUNCH_SPACING,
                f'--launch-spacing-seconds must be at least {MIN_LAUNCH_SPACING:g}: Kit rewrites shared SDK files '
                'and names its log by the second during its first minute')
        require(self.gpu_index >= 0, 'GPU index must be nonnegative')
        require(self.max_idle_wait_seconds is None or (math.isfinite(self.max_idle_wait_seconds) and self.max_idle_wait_seconds > 0),
                'Idle wait limit must be positive')
        require(self.foreign_kit in FOREIGN_KIT, 'Foreign Kit policy must be wait or fail')

    def as_dict(self):
        return dict(vars(self))


def native_floor(plan, output):
    """The native collector's own --min-free-gib refusal floor, from the exact serial command."""
    command = helpers.command_for(plan, PROFILES[0][1], plan['first_seed'], output / 'attempt')
    return float(command[command.index('--min-free-gib') + 1])


class Launch:
    """One child process requested by an attempt task; profile marks a native Isaac launch."""

    def __init__(self, label, command, attempt=None, profile=None, inherited=()):
        self.label, self.command, self.attempt, self.profile, self.inherited = label, command, attempt, profile, inherited


class Slot:
    """One concurrent attempt: its generator task, record, current child and every process it owns."""

    def __init__(self, index, tmpdir):
        self.index, self.tmpdir, self.log = index, tmpdir, None
        self.clear()

    @property
    def busy(self):
        return self.task is not None

    @property
    def phase(self):
        if self.request is None:
            return 'dispatching' if self.busy else 'idle'
        return 'native' if self.request.profile else self.request.label

    def clear(self):
        if self.log is not None:
            self.log.close()
        self.task = self.record = self.request = self.child = self.log = self.log_path = None
        self.owned, self.stopping, self.started, self.started_utc, self.native_started = {}, None, None, None, None


class Context:
    def __init__(self, *, args, bundle, output, plan, state, sampler, merger, frozen, policy, slots, environment):
        self.args, self.bundle, self.output, self.plan, self.state = args, bundle, output, plan, state
        self.sampler, self.merger, self.frozen, self.policy, self.environment = sampler, merger, frozen, policy, environment
        self.slots = slots
        self.dataset = plan['dataset_python']
        self.path, self.process_path = output / 'batch.json', output / 'process.json'
        self.checkpoint = args.stop_after_total_successes
        self.token = uuid.uuid4().hex
        self.stop = False
        self.hangup = False  # SIGHUP: drain at the next scheduler iteration (never written from a signal handler)
        self.drain = None  # {'mode', 'reason', 'since_utc'}
        self.gpu = None  # GpuSampler, started after the locks are held
        self.kit_settings = {}  # {user.config.json: tracked values} snapshotted at session start
        self.local_error = None
        self.accepted = {name: [] for _, name in PROFILES}
        self.last_native_launch = None
        self.gate = {'satisfied': None, 'kind': None, 'reason': 'not evaluated', 'since_utc': now()}
        self.gate_retry_at = 0.
        self.gate_closed_since = None
        self.idle_since = None
        self.last_heartbeat = self.last_progress = self.last_token_scan = 0.

    # -- state files
    def save(self):
        active = [slot.record['directory'] for slot in self.slots if slot.record is not None]
        self.state.update(updated_utc=now(), counts=helpers.counts(self.state), pid=os.getpid(), active=active or None)
        write(self.path, self.state)

    def write_process(self, owned=None, **extra):
        union, slots = {}, []
        for slot in self.slots:
            union.update(slot.owned)
            entry = {'slot': slot.index, 'phase': slot.phase, 'tmpdir': str(slot.tmpdir)}
            if slot.busy:
                record, request = slot.record or {}, slot.request
                entry.update(label=request.label if request else None, attempt=record.get('directory'),
                             profile=record.get('profile'), seed=record.get('seed'),
                             pid=slot.child.pid if slot.child is not None else None, started_utc=slot.started_utc,
                             log=str(slot.log_path) if slot.log_path else None, owned_processes=dict(slot.owned))
            slots.append(entry)
        write(self.process_path, {'schema': 'nepes_parallel_process_v1',
                                  'owned_processes': union if owned is None else owned,
                                  'owner_token': self.token, 'orchestrator_pid': os.getpid(), 'updated_utc': now(),
                                  'parallel': len(self.slots), 'slots': slots, 'gate': self.gate, **extra})

    # -- requests and bookkeeping
    def poll_stop(self):
        if (self.output / 'STOP').exists():
            self.stop = True

    def poll_requests(self):
        self.poll_stop()
        if self.drain is None and self.hangup:
            self.begin_drain('hangup', 'Terminal hang-up (SIGHUP): no new attempts; in-flight attempts finish')
        if self.drain is None and (self.output / 'DRAIN').exists():
            self.begin_drain('drain', 'DRAIN requested: no new attempts; in-flight attempts finish')

    def begin_drain(self, mode, reason):
        if self.drain is None:
            self.drain = {'mode': mode, 'reason': reason, 'since_utc': now()}
            self.state['draining'] = dict(self.drain)
            say(f'Draining ({mode}): {reason}')
            self.save()

    def in_flight(self):
        result = {name: 0 for _, name in PROFILES}
        for slot in self.slots:
            if slot.record is not None:
                result[slot.record['profile']] += 1
        return result

    def own_pids(self):
        pids = set()
        for slot in self.slots:
            if slot.child is not None:
                slot.owned.update(descendants_all(slot.child.pid))
            pids |= {int(pid) for pid in helpers.alive(slot.owned)}
        return pids | {int(pid) for pid in token_pids(self.token)}

    def prefix(self, slot):
        record = slot.record or {}
        return f'slot {slot.index} {record.get("profile", "-")} seed {record.get("seed", "-")}'

    def heartbeat(self):
        moment = time.monotonic()
        if moment - self.last_token_scan >= TOKEN_SCAN_EVERY:
            self.last_token_scan = moment
            marks = owner_marks(self.token)
            for slot in self.slots:
                slot.owned.update({pid: started for pid, (_, mark, started) in marks.items() if mark == str(slot.index)})
        if moment - self.last_heartbeat >= HEARTBEAT:
            self.last_heartbeat = moment
            self.write_process()
        if moment - self.last_progress >= PROGRESS_EVERY:
            self.last_progress = moment
            busy = [slot for slot in self.slots if slot.busy]
            for slot in busy:
                say(f'{self.prefix(slot)}: {slot.phase}; log {slot.log_path}')
            say(f'accepted {helpers.counts(self.state)}; in flight {self.in_flight()}'
                + ('' if self.gate.get('satisfied') is not False else f'; waiting: {self.gate.get("reason")}'))


def check_frozen(ctx):
    require(helpers.verify_bundle(ctx.bundle) == ctx.plan['bundle_manifest_sha256'], 'Bundle manifest changed')
    require({name: sha(TOOLS / name) for name in ctx.frozen} == ctx.frozen, 'Frozen batch tool changed')


# ----------------------------------------------------------------------------- per-attempt logic (serial semantics)
def trials_of(directory):
    document = Path(directory) / 'collection/collection.json'
    return read(document).get('attempts', []) if document.exists() else []


def attempt_task(ctx, record):
    """collect_seeded.py main loop body after record creation, yielding child launches."""
    directory, profile, seed = Path(record['directory']), record['profile'], record['seed']
    started = time.monotonic()
    code = yield Launch('collect', helpers.command_for(ctx.plan, profile, seed, directory), attempt=directory, profile=profile)
    record.update(returncode=code, wall_seconds=time.monotonic() - started, finished_utc=now())
    ctx.save()  # a later crash (e.g. during verification) keeps the observed exit status for resume
    trials = trials_of(directory)
    if ctx.stop:
        record.update(status='interrupted', reason='Stop requested; retained all partial artifacts'); ctx.save()
        say(f'{profile} seed {seed}: interrupted by stop request')
        return
    if len(trials) == 1 and trials[0].get('accepted') is True:
        require(code == 0 and trials[0].get('storage_complete') is True, 'Accepted physics episode did not finish export; investigate storage before continuing')
        yield from verify_task(ctx, record)
        if record['status'] == 'accepted':
            say(f'{profile} seed {seed}: verified distinct success; totals {helpers.counts(ctx.state)}')
        else:
            say(f'{profile} seed {seed}: verified success recorded as surplus; totals {helpers.counts(ctx.state)}')
    else:
        require(len(trials) == 1 and trials[0].get('accepted') is False, 'Missing/ambiguous native outcome; refusing to continue')
        failure = yield from verify_failure_task(ctx, record)
        record.update(status='failed', reason=failure['reason'],
                      failure_verification_sha256=sha(directory / 'failure_verification.json'))
        say(f'{profile} seed {seed}: physical failure {record["reason"]}; totals {helpers.counts(ctx.state)}')
    ctx.save()


def verify_task(ctx, record, revalidate=False):
    """collect_seeded.verify(); a new success beyond the quota becomes surplus, never accepted."""
    directory, plan, profile = Path(record['directory']), ctx.plan, record['profile']
    command = [ctx.dataset, '-B', str(TOOLS / 'merge_seeded.py'), 'verify-attempt', '--bundle', str(ctx.bundle),
               '--attempt', str(directory), '--output', str(directory / 'verification.json')]
    code = yield Launch('verify', command, attempt=directory)
    require(code == 0, f'Attempt integrity/reset verification failed; inspect {directory / "verify.log"}')
    evidence = read(directory / 'verification.json')
    require(evidence['profile'] == record['profile'] and evidence['pair'] == record['pair'] and evidence['seed'] == record['seed'] and evidence['xy_half_range_m'] == ctx.sampler.effective_xy_range(record['pair'], record['seed'], plan['xy_half_range_by_profile_m'][record['profile']]), 'Verification profile/pair/seed/effective range differs from batch record')
    ctx.merger.validate_record_revision(plan, record, read(directory / 'profile/sample.json'))
    if revalidate and record['status'] == 'surplus':
        status = 'surplus'
    else:
        ctx.merger.reject_duplicates(ctx.accepted[profile], evidence)
        keep = revalidate and record['status'] == 'accepted'
        status = 'accepted' if keep or helpers.counts(ctx.state)[profile] < helpers.quota_for(plan, profile) else 'surplus'
        if status == 'accepted':
            ctx.accepted[profile].append(evidence)
    record.update(status=status, verification_sha256=sha(directory / 'verification.json'),
                  config_sha256=evidence['config_sha256'], config_file_sha256=evidence['config_file_sha256'], sampled_offset_xy_m=evidence['sampled_offset_xy_m'],
                  measured_reset_pose_w=evidence['startup_reset']['measured_pose_w'],
                  measured_initial_world_poses=evidence['initial_measurement']['poses'],
                  motion_exact_sha256=evidence['motion_exact_sha256'], motion_quantized_sha256=evidence['motion_quantized_sha256'])
    if status == 'surplus':
        record.setdefault('surplus_reason', f'Verified success beyond the frozen {profile} quota '
                          f'{helpers.quota_for(plan, profile)}; excluded from counts and merge, kept for audit')


def verify_failure_task(ctx, record):
    """collect_seeded.verify_failure()."""
    directory = Path(record['directory'])
    command = [ctx.dataset, '-B', str(TOOLS / 'merge_seeded.py'), 'verify-failure', '--bundle', str(ctx.bundle),
               '--attempt', str(directory), '--returncode', str(record['returncode']),
               '--output', str(directory / 'failure_verification.json')]
    code = yield Launch('verify_failure', command, attempt=directory)
    require(code == 0, f'Unclassified failure/crash/storage error; inspect {directory / "verify_failure.log"}')
    evidence = read(directory / 'failure_verification.json')
    require(evidence.get('retryable_physical_failure') is True and evidence.get('success') is False
            and evidence['profile'] == record['profile'] and evidence['pair'] == record['pair']
            and evidence['seed'] == record['seed'], 'Failure verification identity differs')
    ctx.merger.validate_record_revision(ctx.plan, record, read(directory / 'profile/sample.json'))
    return evidence


def child_task(request):
    code = yield request
    return code


# ----------------------------------------------------------------------------- slot driver
def launch(ctx, slot, request):
    check_frozen(ctx)
    env = ctx.environment.copy()
    env[OWNER_ENV] = f'{ctx.token}:{slot.index}'
    env['TMPDIR'] = str(slot.tmpdir)
    command = request.command
    if request.profile:
        runtime = ctx.bundle / 'runtime' / request.profile
        env['PYTHONPATH'] = os.pathsep.join(str(runtime / 'source' / part) for part in ('robros_motion_planning', 'robros_lab'))
        command = wrapped_command(request.command, request.attempt / NATIVE_EXIT)
    log_path = (request.attempt if request.attempt else ctx.output) / (request.label + '.log')
    log = log_path.open('a', buffering=1)
    try:
        log.write(f'\n[{now()}] {json.dumps(command)}\n')
        child = subprocess.Popen(command, cwd=ctx.bundle, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, pass_fds=request.inherited)
    except BaseException:
        log.close()
        raise
    slot.request, slot.child, slot.log, slot.log_path = request, child, log, log_path
    slot.owned, slot.stopping = descendants_all(child.pid), None
    slot.started, slot.started_utc = time.monotonic(), now()
    if request.profile:
        slot.native_started = ctx.last_native_launch = slot.started
    ctx.write_process()


def service(ctx, slot):
    """One iteration of collect_seeded.run() for this slot; the exit code once every owned process ended."""
    child = slot.child
    slot.owned.update(descendants_all(child.pid))
    if ctx.stop and slot.stopping is None:
        slot.stopping = time.monotonic()
        key = str(child.pid)
        helpers.send({key: slot.owned[key]} if key in slot.owned else {}, signal.SIGINT)
        attempt = slot.request.attempt
        if attempt and (attempt / 'collection').exists():
            (attempt / 'collection/STOP').touch(exist_ok=True)
    code = child.poll(); remaining = helpers.alive(slot.owned)
    if code is not None and not remaining:
        slot.owned.update(token_pids(ctx.token, slot.index))  # reparented children of this slot
        remaining = helpers.alive(slot.owned)
        if not remaining:
            label = slot.request.label
            slot.log.close()
            slot.request = slot.child = slot.log = slot.log_path = None
            slot.owned, slot.stopping = {}, None
            ctx.write_process(last_exit={'slot': slot.index, 'label': label, 'returncode': code, 'utc': now()})
            return code
    if code is not None and slot.stopping is None:
        slot.stopping = time.monotonic(); helpers.send(slot.owned, signal.SIGINT)
    if slot.stopping is not None and time.monotonic() - slot.stopping > TERM_AFTER:
        helpers.send(slot.owned, signal.SIGTERM if time.monotonic() - slot.stopping < KILL_AFTER else signal.SIGKILL)
    if slot.stopping is not None and time.monotonic() - slot.stopping > FATAL_AFTER and remaining:
        raise RuntimeError('Owned processes survived cleanup; collection blocked')
    return None


def release(ctx, slot):
    slot.clear()
    ctx.save()
    ctx.write_process()


def local_failure(ctx, slot, exc):
    """Attempt-local error: the record stays as it is; other slots finish (drain) unless a stop is active."""
    message = f'{type(exc).__name__}: {exc}'
    entry = {'utc': now(), 'slot': slot.index if slot is not None else None,
             'attempt': slot.record.get('directory') if slot is not None and slot.record else None, 'error': message}
    ctx.state.setdefault('errors', []).append(entry)
    if ctx.stop:
        say(f'{ctx.prefix(slot) if slot else "dispatch"}: {message} (stop requested; record kept for resume)')
        ctx.save()
        return
    if ctx.local_error is None:
        ctx.local_error = exc
    say(f'{ctx.prefix(slot) if slot else "dispatch"}: {message}; finishing other attempts, then review is required')
    ctx.begin_drain('local_error', message)
    ctx.save()


def advance(ctx, slot, value):
    try:
        request = slot.task.send(value)
    except StopIteration:
        release(ctx, slot)
        return
    except Exception as exc:  # attempt-local (verification, identity, duplicate, ambiguous outcome)
        local_failure(ctx, slot, exc)
        release(ctx, slot)
        return
    launch(ctx, slot, request)  # global on failure: frozen sources changed or Popen failed


def run_blocking(ctx, task):
    """Serial run() semantics for revalidation and merge: one child at a time in slot 0."""
    slot = ctx.slots[0]
    require(not slot.busy, 'Internal error: blocking run requires an idle slot')
    try:
        request = task.send(None)
    except StopIteration as done:
        return done.value
    slot.task = task
    try:
        while True:
            launch(ctx, slot, request)
            while True:
                ctx.poll_stop()  # never batch.json writes here: merge rejects a batch changed during merge
                code = service(ctx, slot)
                if code is not None:
                    break
                ctx.heartbeat()
                time.sleep(POLL)
            try:
                request = task.send(code)
            except StopIteration as done:
                return done.value
    finally:
        if slot.child is None:
            slot.clear()


# ----------------------------------------------------------------------------- resource gate
def evaluate_gate(ctx):
    policy, moment = ctx.policy, time.monotonic()
    snapshot = {'checked_utc': now()}
    if ctx.last_native_launch is not None and moment - ctx.last_native_launch < policy.launch_spacing_seconds:
        left = policy.launch_spacing_seconds - (moment - ctx.last_native_launch)
        return False, {**snapshot, 'kind': 'spacing', 'reason': f'launch spacing: {left:.0f} s left'}, min(GATE_RETRY, left)
    foreign = sorted(set(native_pids()) - ctx.own_pids())
    snapshot['foreign_kits'] = foreign
    if foreign:
        message = f'Another native Kit process is active: {foreign}'
        if policy.foreign_kit == 'fail':
            ctx.state['error'] = 'ValueError: ' + message
            ctx.begin_drain('foreign_kit', message)
        return False, {**snapshot, 'kind': 'foreign_kit', 'reason': message}, GATE_RETRY
    problem = kit_settings_problem(ctx.kit_settings)
    if problem is not None:
        return False, {**snapshot, 'kind': 'kit_settings', 'reason': problem}, GATE_RETRY
    young = [slot for slot in ctx.slots if slot.phase == 'native' and slot.native_started is not None
             and moment - slot.native_started < policy.warmup_seconds]
    gpu, problem = ctx.gpu.latest(GPU_SAMPLE_MAX_AGE)
    if gpu is None and problem is None:  # a fresh sample was requested from the sampler thread
        return False, {**snapshot, 'kind': 'gpu_pending', 'reason': 'waiting for a fresh nvidia-smi sample'}, POLL
    if gpu is None:  # failed or stalled query keeps the gate closed
        return False, {**snapshot, 'kind': 'gpu_query', 'reason': problem}, GATE_RETRY
    reserved = sum(max(0., policy.kit_vram_gib * 1024 - sum(gpu['apps'].get(int(pid), 0.) for pid in helpers.alive(slot.owned)))
                   for slot in young)
    floor = policy.min_free_vram_gib * 1024
    snapshot.update(gpu_uuid=gpu['uuid'], vram_free_mib=gpu['free_mib'], vram_reserved_mib=round(reserved, 1), vram_floor_mib=floor)
    if gpu['free_mib'] - reserved < floor:
        return False, {**snapshot, 'kind': 'vram', 'reason': f'GPU memory: {gpu["free_mib"] - reserved:.0f} MiB free after warm-up reservations < {floor:.0f} MiB'}, GATE_RETRY
    try:
        ram = available_ram_mib()
        ram_reserved = sum(max(0., policy.kit_ram_gib * 1024 - rss_mib(list(helpers.alive(slot.owned)))) for slot in young)
        disk = disk_free_gib(ctx.output)
    except Exception as exc:
        return False, {**snapshot, 'kind': 'host_query', 'reason': f'Host resource query failed: {exc}'}, GATE_RETRY
    running = sum(slot.phase == 'native' for slot in ctx.slots)
    disk_need = policy.native_floor_gib + policy.disk_reserve_gib_per_attempt * (running + 1)
    snapshot.update(ram_available_mib=round(ram, 1), ram_reserved_mib=round(ram_reserved, 1),
                    ram_floor_mib=policy.min_available_ram_gib * 1024, disk_free_gib=round(disk, 2), disk_required_gib=disk_need)
    if ram - ram_reserved < policy.min_available_ram_gib * 1024:
        return False, {**snapshot, 'kind': 'ram', 'reason': f'RAM: {ram - ram_reserved:.0f} MiB available after reservations < {policy.min_available_ram_gib * 1024:.0f} MiB'}, GATE_RETRY
    if disk < disk_need:
        return False, {**snapshot, 'kind': 'disk', 'reason': f'Disk: {disk:.1f} GiB free < {disk_need:.1f} GiB'}, GATE_RETRY
    return True, {**snapshot, 'kind': None, 'reason': None}, 0.


def gate_open(ctx):
    moment = time.monotonic()
    if moment < ctx.gate_retry_at:
        return False
    satisfied, snapshot, retry = evaluate_gate(ctx)
    if snapshot.get('kind') == 'gpu_pending':  # keep the previous verdict until the requested sample arrives
        ctx.gate_retry_at = moment + retry
        return False
    changed = ctx.gate.get('satisfied') != satisfied or ctx.gate.get('kind') != snapshot.get('kind')
    since = now() if changed else ctx.gate.get('since_utc')
    ctx.gate = {'satisfied': satisfied, **snapshot, 'since_utc': since}
    ctx.gate_retry_at = moment + retry if not satisfied else 0.
    if satisfied:
        ctx.gate_closed_since = None
    elif ctx.gate_closed_since is None:
        ctx.gate_closed_since = moment
    if changed and not satisfied and snapshot['kind'] != 'spacing':
        say(f'Waiting before the next attempt: {snapshot["reason"]}')
        ctx.write_process()
    return satisfied


# ----------------------------------------------------------------------------- scheduler
def dispatch(ctx, slot, pair, profile):
    """The serial record-creation block, after the gate; errors drain instead of killing other attempts."""
    check_frozen(ctx)
    state, plan = ctx.state, ctx.plan
    try:
        index = sum(a['profile'] == profile for a in state['attempts'])
        require(index < helpers.attempts_for(plan, profile), f'{profile} exhausted attempt budget')
        seed = plan['first_seed'] + index
        directory = ctx.output / profile / f'attempt_{index:06d}_seed_{seed:06d}'
        require(not directory.exists(), 'Attempt directory already exists; refusing overwrite')
        directory.mkdir(parents=True)
        config, sample = ctx.sampler.build_config(ctx.bundle, pair, seed)
        ctx.sampler.materialize(directory / 'profile', config, sample)
        record = {'profile': profile, 'pair': pair, 'seed': seed, 'index': index, 'directory': str(directory),
                  'sampled_offset_xy_m': sample['applied_offset_xy_m'], 'config_sha256': sample['config_sha256'],
                  'sample_sha256': sha(directory / 'profile/sample.json'), 'source_revision': ctx.sampler.lineage(sample), 'status': 'prepared', 'started_utc': now(),
                  'slot': slot.index}
        state['attempts'].append(record); state.update(status='running')
        slot.record = record; ctx.save()
        record['status'] = 'running'; ctx.save()
    except Exception as exc:
        slot.record = None
        local_failure(ctx, None, exc)
        return
    ctx.idle_since = None
    say(f'slot {slot.index} {profile} seed {seed}: started attempt {index}; accepted {helpers.counts(state)}, in flight {ctx.in_flight()}')
    slot.task = attempt_task(ctx, record)
    advance(ctx, slot, None)


def try_dispatch(ctx):
    while not (ctx.stop or ctx.drain or ctx.local_error):
        slot = next((slot for slot in ctx.slots if not slot.busy), None)
        if slot is None:
            return
        selected = next_dispatch(ctx.state, ctx.in_flight(), ctx.checkpoint)
        if selected is None or not gate_open(ctx):
            return
        dispatch(ctx, slot, *selected)


def idle_outcome(ctx):
    totals = helpers.counts(ctx.state)
    if ctx.local_error is not None:
        raise ctx.local_error
    if ctx.stop:
        return 'stopped'
    if all(value == helpers.quota_for(ctx.plan, profile) for profile, value in totals.items()):
        return 'collected'
    if ctx.checkpoint is not None and sum(totals.values()) >= ctx.checkpoint:
        return 'checkpoint'
    require(next_dispatch(ctx.state, None, None) is not None, 'Profiles exhausted attempt budget before exact quotas: ' + str(totals))
    if ctx.drain is not None:
        return 'drained'
    limit = ctx.policy.max_idle_wait_seconds
    if limit is not None and ctx.gate.get('satisfied') is False and ctx.gate_closed_since is not None:
        if time.monotonic() - max(ctx.gate_closed_since, ctx.idle_since or 0.) > limit:
            return 'gate_timeout'
    return None


def schedule(ctx):
    ctx.state['status'] = 'running'; ctx.save()
    while True:
        ctx.poll_requests()
        for slot in ctx.slots:
            if slot.child is not None:
                code = service(ctx, slot)
                if code is not None:
                    advance(ctx, slot, code)
        if not any(slot.busy for slot in ctx.slots):
            if ctx.idle_since is None:
                ctx.idle_since = time.monotonic()
            outcome = idle_outcome(ctx)
            if outcome is not None:
                return outcome
        try_dispatch(ctx)
        ctx.heartbeat()
        time.sleep(POLL)


def cleanup_all(ctx, exc):
    """Global error: interrupt every slot at once with the serial timers; returns survivors."""
    reason = f'{type(exc).__name__}: {exc}'
    exited = {slot.index: slot.child.poll() for slot in ctx.slots if slot.child is not None}
    owned = {}
    for slot in ctx.slots:
        for pid in list(slot.owned):
            if helpers.process(pid) == slot.owned[pid]:
                slot.owned.update(descendants_all(int(pid)))
        attempt = slot.request.attempt if slot.request is not None else None
        if slot.child is not None and attempt and (attempt / 'collection').exists():
            (attempt / 'collection/STOP').touch(exist_ok=True)
        owned.update(slot.owned)
    owned.update(token_pids(ctx.token))
    helpers.send(owned, signal.SIGINT); started = time.monotonic()
    while helpers.alive(owned) and time.monotonic() - started < CLEANUP_GIVE_UP:
        elapsed = time.monotonic() - started
        if elapsed > KILL_AFTER:
            helpers.send(owned, signal.SIGKILL)
        elif elapsed > TERM_AFTER:
            helpers.send(owned, signal.SIGTERM)
        time.sleep(POLL)
        owned.update(token_pids(ctx.token))
    for slot in ctx.slots:
        if slot.child is not None:
            code = slot.child.poll()
            if slot.record is not None and slot.phase == 'native' and slot.record.get('status') == 'running':
                if exited.get(slot.index) is None:
                    slot.record.update(status='interrupted', returncode=code, finished_utc=now(),
                                       wall_seconds=time.monotonic() - slot.native_started,
                                       reason=f'Orchestrator cleanup after {reason}; retained all partial artifacts')
                else:  # the native had exited before this cleanup (its slot was reaping descendants): resume classifies it
                    slot.record.update(returncode=exited[slot.index], finished_utc=now(),
                                       wall_seconds=time.monotonic() - slot.native_started)
        slot.clear()
    return helpers.alive(owned)


def slot_directories(output, count):
    """Short private TMPDIR per slot (IsaacLab logs and any temp file a child names after the clock)."""
    base = Path(tempfile.gettempdir()) / f'igris-nepes-{os.getuid()}-slots'
    key = hashlib.sha256(str(output).encode()).hexdigest()[:8]
    result = []
    for path in [base] + [base / f'{key}-{index}' for index in range(count)]:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = path.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(), f'Unsafe per-slot temporary directory: {path}')
        if stat.S_IMODE(info.st_mode) != 0o700:
            path.chmod(0o700)
        if path != base:
            result.append(path)
    return result


def revalidate(ctx):
    """collect_seeded.py resume revalidation in the same order and with the same rules (surplus kept as surplus),
    plus native exit records: an exit recorded by the wrapper while the orchestrator was gone supplies the
    missing return code, and a forwarded stop signal means the attempt was being interrupted."""
    for record in ctx.state['attempts']:
        if record['status'] in ('accepted', 'surplus'):
            previous_hash = record['verification_sha256']
            run_blocking(ctx, verify_task(ctx, record, revalidate=True))
            require(previous_hash == record['verification_sha256'], 'Accepted verification changed since checkpoint')
        elif record['status'] in ('prepared', 'running'):
            directory = Path(record['directory'])
            outcome, exit_record = pending_outcome(ctx.plan, record)
            if exit_record is not None and 'returncode' not in record:  # ended while the orchestrator was gone
                record.update(returncode=exit_record['returncode'], wall_seconds=exit_record['wall_seconds'],
                              finished_utc=exit_record['finished_utc'], returncode_source=NATIVE_EXIT)
            if outcome == 'interrupted':
                record.update(status='interrupted', reason='Stop signal recorded by the native exit wrapper before the '
                              'orchestrator classified it; retained all partial artifacts')
            elif outcome == 'success':
                run_blocking(ctx, verify_task(ctx, record))
            elif outcome == 'failure':
                failure = run_blocking(ctx, verify_failure_task(ctx, record))
                record.update(status='failed', reason=failure['reason'],
                              failure_verification_sha256=sha(directory / 'failure_verification.json'))
            elif outcome == 'unfinished' and ctx.args.interrupt_unfinished:
                record.update(status='interrupted', reason='The native never reported an exit (it ended with the orchestrator, '
                              'e.g. a PC restart); recorded by --resume --interrupt-unfinished; retained all partial artifacts')
            else:
                raise ValueError('Previous pending attempt lacks a complete classified outcome; preserved for review')
        elif record['status'] == 'failed':
            run_blocking(ctx, verify_failure_task(ctx, record))


def resource_preview(policy, output):
    preview = {}
    try:
        gpu = query_gpu(policy.gpu_index)
        preview.update(gpu_uuid=gpu['uuid'], vram_total_mib=gpu['total_mib'], vram_used_mib=gpu['used_mib'], vram_free_mib=gpu['free_mib'])
        if gpu['free_mib'] is not None:
            spare = gpu['free_mib'] - policy.min_free_vram_gib * 1024
            preview['concurrent_kits_admitted_by_vram_now'] = 0 if spare < 0 else 1 + int(spare // max(policy.kit_vram_gib * 1024, 1.))
    except Exception as exc:
        preview['gpu_error'] = f'{type(exc).__name__}: {exc}'
    try:
        preview['ram_available_mib'] = round(available_ram_mib(), 1)
        preview['disk_free_gib'] = round(disk_free_gib(output), 2)
        preview['foreign_kits'] = native_pids()
    except Exception as exc:
        preview['host_error'] = f'{type(exc).__name__}: {exc}'
    return preview


def main(argv=None):
    session_started = time.time()  # STOP/DRAIN created after this belong to this session
    args = arguments(argv)
    require(args.resume or not args.interrupt_unfinished, '--interrupt-unfinished applies only to --resume')
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
    slots = resolve_slots(args, output)
    policy = ResourcePolicy(args)
    sampler, merger = helpers.module('seeded_config'), helpers.module('merge_seeded')
    frozen = frozen_tools()
    plan = {'bundle': str(bundle), 'output': str(output), 'isaac_python': sim, 'dataset_python': dataset,
            **quota_fields,
            'first_seed': args.first_seed, 'xy_half_range_by_profile_m': {profile: sampler.effective_xy_range(pair, args.first_seed) for pair, profile in PROFILES},
            'max_wall_seconds': args.max_wall_seconds,
            'physical_failure_continuation': dict(helpers.PHYSICAL_FAILURE_CONTINUATION),
            'source_revisions': sampler.revision_plan(),
            'ordering': ORDERING, 'bundle_manifest_sha256': helpers.verify_bundle(bundle),
            'tools_sha256': frozen, 'cameras': 'four_existing_cameras_30fps_8spp',
            'motion_quantization_step': 1e-4, 'initial_xy_quantization_m': 1e-5,
            'parallel': parallel_plan(slots)}
    ik_plan = helpers.upper_body_ik_plan(bundle, sim)  # Pink release preflight; after verify_bundle, before any output
    if ik_plan is not None:
        plan['upper_body_ik'] = ik_plan
    policy.native_floor_gib = native_floor(plan, output)
    if args.dry_run:
        examples = []
        for pair, profile in PROFILES:
            _, sample = sampler.build_config(bundle, pair, args.first_seed)
            attempt = output / profile / f'attempt_000000_seed_{args.first_seed:06d}'
            examples.append({'profile': profile, 'sample': sample,
                             'command': helpers.command_for(plan, profile, args.first_seed, attempt)})
        print(json.dumps({'plan': plan, 'checkpoint_total_successes': args.stop_after_total_successes, 'examples': examples,
                          'resource_policy': policy.as_dict(), 'resource_preview': resource_preview(policy, output)}, indent=2))
        return 0
    try:
        query_gpu(policy.gpu_index)
    except Exception as exc:
        raise RuntimeError(f'--parallel needs a working nvidia-smi for the GPU memory gate: {exc}') from exc
    global_lock = (Path(tempfile.gettempdir()) / f'igris-nepes-{os.getuid()}.lock').open('a')
    batch_lock = None
    try:
        fcntl.flock(global_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            require((output / 'batch.json').is_file(), 'Resume batch does not exist')
        else:
            require(not output.exists(), 'Output exists; use --resume with unchanged plan or choose a fresh directory')
        tmpdirs = slot_directories(output, slots)  # before a fresh output exists: a failure here leaves nothing behind
        if not args.resume:
            output.mkdir(parents=True)
        batch_lock = (output / '.batch.lock').open('a')
        fcntl.flock(batch_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path, process_path = output / 'batch.json', output / 'process.json'
        if args.resume:
            state = read(path)
            require(state['plan'].get('upper_body_ik') == plan.get('upper_body_ik'), 'Resume upper-body IK backend/worker differs from the frozen batch plan; use the same NEPES_IK_BACKEND/NEPES_IK_PYTHON and IK environment')
            require(state['plan'] == plan, 'Resume plan/source/scripts differ; never mix incompatible batches')
            require(not helpers.alive(read(process_path).get('owned_processes', {})) if process_path.exists() else True, 'Previous owned processes remain alive')
            tokens = [session['owner_token'] for session in state.get('sessions', []) if session.get('owner_token')]
            orphans = token_pids(tokens) if tokens else {}
            require(not orphans, f'Previous owned processes remain alive: {sorted(orphans, key=int)}')
            require(state.get('status') not in ('rejected', 'needs_review'), 'Batch requires investigation; automatic resume is blocked')
            unfinished = [record for record in state['attempts'] if record['status'] in ('prepared', 'running')
                          and pending_outcome(plan, record)[0] == 'unfinished']
            require(not unfinished or args.interrupt_unfinished,
                    'Resume refused and nothing was changed: '
                    + ', '.join(f'{record["profile"]} seed {record["seed"]} ({record["directory"]})' for record in unfinished)
                    + ' never reported a native exit (the native ended together with the orchestrator, e.g. a PC restart). '
                    'Inspect them, then resume with --interrupt-unfinished to record them as interrupted '
                    '(their seeds are consumed; partial files are kept)')
            for name in ('STOP', 'DRAIN'):
                try:
                    stale = (output / name).stat().st_mtime < session_started - REQUEST_CLOCK_SLACK
                except FileNotFoundError:
                    continue
                if stale:  # the request that ended an earlier session
                    (output / name).rename(output / f'{name}_resumed_{time.time_ns()}')
                else:
                    say(f'{name} was requested while this resume started; it applies to this session')
            state.pop('draining', None); state.pop('stop_mode', None)
        else:
            state = {'schema': 'nepes_seeded_batch_v1', 'plan': plan, 'created_utc': now(), 'attempts': [], 'status': 'prepared'}
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        environment.pop('PYTHONPATH', None); environment.pop(OWNER_ENV, None)
        ctx = Context(args=args, bundle=bundle, output=output, plan=plan, state=state, sampler=sampler, merger=merger,
                      frozen=frozen, policy=policy, slots=[Slot(index, path) for index, path in enumerate(tmpdirs)],
                      environment=environment)

        def request_stop(*_):
            ctx.stop = True

        def request_drain(*_):
            ctx.hangup = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, request_stop)
        # A hang-up (closed terminal or SSH session) drains instead of killing in-flight attempts; nohup keeps ignoring it.
        hangup = 'ignored' if signal.getsignal(signal.SIGHUP) is signal.SIG_IGN else 'drain'
        if hangup == 'drain':
            signal.signal(signal.SIGHUP, request_drain)
        ctx.kit_settings, unreadable = snapshot_kit_settings(sim)
        ctx.gpu = GpuSampler(policy.gpu_index)
        state.setdefault('sessions', []).append({'started_utc': now(), 'pid': os.getpid(), 'owner_token': ctx.token,
                                                 'slots': slots, 'resource_policy': policy.as_dict(),
                                                 'checkpoint_total_successes': args.stop_after_total_successes,
                                                 'sighup': hangup, 'interrupt_unfinished': args.interrupt_unfinished,
                                                 'kit_settings': ctx.kit_settings or None,
                                                 'kit_settings_unreadable': unreadable or None})
        say(f'Parallel collection with {slots} slots; resource gate {policy.as_dict()}; SIGHUP {hangup}; '
            f'Kit settings guard {sorted(ctx.kit_settings) or "inactive (no user.config.json found)"}')
        for path, problem in unreadable.items():
            say(f'Kit settings not guarded, unreadable at start: {path}: {problem}')
        try:
            ctx.save()
            ctx.write_process()
            # Revalidate old records before resuming, exactly as the serial engine does.
            revalidate(ctx)
            ctx.save()
            outcome = schedule(ctx)
            if outcome in ('stopped', 'drained', 'gate_timeout'):
                mode = {'stopped': 'stop', 'gate_timeout': 'gate_timeout'}.get(outcome) or ctx.drain['mode']
                if outcome == 'gate_timeout':
                    state['error'] = f'Resource gate stayed closed: {ctx.gate.get("reason")}'
                state.update(status='stopped', stop_mode=mode, active=None); ctx.save(); ctx.write_process()
                say(f'Stopped ({mode}); accepted {helpers.counts(state)}; resume with the same command and --resume')
                return 130
            if outcome == 'checkpoint':
                state.update(status='checkpoint', active=None); ctx.save(); ctx.write_process()
                say(f'Checkpoint reached: {helpers.counts(state)}'); return 0
            state.update(status='collected', active=None); ctx.save()
            target = output / f'dataset_{sum(helpers.quota_for(plan, profile) for _, profile in PROFILES)}'
            state.update(status='merging'); ctx.save()
            expected_args = (['--expected-per-pair', str(plan['quota_each'])] if 'quota_each' in plan
                             else ['--expected-left', str(plan['quota_by_profile']['left2']), '--expected-right', str(plan['quota_by_profile']['right4'])])
            command = [dataset, '-B', '-u', str(TOOLS / 'merge_seeded.py'), 'merge', '--batch', str(output), '--output', str(target), *expected_args, '--batch-lock-fd', str(batch_lock.fileno())]
            if target.exists():
                command.append('--verify-existing')
            code = run_blocking(ctx, child_task(Launch('merge', command, inherited=(batch_lock.fileno(),))))
            if code != 0:
                raise helpers.MergeFailure('Merge did not complete; inspect merge.log; source episodes are preserved')
            state.update(status='complete', dataset=str(target), finished_utc=now()); ctx.save(); ctx.write_process()
            say(f'Verified dataset ready: {target}'); return 0
        except BaseException as exc:
            survivors = cleanup_all(ctx, exc)
            local = exc is ctx.local_error
            status = 'needs_review' if local else 'stopped' if ctx.stop else 'merge_failed' if isinstance(exc, helpers.MergeFailure) else 'needs_review'
            state.update(status=status, active=None, error=f'{type(exc).__name__}: {exc}')
            if status == 'stopped':
                state['stop_mode'] = 'stop'
            ctx.write_process(owned=survivors); ctx.save()
            if status == 'stopped':
                return 130
            raise
        finally:
            for slot in ctx.slots:
                try:
                    slot.tmpdir.rmdir()  # only when empty; kept otherwise for inspection
                except OSError:
                    pass
    finally:
        global_lock.close()
        if batch_lock is not None:
            batch_lock.close()


if __name__ == '__main__':
    if sys.argv[1:2] == [WRAPPER_FLAG]:  # one native attempt, run by the orchestrator through wrapped_command()
        raise SystemExit(native_wrapper(sys.argv[2:]))
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f'Seeded collection stopped: {exc}') from exc
