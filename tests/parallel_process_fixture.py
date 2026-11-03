#!/usr/bin/env python3
"""Synthetic children for the parallel collector (no Isaac, no real collection paths).

collect_parallel.py and collect_seeded.py are copied byte-for-byte into a
temporary tools directory. The harness replaces only Kit discovery (own running
fake natives plus control['foreign_pids']) and host RAM/RSS/disk readings, and
shrinks the timers. GPU memory comes from a fake nvidia-smi on PATH, so the real
query and parsing code runs. Natives run through the engine's real exit wrapper.
Every fake child is controlled per "pair:seed".
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from seeded_process_fixture import COMMON, HARNESS as SERIAL_HARNESS, read, running, sha

SEEDED = Path(__file__).resolve().parents[1] / 'scripts/seeded'

SAMPLER = r'''
import json,hashlib
def build_config(bundle,pair,seed,width=None):
    document={'pair':pair,'seed':seed}
    sample={'schema':'synthetic_sample_v1','applied_offset_xy_m':[seed*1e-6,pair*1e-6],
            'config_sha256':hashlib.sha256(json.dumps(document).encode()).hexdigest()}
    sample.update(profile='left2' if pair==2 else 'right4',pair=pair,seed=seed)
    return document,sample
def materialize(path,document,sample):
    path.mkdir(parents=True)
    (path/'config.json').write_text(json.dumps(document))
    (path/'sample.json').write_text(json.dumps(sample))
def revision_plan():return {'test_fixture_only':True}
def effective_xy_range(pair,seed,requested=None):return .003 if pair==2 else .005
def lineage(sample):return {'test_fixture_only':True}
'''

NATIVE = COMMON + r'''
p=argparse.ArgumentParser();p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);p.add_argument('--export',dest='dataset',type=Path);p.add_argument('--seed',type=int);args,_=p.parse_known_args()
doc=json.loads(args.config.read_text());pair=doc['pair'];seed=doc['seed'];key=f'{pair}:{seed}'
def spec(name,default=None):
    table=control().get(name,{})
    return table.get(key,table.get('*',default))
args.output.mkdir(parents=True)
event('native_started',pair=pair,seed=seed,time=time.time(),tmpdir=os.environ.get('TMPDIR'),
      owner=os.environ.get('NEPES_PARALLEL_OWNER'),pythonpath=os.environ.get('PYTHONPATH'))
child=None
if spec('grandchild'):
    child=subprocess.Popen([sys.executable,'-u','-c','import signal,time; signal.signal(signal.SIGINT,lambda *_:exit(0)); time.sleep(600)'])
    event('grandchild_started',child_pid=child.pid,pair=pair,seed=seed)
if spec('daemon'):
    daemon=subprocess.Popen([sys.executable,'-c','import sys,time; time.sleep(float(sys.argv[1]))',str(spec('daemon'))],start_new_session=True)
    event('daemon_started',child_pid=daemon.pid,pair=pair,seed=seed)
if spec('stubborn'):  # survives the native, ignores SIGINT/SIGTERM; only SIGKILL ends it
    stubborn=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(600)'],start_new_session=True)
    event('stubborn_started',child_pid=stubborn.pid,pair=pair,seed=seed,time=time.time())
def interrupt(*_):
    if child and child.poll() is None:
        child.send_signal(signal.SIGINT);child.wait(timeout=5)
    event('native_interrupted',pair=pair,seed=seed,time=time.time())
    raise SystemExit(130)
signal.signal(signal.SIGINT,interrupt)
deadline=time.monotonic()+float(spec('delay',0))
while time.monotonic()<deadline or key in control().get('block',[]):
    time.sleep(.02)
outcome=spec('outcome','success')
if outcome=='ambiguous':
    write(args.output/'collection.json',{'attempts':[]})
else:
    failed=outcome=='fail'
    write(args.output/'collection.json',{'attempts':[{'accepted':not failed,'storage_complete':not failed,'status':'complete',
                                                      'failure_reason':'synthetic_contact_failure' if failed else None}]})
    if not failed:write(args.dataset/'meta/info.json',{'synthetic':True})
if child and child.poll() is None:
    child.send_signal(signal.SIGINT);child.wait(timeout=5)
event('native_finished',pair=pair,seed=seed,outcome=outcome,time=time.time())
if outcome!='success':raise SystemExit(2)
'''

MERGER = COMMON + r'''
def spec(name,key,default=None):
    table=control().get(name,{})
    return table.get(key,table.get('*',default))
def validate_record_revision(*args):return {}
def reject_duplicates(previous,evidence):
    if any(x['motion_exact_sha256']==evidence['motion_exact_sha256'] for x in previous):
        raise ValueError('synthetic duplicate motion')
def identity(attempt):
    doc=json.loads((attempt/'profile/config.json').read_text())
    return doc['pair'],doc['seed'],f"{doc['pair']}:{doc['seed']}"
def main():
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--attempt',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--batch',type=Path);p.add_argument('--verify-existing',action='store_true');args,_=p.parse_known_args()
    if args.mode=='verify-failure':
        pair,seed,key=identity(args.attempt)
        event('verify_failure_started',pair=pair,seed=seed,time=time.time())
        while key in control().get('verify_failure_block',[]):time.sleep(.02)
        if spec('classification_error',key):raise SystemExit(7)
        write(args.output,{'retryable_physical_failure':True,'success':False,'profile':'left2' if pair==2 else 'right4',
                           'pair':pair,'seed':seed,'reason':'jig_displaced'})
    elif args.mode=='verify-attempt':
        pair,seed,key=identity(args.attempt)
        event('verify_started',pair=pair,seed=seed,time=time.time())
        while key in control().get('verify_block',[]):time.sleep(.02)
        if spec('verify_error',key):raise SystemExit(7)
        motion=spec('motion',key,f'exact-{pair}-{seed}')
        evidence={'pair':pair,'profile':'left2' if pair==2 else 'right4','seed':seed,'xy_half_range_m':.003 if pair==2 else .005,
          'config_sha256':f'config-{pair}-{seed}','config_file_sha256':f'file-{pair}-{seed}',
          'sampled_offset_xy_m':[seed*1e-6,pair*1e-6],'startup_reset':{'measured_pose_w':[0,0,1,1,0,0,0]},
          'initial_measurement':{'poses':{}},'motion_exact_sha256':motion,'motion_quantized_sha256':'q-'+motion,
          'archive':str(args.attempt/'collection/archives/attempt_000000'),'archive_receipt_sha256':f'receipt-{pair}-{seed}'}
        if args.output.exists():assert json.loads(args.output.read_text())==evidence
        else:write(args.output,evidence)
        event('verify_finished',pair=pair,seed=seed,time=time.time())
    else:
        event('merge_started',verify_existing=args.verify_existing)
        state=json.loads((args.batch/'batch.json').read_text())
        # Same order as merge_seeded.merge_batch: per profile, accepted records in record order.
        episodes=[[a['profile'],a['seed']] for profile in ('left2','right4') for a in state['attempts']
                  if a['profile']==profile and a['status']=='accepted']
        artifact=args.output/'meta/info.json'
        if args.verify_existing:assert json.loads(artifact.read_text())=={'synthetic_published':True,'episodes':episodes}
        else:
            assert not args.output.exists();write(artifact,{'synthetic_published':True,'episodes':episodes})
        event('merge_finished',verify_existing=args.verify_existing)
if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
'''

HARNESS = r'''
import importlib.util,json,os,sys
from pathlib import Path
TOOLS=Path(__file__).resolve().parent
CONTROL=Path(os.environ['COLLECTOR_SYNTHETIC_CONTROL']);EVENTS=CONTROL.parent/'events.jsonl'
def control():return json.loads(CONTROL.read_text())
def running(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:return False
spec=importlib.util.spec_from_file_location('parallel_under_test',TOOLS/'collect_parallel.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for name,value in control().get('timing',{}).items():setattr(m,name,value)
def native_pids():
    events=[]
    for line in (EVENTS.read_text().splitlines() if EVENTS.exists() else []):
        try:events.append(json.loads(line))
        except ValueError:pass
    own={e['pid'] for e in events if e['kind']=='native_started' and running(e['pid'])}
    return sorted(own|set(control().get('foreign_pids',[])))
m.native_pids=native_pids
m.available_ram_mib=lambda:float(control().get('ram_available_mib',65536))
m.rss_mib=lambda pids:float(control().get('rss_mib',0))
m.disk_free_gib=lambda path:float(control().get('disk_free_gib',1000))
try:raise SystemExit(m.main())
except (OSError,ValueError,RuntimeError,KeyError) as exc:raise SystemExit('Seeded collection stopped: '+str(exc))
'''

NVIDIA_SMI = r'''
import json,os,sys,time
from pathlib import Path
path=Path(os.environ['COLLECTOR_SYNTHETIC_CONTROL']);control=json.loads(path.read_text())
query=' '.join(sys.argv[1:])
with (path.parent/'nvidia_smi_calls.log').open('a') as log:log.write(query+'\n')
time.sleep(float(control.get('nvidia_smi_delay',0)))  # a slow or busy driver
if control.get('nvidia_smi_error') or not control.get('gpus'):
    sys.stderr.write('NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.\n');raise SystemExit(9)
if '--query-gpu=index,uuid,memory.total,memory.used,memory.free' in query:
    for g in control['gpus']:print(f"{g['index']}, {g['uuid']}, {g['total']}, {g['used']}, {g['free']}")
elif '--query-compute-apps=gpu_uuid,pid,used_memory' in query:
    for a in control.get('apps',[]):print(f"{a['uuid']}, {a['pid']}, {a['used']}")
else:raise SystemExit(2)
'''

DEFAULT_CONTROL = {
    'timing': {'POLL': .05, 'GATE_RETRY': .2, 'HEARTBEAT': .3, 'PROGRESS_EVERY': 60., 'TOKEN_SCAN_EVERY': .3,
               'TERM_AFTER': 3., 'KILL_AFTER': 4., 'FATAL_AFTER': 6., 'CLEANUP_GIVE_UP': 5.,
               'MIN_LAUNCH_SPACING': 0., 'GPU_SAMPLE_MAX_AGE': .3, 'GPU_QUERY_STALL': 3.},
    'gpus': [{'index': 0, 'uuid': 'GPU-synthetic-0', 'total': 16303, 'used': 4186, 'free': 12117}],
    'apps': [], 'ram_available_mib': 65536, 'disk_free_gib': 1000, 'rss_mib': 0,
}


class ParallelFixture:
    """One synthetic batch; engine 'parallel' (collect_parallel.py) or 'serial' (collect_seeded.py)."""

    def __init__(self, root, name, control=None, *, engine='parallel', slots=2, quotas=(1, 1), budget=5,
                 first_seed=0, spacing=0., extra=(), isaac_prefix=False):
        self.root = Path(root) / name; self.root.mkdir()
        self.control = self.root / 'control.json'
        self.set_control(**(control or {}))
        self.tools = self.root / 'tools'; self.tools.mkdir()
        for filename in ('collect_seeded.py', 'collect_parallel.py'):
            shutil.copyfile(SEEDED / filename, self.tools / filename)
        for filename, content in [('seeded_config.py', SAMPLER), ('merge_seeded.py', MERGER), ('launch.py', HARNESS),
                                  ('serial_launch.py', SERIAL_HARNESS), ('fake_nvidia_smi.py', NVIDIA_SMI)]:
            (self.tools / filename).write_text(content)
        binary = self.root / 'bin'; binary.mkdir()
        wrapper = binary / 'nvidia-smi'
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -B "{self.tools / "fake_nvidia_smi.py"}" "$@"\n')
        wrapper.chmod(0o755)
        self.bundle = self.root / 'bundle'; self.bundle.mkdir()
        files = {}
        for profile in ('left2', 'right4'):
            path = self.bundle / 'runtime' / profile / 'scripts/nepes/collect_stationary.py'
            path.parent.mkdir(parents=True); path.write_text(NATIVE)
            files[path.relative_to(self.bundle).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        (self.bundle / 'manifest.json').write_text(json.dumps({'schema': 'igris_nepes_portable_release_v1', 'files': files}))
        self.output = self.root / 'batch'
        self.tmp = self.root / 'tmp'; self.tmp.mkdir()
        isaac_python = sys.executable
        if isaac_prefix:  # an Isaac-like environment whose Kit settings file the parallel engine guards
            isaac_python = self.root / 'isaac/bin/python'; isaac_python.parent.mkdir(parents=True)
            isaac_python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n'); isaac_python.chmod(0o755)
            self.kit_settings = self.root / 'isaac/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/user.config.json'
            self.kit_settings.parent.mkdir(parents=True)
            self.kit_settings.write_text(json.dumps({'persistent': {'physics': {'numThreads': 8, 'useLocalMeshCache': True,
                                                                                'localMeshCacheSizeMB': 1024}}}))
        self.environment = dict(os.environ, COLLECTOR_SYNTHETIC_CONTROL=str(self.control), TMPDIR=str(self.tmp),
                                PYTHONDONTWRITEBYTECODE='1', PATH=f'{binary}{os.pathsep}{os.environ.get("PATH", "")}')
        self.environment.pop('PYTHONPATH', None); self.environment.pop('NEPES_PARALLEL_OWNER', None)
        launcher = 'launch.py' if engine == 'parallel' else 'serial_launch.py'
        left, right = quotas
        quota = (['--max-successes-per-pair', str(left)] if left == right
                 else ['--left-successes', str(left), '--right-successes', str(right)])
        self.command = [sys.executable, '-B', str(self.tools / launcher), '--bundle', str(self.bundle),
                        '--output', str(self.output), '--isaac-python', str(isaac_python), '--dataset-python', sys.executable,
                        *quota, '--max-attempts-per-pair', str(budget), '--first-seed', str(first_seed)]
        if engine == 'parallel':
            self.command += ['--parallel', str(slots), '--launch-spacing-seconds', str(spacing)]
        self.command += list(extra)
        self.children, self.logs = [], []

    # -- control and observation
    def set_control(self, **value):
        document = {**DEFAULT_CONTROL, **value}
        temporary = self.control.with_suffix('.tmp')
        temporary.write_text(json.dumps(document)); temporary.replace(self.control)

    def update_control(self, **value):
        self.set_control(**{**read(self.control), **value})

    def start(self, resume=False, extra=(), command=None):
        log = (self.root / f'run_{len(self.children)}.log').open('w'); self.logs.append(log)
        argv = (command or self.command) + (['--resume'] if resume else []) + list(extra)
        child = subprocess.Popen(argv, env=self.environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(child)
        return child

    def log(self, index=-1):
        return (self.root / f'run_{len(self.children) + index if index < 0 else index}.log').read_text()

    def events(self, kind=None):
        path = self.root / 'events.jsonl'
        result = []
        for line in (path.read_text().splitlines() if path.exists() else []):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if kind is None or event['kind'] == kind:
                result.append(event)
        return result

    def wait_for(self, predicate, what, timeout=20, watch=True):
        """Poll until predicate() is truthy; with watch, fail early if the newest collector exited."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                value = predicate()
            except (OSError, ValueError, KeyError):  # files are replaced atomically but may not exist yet
                value = None
            if value:
                return value
            if watch and self.children and self.children[-1].poll() is not None:
                raise AssertionError(f'Collector exited before {what}; see {self.root}\n{self.log()}')
            time.sleep(.03)
        raise AssertionError(f'Timeout waiting for {what}: {self.root}\n{self.log()}')

    def wait_events(self, kind, count, timeout=20):
        return self.wait_for(lambda: len(self.events(kind)) >= count and self.events(kind), f'{count} x {kind}', timeout)

    def state(self):
        return read(self.output / 'batch.json')

    def process(self):
        return read(self.output / 'process.json')

    def wait(self, child, expected, timeout=40):
        code = child.wait(timeout=timeout)
        assert code == expected, (self.root, code, expected, self.log())
        state = self.state()
        if (self.output / 'process.json').exists():
            processes = self.process().get('owned_processes', {})
            assert not any(running(int(pid)) for pid in processes), processes
        return state

    def wrapper_pid(self, native_pid):
        """The native exit wrapper that runs a fake native (its parent process)."""
        return int(Path(f'/proc/{native_pid}/stat').read_text().rsplit(')', 1)[1].split()[1])

    def records(self, state=None, profile=None):
        state = state or self.state()
        return [(a['profile'], a['index'], a['seed'], a['status']) for a in state['attempts']
                if profile is None or a['profile'] == profile]

    def synthetic_pids(self):
        """Every live process started for this fixture (they all inherit its control path)."""
        marker = f'COLLECTOR_SYNTHETIC_CONTROL={self.control}'.encode()
        result = []
        for proc in Path('/proc').iterdir():
            if proc.name.isdigit() and int(proc.name) != os.getpid():
                try:
                    if marker in (proc / 'environ').read_bytes().split(b'\0') and running(int(proc.name)):
                        result.append(int(proc.name))
                except OSError:
                    pass
        return result

    def close(self):
        for child in self.children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5)
        for _ in range(3):
            pids = self.synthetic_pids()
            if not pids:
                break
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(.1)
        for log in self.logs:
            log.close()
