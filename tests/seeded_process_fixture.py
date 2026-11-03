#!/usr/bin/env python3
"""Exercise actual collector subprocess cancellation/resume with synthetic children.

No Isaac imports or real collection paths. The collector bytes are copied intact;
only native-PID discovery is stubbed in the test launcher because all executable
children belong to a generated fake bundle. Sampler/verifier semantics are small
fixtures: these tests cover orchestration, not native acceptance/media validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


SAMPLER = r'''
import json,hashlib
def build_config(bundle,pair,seed,width=None):
    document={'pair':pair,'seed':seed}
    sample={'applied_offset_xy_m':[seed*1e-6,pair*1e-6],
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

COMMON = r'''
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
CONTROL=Path(os.environ['COLLECTOR_SYNTHETIC_CONTROL'])
EVENTS=CONTROL.parent/'events.jsonl'
def control():return json.loads(CONTROL.read_text())
def event(kind,**data):
    with EVENTS.open('a') as f:f.write(json.dumps({'kind':kind,'pid':os.getpid(),**data})+'\n')
def blocked(stage):
    if control().get('block_stage')==stage:
        event('blocked',stage=stage)
        while True:time.sleep(.1)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True));temporary.replace(path)
'''

NATIVE = COMMON + r'''
p=argparse.ArgumentParser();p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);p.add_argument('--export',dest='dataset',type=Path);p.add_argument('--seed',type=int);args,_=p.parse_known_args()
doc=json.loads(args.config.read_text());pair=doc['pair'];seed=doc['seed'];args.output.mkdir(parents=True)
event('native_started',pair=pair,seed=seed)
child=None
if control().get('grandchild'):
    child=subprocess.Popen([sys.executable,'-u','-c','import signal,time; signal.signal(signal.SIGINT,lambda *_:exit(0)); time.sleep(600)'])
    event('grandchild_started',child_pid=child.pid)
def interrupt(*_):
    if child and child.poll() is None:
        child.send_signal(signal.SIGINT);child.wait(timeout=5)
    raise SystemExit(130)
signal.signal(signal.SIGINT,interrupt)
blocked('collect')
failed=control().get('collect_failure')=='all' or (control().get('collect_failure')=='left' and pair==2)
write(args.output/'collection.json',{'attempts':[{'accepted':not failed,'storage_complete':not failed,'status':'complete','failure_reason':'synthetic_contact_failure' if failed else None}]})
if not failed:write(args.dataset/'meta/info.json',{'synthetic':True})
if child and child.poll() is None:
    child.send_signal(signal.SIGINT);child.wait(timeout=5)
event('native_finished',pair=pair,seed=seed,accepted=not failed)
if failed:raise SystemExit(2)
'''

MERGER = COMMON + r'''
def validate_record_revision(*args):return {}
def reject_duplicates(previous,evidence):
    if any(x['motion_exact_sha256']==evidence['motion_exact_sha256'] for x in previous):
        raise ValueError('synthetic duplicate')
def main():
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--attempt',type=Path);p.add_argument('--output',type=Path);p.add_argument('--verify-existing',action='store_true');args,_=p.parse_known_args()
    if args.mode=='verify-failure':
        if control().get('classification_error'):raise SystemExit(7)
        doc=json.loads((args.attempt/'profile/config.json').read_text())
        write(args.output,{'retryable_physical_failure':True,'success':False,'profile':'left2' if doc['pair']==2 else 'right4','pair':doc['pair'],'seed':doc['seed'],'reason':'jig_displaced'})
    elif args.mode=='verify-attempt':
        doc=json.loads((args.attempt/'profile/config.json').read_text());pair,seed=doc['pair'],doc['seed']
        event('verify_started',pair=pair,seed=seed);blocked('verify')
        evidence={'pair':pair,'profile':'left2' if pair==2 else 'right4','seed':seed,'xy_half_range_m':.003 if pair==2 else .005,
          'config_sha256':f'config-{pair}-{seed}','config_file_sha256':f'file-{pair}-{seed}',
          'sampled_offset_xy_m':[seed*1e-6,pair*1e-6],'startup_reset':{'measured_pose_w':[0,0,1,1,0,0,0]},
          'initial_measurement':{'poses':{}},'motion_exact_sha256':f'exact-{pair}-{seed}',
          'motion_quantized_sha256':f'quantized-{pair}-{seed}'}
        if args.output.exists():assert json.loads(args.output.read_text())==evidence
        else:write(args.output,evidence)
        event('verify_finished',pair=pair,seed=seed)
    else:
        event('merge_started',verify_existing=args.verify_existing)
        blocked('merge_before')
        if control().get('merge_failure'):raise SystemExit(7)
        artifact=args.output/'meta/info.json'
        if args.verify_existing:assert json.loads(artifact.read_text())=={'synthetic_published':True}
        else:
            assert not args.output.exists();write(artifact,{'synthetic_published':True})
            event('merge_published')
        blocked('merge_after');event('merge_finished',verify_existing=args.verify_existing)
if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
'''

HARNESS = r'''
import importlib.util,sys
from pathlib import Path
path=Path(__file__).with_name('collect_seeded.py')
spec=importlib.util.spec_from_file_location('collector_under_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.native_pids=lambda:[]
try:raise SystemExit(m.main())
except (OSError,ValueError,RuntimeError,KeyError) as exc:raise SystemExit(str(exc))
'''


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def running(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


class Fixture:
    def __init__(self, root, source, name, control=None):
        self.root = root / name; self.root.mkdir()
        self.control = self.root / 'control.json'
        self.set_control(**(control or {}))
        self.tools = self.root / 'tools'; self.tools.mkdir()
        shutil.copyfile(source, self.tools / 'collect_seeded.py')
        for filename, content in [('seeded_config.py', SAMPLER), ('merge_seeded.py', MERGER), ('launch.py', HARNESS)]:
            (self.tools / filename).write_text(content)
        self.bundle = self.root / 'bundle'; self.bundle.mkdir()
        files = {}
        for profile in ('left2', 'right4'):
            path = self.bundle / 'runtime' / profile / 'scripts/nepes/collect_stationary.py'
            path.parent.mkdir(parents=True); path.write_text(NATIVE)
            files[path.relative_to(self.bundle).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        (self.bundle / 'manifest.json').write_text(json.dumps({'schema': 'igris_nepes_portable_release_v1', 'files': files}))
        self.output = self.root / 'batch'
        self.environment = dict(os.environ, COLLECTOR_SYNTHETIC_CONTROL=str(self.control),
                                TMPDIR=str(self.root), PYTHONDONTWRITEBYTECODE='1')
        self.environment.pop('PYTHONPATH', None)
        self.command = [sys.executable, '-B', str(self.tools / 'launch.py'), '--bundle', str(self.bundle),
            '--output', str(self.output), '--isaac-python', sys.executable, '--dataset-python', sys.executable,
            '--max-successes-per-pair', '1', '--max-attempts-per-pair', '5', '--first-seed', '0']
        self.children, self.logs = [], []

    def set_control(self, **value):
        temporary = self.control.with_suffix('.tmp'); temporary.write_text(json.dumps(value)); temporary.replace(self.control)

    def start(self, resume=False):
        log = (self.root / f'run_{len(self.children)}.log').open('w'); self.logs.append(log)
        child = subprocess.Popen(self.command + (['--resume'] if resume else []), env=self.environment,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(child); return child

    def events(self):
        path = self.root / 'events.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def wait_event(self, kind, stage=None, timeout=15):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for event in self.events():
                if event['kind'] == kind and (stage is None or event.get('stage') == stage):
                    return event
            if self.children[-1].poll() is not None:
                raise AssertionError(f'Collector exited before {kind}/{stage}; see {self.root}')
            time.sleep(.05)
        raise AssertionError(f'Timeout waiting for {kind}/{stage}: {self.root}')

    def wait(self, child, expected):
        code = child.wait(timeout=20)
        assert code == expected, (self.root, code, expected)
        state = read(self.output / 'batch.json')
        if (self.output / 'process.json').exists():
            processes = read(self.output / 'process.json').get('owned_processes', {})
            assert not any(running(int(pid)) for pid in processes), processes
        return state

    def stop(self, child, mode):
        if mode == 'file': (self.output / 'STOP').touch()
        else: child.send_signal(signal.SIGINT if mode == 'sigint' else signal.SIGTERM)

    def close(self):
        for child in self.children:
            if child.poll() is None:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                child.wait(timeout=5)
        for event in self.events():
            if event['kind'] in ('native_started', 'verify_started', 'merge_started') and running(event['pid']):
                cmdline = Path(f"/proc/{event['pid']}/cmdline").read_bytes()
                if str(self.root).encode() in cmdline:
                    try: os.killpg(event['pid'], signal.SIGKILL)
                    except ProcessLookupError: pass
            if event['kind'] == 'grandchild_started' and running(event['child_pid']):
                os.kill(event['child_pid'], signal.SIGKILL)
        for log in self.logs: log.close()

