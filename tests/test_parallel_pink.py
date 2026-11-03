"""Pink IK x --parallel: the merged behaviour of this release, with synthetic children (no Isaac, no real lock).

The package's ParallelFixture bundle also carries the real Pink release pieces (scripts/check_release.py,
the runtime client upper_body_ik_backend.py, ik/pink/*.py, requirements/ik_pink.txt) and a manifest with this
release's real ``ik_backend`` block, so both engines run the real batch preflight (``worker.py --check`` in the
real IK Python) and freeze the real ``upper_body_ik`` plan block. The fake native writes the run.json IK record
and the episode provenance in the runtime's shape and keeps a stand-in worker child (ignores SIGINT, ends with
its owner, like the Pink WorkerClient); the fake verifier calls the real merge_seeded.upper_body_ik_evidence()
for verify-attempt, verify-failure and merge re-verification. Every fixture has its own TMPDIR.
Checked: the frozen IK plan, per-attempt IK evidence, preflight before any output/lock/GPU query, resume refusal
on another IK backend or environment, DRAIN/SIGHUP/STOP/crash recovery with the IK plan, N=1, and render planning
from a parallel batch. Needs the Pink IK environment (NEPES_IK_PYTHON from env.sh, or <release>/.venv/ik).
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

from parallel_process_fixture import ParallelFixture, read, running, sha
from seeded_process_fixture import COMMON

RELEASE = Path(__file__).resolve().parents[1]
REAL_MERGER = RELEASE / 'scripts/seeded/merge_seeded.py'
CLIENT = 'source/robros_motion_planning/robros_motion_planning/upper_body_ik_backend.py'
PINK_FILES = ['scripts/check_release.py', 'requirements/ik_pink.txt',
              *(f'ik/pink/{name}' for name in ('worker.py', 'servo_factory.py', 'pin_backend.py', 'pink_qp.py')),
              *(f'runtime/{profile}/{CLIENT}' for profile in ('left2', 'right4'))]


def ik_python():
    """The IK interpreter the documented test command uses (as tests/test_ik_pink_bridge.py), else None."""
    value = os.environ.get('NEPES_IK_PYTHON', '').strip()
    path = Path(value).expanduser() if value else RELEASE / '.venv' / 'ik' / 'bin' / 'python'
    return str(Path(os.path.abspath(path))) if path.is_file() and os.access(path, os.X_OK) else None


IK_PYTHON = ik_python()

NATIVE_IK = COMMON + r'''
import hashlib
p=argparse.ArgumentParser();p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);p.add_argument('--export',dest='dataset',type=Path);p.add_argument('--seed',type=int);args,_=p.parse_known_args()
doc=json.loads(args.config.read_text());pair=doc['pair'];seed=doc['seed'];key=f'{pair}:{seed}'
BUNDLE=Path(__file__).resolve().parents[4]
def spec(name,default=None):
    table=control().get(name,{})
    return table.get(key,table.get('*',default))
def fsha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
args.output.mkdir(parents=True)
raw=os.environ.get('NEPES_IK_PYTHON','').strip()
ik_python=os.path.abspath(os.path.expanduser(raw)) if raw else str(BUNDLE/'.venv/ik/bin/python')  # as resolve_worker_python()
event('native_started',pair=pair,seed=seed,time=time.time(),tmpdir=os.environ.get('TMPDIR'),owner=os.environ.get('NEPES_PARALLEL_OWNER'),
      ik_python=os.environ.get('NEPES_IK_PYTHON'),ik_backend=os.environ.get('NEPES_IK_BACKEND'),pythonpath=os.environ.get('PYTHONPATH'),cwd=os.getcwd(),resolved_ik_python=ik_python)
# Stand-in Pink worker: same environment (NEPES_PARALLEL_OWNER), ignores SIGINT, ends when its owner ends.
worker=subprocess.Popen([sys.executable,'-c','import os,signal,time\nsignal.signal(signal.SIGINT,signal.SIG_IGN)\nowner=os.getppid()\nwhile os.getppid()==owner: time.sleep(.05)'])
event('ik_worker_started',child_pid=worker.pid,pair=pair,seed=seed)
def interrupt(*_):
    event('native_interrupted',pair=pair,seed=seed,time=time.time())
    raise SystemExit(130)
signal.signal(signal.SIGINT,interrupt)
deadline=time.monotonic()+float(spec('delay',0))
while time.monotonic()<deadline or key in control().get('block',[]):
    time.sleep(.02)
outcome=spec('outcome','success');failed=outcome=='fail'
run=args.output/'runs/attempt_000000'
worker_pid=40000+seed*10+pair
sources={'ik/pink/'+n:fsha(BUNDLE/'ik/pink'/n) for n in ('worker.py','servo_factory.py','pin_backend.py','pink_qp.py')}
status=spec('ik_status','closed')
block={'schema':'nepes_upper_body_ik_backend_v1','backend':spec('ik_backend_recorded',os.environ.get('NEPES_IK_BACKEND') or 'pink'),
       'selected_by':'package_default','status':status,'error':None if status=='closed' else 'IKBackendError: synthetic worker exit',
       'worker':{'python':ik_python,'pid':worker_pid,'python_version':'3.11.15','packages':{'pin':'3.8.0'},
                 'sources_sha256':spec('ik_sources',sources),'returncode':0 if status=='closed' else -9},
       'requirements':{'path':'requirements/ik_pink.txt','sha256':fsha(BUNDLE/'requirements/ik_pink.txt')},
       'native_solver':{'native_prefix_sha256':'synthetic'},'urdf_sha256':'synthetic'}
write(run/'run.json',{'schema':'nepes_task_run_v1','seed':seed,'upper_body_ik_backend':block})
write(args.output/'collection.json',{'attempts':[{'accepted':not failed,'storage_complete':not failed,'status':'complete',
      'failure_reason':'synthetic_contact_failure' if failed else None}]})
if not failed:
    write(args.dataset/'meta/info.json',{'synthetic':True})
    archive=args.output/'archives/attempt_000000'
    write(archive/'meta/nepes_storage_verification.json',{'synthetic_archive_receipt':key})
    write(archive/'meta/nepes_collection.json',{'episodes':[{'provenance':{'kind':'original','seed':seed,
          'upper_body_ik_backend':{'backend':block['backend'],'status':'ready','error':None,'worker':{'pid':spec('provenance_pid',worker_pid)}}}}]})
worker.kill();worker.wait()
event('native_finished',pair=pair,seed=seed,outcome=outcome,time=time.time())
if failed:raise SystemExit(2)
'''

MERGER_IK = COMMON + r'''
import hashlib,importlib.util
def fsha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def real():
    s=importlib.util.spec_from_file_location('nepes_test_real_merge_seeded',os.environ['NEPES_TEST_REAL_MERGER'])
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def validate_record_revision(*args):return {}
def reject_duplicates(previous,evidence):
    if any(x['motion_exact_sha256']==evidence['motion_exact_sha256'] for x in previous):
        raise ValueError('synthetic duplicate motion')
def identity(attempt):
    doc=json.loads((attempt/'profile/config.json').read_text())
    return doc['pair'],doc['seed'],f"{doc['pair']}:{doc['seed']}"
def ik_evidence(bundle,attempt,success):
    """The REAL Pink per-attempt check (merge_seeded.upper_body_ik_evidence) on this attempt's files."""
    run=json.loads((attempt/'collection/runs/attempt_000000/run.json').read_text())
    provenance=None
    if success:
        provenance=json.loads((attempt/'collection/archives/attempt_000000/meta/nepes_collection.json').read_text())['episodes'][0]['provenance']
    return real().upper_body_ik_evidence(bundle,run,provenance)
def evidence_for(bundle,attempt):
    pair,seed,key=identity(attempt)
    ik=ik_evidence(bundle,attempt,True)
    archive=attempt/'collection/archives/attempt_000000'
    motion=f'exact-{pair}-{seed}'
    return {'verified':True,'pair':pair,'profile':'left2' if pair==2 else 'right4','seed':seed,'xy_half_range_m':.003 if pair==2 else .005,
            'config_sha256':f'config-{pair}-{seed}','config_file_sha256':f'file-{pair}-{seed}',
            'sampled_offset_xy_m':[seed*1e-6,pair*1e-6],'startup_reset':{'measured_pose_w':[0,0,1,1,0,0,0]},
            'initial_measurement':{'poses':{}},'motion_exact_sha256':motion,'motion_quantized_sha256':'q-'+motion,
            'attempt':str(attempt),'archive':str(archive),'archive_receipt_sha256':fsha(archive/'meta/nepes_storage_verification.json'),
            'upper_body_ik_backend':ik}
def main():
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--attempt',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--bundle',type=Path);p.add_argument('--batch',type=Path);p.add_argument('--verify-existing',action='store_true')
    p.add_argument('--expected-per-pair',type=int);p.add_argument('--expected-left',type=int);p.add_argument('--expected-right',type=int)
    args,_=p.parse_known_args()
    if args.mode=='verify-failure':
        pair,seed,key=identity(args.attempt)
        event('verify_failure_started',pair=pair,seed=seed,time=time.time())
        while key in control().get('verify_failure_block',[]):time.sleep(.02)
        ik=ik_evidence(args.bundle,args.attempt,False)
        write(args.output,{'retryable_physical_failure':True,'success':False,'profile':'left2' if pair==2 else 'right4',
                           'pair':pair,'seed':seed,'reason':'jig_displaced','upper_body_ik_backend':ik})
        event('verify_failure_finished',pair=pair,seed=seed)
    elif args.mode=='verify-attempt':
        pair,seed,key=identity(args.attempt)
        event('verify_started',pair=pair,seed=seed,time=time.time())
        while key in control().get('verify_block',[]):time.sleep(.02)
        evidence=evidence_for(args.bundle,args.attempt)
        if args.output.exists():assert json.loads(args.output.read_text())==evidence,'saved verification differs'
        else:write(args.output,evidence)
        event('verify_finished',pair=pair,seed=seed)
    else:
        event('merge_started',verify_existing=args.verify_existing)
        state=json.loads((args.batch/'batch.json').read_text());plan=state['plan'];bundle=Path(plan['bundle'])
        records=[a for profile in ('left2','right4') for a in state['attempts'] if a['profile']==profile and a['status']=='accepted']
        mapping=[]
        for index,record in enumerate(records):  # merge re-verifies every accepted attempt (real merge_seeded does)
            attempt=Path(record['directory'])
            assert evidence_for(bundle,attempt)==json.loads((attempt/'verification.json').read_text()),'re-verification differs'
            assert fsha(attempt/'verification.json')==record['verification_sha256'],'verification binding differs'
            mapping.append({'episode_index':index,'profile':record['profile'],'seed':record['seed'],'verification_sha256':record['verification_sha256']})
        episodes=[[m['profile'],m['seed']] for m in mapping]
        expected=({'expected_per_pair':plan['quota_each']} if 'quota_each' in plan else {'expected_by_profile':plan['quota_by_profile']})
        if args.verify_existing:
            assert json.loads((args.output/'meta/info.json').read_text())=={'synthetic_published':True,'episodes':episodes}
        else:
            assert not args.output.exists()
            write(args.output/'meta/info.json',{'synthetic_published':True,'episodes':episodes})
            write(args.output/'meta/nepes_storage_verification.json',{'verified':True,'kind':'merged_dataset','total_episodes':len(mapping)})
            write(args.output/'meta/nepes_stationary_collection.json',{'schema':'nepes_seeded_collection_aggregate_v1','total_episodes':len(mapping),**expected,'episodes':mapping})
        event('merge_finished',verify_existing=args.verify_existing)
if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
'''


class PinkFixture(ParallelFixture):
    """ParallelFixture whose bundle declares the real Pink backend (see module docstring)."""

    def __init__(self, root, name, control=None, *, ik_python=None, ik_backend=None, **options):
        super().__init__(root, name, control, **options)
        files = {}
        for relative in PINK_FILES:
            target = self.bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(RELEASE / relative, target)
        for profile in ('left2', 'right4'):
            (self.bundle / 'runtime' / profile / 'scripts/nepes/collect_stationary.py').write_text(NATIVE_IK)
        for path in sorted(p for p in self.bundle.rglob('*') if p.is_file() and p.name != 'manifest.json'):
            files[path.relative_to(self.bundle).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        declared = json.loads((RELEASE / 'manifest.json').read_text())['ik_backend']
        (self.bundle / 'manifest.json').write_text(json.dumps({'schema': 'igris_nepes_portable_release_v1', 'files': files,
                                                               'ik_backend': declared}))
        (self.tools / 'merge_seeded.py').write_text(MERGER_IK)
        self.environment['NEPES_TEST_REAL_MERGER'] = str(REAL_MERGER)
        self.set_ik(IK_PYTHON if ik_python is None else ik_python, ik_backend)

    def set_ik(self, ik_python, ik_backend=None):
        for key, value in (('NEPES_IK_PYTHON', ik_python), ('NEPES_IK_BACKEND', ik_backend)):
            if value is None:
                self.environment.pop(key, None)
            else:
                self.environment[key] = value


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@unittest.skipUnless(IK_PYTHON, 'Pink IK environment not installed (bash scripts/setup.sh; source env.sh)')
class PinkParallelTests(unittest.TestCase):
    def fixture(self, name, control=None, **options):
        root = tempfile.TemporaryDirectory(prefix='nepes-pink-parallel-'); self.addCleanup(root.cleanup)
        fixture = PinkFixture(Path(root.name), name, control, **options)
        self.addCleanup(fixture.close)
        return fixture

    def alternate_ik_python(self, fixture, where=None):
        """A second valid IK interpreter path (shell exec of the same venv python): a different frozen 'python'."""
        path = Path(where) if where else fixture.root / 'alt_ik/bin/python'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'#!/bin/sh\nexec "{IK_PYTHON}" "$@"\n'); path.chmod(0o755)
        return str(path)

    def expected_ik_block(self, fixture, ik_python=IK_PYTHON):
        checker = load('pink_parallel_check_' + fixture.root.name, fixture.bundle / 'scripts/check_release.py')
        environ = dict(fixture.environment, NEPES_IK_PYTHON=ik_python); environ.pop('NEPES_IK_BACKEND', None)
        return checker.batch_preflight(sim_python=sys.executable, environ=environ)

    # ------------------------------------------------------------------ freeze + per-attempt IK evidence
    def test_parallel_batch_freezes_ik_plan_and_admits_only_ik_verified_attempts(self):
        fixture = self.fixture('freeze', {'outcome': {'2:1': 'fail'}, 'delay': {'*': .2}}, quotas=(2, 2))
        state = fixture.wait(fixture.start(), 0)
        plan = state['plan']
        self.assertEqual(state['status'], 'complete')
        self.assertEqual((plan['ordering'], plan['parallel']['slots']), ('round_robin_parallel_v1', 2))
        self.assertEqual(plan['upper_body_ik'], self.expected_ik_block(fixture))
        self.assertEqual((plan['upper_body_ik']['backend'], plan['upper_body_ik']['python']), ('pink', IK_PYTHON))
        natives = fixture.events('native_started')
        self.assertEqual({(e['ik_python'], e['ik_backend']) for e in natives}, {(IK_PYTHON, None)})
        self.assertTrue(all(e['owner'] and e['pythonpath'] for e in natives))
        statuses = [record[3] for record in fixture.records(state)]
        self.assertEqual(sorted(statuses), ['accepted'] * 4 + ['failed'])
        for record in state['attempts']:
            name = 'verification.json' if record['status'] == 'accepted' else 'failure_verification.json'
            evidence = read(Path(record['directory']) / name)
            self.assertEqual((evidence['upper_body_ik_backend']['backend'], evidence['upper_body_ik_backend']['status']), ('pink', 'closed'))
            self.assertEqual(evidence['upper_body_ik_backend']['python'], IK_PYTHON)
        self.assertTrue((fixture.output / 'dataset_4/meta/nepes_storage_verification.json').exists())
        for event in fixture.events('ik_worker_started'):
            self.assertFalse(running(event['child_pid']))

    def test_attempts_without_clean_pink_evidence_are_never_admitted(self):
        cases = {'failed_worker_success': ({'ik_status': {'2:0': 'failed'}}, 'Attempt integrity/reset verification failed', 'verify.log',
                                           'Upper-body IK backend did not close cleanly: failed'),
                 'native_recorded_success': ({'ik_backend_recorded': {'2:0': 'native'}}, 'Attempt integrity/reset verification failed', 'verify.log',
                                             "Upper-body IK backend 'native' is not this release's 'pink'"),
                 'provenance_pid_differs': ({'provenance_pid': {'2:0': 1}}, 'Attempt integrity/reset verification failed', 'verify.log',
                                            'Episode provenance IK backend differs from the run report'),
                 'foreign_worker_sources': ({'ik_sources': {'2:0': {'ik/pink/worker.py': '0' * 64}}}, 'Attempt integrity/reset verification failed',
                                            'verify.log', 'IK worker sources/requirements differ from this release'),
                 'failed_worker_physical_failure': ({'outcome': {'2:0': 'fail'}, 'ik_status': {'2:0': 'failed'}},
                                                    'Unclassified failure/crash/storage error', 'verify_failure.log',
                                                    'Upper-body IK backend did not close cleanly: failed')}
        for name, (control, message, logname, reason) in cases.items():
            with self.subTest(case=name):
                fixture = self.fixture(name, dict(control, block=['4:0']), quotas=(2, 2))
                child = fixture.start()
                fixture.wait_for(lambda: fixture.state().get('draining'), 'drain after the IK evidence refusal')
                self.assertEqual(fixture.state()['draining']['mode'], 'local_error')
                fixture.update_control(block=[])
                state = fixture.wait(child, 1)
                self.assertEqual(state['status'], 'needs_review')
                self.assertIn(message, state['error'])
                left = [a for a in state['attempts'] if a['profile'] == 'left2'][0]
                self.assertNotIn(left['status'], ('accepted', 'failed', 'surplus'))
                log = (Path(left['directory']) / logname).read_text()
                self.assertIn('ValueError: ' + reason, log)  # raised by the REAL merge_seeded.upper_body_ik_evidence
                self.assertEqual(fixture.start(resume=True).wait(timeout=60), 1)
                self.assertIn('automatic resume is blocked', fixture.log())

    # ------------------------------------------------------------------ preflight before output
    def test_pink_preflight_refuses_before_any_output_lock_or_gpu_query(self):
        stale = None
        for name, ik_python, backend, pattern, dry in (
                ('missing_ik', '/nonexistent/ik/bin/python', None, 'not usable', False),
                ('missing_ik_dry', '/nonexistent/ik/bin/python', None, 'not usable', True),
                ('native', IK_PYTHON, 'native', 'single-run debugging', False),
                ('bogus', IK_PYTHON, 'scipy', 'is not supported', False),
                ('stale', 'STALE', None, 'before the batch', False),
                ('unset_no_venv', None, None, 'does not exist', False)):
            with self.subTest(case=name):
                fixture = self.fixture(name)
                if ik_python == 'STALE':
                    stale = fixture.root / 'stale/bin/python'; stale.parent.mkdir(parents=True)
                    stale.write_text('#!/bin/sh\necho \'{"ok": false, "error": "Installed packages differ", "mismatches": {"pin": {"expected": "3.8.0", "installed": "2.7.0"}}}\'\nexit 1\n')
                    stale.chmod(0o755); ik_python = str(stale)
                fixture.set_ik(ik_python, backend)
                code = fixture.start(extra=['--dry-run'] if dry else []).wait(timeout=60)
                self.assertEqual(code, 1, fixture.log())
                self.assertRegex(fixture.log(), pattern)
                self.assertFalse(fixture.output.exists())
                self.assertEqual(sorted(p.name for p in fixture.tmp.iterdir()), [])  # no global lock, no slot TMPDIRs
                self.assertFalse((fixture.root / 'nvidia_smi_calls.log').exists())  # refused before the GPU preflight
                self.assertEqual(fixture.events('native_started'), [])

    # ------------------------------------------------------------------ resume refusal on a changed IK environment
    def test_resume_refuses_another_ik_backend_or_environment_without_writing(self):
        fixture = self.fixture('resume_ik', {'block': ['2:0', '4:0']}, quotas=(2, 2))
        child = fixture.start()
        fixture.wait_events('native_started', 2)
        (fixture.output / 'STOP').touch()
        state = fixture.wait(child, 130)
        self.assertEqual(fixture.records(state), [('left2', 0, 0, 'interrupted'), ('right4', 0, 0, 'interrupted')])
        frozen = state['plan']['upper_body_ik']
        before = (fixture.output / 'batch.json').read_bytes()
        venv_python = fixture.bundle / '.venv/ik/bin/python'
        for name, ik_python, backend, pattern, make_venv in (
                ('other_ik_python', self.alternate_ik_python(fixture), None, 'IK backend/worker differs from the frozen batch plan', False),
                ('native_backend', IK_PYTHON, 'native', 'single-run debugging', False),
                ('pink_explicit_same_python', IK_PYTHON, 'pink', None, False),
                ('unset_without_venv', None, None, 'does not exist', False),
                ('unset_with_release_venv', None, None, 'IK backend/worker differs from the frozen batch plan', True)):
            with self.subTest(case=name):
                if make_venv:
                    self.alternate_ik_python(fixture, venv_python)
                fixture.set_ik(ik_python, backend)
                if pattern is None:
                    continue  # explicit NEPES_IK_BACKEND=pink is the same frozen block; exercised below with the run
                code = fixture.start(resume=True).wait(timeout=60)
                self.assertEqual(code, 1, fixture.log())
                self.assertIn(pattern, fixture.log())
                self.assertEqual((fixture.output / 'batch.json').read_bytes(), before)  # refused before any write
                self.assertTrue((fixture.output / 'STOP').exists())  # not even the stale STOP was renamed
                self.assertEqual(len(fixture.events('native_started')), 2)
        if venv_python.exists():
            venv_python.unlink()
        fixture.update_control(block=[])
        fixture.set_ik(IK_PYTHON, 'pink')  # explicit 'pink' resolves to the same frozen block (selected_by is not frozen)
        state = fixture.wait(fixture.start(resume=True), 0)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['plan']['upper_body_ik'], frozen)
        self.assertEqual({e['ik_backend'] for e in fixture.events('native_started')[2:]}, {'pink'})

    # ------------------------------------------------------------------ DRAIN / STOP / hang-up / crash with the IK plan
    def test_drain_and_hangup_finish_in_flight_attempts_through_ik_verification(self):
        for mode in ('drain', 'hangup'):
            with self.subTest(mode=mode):
                fixture = self.fixture('drain_' + mode, {'block': ['2:0', '4:0']}, quotas=(2, 2))
                child = fixture.start()
                fixture.wait_events('native_started', 2)
                if mode == 'drain':
                    (fixture.output / 'DRAIN').touch()
                else:
                    child.send_signal(signal.SIGHUP)
                fixture.wait_for(lambda: fixture.state().get('draining'), 'drain recorded')
                fixture.update_control(block=[])
                state = fixture.wait(child, 130)
                self.assertEqual((state['status'], state['stop_mode']), ('stopped', mode))
                self.assertEqual(fixture.records(state), [('left2', 0, 0, 'accepted'), ('right4', 0, 0, 'accepted')])
                for record in state['attempts']:
                    self.assertEqual(read(Path(record['directory']) / 'verification.json')['upper_body_ik_backend']['status'], 'closed')
                state = fixture.wait(fixture.start(resume=True), 0)
                self.assertEqual((state['status'], state['counts']), ('complete', {'left2': 2, 'right4': 2}))
                self.assertIn('upper_body_ik', state['plan'])

    def test_stop_and_crash_recovery_keep_the_ik_plan_and_interrupt_unfinished_still_works(self):
        fixture = self.fixture('crash', {'block': ['2:0', '4:0']}, quotas=(2, 2))
        child = fixture.start()
        natives = [event['pid'] for event in fixture.wait_events('native_started', 2)]
        workers = [event['child_pid'] for event in fixture.wait_events('ik_worker_started', 2)]
        wrappers = [fixture.wrapper_pid(pid) for pid in natives]
        os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)  # orchestrator only
        early = fixture.start(resume=True).wait(timeout=60)  # natives and their IK workers still run
        self.assertEqual(early, 1)
        self.assertIn('Previous owned processes remain alive', fixture.log())
        for wrapper in wrappers:  # then a "PC restart": wrapper+native+worker die with no exit record
            os.killpg(wrapper, signal.SIGKILL)
        fixture.wait_for(lambda: not any(running(pid) for pid in natives + wrappers + workers), 'natives to die', watch=False)
        before = (fixture.output / 'batch.json').read_bytes()
        fixture.set_ik(self.alternate_ik_python(fixture))
        self.assertEqual(fixture.start(resume=True, extra=['--interrupt-unfinished']).wait(timeout=60), 1)
        self.assertIn('IK backend/worker differs from the frozen batch plan', fixture.log())  # IK refusal wins, nothing written
        self.assertEqual((fixture.output / 'batch.json').read_bytes(), before)
        fixture.set_ik(IK_PYTHON)
        self.assertEqual(fixture.start(resume=True).wait(timeout=60), 1)
        self.assertIn('never reported a native exit', fixture.log())
        self.assertEqual((fixture.output / 'batch.json').read_bytes(), before)
        fixture.update_control(block=[])
        state = fixture.wait(fixture.start(resume=True, extra=['--interrupt-unfinished']), 0)
        self.assertEqual(fixture.records(state), [('left2', 0, 0, 'interrupted'), ('right4', 0, 0, 'interrupted'),
                                                  ('left2', 1, 1, 'accepted'), ('right4', 1, 1, 'accepted'),
                                                  ('left2', 2, 2, 'accepted'), ('right4', 2, 2, 'accepted')])
        stop = self.fixture('stop', {'block': ['2:0', '4:0']}, quotas=(1, 1))
        child = stop.start()
        stop.wait_events('ik_worker_started', 2)
        (stop.output / 'STOP').touch()
        state = stop.wait(child, 130)
        self.assertEqual([r[3] for r in stop.records(state)], ['interrupted', 'interrupted'])
        for event in stop.events('ik_worker_started'):  # SIGINT-immune workers ended with their owners
            self.assertFalse(running(event['child_pid']))
        stop.update_control(block=[])
        self.assertEqual(stop.wait(stop.start(resume=True), 0)['status'], 'complete')

    def test_exit_record_recovery_still_requires_clean_pink_evidence(self):
        for name, control, expected in (
                ('orphans_clean', {'outcome': {'2:0': 'fail'}}, ('failed', 'accepted')),
                ('orphan_success_ik_failed', {'ik_status': {'4:0': 'failed'}}, None)):
            with self.subTest(case=name):
                fixture = self.fixture(name, dict(control, block=['2:0', '4:0']), quotas=(2, 2))
                child = fixture.start()
                natives = [event['pid'] for event in fixture.wait_events('native_started', 2)]
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)  # orchestrator crash; natives keep running
                fixture.update_control(block=[])
                fixture.wait_for(lambda: not any(running(pid) for pid in natives), 'orphaned natives to finish', watch=False)
                fixture.wait_for(lambda: all((Path(a['directory']) / 'native_exit.json').exists() for a in fixture.state()['attempts']),
                                 'exit records', watch=False)
                if expected:
                    state = fixture.wait(fixture.start(resume=True), 0)
                    left, right = state['attempts'][:2]
                    self.assertEqual((left['status'], right['status']), expected)
                    self.assertEqual((left['returncode_source'], right['returncode_source']), ('native_exit.json',) * 2)
                    self.assertEqual(read(Path(left['directory']) / 'failure_verification.json')['upper_body_ik_backend']['status'], 'closed')
                    self.assertEqual(read(Path(right['directory']) / 'verification.json')['upper_body_ik_backend']['status'], 'closed')
                else:
                    code = fixture.start(resume=True).wait(timeout=60)
                    self.assertEqual(code, 1, fixture.log())
                    state = fixture.state()
                    self.assertEqual(state['status'], 'needs_review')
                    right = [a for a in state['attempts'] if a['profile'] == 'right4'][0]
                    self.assertNotEqual(right['status'], 'accepted')
                    self.assertIn('Upper-body IK backend did not close cleanly', (Path(right['directory']) / 'verify.log').read_text())

    def test_resume_revalidation_rejects_tampered_ik_evidence(self):
        fixture = self.fixture('tamper', {'block': ['2:1', '4:1']}, quotas=(3, 3))
        child = fixture.start()
        fixture.wait_for(lambda: len([a for a in fixture.state()['attempts'] if a['status'] == 'accepted']) == 2, 'two accepted')
        (fixture.output / 'DRAIN').touch()
        fixture.wait_for(lambda: fixture.state().get('draining'), 'drain recorded')
        fixture.update_control(block=[])
        self.assertEqual(fixture.wait(child, 130)['stop_mode'], 'drain')
        accepted = [a for a in fixture.state()['attempts'] if a['status'] == 'accepted'][0]
        run = Path(accepted['directory']) / 'collection/runs/attempt_000000/run.json'
        document = read(run); document['upper_body_ik_backend']['status'] = 'failed'
        run.write_text(json.dumps(document))
        self.assertEqual(fixture.start(resume=True).wait(timeout=60), 1, fixture.log())
        state = fixture.state()
        self.assertEqual(state['status'], 'needs_review')
        self.assertIn('Attempt integrity/reset verification failed', state['error'])
        self.assertIn('Upper-body IK backend did not close cleanly', (Path(accepted['directory']) / 'verify.log').read_text())

    # ------------------------------------------------------------------ N=1: the Pink serial engine with the same fixture
    def test_serial_engine_freezes_and_refuses_the_same_way(self):
        fixture = self.fixture('serial', {'block': ['2:0']}, engine='serial', quotas=(1, 1))
        child = fixture.start()
        fixture.wait_events('native_started', 1)
        (fixture.output / 'STOP').touch()
        state = fixture.wait(child, 130)
        self.assertEqual(state['plan']['ordering'], 'round_robin_serial')
        self.assertNotIn('parallel', state['plan'])
        self.assertEqual(state['plan']['upper_body_ik'], self.expected_ik_block(fixture))
        before = (fixture.output / 'batch.json').read_bytes()
        fixture.set_ik(self.alternate_ik_python(fixture))
        self.assertEqual(fixture.start(resume=True).wait(timeout=60), 1)
        self.assertIn('IK backend/worker differs from the frozen batch plan', fixture.log())
        self.assertEqual((fixture.output / 'batch.json').read_bytes(), before)
        fixture.set_ik(IK_PYTHON)
        fixture.update_control(block=[])
        state = fixture.wait(fixture.start(resume=True), 0)
        self.assertEqual(state['status'], 'complete')
        accepted = [a for a in state['attempts'] if a['status'] == 'accepted']
        self.assertEqual(len(accepted), 2)
        for record in accepted:
            self.assertIn('upper_body_ik_backend', read(Path(record['directory']) / 'verification.json'))

    # ------------------------------------------------------------------ render (serial) on a parallel batch
    def test_render_collect_plans_from_a_parallel_batch(self):
        fixture = self.fixture('render_source', {'outcome': {'2:1': 'fail'}, 'delay': {'*': .2}}, quotas=(2, 2))
        state = fixture.wait(fixture.start(), 0)
        render = RELEASE / 'scripts/augmentation/render_collect.py'
        output = fixture.root / 'render_out'
        result = subprocess.run([sys.executable, '-B', str(render), '--batch', str(fixture.output), '--bundle', str(fixture.bundle),
                                 '--output', str(output), '--isaac-python', sys.executable, '--dataset-python', sys.executable],
                                capture_output=True, text=True, env=fixture.environment, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        plan = json.loads(result.stdout)
        self.assertFalse(output.exists())
        self.assertEqual(plan['upper_body_ik'], state['plan']['upper_body_ik'])
        self.assertIn('collect_parallel.py', plan['parent_collection_tools_sha256'])
        accepted = {(a['profile'], a['seed']) for a in state['attempts'] if a['status'] == 'accepted'}
        self.assertEqual({(p['profile'], p['seed']) for p in plan['parents']}, accepted)
        self.assertEqual([(p['profile'], p['seed']) for p in plan['parents']],
                         [('left2', 0), ('right4', 0), ('left2', 2), ('right4', 1)])  # per-profile record order, failure skipped
        # what render_collect's frozen() checks at --execute against this release: every frozen tool exists here
        for name in plan['parent_collection_tools_sha256']:
            self.assertTrue((RELEASE / 'scripts/seeded' / name).is_file(), name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
