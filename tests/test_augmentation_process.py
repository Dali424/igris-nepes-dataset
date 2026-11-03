"""CPU-only real dummy children exercise render STOP, ownership, and resume.

No Kit is launched. Each test redirects the coordinator lock to its own
TemporaryDirectory and explicitly substitutes native/parent fixtures.
"""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'scripts/augmentation'


def running(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


class AugmentationProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nepes-render-process-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = self.root / 'fixture.py'
        self.fixture.write_text('''import json,os,subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
(p/'native_started.json').write_text(json.dumps({'pid':os.getpid(),'grandchild':child.pid}))
try: time.sleep(60)
except KeyboardInterrupt: pass
finally:
 if child.poll() is None: child.terminate()
 child.wait()
''')
        self.wrapper = self.root / 'worker.py'
        self.wrapper.write_text('''import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import render_collect as c
root=Path(sys.argv[2]); resume=len(sys.argv)>3
tempfile.tempdir=str(root/'locks');Path(tempfile.tempdir).mkdir(exist_ok=True)
bundle=root/'bundle';bundle.mkdir(exist_ok=True)
batch=root/'parent';batch.mkdir(exist_ok=True)
attempt=batch/'attempt';(attempt/'profile').mkdir(parents=True,exist_ok=True)
for p in (batch/'batch.json',batch/'merged/meta/nepes_storage_verification.json',attempt/'verification.json',attempt/'profile/config.json'):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.write_text(json.dumps({'plan':{'tools_sha256':{}}}) if p.name=='batch.json' else '{}')
parent={'index':0,'profile':'left2','seed':0,'case':'dim_matte','parent_attempt':str(attempt),'parent_verification':str(attempt/'verification.json'),'parent_verification_sha256':c.sha(attempt/'verification.json'),'parent_config_sha256':c.sha(attempt/'profile/config.json')}
c.parent_plan=lambda *a:{'ready':True,'parent_batch':str(batch),'parent_batch_sha256':c.sha(batch/'batch.json'),'parent_merged_dataset':str(batch/'merged'),'parent_merge_receipt_sha256':c.sha(batch/'merged/meta/nepes_storage_verification.json'),'parents':[parent],'selected_episodes':1,'source_total_episodes':1,'expected_profile_counts':{'left2':1,'right4':0}}
c.native_pids=lambda:[]
original=c.subprocess.Popen
def popen(argv,**kw):
 label='prepare' if 'render_scene.py' in str(argv) else 'native'
 if label=='prepare':argv=[sys.executable,'-c','pass']
 else:argv=[sys.executable,str(root/'fixture.py'),str(root/('signal' if resume else 'stopfile'))]
 return original(argv,**kw)
c.subprocess.Popen=popen
sys.argv=['render_collect.py','--batch',str(batch),'--bundle',str(bundle),'--output',str(root/'output'),'--isaac-python',sys.executable,'--dataset-python',sys.executable,'--execute']+(['--resume'] if resume else [])
raise SystemExit(c.main())
''')
        self.sentinel = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        self.addCleanup(self.stop_process, self.sentinel)

    def stop_process(self, process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)

    def exercise(self, mode, resume=False):
        command = [sys.executable, '-B', str(self.wrapper), str(TOOLS), str(self.root)]
        if resume:
            command.append('resume')
        with (self.root / (mode + '.log')).open('w') as log:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            self.addCleanup(self.stop_process, worker)
            marker = self.root / mode / 'native_started.json'
            deadline = time.monotonic() + 15
            while not marker.exists():
                if worker.poll() is not None:
                    self.fail('Dummy worker exited early: ' + (self.root / (mode + '.log')).read_text())
                if time.monotonic() > deadline:
                    self.fail('Dummy worker did not start before timeout')
                time.sleep(.05)
            ids = json.loads(marker.read_text())
            time.sleep(.65)
            if mode == 'stopfile':
                (self.root / 'output/STOP').touch()
            else:
                worker.send_signal(signal.SIGTERM)
            self.assertEqual(worker.wait(timeout=20), 130)
        state = json.loads((self.root / 'output/state.json').read_text())
        self.assertEqual(state['status'], 'stopped')
        self.assertEqual(state['accepted'], 0)
        self.assertFalse(running(ids['pid']))
        self.assertFalse(running(ids['grandchild']))
        self.assertIsNone(self.sentinel.poll())
        return state

    def test_stop_file_cleans_owned_children_and_preserves_foreign_process(self):
        state = self.exercise('stopfile')
        self.assertEqual(state['jobs'][0]['attempts'][0]['status'], 'interrupted')

    def test_signal_resume_preserves_previous_attempt_and_cleans_children(self):
        original = self.exercise('stopfile')
        resumed = self.exercise('signal', resume=True)
        self.assertEqual(len(resumed['jobs'][0]['attempts']), 2)
        self.assertEqual(resumed['jobs'][0]['attempts'][0], original['jobs'][0]['attempts'][0])
        self.assertTrue(all(item['status'] == 'interrupted' for item in resumed['jobs'][0]['attempts']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
