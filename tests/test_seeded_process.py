"""Dummy subprocesses verify cancellation, resume and failure budget orchestration."""
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from seeded_process_fixture import Fixture, running

TOOLS=Path(__file__).resolve().parents[1]/'scripts/seeded'

class ProcessTests(unittest.TestCase):
    def fixture(self,name,control=None):
        root=tempfile.TemporaryDirectory(prefix='nepes-collect-process-');self.addCleanup(root.cleanup)
        fixture=Fixture(Path(root.name),TOOLS/'collect_seeded.py',name,control)
        self.addCleanup(fixture.close)
        return fixture

    def test_physical_failures_continue_past_three_until_budget(self):
        fixture=self.fixture('physical',{'collect_failure':'left'})
        state=fixture.wait(fixture.start(),1)
        self.assertEqual(state['status'],'needs_review')
        self.assertEqual(state['counts'],{'left2':0,'right4':1})
        self.assertEqual(sum(a['status']=='failed' for a in state['attempts']),5)
        self.assertIn('exhausted attempt budget',state['error'])

    def test_unknown_failure_stops_at_first_and_resume_stays_blocked(self):
        fixture=self.fixture('unknown',{'collect_failure':'all','classification_error':True})
        state=fixture.wait(fixture.start(),1)
        self.assertEqual(len(state['attempts']),1)
        self.assertEqual(state['status'],'needs_review')
        self.assertEqual(fixture.start(resume=True).wait(timeout=20),1)
        self.assertEqual(len([e for e in fixture.events() if e['kind']=='native_started']),1)

    def test_stop_resume_preserves_published_output_and_other_process(self):
        foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
        self.addCleanup(lambda: foreign.poll() is None and (foreign.terminate(),foreign.wait(timeout=5)))
        for stage,mode in [('collect','file'),('verify','sigint'),('merge_before','file'),('merge_after','sigterm')]:
            with self.subTest(stage=stage):
                fixture=self.fixture(stage,{'block_stage':stage,'grandchild':stage=='collect'})
                child=fixture.start();fixture.wait_event('blocked',stage);fixture.stop(child,mode)
                stopped=fixture.wait(child,130)
                self.assertEqual(stopped['status'],'stopped')
                self.assertIsNone(foreign.poll())
                for event in fixture.events():
                    if event['kind'] in ('native_started','verify_started','merge_started'):
                        self.assertFalse(running(event['pid']))
                    if event['kind']=='grandchild_started':self.assertFalse(running(event['child_pid']))
                artifact=fixture.output/'dataset_2/meta/info.json'
                before=artifact.read_bytes() if artifact.exists() else None
                fixture.set_control()
                complete=fixture.wait(fixture.start(resume=True),0)
                self.assertEqual(complete['status'],'complete')
                self.assertEqual(complete['counts'],{'left2':1,'right4':1})
                self.assertEqual(len([e for e in fixture.events() if e['kind']=='native_started']),3 if stage=='collect' else 2)
                if before is not None:self.assertEqual(artifact.read_bytes(),before)

if __name__=='__main__':unittest.main(verbosity=2)
