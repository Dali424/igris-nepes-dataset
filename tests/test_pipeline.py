import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/pipeline.py"


class PipelineTests(unittest.TestCase):
    def test_mixed_plan_is_read_only_and_has_exact_total(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "run"
            result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--per-pair", "42",
                                     "--augmentation-mode", "both", "--augmentation-episodes", "42",
                                     "--output", str(target), "--dry-run"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["original_episodes"], 84)
            self.assertEqual(plan["video_augmented_episodes"] + plan["render_augmented_episodes"], 84)
            self.assertFalse(target.exists())

    def test_reject_duplicate_video_variant(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--video-presets", "darker", "darker",
                                     "--output", str(Path(temp) / "run"), "--dry-run"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unique", result.stderr)

    @unittest.skipUnless(os.name == "posix", "POSIX signal delivery")
    def test_terminate_forwards_to_owned_stage_and_waits(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            ready, stopped = temp / "ready", temp / "stopped"
            child = temp / "child.py"
            child.write_text("import signal,time,sys\nfrom pathlib import Path\n"
                             "def stop(*_):\n Path(sys.argv[2]).write_text('stopped');sys.exit(0)\n"
                             "signal.signal(signal.SIGTERM, stop)\n"
                             "Path(sys.argv[1]).write_text('ready')\n"
                             "while True:time.sleep(.05)\n")
            runner = ("import sys;sys.path.insert(0,sys.argv[1]);from pipeline import run_stage;"
                      "sys.exit(run_stage([sys.executable,'-B',*sys.argv[2:]]))")
            proc = subprocess.Popen([sys.executable, "-B", "-c", runner, str(ROOT / "scripts"),
                                     str(child), str(ready), str(stopped)])
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists())
                proc.terminate()
                self.assertEqual(proc.wait(timeout=10), 143)
                self.assertEqual(stopped.read_text(), "stopped")
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
