import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/campaign.py"
spec = importlib.util.spec_from_file_location("quota_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def write(self, path, document):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))

    def make_args(self, root, left=14, right=21):
        ledger = root / "imports.json"
        ledger.write_text(json.dumps({"imports": [
            {"profile": name, "attempt": str(root / name / str(index))}
            for name, count in (("left2", left), ("right4", right)) for index in range(count)]}))
        return argparse.Namespace(imports=ledger, output=root / "run", base_per_pair=100,
                                  augmentation_episodes=100, first_seed=10000,
                                  max_attempts_per_pair=None, max_attempts_per_parent=5,
                                  workers=2, video_preset="darker", isaac_python=sys.executable,
                                  dataset_python=sys.executable)

    def test_exact_missing_quotas_and_separate_derivatives(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, "verify_bundle", return_value="frozen"):
            args = self.make_args(Path(directory))
            plan = campaign.build_plan(args)
            self.assertEqual(plan["new_base_by_profile"], {"left2": 86, "right4": 79})
            self.assertEqual(plan["target_base"], 200)
            self.assertEqual(plan["target_brightness"], 100)
            self.assertEqual(plan["target_render"], 100)
            stages = dict(campaign.commands(plan))
            collect = stages["collect_base"]
            self.assertEqual(collect[collect.index("--left-successes") + 1], "86")
            self.assertEqual(collect[collect.index("--right-successes") + 1], "79")
            self.assertEqual(stages["augment"][stages["augment"].index("--episodes") + 1], "100")
            self.assertNotIn("--resume", collect)

    def test_duplicate_import_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate imported attempt"):
            campaign.imported_counts({"imports": [{"profile": "left2", "attempt": "/same"}] * 2})

    def test_insufficient_balanced_parents_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(Path(directory), left=60)
            with self.assertRaisesRegex(ValueError, "Insufficient new parents"):
                campaign.build_plan(args)

    def test_output_must_not_overlap_historical_sources(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, "verify_bundle", return_value="frozen"):
            root = Path(directory)
            args = self.make_args(root)
            args.output = root / "left2/0/new_campaign"
            with self.assertRaisesRegex(ValueError, "Output overlaps imported source"):
                campaign.build_plan(args)

    def completed_fixture(self, directory):
        args = self.make_args(Path(directory), left=1, right=1)
        args.base_per_pair = 2
        args.augmentation_episodes = 2
        plan = campaign.build_plan(args)
        run = args.output
        self.write(run / "new_base/batch.json", {
            "schema": "nepes_seeded_batch_v1", "status": "complete", "counts": {"left2": 1, "right4": 1},
            "plan": {"quota_by_profile": plan["new_base_by_profile"],
                     "max_attempts_by_profile": {p: plan["max_attempts_per_pair"] for p in campaign.PROFILES},
                     "bundle": plan["bundle"], "output": str(run / "new_base"),
                     "first_seed": plan["first_seed"], "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
                     "isaac_python": plan["isaac_python"], "dataset_python": plan["dataset_python"]},
            "attempts": [{"profile": profile, "status": "accepted"} for profile in campaign.PROFILES]})
        basic = run / "basic_4"
        video = run / "augmentation/video/visible_darker_v2"
        render = run / "augmentation/render/dataset_2"
        for path, episodes in ((basic, 4), (video, 2), (render, 2)):
            self.write(path / "meta/info.json", {"total_episodes": episodes})
            if path != video:
                self.write(path / "meta/nepes_storage_verification.json", {
                    "schema": "nepes_verified_lerobot_storage_v1", "verified": True,
                    "kind": "merged_dataset", "total_episodes": episodes})
        audit = video / "meta/video_augmentation_audit.json"
        self.write(audit, {"schema": "nepes_seeded_video_dataset_audit_v1", "verified": True, "episodes": 2})
        video_result = {"preset": "visible_darker_v2", "verified": True, "episodes": 2,
                        "dataset": str(video), "audit_sha256": campaign.helpers.sha(audit)}
        self.write(run / "augmentation/video/state.json", {
            "schema": "nepes_seeded_video_batch_v1", "status": "complete", "total_augmented_episodes": 2,
            "results": [video_result]})
        self.write(run / "augmentation/render/state.json", {
            "schema": "nepes_native_render_batch_v1", "status": "complete", "accepted": 2,
            "counts": {"left2": 1, "right4": 1}, "dataset": str(render),
            "jobs": [{"profile": profile, "status": "accepted"} for profile in campaign.PROFILES]})
        self.write(run / "augmentation/state.json", {
            "schema": "nepes_portable_augmentation_batch_v1", "status": "complete",
            "results": {"video": [video_result], "render": {
                "dataset": str(render), "episodes": 2, "new_physics_execution": True}}})
        return plan

    def test_completion_binds_exact_quotas_verified_receipts_and_paths(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, "verify_bundle", return_value="frozen"):
            plan = self.completed_fixture(directory)
            totals, paths = campaign.validate_completion(plan)
            self.assertEqual(totals["base_by_profile"], {"left2": 2, "right4": 2})
            self.assertEqual(Path(paths["lighting_material"]).name, "dataset_2")

    def test_complete_counters_cannot_hide_incomplete_worker(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, "verify_bundle", return_value="frozen"):
            plan = self.completed_fixture(directory)
            path = Path(plan["output"]) / "augmentation/state.json"
            state = campaign.helpers.read(path); state["status"] = "needs_review"
            self.write(path, state)
            with self.assertRaisesRegex(ValueError, "supervisor is not complete"):
                campaign.validate_completion(plan)

    def test_completed_batch_from_another_seed_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, "verify_bundle", return_value="frozen"):
            plan = self.completed_fixture(directory)
            path = Path(plan["output"]) / "new_base/batch.json"
            state = campaign.helpers.read(path); state["plan"]["first_seed"] += 1
            self.write(path, state)
            with self.assertRaisesRegex(ValueError, "quota/seed plan differs"):
                campaign.validate_completion(plan)

    def test_equal_total_does_not_hide_wrong_arm_quota(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, "verify_bundle", return_value="frozen"):
            plan = self.completed_fixture(directory)
            path = Path(plan["output"]) / "augmentation/render/state.json"
            state = campaign.helpers.read(path); state["jobs"][1]["profile"] = "left2"
            self.write(path, state)
            with self.assertRaisesRegex(ValueError, "render profile quotas"):
                campaign.validate_completion(plan)

    def test_tampered_receipt_cannot_publish_completion(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, "verify_bundle", return_value="frozen"):
            plan = self.completed_fixture(directory)
            path = Path(plan["output"]) / "augmentation/video/visible_darker_v2/meta/video_augmentation_audit.json"
            state = campaign.helpers.read(path); state["unbound_change"] = True
            self.write(path, state)
            with self.assertRaisesRegex(ValueError, "Video verification receipt differs"):
                campaign.validate_completion(plan)

    def test_real_subprocess_stop_then_resume_and_frozen_plan(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, "verify_bundle", return_value="frozen"):
            root = Path(directory)
            args = self.make_args(root)
            child = root / "child.py"
            child.write_text("import signal,time,sys\nfrom pathlib import Path\n"
                             "def stop(*_):\n Path(sys.argv[1]).write_text('stopped')\n raise SystemExit(130)\n"
                             "signal.signal(signal.SIGINT,stop)\nPath(sys.argv[2]).write_text('ready')\n"
                             "while True: time.sleep(.05)\n")
            ready, stopped = root / "ready", root / "child_stopped"
            argv = ["--imports", str(args.imports), "--output", str(args.output),
                    "--isaac-python", sys.executable, "--dataset-python", sys.executable]
            def request_stop():
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                if ready.exists():
                    (args.output / "STOP").write_text("test")
            watcher = threading.Thread(target=request_stop, daemon=True); watcher.start()
            commands = [("verify_imports", [sys.executable, str(child), str(stopped), str(ready)])]
            with mock.patch.object(campaign, "commands", return_value=commands):
                self.assertEqual(campaign.main(argv), 130)
            watcher.join(timeout=1)
            self.assertTrue(stopped.exists())
            state = campaign.helpers.read(args.output / "campaign.json")
            self.assertEqual(state["status"], "stopped")
            self.assertEqual(state["owned_processes"], {})
            with self.assertRaisesRegex(ValueError, "plan/source/code changed"):
                campaign.main(argv + ["--resume", "--workers", "1"])
            resumed = root / "resumed"
            cmd = [sys.executable, "-c", "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('yes')", str(resumed)]
            with mock.patch.object(campaign, "commands", return_value=[("verify_imports", cmd)]), \
                    mock.patch.object(campaign, "validate_completion", side_effect=ValueError("synthetic incomplete")):
                with self.assertRaisesRegex(ValueError, "synthetic incomplete"):
                    campaign.main(argv + ["--resume"])
            self.assertTrue(resumed.exists())
            self.assertFalse((args.output / "STOP").exists())
            self.assertEqual(len(list(args.output.glob("STOP_resumed_*"))), 1)
            self.assertEqual(campaign.helpers.read(args.output / "campaign.json")["status"], "needs_review")
            self.assertEqual(campaign.helpers.read(args.output / "campaign.json")["error"], "ValueError: synthetic incomplete")

    def test_progress_counts_only_accepted_outcomes(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.helpers, "verify_bundle", return_value="frozen"):
            root = Path(directory)
            args = self.make_args(root)
            plan = campaign.build_plan(args)
            run = args.output
            (run / "new_base").mkdir(parents=True)
            (run / "new_base/batch.json").write_text(json.dumps({"attempts": [
                {"profile": "left2", "status": "accepted"}, {"profile": "left2", "status": "failed"},
                {"profile": "right4", "status": "running"}]}))
            (run / "augmentation/render").mkdir(parents=True)
            (run / "augmentation/render/state.json").write_text(json.dumps({"jobs": [
                {"status": "accepted"}, {"status": "failed"}, {"status": "pending"}]}))
            (run / "augmentation/video").mkdir(parents=True)
            video = run / "augmentation/video/state.json"
            video.write_text(json.dumps({"status": "running", "total_augmented_episodes": 100}))
            result = campaign.progress(plan)
            self.assertEqual(result["base"], 36)
            self.assertEqual(result["new_attempts"], {"left2": 2, "right4": 1})
            self.assertEqual(result["brightness"], 0)
            self.assertEqual(result["render"], 1)
            video.write_text(json.dumps({"status": "complete", "total_augmented_episodes": 100}))
            self.assertEqual(campaign.progress(plan)["brightness"], 100)


if __name__ == "__main__":
    unittest.main()
