#!/usr/bin/env python3
"""Regenerate file checksums and make a clean ZIP; never includes recorded runs."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"assets", "configs", "ik", "requirements", "runtime", "scripts", "tests"}
# Upper-body IK backend of this release (see README, "Pink IK 백엔드").
IK_BACKEND = {
    "default": "pink", "choices": ["pink", "native"],
    "selection_environment": "NEPES_IK_BACKEND", "worker_python_environment": "NEPES_IK_PYTHON",
    "worker": "ik/pink/worker.py", "requirements": "requirements/ik_pink.txt",
    "client": "runtime/{profile}/source/robros_motion_planning/robros_motion_planning/upper_body_ik_backend.py",
    "transport": "persistent_stdio_jsonl_one_worker_per_isaac_process",
    "failure_policy": "fail_attempt_no_fallback",
    "run_report_key": "upper_body_ik_backend",
    # Batches (collect_seeded, collect_parallel, render_collect) run scripts/check_release.py batch_preflight
    # before any output or Isaac start, freeze the resolved backend/worker in the plan ("upper_body_ik") and never
    # run the native debugging backend; merge_seeded admits only attempts that recorded this backend.
    "batch_plan_key": "upper_body_ik",
    "batch_backends": ["pink"],
    "attempt_verification": "declared_backend_closed_cleanly",
}
ROOT_FILES = {"README.md", "LICENSE", "THIRD_PARTY_NOTICES.txt", ".gitignore", ".gitattributes"}
# Top folder inside the ZIP (README: cd igris_nepes_dataset), whatever the build folder is called.
ZIP_ROOT = "igris_nepes_dataset"


def release_files():
    result = []
    candidates = [ROOT / name for name in ROOT_FILES if (ROOT / name).exists()]
    for name in DIRECTORIES:
        candidates.extend((ROOT / name).rglob("*") if (ROOT / name).is_dir() else [])
    for path in sorted(candidates):
        relative = path.relative_to(ROOT)
        if any(p in {"__pycache__", ".pytest_cache", ".git", ".venv"} for p in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink cannot be shipped: {relative}")
        if not path.is_file():
            continue
        if relative.parts[0] not in DIRECTORIES and str(relative) not in ROOT_FILES:
            continue
        if path.suffix in {".pyc", ".pyo", ".log"}:
            continue
        result.append((relative.as_posix(), path))
    return result


def checksum(path):
    """SHA-256 of a file; chunked, so it also runs on Python < 3.11 (no hashlib.file_digest)."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="Optional ZIP destination outside the release")
    args = parser.parse_args()
    files = release_files()
    manifest = {"schema": "igris_nepes_portable_release_v1", "release": date.today().isoformat(),
                "profiles": {"left2": {"pair": "2 -> 2", "arm": "left", "config": "configs/left2.json"},
                             "right4": {"pair": "4 -> 4", "arm": "right", "config": "configs/right4.json"}},
                "capabilities": ["seeded_collection", "parallel_collection", "video_augmentation", "native_render_augmentation",
                                 "state_replay", "quota_campaign", "pink_ik_backend"],
                "ik_backend": IK_BACKEND,
                "files": {name: {"bytes": path.stat().st_size,
                                 "sha256": checksum(path)}
                          for name, path in files}}
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Manifest: {len(files)} files, {sum(p.stat().st_size for _, p in files) / 2**20:.1f} MiB")
    if args.zip:
        target = args.zip.expanduser().resolve()
        if target.is_relative_to(ROOT):
            parser.error("ZIP must be outside the release")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            parser.error("ZIP exists; select a new filename")
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in files + [("manifest.json", manifest_path)]:
                archive.write(path, f"{ZIP_ROOT}/{name}")
        digest = checksum(target)
        target.with_name(target.name + ".sha256").write_text(f"{digest}  {target.name}\n")
        print(f"ZIP: {target} ({target.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
