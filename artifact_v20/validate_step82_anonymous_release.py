"""Validate V7 artifact integrity, anonymity, dependencies, and replay suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "STEP82_RELEASE_MANIFEST.json"
INVENTORY = ROOT / "STEP68_REPLAY_DEPENDENCY_INVENTORY_2026-08-10.json"
RECEIPT = ROOT / "STEP68_FETCHED_DEPENDENCY_RECEIPT.json"
TEXT_SUFFIXES = {".py", ".md", ".tex", ".bib", ".json", ".txt", ".sty", ".bst"}
WINDOWS_USER_PATH = re.compile(r"(?i)[A-Z]:\\Users\\[^\\\"'\s]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_files(manifest: dict) -> None:
    for relative, record in manifest["files"].items():
        path = ROOT / Path(relative)
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise AssertionError(f"missing or drifting bundled file: {relative}")


def verify_anonymity(manifest: dict) -> None:
    failures = []
    for relative in manifest["files"]:
        path = ROOT / Path(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if WINDOWS_USER_PATH.search(line):
                failures.append({"file": relative, "line": line_number})
    if failures:
        raise AssertionError({"workstation_path_hits": failures})


def verify_redactions() -> None:
    inherited_path = ROOT / "STEP68_PATH_REDACTION_MAP.json"
    inherited = load(inherited_path)
    if len(inherited["files"]) != 6:
        raise AssertionError("inherited rewrite count drift")
    for relative, record in inherited["files"].items():
        if sha256(ROOT / Path(relative)) != record["portable_sha256"]:
            raise AssertionError(f"inherited portable rewrite drift: {relative}")
    current = load(ROOT / "STEP82_PATH_REDACTION_MAP.json")
    if current["inherited_step68_map_sha256"] != sha256(inherited_path):
        raise AssertionError("Step 82 redaction map is bound to a different inherited map")
    if set(current["new_rewrites"]) != {"run_step77_v2_stable_acquisition.py"}:
        raise AssertionError("unexpected Step 82 rewrite set")
    record = current["new_rewrites"]["run_step77_v2_stable_acquisition.py"]
    if sha256(ROOT / "run_step77_v2_stable_acquisition.py") != record["portable_sha256"]:
        raise AssertionError("Step 77 portable rewrite drift")


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def verify_public_dependencies() -> None:
    inventory = load(INVENTORY)
    if not RECEIPT.is_file():
        raise AssertionError("run fetch_step68_public_dependencies.py first")
    receipt = load(RECEIPT)
    if receipt["inventory_sha256"] != sha256(INVENTORY):
        raise AssertionError("dependency receipt/inventory mismatch")
    for relative, record in inventory["files"].items():
        if record["classification"] not in {"public_helm_cache_fetch_only", "external_repository_fetch_only"}:
            continue
        path = ROOT / Path(relative)
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise AssertionError(f"public dependency mismatch: {relative}")
    for name, record in inventory["external_repositories"].items():
        if git_commit(ROOT / "external" / name) != record["commit"]:
            raise AssertionError(f"repository commit drift: {name}")


def run_full() -> None:
    completed = subprocess.run(
        [sys.executable, "run_step82_portable_validators.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or "PASS_STEP82_PORTABLE_VALIDATORS" not in output:
        raise AssertionError({"full_suite_returncode": completed.returncode, "output_tail": output[-8000:]})


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--integrity-only", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    manifest = load(MANIFEST)
    verify_files(manifest)
    verify_anonymity(manifest)
    verify_redactions()
    if args.full:
        verify_public_dependencies()
        run_full()
    print(json.dumps({
        "verdict": "PASS_STEP82_ANONYMOUS_RELEASE",
        "mode": "full" if args.full else "integrity-only",
        "bundled_files": len(manifest["files"]),
        "path_rewrites": 7,
        "workstation_path_hits": 0,
        "step80_independent_reconstruction": bool(args.full),
    }, indent=2))


if __name__ == "__main__":
    main()

