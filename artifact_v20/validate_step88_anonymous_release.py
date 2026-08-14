"""Validate V11 integrity, anonymity, inherited closure, and Step 88 replay."""

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
TEXT_SUFFIXES = {".py", ".md", ".tex", ".bib", ".json", ".txt", ".sty", ".bst"}
WINDOWS_USER_PATH = re.compile(r"(?i)[A-Z]:\\Users\\[^\\\"'\s]+")
UNBUNDLED_PREFIXES = (
    "external/llm-selector/",
    "external_data/step88/",
)
EXPECTED_BINDINGS = {
    "preregistration": "b630370652c88cb4150a5630d04cdf6fd93fde90cf67cb4e64ca694bb03ee4df",
    "runner": "751f4f250627a8854c460c028ee79fc218dba18df3fdd611a2cd04f9b41f8ccb",
    "results": "24c3fcdab2016bf6004844c242603bfd346fe0222911124f304fd81e4cea7020",
    "independent_validator": "4eee854c0a958877072efd1447ce9bc5bc7c071ab5d88875fa1b9d645c345a09",
    "independent_receipt": "9e3f062cbc2e76f552738a5e0fb4e3f62b043f3b8eb6d51c146ac4a41eb993fd",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], marker: str) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or marker not in output:
        raise AssertionError(
            {
                "command": command,
                "returncode": completed.returncode,
                "marker": marker,
                "output_tail": output[-12000:],
            }
        )
    return {"command": command, "marker": marker, "passed": True}


def verify_integrity() -> dict:
    manifest = load(MANIFEST)
    if manifest.get("schema") != "step88.anonymous_validation_closure_release.v5":
        raise AssertionError("unexpected V11 schema")
    refresh = manifest.get("refresh", {})
    if refresh.get("numerical_evidence_changed") is not True:
        raise AssertionError("Step 88 numerical refresh not declared")
    if refresh.get("public_source_packages_embedded") is not False:
        raise AssertionError("public-source embedding declaration drift")
    records = manifest["files"]
    for relative, record in records.items():
        path = ROOT / Path(relative)
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise AssertionError(f"missing or drifting bundled file: {relative}")
    bundled_public = [
        relative
        for relative in records
        if any(relative.startswith(prefix) for prefix in UNBUNDLED_PREFIXES)
    ]
    if bundled_public:
        raise AssertionError({"public_step88_sources_embedded": bundled_public})
    workstation_hits = []
    for relative in records:
        path = ROOT / Path(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if WINDOWS_USER_PATH.search(text):
            workstation_hits.append(relative)
    if workstation_hits:
        raise AssertionError({"workstation_path_hits": workstation_hits})
    if manifest["bindings"].get("step88") != EXPECTED_BINDINGS:
        raise AssertionError("Step 88 binding drift")
    if sha256(ROOT / "paper_draft/main.pdf") != manifest["bindings"]["paper_pdf_sha256"]:
        raise AssertionError("paper PDF binding drift")
    expected_summary = {
        "step88_datasets": 6,
        "step88_passing_datasets": 2,
        "step88_paired_trajectories": 6000,
        "step88_independent_checks": 85934,
        "step88_literal_official_path_replays": 300,
    }
    for key, value in expected_summary.items():
        if manifest["summary"].get(key) != value:
            raise AssertionError({"summary_key": key, "observed": manifest["summary"].get(key)})
    if manifest["summary"].get("files") != len(records):
        raise AssertionError("manifest file count drift")
    if manifest["summary"].get("bytes") != sum(record["bytes"] for record in records.values()):
        raise AssertionError("manifest byte count drift")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--integrity-only", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()

    manifest = verify_integrity()
    checks = [
        run(
            [
                sys.executable,
                "validate_step87_anonymous_release.py",
                "--full" if args.full else "--integrity-only",
            ],
            "PASS_STEP87_ANONYMOUS_RELEASE",
        )
    ]
    if args.full:
        checks.append(
            run(
                [sys.executable, "validate_step88_llm_selector_audit.py"],
                "PASS_STEP88_INDEPENDENT_VALIDATION",
            )
        )
        checks.append(
            run(
                [sys.executable, "validate_step88_manuscript_integration.py"],
                "PASS_STEP88_MANUSCRIPT_INTEGRATION",
            )
        )
        # The independent validator rewrites its receipt deterministically.  A
        # second integrity pass proves that full replay did not drift any byte.
        verify_integrity()
    print(
        json.dumps(
            {
                "verdict": "PASS_STEP88_ANONYMOUS_RELEASE",
                "mode": "full" if args.full else "integrity-only",
                "bundled_files": len(manifest["files"]),
                "public_step88_source_packages_bundled": 0,
                "workstation_path_hits": 0,
                "step88_independently_reconstructed": bool(args.full),
                "step88_passing_datasets": ["alpacaeval", "bingo"],
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
