"""Validate V10 artifact integrity, anonymity, prior closure, and Step 87 replay."""

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
UNBUNDLED = {
    "STEP87B_LABELED_CALIBRATION_PACKAGE_2026-08-12.json",
    "external_data/step87b_sealed/STEP87B_SEALED_HOLDOUT_OUTCOMES_2026-08-12.json",
}
RAW_MANIFEST = ROOT / "STEP87B_RAW_SOURCE_MANIFEST_2026-08-12.json"


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
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or marker not in output:
        raise AssertionError(
            {
                "command": command,
                "returncode": completed.returncode,
                "marker": marker,
                "output_tail": output[-8000:],
            }
        )
    return {"command": command, "marker": marker, "passed": True}


def verify_integrity() -> dict:
    manifest = load(MANIFEST)
    if manifest.get("schema") not in {
        "step87.anonymous_validation_closure_release.v4",
        "step88.anonymous_validation_closure_release.v5",
    }:
        raise AssertionError("unexpected V10/V11 schema")
    if manifest.get("refresh", {}).get("numerical_evidence_changed") is not True:
        raise AssertionError("Step 87 numerical refresh not declared")
    records = manifest["files"]
    for relative, record in records.items():
        path = ROOT / Path(relative)
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise AssertionError(f"missing or drifting bundled file: {relative}")
    if UNBUNDLED & set(records):
        raise AssertionError("public Step 87 source package was embedded")
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
    expected_bindings = {
        "preregistration": "a95c6f0b2a11177577cd03574d019475bc7d8c169b71f02ab8de778a99ed8093",
        "stage_b_lock": "0fa650a7c60429711f5a0b293802f23f96213ee3cf030b368250193e7011ce9c",
        "frozen_attack": "ac44b311dff31cac2911319bf39fdc7aa9d6097fc932f90eeed07a1eda738ec2",
        "results": "fc953c876ef67e3b5a6b915434667e4051d6ece47973574f8d82ee3e917cf556",
        "raw": "26422130d6b7d053c4b78fbc292e3deceb05db5eb72113c2b6e9175f2cb0ddad",
        "independent_validation": "1ef6c0f7df2c11f871ead0f7162c480d600a3730d0f4d455f9e88500841cd9db",
    }
    if manifest["bindings"].get("step87b") != expected_bindings:
        raise AssertionError("Step 87 binding drift")
    return manifest


def normalized_source_manifest(payload: dict) -> dict:
    return {
        "manifest_id": payload["manifest_id"],
        "preregistration_sha256": payload["preregistration_sha256"],
        "stage0_lock_sha256": payload["stage0_lock_sha256"],
        "records": [
            {key: value for key, value in record.items() if key != "source"}
            for record in payload["records"]
        ],
        "record_count": payload["counts"]["records"],
        "output_sha256": payload["output_sha256"],
    }


def verify_prepared_public_data(frozen_raw_manifest: bytes) -> None:
    stage_a = load(ROOT / "STEP87B_STAGEA_EXECUTION_LOCK_2026-08-12.json")
    stage_b = load(ROOT / "STEP87B_STAGEB_EXECUTION_LOCK_2026-08-12.json")
    if hashlib.sha256(frozen_raw_manifest).hexdigest() != stage_a["input_sha256"]["raw_manifest"]:
        raise AssertionError("frozen source manifest/Stage-A lock mismatch")
    expected = {
        "calibration": ROOT / "STEP87B_LABELED_CALIBRATION_PACKAGE_2026-08-12.json",
        "blind_manifest": ROOT / "STEP87B_BLIND_PARENT_INPUT_MANIFEST_2026-08-12.json",
        "split_audit": ROOT / "STEP87B_SPLIT_AND_SCHEMA_AUDIT_2026-08-12.json",
    }
    for name, path in expected.items():
        if not path.is_file() or sha256(path) != stage_a["input_sha256"][name]:
            raise AssertionError(f"regenerated Stage-A input mismatch: {name}")
    sealed = ROOT / "external_data/step87b_sealed/STEP87B_SEALED_HOLDOUT_OUTCOMES_2026-08-12.json"
    if not sealed.is_file() or sha256(sealed) != stage_b["input_sha256"]["sealed_holdout"]:
        raise AssertionError("regenerated sealed holdout mismatch")
    frozen = json.loads(frozen_raw_manifest.decode("utf-8"))
    regenerated = load(RAW_MANIFEST)
    if normalized_source_manifest(frozen) != normalized_source_manifest(regenerated):
        raise AssertionError("regenerated source URLs, bytes, hashes, or output hashes drifted")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--integrity-only", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    frozen_raw_manifest = RAW_MANIFEST.read_bytes()
    manifest = verify_integrity()
    checks = [
        run(
            [
                sys.executable,
                "validate_step82_anonymous_release.py",
                "--full" if args.full else "--integrity-only",
            ],
            "PASS_STEP82_ANONYMOUS_RELEASE",
        )
    ]
    if args.full:
        try:
            checks.append(
                run(
                    [sys.executable, "prepare_step87b_blind_tasks.py"],
                    "STEP87B_PREPARATION_COMPLETE_WITH_SEALED_OUTCOMES",
                )
            )
            verify_prepared_public_data(frozen_raw_manifest)
        finally:
            RAW_MANIFEST.write_bytes(frozen_raw_manifest)
        checks.append(
            run(
                [sys.executable, "run_step87b_portable_validator.py"],
                "PASS_STEP87B_INDEPENDENT_RECONSTRUCTION",
            )
        )
        checks.append(
            run(
                [sys.executable, "validate_step87_manuscript_integration.py"],
                "PASS_STEP87_MANUSCRIPT_INTEGRATION",
            )
        )
    print(
        json.dumps(
            {
                "verdict": "PASS_STEP87_ANONYMOUS_RELEASE",
                "mode": "full" if args.full else "integrity-only",
                "bundled_files": len(manifest["files"]),
                "public_source_packages_bundled": 0,
                "workstation_path_hits": 0,
                "step87_independently_reconstructed": bool(args.full),
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
