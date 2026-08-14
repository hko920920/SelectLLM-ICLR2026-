"""Validate V12 integrity, anonymity, inherited closure, and Step 90 replay."""

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
EXPECTED_STEP89 = {
    "results": "efb16dbc16830d2398f98592747018fb7f1da952a4022e76ea0f06270b892ec0",
    "raw": "4e7ad9d8e622700cd3504420876e368733d65ca2b700613682f20a93fa9699d0",
}
EXPECTED_STEP80_INHERITED = {
    "STEP80_MAIN_T2A_V2_TRANSFER_VALIDATION_2026-08-10.json": "62590fb9a35fa828b05709d172a66be67512e8fe41e960c578eaa892a307cdc6",
    "validate_step80_main_t2a_v2_transfer.py": "e0c1d4d54fb405a2d0d8de89bb5b7a4a54b6955781b95e8301eeec8322b9a1a2",
}
EXPECTED_V11_MANIFEST = "616be697a0424cb674a5a8157bef5c3ee0e52c8f36fa479dd5b90504be9e4c34"
EXPECTED_STEP90 = {
    "stage0_manifest": "4aa672d6657d1324636c74a29510c8a06a5ea804c6f5db0fd51d3272c9df87cb",
    "frozen_config": "3dfda0e5deb6db51bba510fcf074360f4a8764a95271c16f427bbec1b585ecb4",
    "preregistration": "70147e0b5b1c70024e95492f23d31dbe9b739866468a407237bc0cb661b05c55",
    "execution_lock": "b73730c970f6e8d78a59978fe1e379168ebed498ca4bbc1123c39b8bf3ff42b3",
    "results": "f699eee78c4fd84139933e469d1c1c2bfbefb9ffe4a6cc3f278a2996f55d5654",
    "raw": "5ef206b57d2204c128e621f07c83f8a621d1e94ab6856cbddc05d8e3bbd44d6d",
    "independent_validator": "ee3aa0b1377733c25fcf3d6c356faf8d0a6696eee368127b83ebdceadc5aa918",
    "independent_receipt": "0226d76e493905bfed74b62647729f4ed3e83d37bc127c57d81d20a41bf4aa79",
    "manuscript_validator": "77beaa671e314a89e7be0e861c7e1ab7ddec7021cd61b15113317aa6f3cf9d8e",
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
        raise AssertionError({
            "command": command,
            "returncode": completed.returncode,
            "marker": marker,
            "output_tail": output[-12000:],
        })
    return {"command": command, "marker": marker, "passed": True}


def verify_integrity() -> dict:
    manifest = load(MANIFEST)
    if manifest.get("schema") != "step90.anonymous_validation_closure_release.v6":
        raise AssertionError("unexpected V12 schema")
    refresh = manifest.get("refresh", {})
    expected_refresh = {
        "numerical_evidence_changed": True,
        "derived_step89_step90_packages_embedded": True,
        "public_parent_weights_embedded": False,
        "third_party_repositories_embedded": False,
    }
    for key, expected in expected_refresh.items():
        if refresh.get(key) is not expected:
            raise AssertionError({"refresh_key": key, "observed": refresh.get(key)})
    if refresh.get("base_manifest_sha256") != EXPECTED_V11_MANIFEST:
        raise AssertionError("inherited V11 manifest binding drift")

    records = manifest["files"]
    for relative, record in records.items():
        path = ROOT / Path(relative)
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise AssertionError(f"missing or drifting bundled file: {relative}")

    forbidden_weight_names = {
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
        "flax_model.msgpack",
    }
    parent_weights = [relative for relative in records if Path(relative).name in forbidden_weight_names]
    if parent_weights:
        raise AssertionError({"public_parent_weights_embedded": parent_weights})

    workstation_hits = []
    identity_hits = []
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
        if re.search(r"(?i)\bSOGANG\b", text):
            identity_hits.append(relative)
    if workstation_hits or identity_hits:
        raise AssertionError({"workstation_path_hits": workstation_hits, "identity_hits": identity_hits})

    if manifest["bindings"].get("step89_retained_negative") != EXPECTED_STEP89:
        raise AssertionError("Step 89 retained-negative binding drift")
    if manifest["bindings"].get("step90") != EXPECTED_STEP90:
        raise AssertionError("Step 90 binding drift")
    for relative, expected in EXPECTED_STEP80_INHERITED.items():
        if records.get(relative, {}).get("sha256") != expected:
            raise AssertionError(f"inherited Step 80 closure drift: {relative}")
    if sha256(ROOT / "paper_draft/main.pdf") != manifest["bindings"]["paper_pdf_sha256"]:
        raise AssertionError("paper PDF binding drift")
    main_source = (ROOT / "paper_draft/main.tex").read_text(encoding="utf-8")
    if "\\author{Anonymous Authors}" not in main_source or "\\iclrfinalcopy" in main_source.splitlines():
        raise AssertionError("anonymous manuscript mode drift")

    expected_summary = {
        "step89_retained_negative_tasks": 1,
        "step90_confirmatory_tasks": 2,
        "step90_passing_tasks": 2,
        "step90_paired_active_trajectories": 4000,
        "step90_fixed_query_replays": 2000,
        "step90_independent_checks": 12077,
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
    checks: list[dict[str, object]] = []
    if args.full:
        checks.extend([
            run([sys.executable, "validate_step90_independent.py"], "PASS_STEP90_INDEPENDENT_VALIDATION"),
            run([sys.executable, "validate_step90_manuscript_integration.py"], "PASS_STEP90_MANUSCRIPT_INTEGRATION"),
        ])
        verify_integrity()
    print(json.dumps({
        "verdict": "PASS_STEP90_ANONYMOUS_RELEASE",
        "mode": "full" if args.full else "integrity-only",
        "bundled_files": len(manifest["files"]),
        "workstation_path_hits": 0,
        "public_parent_weights_bundled": 0,
        "inherited_v11_manifest_bound": True,
        "step90_independently_reconstructed": bool(args.full),
        "step90_passing_tasks": ["mnli", "qqp"],
        "step89_retained_negative": "qnli",
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
