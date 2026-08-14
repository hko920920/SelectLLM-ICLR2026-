"""Portable validator for the Step 105 / V20 anonymous review artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "STEP105_RELEASE_MANIFEST.json"
PARENT_MANIFEST = ROOT / "STEP105_V19_PARENT_MANIFEST.json"
EXPECTED_PARENT_ZIP = "d9a3ede4eee13ce9b8229eb4a998b51d1043861f69f259fde952ec62fb0609e0"
EXPECTED_PARENT_MANIFEST = "065be708df377ca9d6d0c65151d9d361bad946f224d49652fbc43a29a12d300a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, tol: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        raise AssertionError((actual, expected))


def safe_relative(relative: str) -> None:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise AssertionError({"unsafe_manifest_path": relative})


def verify_anonymity(records: dict[str, dict]) -> None:
    account = b"sog" + b"ang"
    banned = (
        b"c:\\users\\" + account,
        b"/users/" + account,
        b"documents\\" + b"iclr 2027 plan",
    )
    text_suffixes = {".py", ".tex", ".bib", ".md", ".json", ".txt", ".csv", ".sty"}
    for relative in records:
        path = ROOT / relative
        if path.suffix.lower() not in text_suffixes or path.stat().st_size > 40_000_000:
            continue
        payload = path.read_bytes().lower()
        for token in banned:
            if token in payload:
                raise AssertionError({"anonymity_token": token.decode(errors="ignore"), "file": relative})
    reader = PdfReader(str(ROOT / "paper_draft/main.pdf"))
    metadata = " ".join(str(value) for value in (reader.metadata or {}).values()).lower()
    if "sogang" in metadata or "c:\\users" in metadata:
        raise AssertionError("PDF metadata anonymity failure")


def verify_integrity() -> dict:
    manifest = load(MANIFEST)
    if manifest.get("schema") != "step105.anonymous_validation_closure_release.v14":
        raise AssertionError("unexpected V20 schema")
    records = manifest.get("files", {})
    if not records:
        raise AssertionError("empty release manifest")
    for relative, record in records.items():
        safe_relative(relative)
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError({"missing": relative})
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise AssertionError({"hash_or_size_drift": relative})
    if manifest["summary"]["files"] != len(records):
        raise AssertionError("manifest file cardinality drift")
    if manifest["summary"]["bytes"] != sum(record["bytes"] for record in records.values()):
        raise AssertionError("manifest byte cardinality drift")
    bindings = manifest["bindings"]
    if bindings["parent_v19_zip_sha256"] != EXPECTED_PARENT_ZIP:
        raise AssertionError("V19 parent ZIP binding drift")
    if bindings["parent_v19_manifest_sha256"] != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("V19 parent manifest binding drift")
    if sha256(PARENT_MANIFEST) != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("embedded V19 manifest drift")
    if sha256(ROOT / "paper_draft/main.pdf") != bindings["paper_pdf_sha256"]:
        raise AssertionError("paper binding drift")

    primary104 = load(ROOT / "STEP104_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json")
    primary105 = load(ROOT / "STEP105_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json")
    raw105 = load(ROOT / "STEP105_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json")
    recon105 = load(ROOT / "STEP105_ARTIFACT_RECONSTRUCTION_2026-08-13.json")
    if primary104["decision"] != "GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY":
        raise AssertionError("Step 104 status drift")
    if primary105["decision"] != "NO_GO_RETAIN_STEP105_PROSPECTIVE_NEGATIVE":
        raise AssertionError("Step 105 status drift")
    if primary105["failed_gates"] != ["terminal_mean_at_least_half_point", "active_minus_fixed_mean_at_least_half_point"]:
        raise AssertionError("Step 105 failed-gate drift")
    if sum(primary105["gates"].values()) != 14 or len(primary105["gates"]) != 16:
        raise AssertionError("Step 105 gate-count drift")
    if raw105["verdict"] != "PASS_STEP105_PRIMARY_INDEPENDENT_RECONSTRUCTION" or raw105["failed_checks"] or not all(raw105["checks"].values()):
        raise AssertionError("Step 105 raw-input receipt failed")
    if recon105["decision"] != "PASS_STEP105_ARTIFACT_RECONSTRUCTION" or recon105["failed_checks"] or not all(recon105["checks"].values()):
        raise AssertionError("Step 105 reconstruction receipt failed")

    summary = manifest["summary"]
    expected = {
        "paper_pages": 28,
        "main_scientific_text_pages": 9,
        "references_begin_page": 10,
        "step104_decision": primary104["decision"],
        "step105_rows": 38000,
        "step105_runs": 3000,
        "step105_target_selector_grid_cells": 0,
        "step105_decision": primary105["decision"],
        "step105_failed_gates": primary105["failed_gates"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise AssertionError({"summary": key, "actual": summary.get(key), "expected": value})
    close(summary["step105_terminal_harm_pp"], 0.35273333333333357)
    close(summary["step105_cumulative_regret_delta"], 0.017924666666666682)
    close(summary["step105_path_change_rate"], 0.999)
    close(summary["step105_final_root_change_rate"], 0.417)
    if summary["step105_fixed_query_exact_zero"] is not True:
        raise AssertionError("fixed-query summary drift")

    reader = PdfReader(str(ROOT / "paper_draft/main.pdf"))
    if len(reader.pages) != 28:
        raise AssertionError({"paper_pages": len(reader.pages)})
    if "LIMITATIONS" not in (reader.pages[8].extract_text() or "").upper():
        raise AssertionError("main scientific text boundary drift")
    if "doi:" not in (reader.pages[9].extract_text() or "").lower():
        raise AssertionError("reference boundary drift")
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted((ROOT / "paper_draft").rglob("*.tex")))
    for token in (
        "app:step105", "99.90\\%", "+.3527", "+.017925",
        "NO\\_GO\\_RETAIN\\_STEP105\\_PROSPECTIVE\\_NEGATIVE",
        "not a second primary success",
    ):
        if token not in source:
            raise AssertionError({"missing_manuscript_token": token})
    verify_anonymity(records)
    return manifest


def copy_relative(destination: Path, relatives: list[str]) -> None:
    for relative in sorted(set(relatives)):
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def run(command: list[str], expected: str, cwd: Path) -> str:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(command, cwd=cwd, env=environment, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or expected not in output:
        raise AssertionError({"command": command, "returncode": completed.returncode, "output_tail": output[-5000:]})
    return output


def step104_inputs(include_raw: bool) -> list[str]:
    lock = load(ROOT / "STEP104_PREOUTCOME_LOCK_2026-08-13.json")
    relatives = [
        "step93_common.py", "step100_common.py", "step102_common.py",
        "step102_executable_adapter_endpoint.py",
        "STEP102_STAGEA_COMPLETE_LEDGER_2026-08-13.json",
        "STEP104_PREOUTCOME_LOCK_2026-08-13.json",
        "STEP104_PREOUTCOME_PREDICTIONS_2026-08-13.npz",
        "STEP104_PRIMARY_CONFIRMATORY_ARRAYS_2026-08-13.npz",
        "STEP104_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json",
        lock["input_file"], lock["sealed_file"],
    ]
    for group in ("authority_sha256", "code_sha256", "model_sha256", "model_audit_sha256", "preoutcome_sha256"):
        relatives.extend(lock[group])
    relatives.append("step104_validate_primary_independent.py" if include_raw else "validate_step104_artifact_reconstruction.py")
    return relatives


def step105_inputs(include_raw: bool) -> list[str]:
    lock = load(ROOT / "STEP105_PREOUTCOME_LOCK_2026-08-13.json")
    relatives = [
        "STEP105_PREOUTCOME_LOCK_2026-08-13.json",
        "STEP105_PREOUTCOME_PREDICTIONS_2026-08-13.npz",
        "STEP105_PRIMARY_CONFIRMATORY_ARRAYS_2026-08-13.npz",
        "STEP105_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json",
        lock["test_input_file"], lock["sealed_file"],
    ]
    relatives.extend(lock["code_sha256"])
    relatives.extend(lock["adapter_sha256"])
    relatives.append("step105_validate_independent.py" if include_raw else "validate_step105_artifact_reconstruction.py")
    return relatives


def reconstruct(script: str, expected: str, output_name: str, inputs: list[str]) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="step105_v20_reconstruct_") as raw:
        temp_root = Path(raw)
        copy_relative(temp_root, inputs)
        run([sys.executable, script], expected, temp_root)
        generated = temp_root / output_name
        if sha256(generated) != sha256(ROOT / output_name):
            raise AssertionError({"receipt_not_byte_reproduced": output_name})
    return {"name": script, "passed": True}


def rebuild_paper() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="step105_v20_paper_") as raw:
        temp_paper = Path(raw) / "paper_draft"
        shutil.copytree(ROOT / "paper_draft", temp_paper)
        run(["latexmk", "-pdf", "-g", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], "Output written on", temp_paper)
        reader = PdfReader(str(temp_paper / "main.pdf"))
        if len(reader.pages) != 28:
            raise AssertionError({"rebuilt_pages": len(reader.pages)})
        log = (temp_paper / "main.log").read_text(encoding="utf-8", errors="replace")
        if "undefined references" in log.lower() or "undefined citations" in log.lower() or "overfull \\hbox" in log.lower():
            raise AssertionError("paper rebuild warning")
    return {"name": "paper_rebuild", "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--integrity-only", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--raw-input", action="store_true")
    args = parser.parse_args()

    manifest = verify_integrity()
    checks: list[dict[str, object]] = [{"name": "manifest_anonymity_and_scientific_status", "passed": True}]
    if args.full or args.raw_input:
        run([sys.executable, "paper_draft/validate_paper_tables.py"], "PASS_PAPER_TABLE_SOURCE_CONSISTENCY", ROOT)
        checks.append({"name": "paper_number_source_consistency", "passed": True})
        checks.append(rebuild_paper())
        checks.append(reconstruct(
            "validate_step104_artifact_reconstruction.py",
            "PASS_STEP104_ARTIFACT_RECONSTRUCTION",
            "STEP104_ARTIFACT_RECONSTRUCTION_2026-08-13.json",
            step104_inputs(False),
        ))
        checks.append(reconstruct(
            "validate_step105_artifact_reconstruction.py",
            "PASS_STEP105_ARTIFACT_RECONSTRUCTION",
            "STEP105_ARTIFACT_RECONSTRUCTION_2026-08-13.json",
            step105_inputs(False),
        ))
    if args.raw_input:
        checks.append(reconstruct(
            "step104_validate_primary_independent.py",
            "PASS_STEP104_PRIMARY_INDEPENDENT_RECONSTRUCTION",
            "STEP104_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json",
            step104_inputs(True),
        ))
        checks.append(reconstruct(
            "step105_validate_independent.py",
            "PASS_STEP105_PRIMARY_INDEPENDENT_RECONSTRUCTION",
            "STEP105_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json",
            step105_inputs(True),
        ))

    payload = {
        "verdict": "PASS_STEP105_ANONYMOUS_RELEASE",
        "mode": "raw-input" if args.raw_input else "full" if args.full else "integrity-only",
        "bundled_files": len(manifest["files"]),
        "paper_pages": manifest["summary"]["paper_pages"],
        "main_scientific_text_pages": manifest["summary"]["main_scientific_text_pages"],
        "step104_decision": manifest["summary"]["step104_decision"],
        "step105_decision": manifest["summary"]["step105_decision"],
        "step105_materiality_gate_passed": False,
        "checks": checks,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
