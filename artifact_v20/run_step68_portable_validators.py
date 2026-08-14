"""Run the locked validators against the path-neutral Step 68 release tree.

Historical runner hashes are resolved only through the audited path-redaction
map.  Validators and result files remain byte-identical to the Step 65 bundle.
Each checker runs in a fresh Python process to avoid cross-checker module state.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

from step68_portable_hash import historical_sha256_file


ROOT = Path(__file__).resolve().parent

CHECKERS = (
    ("verify_step27_sequential_frame_lower_bound", '"status": "PASS"'),
    ("validate_step30_phase_b1", '"status": "PASS_RESULT_CONSISTENCY"'),
    ("validate_step31_phase_b2", '"status": "PASS_RESULT_CONSISTENCY"'),
    ("validate_step35_coda_cross_method", "PASS_STEP35_RESULT_CONSISTENCY"),
    ("validate_step46_clone_robust_soft_weighting", "PASS_STEP46_CLONE_ROBUST_SOFT_WEIGHTING"),
    ("validate_step50_natural_alias_audit", "PASS_STEP50_NATURAL_ALIAS_AUDIT"),
    ("validate_step54_matrix_free_realism_audit", "PASS_STEP54_MATRIX_FREE_REALISM_AUDIT"),
    ("validate_step55_executable_endpoint_realism_audit", "PASS_STEP55_EXECUTABLE_ENDPOINT_REALISM_AUDIT"),
    ("validate_step56_calibration_to_holdout", "PASS_STEP56_CALIBRATION_TO_HOLDOUT_AUDIT"),
    ("validate_step61b_results", '"verdict": "VALIDATED"'),
)


def patch_historical_hashes(name: str, module: object) -> None:
    if name == "validate_step46_clone_robust_soft_weighting":
        setattr(module, "sha256", historical_sha256_file)
    elif name == "validate_step50_natural_alias_audit":
        module.sha = historical_sha256_file
        module.audit.file_sha256 = historical_sha256_file
    elif name == "validate_step54_matrix_free_realism_audit":
        module.step54.file_sha256 = historical_sha256_file
        module.step54.step50.file_sha256 = historical_sha256_file
    elif name == "validate_step55_executable_endpoint_realism_audit":
        module.file_sha256 = historical_sha256_file
        module.step54.file_sha256 = historical_sha256_file
        module.step54.step50.file_sha256 = historical_sha256_file
    elif name == "validate_step56_calibration_to_holdout":
        module.step54.file_sha256 = historical_sha256_file
        module.step54.step50.file_sha256 = historical_sha256_file


def run_one(name: str) -> None:
    module = importlib.import_module(name)
    patch_historical_hashes(name, module)
    module.main()


def orchestrate(compile_paper: bool) -> None:
    results = []
    for name, marker in CHECKERS:
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--one", name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            errors="replace",
        )
        elapsed = time.perf_counter() - started
        output = completed.stdout + completed.stderr
        passed = completed.returncode == 0 and marker in output
        results.append(
            {
                "checker": name,
                "passed": passed,
                "seconds": round(elapsed, 3),
                "returncode": completed.returncode,
            }
        )
        if not passed:
            raise AssertionError(
                {
                    "checker": name,
                    "returncode": completed.returncode,
                    "required_marker": marker,
                    "output_tail": output[-3000:],
                }
            )

    table = subprocess.run(
        [sys.executable, "validate_paper_tables.py"],
        cwd=ROOT / "paper_draft",
        capture_output=True,
        text=True,
        errors="replace",
    )
    table_output = table.stdout + table.stderr
    if table.returncode != 0 or "PASS_PAPER_TABLE_SOURCE_CONSISTENCY" not in table_output:
        raise AssertionError({"paper_table_validation": table_output[-3000:]})
    results.append({"checker": "paper_tables", "passed": True})

    if compile_paper:
        build = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=ROOT / "paper_draft",
            capture_output=True,
            text=True,
            errors="replace",
        )
        if build.returncode != 0:
            raise AssertionError({"latex_tail": (build.stdout + build.stderr)[-3000:]})
        results.append({"checker": "paper_compile", "passed": True})

    print(
        json.dumps(
            {
                "verdict": "PASS_STEP68_PORTABLE_VALIDATORS",
                "checkers": results,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", choices=[name for name, _ in CHECKERS])
    parser.add_argument("--compile-paper", action="store_true")
    args = parser.parse_args()
    if args.one:
        run_one(args.one)
    else:
        orchestrate(args.compile_paper)


if __name__ == "__main__":
    main()
