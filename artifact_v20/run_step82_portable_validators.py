"""Run the complete V7 validation suite inside the anonymous release tree."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path, marker: str, name: str, env: dict[str, str] | None = None) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        env=env,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or marker not in output:
        raise AssertionError(
            {
                "checker": name,
                "returncode": completed.returncode,
                "required_marker": marker,
                "output_tail": output[-6000:],
            }
        )
    return {"checker": name, "passed": True, "seconds": round(time.perf_counter() - started, 3)}


def independent_step80() -> None:
    module = importlib.import_module("validate_step80_main_t2a_v2_transfer")
    expected = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_VALIDATION_2026-08-10.json"
    expected_hash = sha256(expected)
    with tempfile.TemporaryDirectory(prefix="step80_validation_", dir=ROOT) as temporary:
        module.VALIDATION = Path(temporary) / expected.name
        module.validate()
        if sha256(module.VALIDATION) != expected_hash:
            raise AssertionError("independently regenerated Step 80 validation differs")


def orchestrate(skip_step80: bool) -> None:
    checks: list[dict] = []
    checks.append(
        run(
            [sys.executable, "run_step68_portable_validators.py"],
            ROOT,
            "PASS_STEP68_PORTABLE_VALIDATORS",
            "prior_result_validators",
        )
    )
    with tempfile.TemporaryDirectory(prefix="paper_build_", dir=ROOT) as temporary:
        build_dir = Path(temporary)
        checks.append(
            run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={build_dir}", "main.tex"],
                ROOT / "paper_draft",
                "All targets",
                "paper_build_in_temporary_directory",
            )
        )
        environment = dict(os.environ)
        environment["STEP81_BUILD_DIR"] = str(build_dir)
        checks.append(
            run(
                [sys.executable, "validate_step81_manuscript_integration.py"],
                ROOT,
                "PASS_STEP81_MANUSCRIPT_INTEGRATION",
                "current_manuscript_integration",
                env=environment,
            )
        )
    if not skip_step80:
        checks.append(
            run(
                [sys.executable, str(Path(__file__).resolve()), "--step80-independent"],
                ROOT,
                "PASS_STEP80_INDEPENDENT_VALIDATION",
                "step80_independent_reconstruction",
            )
        )
    print(
        json.dumps(
            {
                "verdict": "PASS_STEP82_PORTABLE_VALIDATORS",
                "step80_independently_recomputed": not skip_step80,
                "checks": checks,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step80-independent", action="store_true")
    parser.add_argument("--skip-step80", action="store_true")
    args = parser.parse_args()
    if args.step80_independent:
        independent_step80()
    else:
        orchestrate(args.skip_step80)


if __name__ == "__main__":
    main()
