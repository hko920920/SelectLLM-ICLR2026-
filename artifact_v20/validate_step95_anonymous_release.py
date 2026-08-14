"""Validate the isolated Step 95 / V17 anonymous review release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "STEP82_RELEASE_MANIFEST.json"
PARENT_MANIFEST = ROOT / "STEP95_V16_PARENT_MANIFEST.json"
TEXT_SUFFIXES = {".py", ".md", ".tex", ".bib", ".json", ".txt", ".sty", ".bst"}
DATA_ROLES = {
    "sealed_test_input", "sealed_test_outcome", "development_input",
    "development_output", "complete_development_ledger",
    "quarantined_preanalysis_partial",
}
WINDOWS_USER_PATH = re.compile(r"(?i)[A-Z]:\\Users\\[^\\\"'\s]+")

EXPECTED_PARENT_MANIFEST = "d2cfcf1be836d57a995d3b3b0c6a4371081c8fc76c37c620e35c3a8c0a9f503e"
EXPECTED_PARENT_ZIP = "130c5c528bc5ce94dbeb50867a99094a81eb8073e1c2a2388a5288581db15949"
EXPECTED_PDF = "7852cc5ddb1fd5741c62e6d467dda3f0c331fdf6a6276f3799a72a2ac80d9c30"
EXPECTED_STEP95 = {
    "preregistration": "f1212135928d132a0bcd9c045e982ba22e06b596382dcb31c0fb5f04ce97bdf1",
    "amendment_a": "551a6a1a18ce836577df77b40240c0cadcee0fa57ec386cdc2881613a8f12694",
    "incident_b": "9c4273dbdc196af976604479337ec07f411cb367e951f4248b44603ad9c85376",
    "stage0_manifest": "c7ab140ba2a35a8ce7e6c56a225f130dd93fba040fe62c114a36e4d4c86b74da",
    "development_input": "15203a670d476ffc4bcb4f7bb72412a7121438b7b459a556ffbc9b4e340f8ea7",
    "development_arrays": "3fe4d4f707a25f65511014005c60c5fca2fc333572275709a97d695f90716c57",
    "complete_development_ledger": "b6e9a801ce29a9de0c3f62bba7510ee43e9c7458290ebf3556cb2ffe43317d00",
    "frozen_config": "1a36fc7d02929fd40fd023e2d7c6fc40ca5e11358c664a4a433e6716917359c9",
    "independent_validation_receipt": "e3e09cbd51d48b06632b4ff1ba5b6ea64f5ef374a964bd08edce802c3495f0da",
    "development_stop_receipt": "e7ec557021222257566efc2c7fcd233009246bad79f959fa336c71c3a55c32e9",
    "review_closure_report": "be3a8157f84f3fe3abf4537f8e0719eab5b336ccac3c8f10c8e07e058c5a93ad",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(observed: float, expected: float, atol: float = 1e-14) -> bool:
    return math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=atol)


def run(command: list[str], marker: str | None = None, cwd: Path = ROOT) -> dict[str, object]:
    completed = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, errors="replace", check=False
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or (marker is not None and marker not in output):
        raise AssertionError({
            "command": command, "returncode": completed.returncode,
            "marker": marker, "output_tail": output[-12000:],
        })
    return {"command": command, "marker": marker, "passed": True}


def assert_pdf_boundary(path: Path) -> None:
    reader = PdfReader(str(path))
    if len(reader.pages) != 24:
        raise AssertionError({"paper_pages": len(reader.pages)})
    page9 = (reader.pages[8].extract_text() or "").upper()
    page10 = (reader.pages[9].extract_text() or "").upper()
    page22 = (reader.pages[21].extract_text() or "").upper()
    page9_compact = "".join(page9.split())
    page22_compact = "".join(page22.split())
    if "LIMITATIONS" not in page9 or "TARGETEDEVIDENCE-FRAMEFAILURES" not in page9_compact:
        raise AssertionError("scientific conclusion does not finish on page 9")
    if "REFERENCES" in {line.replace(" ", "").strip() for line in page9.splitlines()}:
        raise AssertionError("references begin before page 10")
    if "REFERENCES" not in {line.replace(" ", "").strip() for line in page10.splitlines()}:
        raise AssertionError("references do not begin on page 10")
    if "MULTINLIFULL-ADAPTERDEVELOPMENTSTOP" not in page22_compact:
        raise AssertionError("Step 95 appendix section missing from page 22")


def verify_integrity() -> dict:
    manifest = load(MANIFEST)
    if manifest.get("schema") != "step95.anonymous_validation_closure_release.v11":
        raise AssertionError("unexpected V17 schema")
    refresh = manifest.get("refresh", {})
    expected_refresh = {
        "numerical_evidence_changed": True,
        "complete_step95_grid_embedded": True,
        "all_step95_learned_files_embedded": True,
        "development_gate_stop": True,
        "matched_test_outcome_opened": False,
        "mismatched_test_substituted": False,
        "confirmatory_output_embedded": False,
        "public_base_model_weights_embedded": False,
        "third_party_repositories_embedded": False,
        "isolated_from_worktree_by_byte_copy": True,
    }
    for key, expected in expected_refresh.items():
        if refresh.get(key) is not expected:
            raise AssertionError({"refresh_key": key, "observed": refresh.get(key)})
    if refresh.get("parent_v16_manifest_sha256") != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("parent V16 manifest binding drift")
    if refresh.get("parent_v16_zip_sha256") != EXPECTED_PARENT_ZIP:
        raise AssertionError("parent V16 ZIP binding drift")

    records = manifest.get("files", {})
    if not records:
        raise AssertionError("empty release manifest")
    for relative, record in records.items():
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or posix.as_posix() != relative:
            raise AssertionError({"unsafe_manifest_path": relative})
        path = ROOT / Path(relative)
        if not path.is_file() or path.stat().st_size != record.get("bytes") or sha256(path) != record.get("sha256"):
            raise AssertionError({"missing_or_drifting_file": relative})
    forbidden_weight_names = {"model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack"}
    public_weights = [relative for relative in records if Path(relative).name in forbidden_weight_names]
    if public_weights:
        raise AssertionError({"public_base_model_weights_embedded": public_weights})
    confirmatory_names = {
        "STEP95_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json",
        "STEP95_CONFIRMATORY_RAW_2026-08-13.npz",
        "STEP95_CONFIRMATORY_RESULTS_2026-08-13.json",
        "STEP95_INDEPENDENT_VALIDATION_2026-08-13.json",
    }
    if confirmatory_names.intersection(records):
        raise AssertionError("Step 95 confirmatory output is bundled despite development stop")

    workstation_hits: list[str] = []
    identity_hits: list[str] = []
    for relative, record in records.items():
        path = ROOT / Path(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES or record.get("role") in DATA_ROLES:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if WINDOWS_USER_PATH.search(source):
            workstation_hits.append(relative)
        if re.search(r"(?i)\bSOGANG\b", source):
            identity_hits.append(relative)
    if workstation_hits or identity_hits:
        raise AssertionError({"workstation_path_hits": workstation_hits, "identity_hits": identity_hits})

    if sha256(PARENT_MANIFEST) != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("embedded V16 manifest byte drift")
    parent = load(PARENT_MANIFEST)
    if parent.get("schema") != "step94.anonymous_validation_closure_release.v10":
        raise AssertionError("embedded V16 manifest schema drift")
    bindings = manifest.get("bindings", {})
    if bindings.get("parent_v16_manifest_sha256") != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("manifest-to-V16 snapshot binding drift")
    step95 = bindings.get("step95", {})
    for key, expected in EXPECTED_STEP95.items():
        if step95.get(key) != expected:
            raise AssertionError({"step95_binding": key, "observed": step95.get(key)})

    config = load(ROOT / "STEP95_STAGEA_FROZEN_CONFIG_2026-08-13.json")
    ledger = load(ROOT / "STEP95_STAGEA_COMPLETE_LEDGER_2026-08-13.json")
    validation = load(ROOT / "STEP95_STAGEA_INDEPENDENT_VALIDATION_2026-08-13.json")
    stop_receipt = load(ROOT / "STEP95_DEVELOPMENT_STOP_RECEIPT_2026-08-13.json")
    model_map = step95.get("model_weights", {})
    if model_map != config["adapter_sha256"] or len(model_map) != 4:
        raise AssertionError("complete Step 95 adapter map drift")
    if any(records.get(relative, {}).get("sha256") != digest for relative, digest in model_map.items()):
        raise AssertionError("Step 95 adapter record drift")
    if sum(record.get("role") == "learned_error_adapter" for record in records.values()) != 4:
        raise AssertionError("Step 95 adapter role count drift")
    if len({row["checkpoint"]["sha256"] for row in ledger["adapter_audits"]}) != 4:
        raise AssertionError("Step 95 adapters are not hash-distinct")
    if any(row["checkpoint"]["parameter_count"] != 14767874 for row in ledger["adapter_audits"]):
        raise AssertionError("Step 95 adapter parameter count drift")
    if len(step95.get("test_inputs", {})) != 1 or len(step95.get("sealed_outcomes", {})) != 1:
        raise AssertionError("Step 95 test seal binding count drift")

    if bindings.get("paper_pdf_sha256") != EXPECTED_PDF:
        raise AssertionError("paper PDF manifest binding drift")
    pdf = ROOT / "paper_draft/main.pdf"
    if sha256(pdf) != EXPECTED_PDF:
        raise AssertionError("paper PDF byte drift")
    assert_pdf_boundary(pdf)
    main_source = (ROOT / "paper_draft/main.tex").read_text(encoding="utf-8")
    active_lines = [line.strip() for line in main_source.splitlines() if not line.lstrip().startswith("%")]
    if "\\author{Anonymous Authors}" not in main_source or "\\iclrfinalcopy" in active_lines:
        raise AssertionError("anonymous manuscript mode drift")
    manuscript = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / "paper_draft").rglob("*.tex"))
    )
    required_manuscript = (
        "four 14.77M-parameter MNLI adapters",
        "the matched test stays sealed",
        "no audit passes the $.5$-point terminal bridge",
        "MNLI stops at development and leaves its test sealed",
        "NO\\_GO\\_DEVELOPMENT\\_GATE\\_STOP",
    )
    if any(phrase not in manuscript for phrase in required_manuscript):
        raise AssertionError("Step 95 manuscript scope or numerical statement drift")

    if ledger.get("decision") != "NO_GO_DEVELOPMENT_GATE_STOP":
        raise AssertionError("literal development-stop decision drift")
    if len(ledger.get("search_complete_48", [])) != 48 or len(ledger.get("verification_top_8", [])) != 8:
        raise AssertionError("complete Step 95 grid drift")
    selected = ledger["selected_verification"]
    if not close(selected["path_change_rate"], 0.9933333333333333):
        raise AssertionError("path-change result drift")
    if not close(selected["query_set_change_rate"], 0.992):
        raise AssertionError("query-set result drift")
    if not close(selected["final_root_change_rate"], 0.21133333333333335):
        raise AssertionError("terminal-root result drift")
    if not close(selected["inference"]["terminal"]["mean"], 0.0004986666666666671):
        raise AssertionError("terminal result drift")
    if not close(selected["inference"]["cumulative"]["mean"], 0.05055733333333333):
        raise AssertionError("cumulative result drift")
    if not close(selected["inference"]["cumulative"]["bootstrap_95"][0], 0.042609299999999996):
        raise AssertionError("cumulative interval drift")
    expected_failed = {
        "alias_losses_at_most_one_point_threshold_search_verify",
        "search_and_verification_terminal_mean_at_least_half_point",
        "search_and_verification_active_minus_fixed_mean_at_least_half_point",
    }
    if set(ledger.get("failed_gates", [])) != expected_failed:
        raise AssertionError("Step 95 failed-gate set drift")
    if any(ledger["gates"].get(key) is not False for key in expected_failed):
        raise AssertionError("Step 95 no-go gate truth drift")
    if any(value is not True for key, value in ledger["gates"].items() if key not in expected_failed):
        raise AssertionError("Step 95 passed-gate truth drift")
    if validation.get("decision") != ledger["decision"] or validation.get("sealed_test_opened") is not False:
        raise AssertionError("Stage-A independent validation decision drift")
    if any(value is not True for value in validation.get("checks", {}).values()):
        raise AssertionError("Stage-A independent validation receipt drift")
    if stop_receipt.get("decision") != ledger["decision"]:
        raise AssertionError("development-stop receipt decision drift")
    if stop_receipt["sealed_outcome"].get("opened_for_step95_confirmation") is not False:
        raise AssertionError("sealed test opened according to stop receipt")
    if set(stop_receipt.get("confirmatory_outputs_absent", [])) != confirmatory_names:
        raise AssertionError("confirmatory-absence receipt drift")

    closure = " ".join((ROOT / "STEP95_REVIEW_CLOSURE_REPORT_2026-08-13.md").read_text(encoding="utf-8").split())
    required_closure = (
        "NO_GO_DEVELOPMENT_GATE_STOP",
        "sealed `validation_matched` outcome was not opened",
        "The result is retained rather than replaced by another task",
    )
    if any(phrase not in closure for phrase in required_closure):
        raise AssertionError("Step 95 review-closure report drift")

    expected_summary = {
        "paper_pages": 24,
        "main_scientific_text_pages": 9,
        "step95_task": "mnli_validation_matched",
        "step95_public_root_bank": 8,
        "step95_clean_roster": 6,
        "step95_learned_adapters": 4,
        "step95_parameters_per_adapter": 14767874,
        "step95_search_conditions": 48,
        "step95_verification_conditions": 8,
        "step95_verification_runs": 1500,
        "step95_path_change_rate": 0.9933333333333333,
        "step95_query_set_change_rate": 0.992,
        "step95_final_root_change_rate": 0.21133333333333335,
        "step95_cumulative_regret_delta": 0.05055733333333333,
        "step95_terminal_harm_pp": 0.04986666666666671,
        "step95_fixed_query_exact_zero": True,
        "step95_sealed_test_opened": False,
        "step95_decision": "NO_GO_DEVELOPMENT_GATE_STOP",
        "review_closure": "source_faithful_anytime_replication_material_terminal_condition_open",
    }
    summary = manifest.get("summary", {})
    for key, expected in expected_summary.items():
        observed = summary.get(key)
        if isinstance(expected, float):
            if not close(observed, expected):
                raise AssertionError({"summary_key": key, "observed": observed})
        elif observed != expected:
            raise AssertionError({"summary_key": key, "observed": observed})
    if summary.get("files") != len(records):
        raise AssertionError("manifest file-count summary drift")
    if summary.get("bytes") != sum(record["bytes"] for record in records.values()):
        raise AssertionError("manifest byte-count summary drift")
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
            run(
                [sys.executable, "validate_step95_stagea.py", "--check-only"],
                "STEP95_STAGEA_INDEPENDENT_RECONSTRUCTION_V1",
            ),
            run(
                [sys.executable, "paper_draft/validate_paper_tables.py"],
                "PASS_PAPER_TABLE_SOURCE_CONSISTENCY",
            ),
            run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=_validation_build_v17", "main.tex"],
                cwd=ROOT / "paper_draft",
            ),
        ])
        assert_pdf_boundary(ROOT / "paper_draft/_validation_build_v17/main.pdf")
        verify_integrity()
    print(json.dumps({
        "verdict": "PASS_STEP95_ANONYMOUS_RELEASE",
        "mode": "full" if args.full else "integrity-only",
        "bundled_files": len(manifest["files"]),
        "workstation_path_hits": 0,
        "public_base_model_weights_bundled": 0,
        "parent_v16_manifest_bound": True,
        "step95_adapter_weight_files": 4,
        "step95_independently_reconstructed": bool(args.full),
        "step95_sealed_test_opened": False,
        "step95_decision": "NO_GO_DEVELOPMENT_GATE_STOP",
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
