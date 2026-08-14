"""Validate the isolated Step 94 / V16 anonymous review release."""

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
PARENT_MANIFEST = ROOT / "STEP94_V15_PARENT_MANIFEST.json"
TEXT_SUFFIXES = {".py", ".md", ".tex", ".bib", ".json", ".txt", ".sty", ".bst"}
DATA_ROLES = {
    "sealed_test_input",
    "sealed_test_outcome",
    "development_output",
    "complete_development_ledger",
    "quarantined_preanalysis_partial",
}
WINDOWS_USER_PATH = re.compile(r"(?i)[A-Z]:\\Users\\[^\\\"'\s]+")

EXPECTED_PARENT_MANIFEST = "14f4a65b5f7e450ca3f546fb6ea687cf7498162d0303eeaef534c605ff6187af"
EXPECTED_PARENT_ZIP = "1743a7b696b23a02688aa0bcf9e461ccab6c9e67985b8bd0bd8a8611da434a2a"
EXPECTED_PDF = "4fc9466b855f8c9c8a360d9a31022290e2749aca45a015eefe4ba58d2d15e311"
EXPECTED_STEP94 = {
    "preregistration": "31c96d1378d8fd841835ab509a50ee2fcb4b19480e317e43146ec37a01cf56f5",
    "amendment_a": "8498c1d77a12ff701132ea02827ada29655032330e64b8d34937380135698c9b",
    "amendment_b": "704071b0d983fea4f9afd9c38789ab27ea34a34dca05c75e4983f38796461556",
    "amendment_c": "dad7f95ce810647dbf0f4cc74944a7765ffd3b8fdf6a28d101c712c44b901adc",
    "amendment_d": "35c3e68ce088e9112f377b4b9ade2b3bd8b74b960e14ed10a3ae0c6a3944ff5d",
    "stage0_manifest": "cbaa7041bb4ff22fb39f00fdc35e4b779cfcd24dadbb1207dc92fbe9f36cb1d2",
    "complete_development_ledger": "61e3af79ac0824cb89ff57cace0ba1e505f39c5ef74912dbd79fb0d55c6e4428",
    "frozen_config": "32c00b22b458851f0d6a7a169364be4e99cdc4adc84199a053b495db113e2121",
    "execution_lock": "8ece3313a2b1fde60e9dce0943c45dc43c94f730ae94a88a84027845dc87628f",
    "raw": "76aec828348bee627299c6c9f1da1a57f7390ee8f2b4916a89c0d37ea919b700",
    "literal_results": "e93187ecc87341cb24191b46b97372e0e28caf4bb526e3ff8f55319ea0c52ccd",
    "locked_independent_validator": "505df6a7b1588cee5ac9e30c3b746f88dc3a214cfd2a55ca51c25012ecff5c2d",
    "independent_validation_receipt": "a3bd46c14edd3ccaf1cfd823c44b2a0d757e51b581a50474a52306563bcd86f7",
    "artifact_reconstruction_validator": "1dc805179483c3c9892e1ca3998906c0c5cff6858d120bc9f68cdbd35cda353b",
    "review_closure_report": "5e7d10379246f5ebe135de3f3760e4451952a7262a4499a0076430c3f38d3b25",
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
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or (marker is not None and marker not in output):
        raise AssertionError(
            {
                "command": command,
                "returncode": completed.returncode,
                "marker": marker,
                "output_tail": output[-12000:],
            }
        )
    return {"command": command, "marker": marker, "passed": True}


def assert_pdf_boundary(path: Path) -> None:
    reader = PdfReader(str(path))
    if len(reader.pages) != 23:
        raise AssertionError({"paper_pages": len(reader.pages)})
    page9 = (reader.pages[8].extract_text() or "").upper()
    page10 = (reader.pages[9].extract_text() or "").upper()
    if "LIMITATIONS" not in page9 or "TARGETED EVIDENCE-FRAME FAILURES" not in page9:
        raise AssertionError("scientific conclusion does not finish on page 9")
    if "REFERENCES" in {line.replace(" ", "").strip() for line in page9.splitlines()}:
        raise AssertionError("references begin before page 10")
    if "REFERENCES" not in {line.replace(" ", "").strip() for line in page10.splitlines()}:
        raise AssertionError("references do not begin on page 10")


def verify_integrity() -> dict:
    manifest = load(MANIFEST)
    if manifest.get("schema") != "step94.anonymous_validation_closure_release.v10":
        raise AssertionError("unexpected V16 schema")
    refresh = manifest.get("refresh", {})
    expected_refresh = {
        "numerical_evidence_changed": True,
        "three_task_stage0_sealed": True,
        "development_only_task_selection": True,
        "one_selected_test_analyzed": True,
        "unselected_tests_not_used_for_replacement": True,
        "complete_step94_development_ledger_embedded": True,
        "all_step94_learned_files_embedded": True,
        "retained_terminal_negative_embedded": True,
        "public_base_encoder_weights_embedded": False,
        "third_party_repositories_embedded": False,
        "isolated_from_worktree_by_byte_copy": True,
    }
    for key, expected in expected_refresh.items():
        if refresh.get(key) is not expected:
            raise AssertionError({"refresh_key": key, "observed": refresh.get(key)})
    if refresh.get("parent_v15_manifest_sha256") != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("parent V15 manifest binding drift")
    if refresh.get("parent_v15_zip_sha256") != EXPECTED_PARENT_ZIP:
        raise AssertionError("parent V15 ZIP binding drift")

    records = manifest.get("files", {})
    if not records:
        raise AssertionError("empty release manifest")
    for relative, record in records.items():
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or posix.as_posix() != relative:
            raise AssertionError({"unsafe_manifest_path": relative})
        path = ROOT / Path(relative)
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256(path) != record.get("sha256")
        ):
            raise AssertionError({"missing_or_drifting_file": relative})

    forbidden_weight_names = {
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
        "flax_model.msgpack",
    }
    public_weights = [relative for relative in records if Path(relative).name in forbidden_weight_names]
    if public_weights:
        raise AssertionError({"public_base_encoder_weights_embedded": public_weights})

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
        raise AssertionError("embedded V15 manifest byte drift")
    parent = load(PARENT_MANIFEST)
    if parent.get("schema") != "step93.anonymous_validation_closure_release.v9":
        raise AssertionError("embedded V15 manifest schema drift")
    bindings = manifest.get("bindings", {})
    if bindings.get("parent_v15_manifest_sha256") != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("manifest-to-V15 snapshot binding drift")

    step94 = bindings.get("step94", {})
    for key, expected in EXPECTED_STEP94.items():
        if step94.get(key) != expected:
            raise AssertionError({"step94_binding": key, "observed": step94.get(key)})
    config = load(ROOT / "STEP94_STAGEA_FROZEN_CONFIG_2026-08-13.json")
    model_map = step94.get("model_weights", {})
    if model_map != config["all_model_sha256"] or len(model_map) != 128:
        raise AssertionError("complete Step 94 learned-file map drift")
    for relative, digest in model_map.items():
        if records.get(relative, {}).get("sha256") != digest:
            raise AssertionError({"model_record_mismatch": relative})
    role_counts = {
        role: sum(
            relative.startswith("step94_models/") and record.get("role") == role
            for relative, record in records.items()
        )
        for role in ("clean_root_adapter", "learned_error_gate", "integrated_learned_variant")
    }
    if role_counts != {
        "clean_root_adapter": 48,
        "learned_error_gate": 68,
        "integrated_learned_variant": 12,
    }:
        raise AssertionError({"step94_model_roles": role_counts})
    selected_map = {
        row["checkpoint"]: row["checkpoint_audit"]["sha256"]
        for row in config["selected_task"]["integrated_variants"]
    }
    if step94.get("selected_integrated_weights") != selected_map or len(set(selected_map.values())) != 4:
        raise AssertionError("selected integrated-variant binding drift")
    if any(row["checkpoint_audit"]["parameter_count"] != 205245 for row in config["selected_task"]["integrated_variants"]):
        raise AssertionError("selected integrated-variant parameter count drift")
    if len(step94.get("development_outputs", {})) != 3:
        raise AssertionError("development-output binding count drift")
    if len(step94.get("test_inputs", {})) != 3 or len(step94.get("sealed_outcomes", {})) != 3:
        raise AssertionError("three-task seal binding drift")

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
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "paper_draft").rglob("*.tex"))
    )
    required_manuscript = (
        "20 Newsgroups",
        "$.0822$ [$.0385,.1266$]",
        "terminal harm is $+.120$ [$-.044,+.290$]",
        "neither passes the $.5$-point terminal gate",
        "learned single-$s$ Banking77/20 Newsgroups have positive CReg but null terminal harm",
    )
    if any(phrase not in manuscript for phrase in required_manuscript):
        raise AssertionError("Step 94 manuscript scope or numerical statement drift")

    result = load(ROOT / "STEP94_CONFIRMATORY_RESULTS_2026-08-13.json")
    receipt = load(ROOT / "STEP94_INDEPENDENT_VALIDATION_2026-08-13.json")
    if result.get("decision") != "NO_GO_RETAIN_FRESH_TASK_NEGATIVE":
        raise AssertionError("literal retained-negative decision drift")
    if receipt.get("decision") != result.get("decision") or any(
        value is not True for value in receipt.get("checks", {}).values()
    ):
        raise AssertionError("independent validation receipt drift")
    if not close(result["mechanism"]["path_change_rate"], 1.0):
        raise AssertionError("path-change result drift")
    if not close(result["mechanism"]["query_set_change_rate"], 1.0):
        raise AssertionError("query-set result drift")
    if not close(result["mechanism"]["final_root_change_rate"], 0.482):
        raise AssertionError("terminal-root result drift")
    if not close(result["effects"]["terminal"]["mean"], 0.001198):
        raise AssertionError("terminal result drift")
    if not close(result["effects"]["cumulative"]["mean"], 0.082157):
        raise AssertionError("cumulative result drift")
    if not close(result["effects"]["cumulative"]["bootstrap_95"][0], 0.038543625):
        raise AssertionError("cumulative interval drift")
    required_true_gates = {
        "pre_outcome_integrity",
        "same_literal_similarity",
        "four_distinct_integrated_checkpoint_hashes",
        "raw_input_endpoint_signature_has_no_forbidden_input",
        "coordinate_wise_nonimproving",
        "each_alias_loss_at_most_one_point",
        "path_change_at_least_half",
        "cumulative_mean_positive",
        "cumulative_bootstrap_lower_positive",
        "fixed_terminal_exact_zero",
        "fixed_cumulative_exact_zero",
        "fixed_root_history_bitwise_exact",
    }
    required_false_gates = {
        "terminal_mean_at_least_half_point",
        "terminal_bootstrap_lower_positive",
        "terminal_signflip_at_most_point05",
        "active_minus_fixed_mean_at_least_half_point",
        "active_minus_fixed_bootstrap_lower_positive",
    }
    if any(result["gates"].get(key) is not True for key in required_true_gates):
        raise AssertionError("Step 94 implementation/mechanism gate drift")
    if any(result["gates"].get(key) is not False for key in required_false_gates):
        raise AssertionError("Step 94 terminal no-go gate drift")

    closure = (ROOT / "STEP94_REVIEW_CLOSURE_REPORT_2026-08-13.md").read_text(encoding="utf-8")
    normalized_closure = " ".join(closure.split())
    required_closure = (
        "remaining high-value reviewer condition was not logically impossible",
        "NO_GO_RETAIN_FRESH_TASK_NEGATIVE",
        "no unselected test is opened to search for one",
        "Numerical evidence changed: yes",
    )
    if any(phrase not in normalized_closure for phrase in required_closure):
        raise AssertionError("Step 94 review-closure report drift")

    expected_summary = {
        "paper_pages": 23,
        "main_scientific_text_pages": 9,
        "step94_task_menu": 3,
        "step94_selected_task": "20newsgroups",
        "step94_search_conditions_per_task": 648,
        "step94_verification_conditions_per_task": 24,
        "step94_clean_root_weight_files": 48,
        "step94_gate_weight_files": 68,
        "step94_integrated_variant_weight_files": 12,
        "step94_total_learned_files": 128,
        "step94_selected_integrated_variants": 4,
        "step94_parameters_per_selected_variant": 205245,
        "step94_paired_runs": 2000,
        "step94_path_change_rate": 1.0,
        "step94_query_set_change_rate": 1.0,
        "step94_final_root_change_rate": 0.482,
        "step94_cumulative_regret_delta": 0.082157,
        "step94_terminal_harm_pp": 0.1198,
        "step94_fixed_query_exact_zero": True,
        "step94_decision": "NO_GO_RETAIN_FRESH_TASK_NEGATIVE",
        "review_closure": "fresh_task_anytime_replication_terminal_condition_open",
    }
    summary = manifest.get("summary", {})
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise AssertionError({"summary_key": key, "observed": summary.get(key)})
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
        checks.extend(
            [
                run(
                    [sys.executable, "validate_step94_artifact_reconstruction.py"],
                    "PASS_STEP94_ARTIFACT_RECONSTRUCTION",
                ),
                run(
                    [sys.executable, "paper_draft/validate_paper_tables.py"],
                    "PASS_PAPER_TABLE_SOURCE_CONSISTENCY",
                ),
                run(
                    [
                        "latexmk",
                        "-pdf",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-outdir=_validation_build",
                        "main.tex",
                    ],
                    cwd=ROOT / "paper_draft",
                ),
            ]
        )
        assert_pdf_boundary(ROOT / "paper_draft/_validation_build/main.pdf")
        verify_integrity()
    print(
        json.dumps(
            {
                "verdict": "PASS_STEP94_ANONYMOUS_RELEASE",
                "mode": "full" if args.full else "integrity-only",
                "bundled_files": len(manifest["files"]),
                "workstation_path_hits": 0,
                "public_base_encoder_weights_bundled": 0,
                "parent_v15_manifest_bound": True,
                "step94_model_weight_files": 128,
                "step94_independently_reconstructed": bool(args.full),
                "step94_decision": "NO_GO_RETAIN_FRESH_TASK_NEGATIVE",
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
