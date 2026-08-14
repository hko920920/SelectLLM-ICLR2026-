"""Build the isolated V17 anonymous artifact with the Step 95 development stop."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "STEP94_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V16"
TARGET = ROOT / "STEP95_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V17"
MANIFEST_NAME = "STEP82_RELEASE_MANIFEST.json"
PARENT_MANIFEST_NAME = "STEP95_V16_PARENT_MANIFEST.json"
EXPECTED_PARENT_MANIFEST_SHA256 = "d2cfcf1be836d57a995d3b3b0c6a4371081c8fc76c37c620e35c3a8c0a9f503e"
EXPECTED_PARENT_ZIP_SHA256 = "130c5c528bc5ce94dbeb50867a99094a81eb8073e1c2a2388a5288581db15949"

ADDITIONS: dict[str, str] = {
    "README_STEP95_ARTIFACT.md": "release_control",
    "STEP95_REVIEW_CLOSURE_REPORT_2026-08-13.md": "review_closure_report",
    "build_step95_v17_closure.py": "release_builder",
    "package_step95_v17.py": "release_packager",
    "validate_step95_anonymous_release.py": "release_validator",
    "STEP95_MNLI_FULL_ADAPTER_PREREGISTRATION_2026-08-13.md": "preregistration",
    "STEP95_PREREGISTRATION_AMENDMENT_A_2026-08-13.md": "preregistration_amendment",
    "STEP95_PREREGISTRATION_AMENDMENT_B_2026-08-13.md": "execution_incident_record",
    "STEP95_STAGE0_MANIFEST_2026-08-13.json": "stage0_manifest",
    "STEP95_MNLI_DEVELOPMENT_ROWS_2026-08-13.json": "development_input",
    "STEP95_STAGEA_DEVELOPMENT_ARRAYS_2026-08-13.npz": "development_output",
    "STEP95_STAGEA_COMPLETE_LEDGER_2026-08-13.json": "complete_development_ledger",
    "STEP95_STAGEA_FROZEN_CONFIG_2026-08-13.json": "frozen_config",
    "STEP95_STAGEA_INDEPENDENT_VALIDATION_2026-08-13.json": "independent_validation_receipt",
    "STEP95_DEVELOPMENT_STOP_RECEIPT_2026-08-13.json": "development_stop_receipt",
    "step95_common.py": "execution_code",
    "step95_stage0_prepare.py": "execution_code",
    "step95_stagea_develop.py": "execution_code",
    "step95_executable_adapter_endpoint.py": "endpoint_code",
    "validate_step95_stagea.py": "artifact_reconstruction_validator",
    "close_step95_development_stop.py": "closure_code",
    "step95_stageb_confirmatory.py": "unexecuted_confirmatory_code",
    "lock_step95_confirmatory.py": "unexecuted_confirmatory_code",
    "validate_step95_independent.py": "unexecuted_confirmatory_code",
}


def add_tree(directory: str, role: str) -> None:
    base = ROOT / directory
    for path in sorted(base.rglob("*")):
        if path.is_file():
            ADDITIONS[path.relative_to(ROOT).as_posix()] = role


add_tree("step95_test_inputs", "sealed_test_input")
add_tree("external_data/step95_sealed", "sealed_test_outcome")
add_tree("step95_models", "learned_error_adapter")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_record(source: Path, relative: str, role: str) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError({"relative": relative, "source": str(source)})
    destination = TARGET / Path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "role": role,
    }


def bind_tree(records: dict[str, dict[str, object]], prefix: str) -> dict[str, str]:
    return {
        relative: str(record["sha256"])
        for relative, record in records.items()
        if relative.startswith(prefix)
    }


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    base_manifest_path = BASE / MANIFEST_NAME
    if sha256(base_manifest_path) != EXPECTED_PARENT_MANIFEST_SHA256:
        raise AssertionError("V16 parent manifest drift")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("schema") != "step94.anonymous_validation_closure_release.v10":
        raise AssertionError("V16 parent has the wrong schema")

    TARGET.mkdir(parents=True)
    records: dict[str, dict[str, object]] = {}
    for relative, old_record in sorted(base_manifest["files"].items()):
        source = ROOT / Path(relative) if relative.startswith("paper_draft/") else BASE / Path(relative)
        records[relative] = copy_record(source, relative, str(old_record["role"]))
    for relative, role in sorted(ADDITIONS.items()):
        records[relative] = copy_record(ROOT / Path(relative), relative, role)
    records[PARENT_MANIFEST_NAME] = copy_record(
        base_manifest_path, PARENT_MANIFEST_NAME, "parent_release_manifest"
    )

    config = json.loads((TARGET / "STEP95_STAGEA_FROZEN_CONFIG_2026-08-13.json").read_text(encoding="utf-8"))
    ledger = json.loads((TARGET / "STEP95_STAGEA_COMPLETE_LEDGER_2026-08-13.json").read_text(encoding="utf-8"))
    receipt = json.loads((TARGET / "STEP95_DEVELOPMENT_STOP_RECEIPT_2026-08-13.json").read_text(encoding="utf-8"))
    model_records = bind_tree(records, "step95_models/")
    if model_records != config["adapter_sha256"] or len(model_records) != 4:
        raise AssertionError("release adapter map differs from frozen config")
    if ledger["decision"] != "NO_GO_DEVELOPMENT_GATE_STOP" or receipt["decision"] != ledger["decision"]:
        raise AssertionError("Step 95 development-stop decision drift")
    forbidden_confirmatory = (
        "STEP95_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json",
        "STEP95_CONFIRMATORY_RAW_2026-08-13.npz",
        "STEP95_CONFIRMATORY_RESULTS_2026-08-13.json",
        "STEP95_INDEPENDENT_VALIDATION_2026-08-13.json",
    )
    if any(name in records for name in forbidden_confirmatory):
        raise AssertionError("Step 95 confirmatory output must not be bundled")

    step95_binding = {
        "preregistration": sha256(TARGET / "STEP95_MNLI_FULL_ADAPTER_PREREGISTRATION_2026-08-13.md"),
        "amendment_a": sha256(TARGET / "STEP95_PREREGISTRATION_AMENDMENT_A_2026-08-13.md"),
        "incident_b": sha256(TARGET / "STEP95_PREREGISTRATION_AMENDMENT_B_2026-08-13.md"),
        "stage0_manifest": sha256(TARGET / "STEP95_STAGE0_MANIFEST_2026-08-13.json"),
        "development_input": sha256(TARGET / "STEP95_MNLI_DEVELOPMENT_ROWS_2026-08-13.json"),
        "development_arrays": sha256(TARGET / "STEP95_STAGEA_DEVELOPMENT_ARRAYS_2026-08-13.npz"),
        "complete_development_ledger": sha256(TARGET / "STEP95_STAGEA_COMPLETE_LEDGER_2026-08-13.json"),
        "frozen_config": sha256(TARGET / "STEP95_STAGEA_FROZEN_CONFIG_2026-08-13.json"),
        "independent_validation_receipt": sha256(TARGET / "STEP95_STAGEA_INDEPENDENT_VALIDATION_2026-08-13.json"),
        "development_stop_receipt": sha256(TARGET / "STEP95_DEVELOPMENT_STOP_RECEIPT_2026-08-13.json"),
        "review_closure_report": sha256(TARGET / "STEP95_REVIEW_CLOSURE_REPORT_2026-08-13.md"),
        "model_weights": model_records,
        "test_inputs": bind_tree(records, "step95_test_inputs/"),
        "sealed_outcomes": bind_tree(records, "external_data/step95_sealed/"),
    }
    selected = ledger["selected_verification"]
    manifest = {
        **base_manifest,
        "schema": "step95.anonymous_validation_closure_release.v11",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "refresh": {
            "reason": "mnli_full_adapter_development_gate_stop",
            "numerical_evidence_changed": True,
            "parent_v16_manifest_sha256": EXPECTED_PARENT_MANIFEST_SHA256,
            "parent_v16_zip_sha256": EXPECTED_PARENT_ZIP_SHA256,
            "complete_step95_grid_embedded": True,
            "all_step95_learned_files_embedded": True,
            "development_gate_stop": True,
            "matched_test_outcome_opened": False,
            "mismatched_test_substituted": False,
            "confirmatory_output_embedded": False,
            "public_base_model_weights_embedded": False,
            "third_party_repositories_embedded": False,
            "isolated_from_worktree_by_byte_copy": True,
        },
        "bindings": {
            **base_manifest["bindings"],
            "paper_pdf_sha256": sha256(TARGET / "paper_draft/main.pdf"),
            "parent_v16_manifest_sha256": sha256(TARGET / PARENT_MANIFEST_NAME),
            "step95": step95_binding,
        },
        "summary": {
            **base_manifest["summary"],
            "files": len(records),
            "bytes": sum(int(record["bytes"]) for record in records.values()),
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
            "step95_path_change_rate": selected["path_change_rate"],
            "step95_query_set_change_rate": selected["query_set_change_rate"],
            "step95_final_root_change_rate": selected["final_root_change_rate"],
            "step95_cumulative_regret_delta": selected["inference"]["cumulative"]["mean"],
            "step95_terminal_harm_pp": 100.0 * selected["inference"]["terminal"]["mean"],
            "step95_fixed_query_exact_zero": selected["fixed_root_history_exact"]
            and selected["max_abs_fixed_terminal_delta"] == 0
            and selected["max_abs_fixed_cumulative_delta"] == 0,
            "step95_sealed_test_opened": False,
            "step95_decision": "NO_GO_DEVELOPMENT_GATE_STOP",
            "review_closure": "source_faithful_anytime_replication_material_terminal_condition_open",
        },
        "files": records,
    }
    manifest_path = TARGET / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "target": TARGET.name,
        "files": len(records),
        "bytes": manifest["summary"]["bytes"],
        "paper_pdf_sha256": manifest["bindings"]["paper_pdf_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "step95_model_weight_files": len(model_records),
        "step95_decision": ledger["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
