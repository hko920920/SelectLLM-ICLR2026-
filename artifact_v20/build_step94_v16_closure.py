"""Build the isolated V16 anonymous artifact with the Step 94 audit."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "STEP93_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V15"
TARGET = ROOT / "STEP94_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V16"
MANIFEST_NAME = "STEP82_RELEASE_MANIFEST.json"
PARENT_MANIFEST_NAME = "STEP94_V15_PARENT_MANIFEST.json"
EXPECTED_PARENT_MANIFEST_SHA256 = "14f4a65b5f7e450ca3f546fb6ea687cf7498162d0303eeaef534c605ff6187af"
EXPECTED_PARENT_ZIP_SHA256 = "1743a7b696b23a02688aa0bcf9e461ccab6c9e67985b8bd0bd8a8611da434a2a"

ADDITIONS: dict[str, str] = {
    "README_STEP94_ARTIFACT.md": "release_control",
    "STEP94_REVIEW_CLOSURE_REPORT_2026-08-13.md": "review_closure_report",
    "build_step94_v16_closure.py": "release_builder",
    "package_step94_v16.py": "release_packager",
    "validate_step94_anonymous_release.py": "release_validator",
    "validate_step94_artifact_reconstruction.py": "artifact_reconstruction_validator",
    "STEP94_FRESH_TASK_LEARNED_VARIANT_PREREGISTRATION_2026-08-13.md": "preregistration",
    "STEP94_PREREGISTRATION_AMENDMENT_A_2026-08-13.md": "preregistration_amendment",
    "STEP94_PREREGISTRATION_AMENDMENT_B_2026-08-13.md": "preregistration_amendment",
    "STEP94_PREREGISTRATION_AMENDMENT_C_2026-08-13.md": "preregistration_amendment",
    "STEP94_PREREGISTRATION_AMENDMENT_D_2026-08-13.md": "preregistration_amendment",
    "STEP94_STAGE0_MANIFEST_2026-08-13.json": "stage0_manifest",
    "STEP94_STAGEA_COMPLETE_LEDGER_2026-08-13.json": "complete_development_ledger",
    "STEP94_STAGEA_FROZEN_CONFIG_2026-08-13.json": "frozen_config",
    "STEP94_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json": "execution_lock",
    "STEP94_CONFIRMATORY_RAW_2026-08-13.npz": "raw_result",
    "STEP94_CONFIRMATORY_RESULTS_2026-08-13.json": "literal_result",
    "STEP94_INDEPENDENT_VALIDATION_2026-08-13.json": "independent_validation_receipt",
    "step94_common.py": "execution_code",
    "step94_stage0_prepare.py": "execution_code",
    "step94_stagea_train_and_select.py": "execution_code",
    "step94_executable_variant_endpoint.py": "endpoint_code",
    "step94_stageb_confirmatory.py": "execution_code",
    "lock_step94_confirmatory.py": "execution_code",
    "validate_step94_independent.py": "locked_independent_validator",
}


def add_tree(directory: str, role: str) -> None:
    base = ROOT / directory
    for path in sorted(base.rglob("*")):
        if path.is_file():
            ADDITIONS[path.relative_to(ROOT).as_posix()] = role


add_tree("step94_stagea_outputs", "development_output")
add_tree("step94_test_inputs", "sealed_test_input")
add_tree("external_data/step94_sealed", "sealed_test_outcome")
add_tree("STEP94_STAGE0_FAILED_PARTIAL_2026-08-13", "quarantined_preanalysis_partial")
for model_path in sorted((ROOT / "step94_models").rglob("*.safetensors")):
    relative = model_path.relative_to(ROOT).as_posix()
    if "_integrated_variant_" in model_path.name:
        role = "integrated_learned_variant"
    elif "_parent_gate_" in model_path.name:
        role = "learned_error_gate"
    else:
        role = "clean_root_adapter"
    ADDITIONS[relative] = role


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
        raise AssertionError("V15 parent manifest drift")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("schema") != "step93.anonymous_validation_closure_release.v9":
        raise AssertionError("V15 parent has the wrong schema")

    TARGET.mkdir(parents=True)
    records: dict[str, dict[str, object]] = {}
    for relative, old_record in sorted(base_manifest["files"].items()):
        # Current manuscript bytes supersede V15; all inherited evidence comes
        # only from the already sealed V15 candidate.
        source = ROOT / Path(relative) if relative.startswith("paper_draft/") else BASE / Path(relative)
        records[relative] = copy_record(source, relative, str(old_record["role"]))
    for relative, role in sorted(ADDITIONS.items()):
        records[relative] = copy_record(ROOT / Path(relative), relative, role)
    records[PARENT_MANIFEST_NAME] = copy_record(
        base_manifest_path, PARENT_MANIFEST_NAME, "parent_release_manifest"
    )

    config = json.loads(
        (TARGET / "STEP94_STAGEA_FROZEN_CONFIG_2026-08-13.json").read_text(encoding="utf-8")
    )
    model_records = bind_tree(records, "step94_models/")
    if model_records != config["all_model_sha256"]:
        raise AssertionError("release model map differs from frozen config")
    selected_integrated = {
        row["checkpoint"]: row["checkpoint_audit"]["sha256"]
        for row in config["selected_task"]["integrated_variants"]
    }

    step94_binding = {
        "preregistration": sha256(TARGET / "STEP94_FRESH_TASK_LEARNED_VARIANT_PREREGISTRATION_2026-08-13.md"),
        "amendment_a": sha256(TARGET / "STEP94_PREREGISTRATION_AMENDMENT_A_2026-08-13.md"),
        "amendment_b": sha256(TARGET / "STEP94_PREREGISTRATION_AMENDMENT_B_2026-08-13.md"),
        "amendment_c": sha256(TARGET / "STEP94_PREREGISTRATION_AMENDMENT_C_2026-08-13.md"),
        "amendment_d": sha256(TARGET / "STEP94_PREREGISTRATION_AMENDMENT_D_2026-08-13.md"),
        "stage0_manifest": sha256(TARGET / "STEP94_STAGE0_MANIFEST_2026-08-13.json"),
        "complete_development_ledger": sha256(TARGET / "STEP94_STAGEA_COMPLETE_LEDGER_2026-08-13.json"),
        "frozen_config": sha256(TARGET / "STEP94_STAGEA_FROZEN_CONFIG_2026-08-13.json"),
        "execution_lock": sha256(TARGET / "STEP94_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json"),
        "raw": sha256(TARGET / "STEP94_CONFIRMATORY_RAW_2026-08-13.npz"),
        "literal_results": sha256(TARGET / "STEP94_CONFIRMATORY_RESULTS_2026-08-13.json"),
        "locked_independent_validator": sha256(TARGET / "validate_step94_independent.py"),
        "independent_validation_receipt": sha256(TARGET / "STEP94_INDEPENDENT_VALIDATION_2026-08-13.json"),
        "artifact_reconstruction_validator": sha256(TARGET / "validate_step94_artifact_reconstruction.py"),
        "review_closure_report": sha256(TARGET / "STEP94_REVIEW_CLOSURE_REPORT_2026-08-13.md"),
        "model_weights": model_records,
        "selected_integrated_weights": selected_integrated,
        "development_outputs": bind_tree(records, "step94_stagea_outputs/"),
        "test_inputs": bind_tree(records, "step94_test_inputs/"),
        "sealed_outcomes": bind_tree(records, "external_data/step94_sealed/"),
        "quarantined_partial": bind_tree(records, "STEP94_STAGE0_FAILED_PARTIAL_2026-08-13/"),
    }
    manifest = {
        **base_manifest,
        "schema": "step94.anonymous_validation_closure_release.v10",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "refresh": {
            "reason": "fresh_task_learned_single_similarity_replication",
            "numerical_evidence_changed": True,
            "parent_v15_manifest_sha256": EXPECTED_PARENT_MANIFEST_SHA256,
            "parent_v15_zip_sha256": EXPECTED_PARENT_ZIP_SHA256,
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
        },
        "bindings": {
            **base_manifest["bindings"],
            "paper_pdf_sha256": sha256(TARGET / "paper_draft/main.pdf"),
            "parent_v15_manifest_sha256": sha256(TARGET / PARENT_MANIFEST_NAME),
            "step94": step94_binding,
        },
        "summary": {
            **base_manifest["summary"],
            "files": len(records),
            "bytes": sum(int(record["bytes"]) for record in records.values()),
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
        },
        "files": records,
    }
    manifest_path = TARGET / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "target": TARGET.name,
                "files": len(records),
                "bytes": manifest["summary"]["bytes"],
                "paper_pdf_sha256": manifest["bindings"]["paper_pdf_sha256"],
                "manifest_sha256": sha256(manifest_path),
                "step94_model_weight_files": len(model_records),
                "selected_step94_integrated_weights": len(selected_integrated),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
