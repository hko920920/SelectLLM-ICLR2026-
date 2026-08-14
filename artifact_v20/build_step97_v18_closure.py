"""Build the isolated V18 anonymous artifact with Steps 96--97 bound."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "STEP95_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V17"
TARGET = ROOT / "STEP97_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V18"
MANIFEST_NAME = "STEP82_RELEASE_MANIFEST.json"
PARENT_MANIFEST_NAME = "STEP97_V17_PARENT_MANIFEST.json"
EXPECTED_PARENT_MANIFEST_SHA256 = "99108ca6b181b39fcc54b8c4d9260474b3a0df080499d76defa52d70cbec76f6"
EXPECTED_PARENT_ZIP_SHA256 = "f3d51f0f2ddb32d6d0bdf1ff66b8ef6b1dfd7ffbfd045ab2ab7884158159f20e"


ADDITIONS: dict[str, str] = {
    "README_STEP97_ARTIFACT.md": "release_control",
    "STEP97_REVIEW_CLOSURE_REPORT_2026-08-13.md": "review_closure_report",
    "build_step97_v18_closure.py": "release_builder",
    "package_step97_v18.py": "release_packager",
    "validate_step97_anonymous_release.py": "release_validator",
    "paper_draft/tables/step96_terminal_bridge_results.tex": "manuscript_source",
    "diagnose_step95_terminal_bottleneck.py": "diagnostic_code",
    "diagnose_step95_oracle_roster_geometry.py": "diagnostic_code",
    "STEP95_POSTHOC_TERMINAL_BOTTLENECK_DIAGNOSTIC_2026-08-13.json": "diagnostic_output",
    "STEP95_POSTHOC_TERMINAL_BOTTLENECK_DIAGNOSTIC_2026-08-13.md": "diagnostic_report",
    "STEP95_POSTHOC_ORACLE_ROSTER_GEOMETRY_2026-08-13.json": "development_only_oracle_diagnostic",
    "STEP96_SNLI_DIRECTIONAL_TERMINAL_BRIDGE_PREREGISTRATION_2026-08-13.md": "preregistration",
    "STEP96_PREREGISTRATION_AMENDMENT_A_INTEGER_GATE_2026-08-13.md": "preregistration_amendment",
    "STEP96_PREREGISTRATION_AMENDMENT_B_CONFIRMATORY_SEEDS_2026-08-13.md": "preregistration_amendment",
    "STEP96_STAGE0_MANIFEST_2026-08-13.json": "stage0_manifest",
    "STEP96_SNLI_DEVELOPMENT_ROWS_2026-08-13.json": "development_input",
    "STEP96_STAGEA_DEVELOPMENT_ARRAYS_2026-08-13.npz": "development_output",
    "STEP96_STAGEA_COMPLETE_LEDGER_PRE_INTEGER_FIX_2026-08-13.json": "retained_pre_repair_ledger",
    "STEP96_STAGEA_FROZEN_CONFIG_PRE_INTEGER_FIX_2026-08-13.json": "retained_pre_repair_config",
    "STEP96_STAGEA_COMPLETE_LEDGER_2026-08-13.json": "complete_development_ledger",
    "STEP96_STAGEA_FROZEN_CONFIG_2026-08-13.json": "frozen_config",
    "STEP96_INTEGER_QUALITY_GATE_REPAIR_RECEIPT_2026-08-13.json": "repair_receipt",
    "STEP96_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json": "confirmatory_lock",
    "STEP96_CONFIRMATORY_PREOUTCOME_PREDICTIONS_2026-08-13.npz": "preoutcome_prediction",
    "STEP96_CONFIRMATORY_PREOUTCOME_LEDGER_2026-08-13.json": "preoutcome_ledger",
    "STEP96_CONFIRMATORY_RAW_2026-08-13.npz": "confirmatory_raw",
    "STEP96_CONFIRMATORY_RESULTS_2026-08-13.json": "confirmatory_results",
    "STEP96_CONFIRMATORY_INDEPENDENT_VALIDATION_2026-08-13.json": "independent_validation_receipt",
    "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_RAW_2026-08-13.npz": "postconfirmatory_raw",
    "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_SENSITIVITY_2026-08-13.json": "postconfirmatory_sensitivity",
    "step96_common.py": "execution_code",
    "step96_stage0_prepare.py": "execution_code",
    "step96_stagea_develop.py": "execution_code",
    "step96_executable_adapter_endpoint.py": "endpoint_code",
    "repair_step96_integer_quality_gate.py": "repair_code",
    "lock_step96_confirmatory.py": "lock_code",
    "step96_stageb_prepare_predictions.py": "execution_code",
    "step96_stageb_confirmatory.py": "execution_code",
    "validate_step96_independent.py": "artifact_reconstruction_validator",
    "audit_step96_complete_top8_test_sensitivity.py": "postconfirmatory_audit_code",
    "STEP97_SCITAIL_CONSERVATIVE_TERMINAL_BRIDGE_PREREGISTRATION_2026-08-13.md": "preregistration",
    "STEP97_STAGE0_MANIFEST_2026-08-13.json": "stage0_manifest",
    "STEP97_SCITAIL_DEVELOPMENT_ROWS_2026-08-13.json": "development_input",
    "STEP97_STAGEA_ROOT_GEOMETRY_STOP_2026-08-13.json": "development_stop_receipt",
    "STEP97_ROOT_GEOMETRY_STOP_INDEPENDENT_VALIDATION_2026-08-13.json": "independent_validation_receipt",
    "step97_common.py": "execution_code",
    "step97_stage0_prepare.py": "execution_code",
    "step97_stagea_develop.py": "execution_code",
    "step97_executable_adapter_endpoint.py": "endpoint_code",
    "validate_step97_geometry_stop.py": "artifact_reconstruction_validator",
}


def add_tree(directory: str, role: str) -> None:
    base = ROOT / directory
    for path in sorted(base.rglob("*")):
        if path.is_file():
            relative = path.relative_to(ROOT).as_posix()
            if directory == "step96_models" and path.suffix == ".safetensors":
                ADDITIONS[relative] = "learned_error_adapter_weights"
            elif directory == "step96_models":
                ADDITIONS[relative] = "learned_error_adapter_audit"
            else:
                ADDITIONS[relative] = role


add_tree("step96_models", "learned_error_adapter")
add_tree("step96_stagea_work", "development_cache")
add_tree("step96_test_inputs", "sealed_test_input")
add_tree("external_data/step96_sealed", "sealed_test_outcome")
add_tree("step97_stagea_work", "development_cache")
add_tree("step97_test_inputs", "sealed_test_input")
add_tree("external_data/step97_sealed", "sealed_test_outcome")


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
    return {"bytes": destination.stat().st_size, "sha256": sha256(destination), "role": role}


def bind_tree(records: dict[str, dict[str, object]], prefix: str) -> dict[str, str]:
    return {
        relative: str(record["sha256"])
        for relative, record in records.items()
        if relative.startswith(prefix)
    }


def bind_files(records: dict[str, dict[str, object]], names: list[str]) -> dict[str, str]:
    return {name: str(records[name]["sha256"]) for name in names}


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    base_manifest_path = BASE / MANIFEST_NAME
    if sha256(base_manifest_path) != EXPECTED_PARENT_MANIFEST_SHA256:
        raise AssertionError("V17 parent manifest drift")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("schema") != "step95.anonymous_validation_closure_release.v11":
        raise AssertionError("V17 parent has the wrong schema")

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

    stagea = json.loads((TARGET / "STEP96_STAGEA_COMPLETE_LEDGER_2026-08-13.json").read_text(encoding="utf-8"))
    config = json.loads((TARGET / "STEP96_STAGEA_FROZEN_CONFIG_2026-08-13.json").read_text(encoding="utf-8"))
    primary = json.loads((TARGET / "STEP96_CONFIRMATORY_RESULTS_2026-08-13.json").read_text(encoding="utf-8"))
    independent = json.loads((TARGET / "STEP96_CONFIRMATORY_INDEPENDENT_VALIDATION_2026-08-13.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((TARGET / "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_SENSITIVITY_2026-08-13.json").read_text(encoding="utf-8"))
    step97 = json.loads((TARGET / "STEP97_STAGEA_ROOT_GEOMETRY_STOP_2026-08-13.json").read_text(encoding="utf-8"))
    step97_validation = json.loads((TARGET / "STEP97_ROOT_GEOMETRY_STOP_INDEPENDENT_VALIDATION_2026-08-13.json").read_text(encoding="utf-8"))

    weight_records = {
        relative: str(record["sha256"])
        for relative, record in records.items()
        if relative.startswith("step96_models/") and relative.endswith(".safetensors")
    }
    if weight_records != config["adapter_sha256"] or len(weight_records) != 4:
        raise AssertionError("release adapter weights differ from frozen config")
    if stagea["decision"] != "GO_TO_STEP96_ONE_TIME_CONFIRMATORY_LOCK" or stagea["failed_gates"]:
        raise AssertionError("Step 96 development authorization drift")
    if primary["decision"] != "NO_GO_RETAIN_STEP96_NEGATIVE" or primary["failed_gates"] != ["alias_integer_loss_counts_at_most_one_point"]:
        raise AssertionError("Step 96 literal primary decision drift")
    if independent["status"] != "PASS_STEP96_INDEPENDENT_VALIDATION" or not all(independent["checks"].values()):
        raise AssertionError("Step 96 independent validation drift")
    if sensitivity["status"] != "POSTCONFIRMATORY_EXPLORATORY_NOT_PRIMARY_CONFIRMATION" or sensitivity["primary_decision_unchanged"] != primary["decision"]:
        raise AssertionError("Step 96 sensitivity status drift")
    if len(sensitivity["unique_conditions"]) != 2 or not sensitivity["unique_conditions"][1]["all_gates_with_holm"]:
        raise AssertionError("Step 96 complete-top-eight sensitivity drift")
    if step97["decision"] != "NO_GO_STEP97_ROOT_GEOMETRY_STOP" or step97["sealed_test_opened"]:
        raise AssertionError("Step 97 geometry stop drift")
    if step97_validation["status"] != "PASS_STEP97_GEOMETRY_STOP_VALIDATION" or not all(step97_validation["checks"].values()):
        raise AssertionError("Step 97 validation drift")

    step96_names = [
        "STEP96_SNLI_DIRECTIONAL_TERMINAL_BRIDGE_PREREGISTRATION_2026-08-13.md",
        "STEP96_PREREGISTRATION_AMENDMENT_A_INTEGER_GATE_2026-08-13.md",
        "STEP96_PREREGISTRATION_AMENDMENT_B_CONFIRMATORY_SEEDS_2026-08-13.md",
        "STEP96_STAGE0_MANIFEST_2026-08-13.json",
        "STEP96_STAGEA_COMPLETE_LEDGER_2026-08-13.json",
        "STEP96_STAGEA_FROZEN_CONFIG_2026-08-13.json",
        "STEP96_INTEGER_QUALITY_GATE_REPAIR_RECEIPT_2026-08-13.json",
        "STEP96_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json",
        "STEP96_CONFIRMATORY_PREOUTCOME_PREDICTIONS_2026-08-13.npz",
        "STEP96_CONFIRMATORY_PREOUTCOME_LEDGER_2026-08-13.json",
        "STEP96_CONFIRMATORY_RAW_2026-08-13.npz",
        "STEP96_CONFIRMATORY_RESULTS_2026-08-13.json",
        "STEP96_CONFIRMATORY_INDEPENDENT_VALIDATION_2026-08-13.json",
        "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_RAW_2026-08-13.npz",
        "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_SENSITIVITY_2026-08-13.json",
    ]
    step97_names = [
        "STEP97_SCITAIL_CONSERVATIVE_TERMINAL_BRIDGE_PREREGISTRATION_2026-08-13.md",
        "STEP97_STAGE0_MANIFEST_2026-08-13.json",
        "STEP97_STAGEA_ROOT_GEOMETRY_STOP_2026-08-13.json",
        "STEP97_ROOT_GEOMETRY_STOP_INDEPENDENT_VALIDATION_2026-08-13.json",
    ]
    conservative = sensitivity["unique_conditions"][1]
    manifest = {
        **base_manifest,
        "schema": "step97.anonymous_validation_closure_release.v12",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "refresh": {
            "reason": "snli_directional_terminal_bridge_and_scitail_geometry_stop",
            "numerical_evidence_changed": True,
            "parent_v17_manifest_sha256": EXPECTED_PARENT_MANIFEST_SHA256,
            "parent_v17_zip_sha256": EXPECTED_PARENT_ZIP_SHA256,
            "complete_step96_primary_embedded": True,
            "complete_step96_top8_sensitivity_embedded": True,
            "step96_primary_no_go_retained": True,
            "step96_secondary_not_relabelled_confirmatory": True,
            "step97_geometry_stop_embedded": True,
            "step97_test_outcome_opened": False,
            "public_base_model_weights_embedded": False,
            "third_party_repositories_embedded": False,
            "isolated_from_worktree_by_byte_copy": True,
        },
        "bindings": {
            **base_manifest["bindings"],
            "paper_pdf_sha256": sha256(TARGET / "paper_draft/main.pdf"),
            "parent_v17_manifest_sha256": sha256(TARGET / PARENT_MANIFEST_NAME),
            "step96": {
                "files": bind_files(records, step96_names),
                "model_weights": weight_records,
                "model_audits": {
                    relative: str(record["sha256"])
                    for relative, record in records.items()
                    if relative.startswith("step96_models/") and relative.endswith(".json")
                },
                "stagea_cache": bind_tree(records, "step96_stagea_work/"),
                "test_inputs": bind_tree(records, "step96_test_inputs/"),
                "sealed_outcomes": bind_tree(records, "external_data/step96_sealed/"),
            },
            "step97": {
                "files": bind_files(records, step97_names),
                "stagea_cache": bind_tree(records, "step97_stagea_work/"),
                "test_inputs": bind_tree(records, "step97_test_inputs/"),
                "sealed_outcomes": bind_tree(records, "external_data/step97_sealed/"),
            },
        },
        "summary": {
            **base_manifest["summary"],
            "files": len(records),
            "bytes": sum(int(record["bytes"]) for record in records.values()),
            "paper_pages": 25,
            "main_scientific_text_pages": 9,
            "step96_task": "snli_test",
            "step96_confirmatory_runs": primary["runs"],
            "step96_primary_terminal_harm_pp": 100.0 * primary["inference"]["terminal"]["mean"],
            "step96_primary_cumulative_regret_delta": primary["inference"]["cumulative"]["mean"],
            "step96_primary_path_change_rate": primary["mechanism"]["path_change_rate"],
            "step96_primary_final_root_change_rate": primary["mechanism"]["final_root_change_rate"],
            "step96_primary_fixed_query_exact_zero": primary["gates"]["fixed_query_exact_zero"],
            "step96_primary_decision": primary["decision"],
            "step96_primary_failed_gates": primary["failed_gates"],
            "step96_secondary_status": sensitivity["status"],
            "step96_secondary_terminal_harm_pp": 100.0 * conservative["inference"]["terminal"]["mean"],
            "step96_secondary_holm_q": conservative["inference"]["terminal"]["holm_q_across_unique_top8_conditions"],
            "step96_secondary_all_gates_with_holm": conservative["all_gates_with_holm"],
            "step97_task": "scitail_tsv_format_test",
            "step97_decision": step97["decision"],
            "step97_test_opened": False,
            "review_closure": "learned_single_similarity_terminal_mechanism_present_primary_quality_near_miss_secondary_pass",
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
        "step96_weight_files": len(weight_records),
        "step96_primary_decision": primary["decision"],
        "step96_secondary_status": sensitivity["status"],
        "step97_decision": step97["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
