"""Build the isolated V14 anonymous artifact with Step 93 fidelity closure."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "STEP92_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V13"
TARGET = ROOT / "STEP93_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V14"
MANIFEST_NAME = "STEP82_RELEASE_MANIFEST.json"
PARENT_MANIFEST_NAME = "STEP92_V13_PARENT_MANIFEST.json"
EXPECTED_PARENT_MANIFEST_SHA256 = "56de71271154e9ec3ea2f5fbcbe8674324d7c3fbd20f0e3cbdbad51ca2d886e7"
EXPECTED_PARENT_ZIP_SHA256 = "42b5575f5d0a71bdd883627a714f02d4fa3e124e70b517f1c155c25f607d501a"

ADDITIONS = {
    "README_STEP92_ARTIFACT.md": "historical_scope_correction",
    "README_STEP93_ARTIFACT.md": "release_control",
    "STEP93_METHOD_FIDELITY_CLOSURE_REPORT_2026-08-13.md": "closure_report",
    "build_step93_anonymous_artifact_refresh.py": "release_builder",
    "fetch_step93_public_dependencies.py": "dependency_fetcher",
    "validate_step93_anonymous_release.py": "release_validator",
    "validate_step93_manuscript_integration.py": "manuscript_validator",
    "package_step93_v14.py": "release_packager",
    "paper_draft/tables/step93_source_faithful_results.tex": "manuscript_source",
    "STEP93_SOURCE_FAITHFUL_LEARNED_ABSTENTION_PREREGISTRATION_2026-08-13.md": "preregistration",
    "STEP93_PREREGISTRATION_AMENDMENT_A_2026-08-13.md": "preregistration_amendment",
    "STEP93_PREREGISTRATION_AMENDMENT_B_2026-08-13.md": "preregistration_amendment",
    "STEP93_STAGE0_MANIFEST_2026-08-13.json": "manifest",
    "STEP93_BANKING77_CALIBRATION_PACKAGE_2026-08-13.json": "derived_input",
    "STEP93_BANKING77_HOLDOUT_INPUTS_2026-08-13.json": "derived_input",
    "external_data/step93_sealed/STEP93_BANKING77_SEALED_OUTCOMES_2026-08-13.npz": "sealed_outcome",
    "STEP93_STAGEA_CALIBRATION_OUTPUTS_2026-08-13.npz": "development_output",
    "STEP93_STAGEA_COMPLETE_SEARCH_LEDGER_2026-08-13.json": "development_ledger",
    "STEP93_STAGEA_FROZEN_CONFIG_2026-08-13.json": "frozen_config",
    "STEP93_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json": "execution_lock",
    "step93_common.py": "execution_code",
    "step93_stage0_prepare.py": "execution_code",
    "step93_stagea_train_and_develop.py": "execution_code",
    "step93_stageb_confirmatory.py": "execution_code",
    "step93_executable_abstention_endpoint.py": "endpoint_code",
    "lock_step93_confirmatory.py": "execution_code",
    "STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RAW_2026-08-13.npz": "raw_result",
    "STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RESULTS_2026-08-13.json": "literal_result",
    "validate_step93_independent.py": "independent_validator",
    "STEP93_INDEPENDENT_VALIDATION_AND_ADJUDICATION_2026-08-13.json": "validation_and_adjudication",
}

for model_path in sorted((ROOT / "step93_models").glob("*.safetensors")):
    relative = model_path.relative_to(ROOT).as_posix()
    role = "clean_root_adapter" if "clean_root" in model_path.name else "learned_abstention_adapter"
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


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    base_manifest_path = BASE / MANIFEST_NAME
    if sha256(base_manifest_path) != EXPECTED_PARENT_MANIFEST_SHA256:
        raise AssertionError("V13 parent manifest drift")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("schema") != "step92.anonymous_validation_closure_release.v7":
        raise AssertionError("V13 parent has the wrong schema")

    TARGET.mkdir(parents=True)
    records: dict[str, dict[str, object]] = {}
    for relative, old_record in sorted(base_manifest["files"].items()):
        # Current manuscript replaces V13 manuscript bytes. All other inherited
        # evidence is copied only from the already sealed V13 candidate.
        source = ROOT / Path(relative) if relative.startswith("paper_draft/") else BASE / Path(relative)
        records[relative] = copy_record(source, relative, old_record["role"])
    for relative, role in sorted(ADDITIONS.items()):
        records[relative] = copy_record(ROOT / Path(relative), relative, role)
    records[PARENT_MANIFEST_NAME] = copy_record(
        base_manifest_path, PARENT_MANIFEST_NAME, "parent_release_manifest"
    )

    model_records = {
        relative: record["sha256"]
        for relative, record in records.items()
        if relative.startswith("step93_models/")
    }
    selected_parent = 10
    selected = {
        relative: digest
        for relative, digest in model_records.items()
        if f"parent_{selected_parent:02d}_abstention_adapter" in relative
    }
    step93_binding = {
        "preregistration": sha256(TARGET / "STEP93_SOURCE_FAITHFUL_LEARNED_ABSTENTION_PREREGISTRATION_2026-08-13.md"),
        "amendment_a": sha256(TARGET / "STEP93_PREREGISTRATION_AMENDMENT_A_2026-08-13.md"),
        "amendment_b": sha256(TARGET / "STEP93_PREREGISTRATION_AMENDMENT_B_2026-08-13.md"),
        "stage0_manifest": sha256(TARGET / "STEP93_STAGE0_MANIFEST_2026-08-13.json"),
        "complete_search_ledger": sha256(TARGET / "STEP93_STAGEA_COMPLETE_SEARCH_LEDGER_2026-08-13.json"),
        "calibration_outputs": sha256(TARGET / "STEP93_STAGEA_CALIBRATION_OUTPUTS_2026-08-13.npz"),
        "frozen_config": sha256(TARGET / "STEP93_STAGEA_FROZEN_CONFIG_2026-08-13.json"),
        "execution_lock": sha256(TARGET / "STEP93_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json"),
        "literal_results": sha256(TARGET / "STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RESULTS_2026-08-13.json"),
        "raw": sha256(TARGET / "STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RAW_2026-08-13.npz"),
        "independent_validator": sha256(TARGET / "validate_step93_independent.py"),
        "independent_adjudication": sha256(TARGET / "STEP93_INDEPENDENT_VALIDATION_AND_ADJUDICATION_2026-08-13.json"),
        "manuscript_validator": sha256(TARGET / "validate_step93_manuscript_integration.py"),
        "closure_report": sha256(TARGET / "STEP93_METHOD_FIDELITY_CLOSURE_REPORT_2026-08-13.md"),
        "model_weights": model_records,
        "selected_adapter_weights": selected,
    }
    manifest = {
        **base_manifest,
        "schema": "step93.anonymous_validation_closure_release.v8",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "refresh": {
            "reason": "step93_source_faithful_learned_negative_and_method_fidelity_correction",
            "numerical_evidence_changed": True,
            "parent_v13_manifest_sha256": EXPECTED_PARENT_MANIFEST_SHA256,
            "parent_v13_zip_sha256": EXPECTED_PARENT_ZIP_SHA256,
            "derived_step93_packages_embedded": True,
            "all_stagea_step93_weights_embedded": True,
            "literal_negative_and_adjudication_both_embedded": True,
            "public_base_encoder_weights_embedded": False,
            "third_party_repositories_embedded": False,
            "isolated_from_worktree_by_byte_copy": True,
        },
        "bindings": {
            **base_manifest["bindings"],
            "paper_pdf_sha256": sha256(TARGET / "paper_draft/main.pdf"),
            "parent_v13_manifest_sha256": sha256(TARGET / PARENT_MANIFEST_NAME),
            "step93": step93_binding,
        },
        "summary": {
            **base_manifest["summary"],
            "files": len(records),
            "bytes": sum(int(record["bytes"]) for record in records.values()),
            "paper_pages": 23,
            "main_scientific_text_pages": 9,
            "step92_scope": "learned_decoupled_view_positive",
            "step93_task": "banking77",
            "step93_clean_root_weight_files": 12,
            "step93_learned_adapter_weight_files": 28,
            "step93_selected_learned_adapters": 4,
            "step93_parameters_per_selected_adapter": 50817,
            "step93_paired_runs": 1000,
            "step93_independent_checks": 82,
            "step93_raw_input_endpoint_samples": 128,
            "step93_path_change_rate": 0.998,
            "step93_cumulative_regret_delta": 0.040945,
            "step93_terminal_harm_pp": 0.032,
            "step93_fixed_query_exact_zero": True,
            "step93_adjudicated_decision": "NO_GO_RETAIN_SOURCE_FAITHFUL_NEGATIVE",
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
                "step93_model_weight_files": len(model_records),
                "selected_step93_adapter_weights": len(selected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
