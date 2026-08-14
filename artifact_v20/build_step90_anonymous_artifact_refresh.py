"""Build the V12 anonymous artifact with Step 89/90 evidence closure."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "STEP88_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V11"
TARGET = ROOT / "STEP90_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V12"
MANIFEST_NAME = "STEP82_RELEASE_MANIFEST.json"
OVERRIDES = {"validate_step88_manuscript_integration.py"}

ADDITIONS = {
    "README_STEP90_ARTIFACT.md": "release_control",
    "build_step90_anonymous_artifact_refresh.py": "release_builder",
    "validate_step90_manuscript_integration.py": "manuscript_validator",
    "validate_step90_anonymous_release.py": "release_validator",
    "paper_draft/tables/step90_executable_adapter_results.tex": "manuscript_source",
    # Retained Step 89 nonconfirmation.
    "STEP89_QNLI_EXECUTABLE_ADAPTER_STAGE0_PROTOCOL_2026-08-12.md": "protocol",
    "STEP89_QNLI_STAGE0_SPLIT_MANIFEST_2026-08-12.json": "manifest",
    "STEP89_QNLI_CALIBRATION_PACKAGE_2026-08-12.json": "derived_input",
    "STEP89_QNLI_HOLDOUT_INPUTS_2026-08-12.json": "derived_input",
    "external_data/step89_sealed/STEP89_QNLI_SEALED_HOLDOUT_OUTCOMES_2026-08-12.npz": "sealed_outcome",
    "STEP89_QNLI_CALIBRATION_PARENT_OUTPUTS_2026-08-12.npz": "derived_input",
    "STEP89_QNLI_STAGEA_SEARCH_LEDGER_2026-08-12.json": "development_ledger",
    "STEP89_QNLI_STAGEA_FROZEN_CONFIG_2026-08-12.json": "frozen_config",
    "STEP89_QNLI_EXECUTABLE_ADAPTER_CONFIRMATORY_PREREGISTRATION_2026-08-12.md": "preregistration",
    "STEP89_QNLI_CONFIRMATORY_EXECUTION_LOCK_2026-08-12.json": "execution_lock",
    "step89_stage0_prepare.py": "execution_code",
    "step89_stagea_develop.py": "execution_code",
    "step89_stageb_confirmatory.py": "execution_code",
    "step89_models/step89_adapter_0.safetensors": "adapter",
    "step89_models/step89_adapter_1.safetensors": "adapter",
    "step89_models/step89_adapter_2.safetensors": "adapter",
    "step89_models/step89_adapter_3.safetensors": "adapter",
    "STEP89_QNLI_EXECUTABLE_ADAPTER_RAW_2026-08-12.npz": "raw_result",
    "STEP89_QNLI_EXECUTABLE_ADAPTER_RESULTS_2026-08-12.json": "result",
    # Step 90 promoted audit.
    "STEP90_HIGH_CONFIDENCE_ADAPTER_STAGE0_PROTOCOL_2026-08-12.md": "protocol",
    "STEP90_HIGH_CONFIDENCE_ADAPTER_STAGE0_MANIFEST_2026-08-12.json": "manifest",
    "STEP90_MNLI_CALIBRATION_PACKAGE_2026-08-12.json": "derived_input",
    "STEP90_MNLI_HOLDOUT_INPUTS_2026-08-12.json": "derived_input",
    "external_data/step90_sealed/STEP90_MNLI_SEALED_OUTCOMES_2026-08-12.npz": "sealed_outcome",
    "STEP90_MNLI_CALIBRATION_PARENT_OUTPUTS_2026-08-12.npz": "derived_input",
    "STEP90_MNLI_STAGEA_SEARCH_LEDGER_2026-08-12.json": "development_ledger",
    "STEP90_QQP_CALIBRATION_PACKAGE_2026-08-12.json": "derived_input",
    "STEP90_QQP_HOLDOUT_INPUTS_2026-08-12.json": "derived_input",
    "external_data/step90_sealed/STEP90_QQP_SEALED_OUTCOMES_2026-08-12.npz": "sealed_outcome",
    "STEP90_QQP_CALIBRATION_PARENT_OUTPUTS_2026-08-12.npz": "derived_input",
    "STEP90_QQP_STAGEA_SEARCH_LEDGER_2026-08-12.json": "development_ledger",
    "STEP90_HIGH_CONFIDENCE_STAGEA_FROZEN_CONFIG_2026-08-12.json": "frozen_config",
    "STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_PREREGISTRATION_2026-08-12.md": "preregistration",
    "STEP90_HIGH_CONFIDENCE_CONFIRMATORY_EXECUTION_LOCK_2026-08-12.json": "execution_lock",
    "step90_stage0_prepare.py": "execution_code",
    "step90_stagea_develop.py": "execution_code",
    "step90_stageb_confirmatory.py": "execution_code",
    "step90_executable_adapter_endpoint.py": "executable_endpoint",
    "step90_models/step90_mnli_adapter_0.safetensors": "adapter",
    "step90_models/step90_mnli_adapter_1.safetensors": "adapter",
    "step90_models/step90_mnli_adapter_2.safetensors": "adapter",
    "step90_models/step90_mnli_adapter_3.safetensors": "adapter",
    "step90_models/step90_qqp_adapter_0.safetensors": "adapter",
    "step90_models/step90_qqp_adapter_1.safetensors": "adapter",
    "step90_models/step90_qqp_adapter_2.safetensors": "adapter",
    "step90_models/step90_qqp_adapter_3.safetensors": "adapter",
    "STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_RAW_2026-08-12.npz": "raw_result",
    "STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_RESULTS_2026-08-12.json": "result",
    "validate_step90_independent.py": "independent_validator",
    "STEP90_HIGH_CONFIDENCE_INDEPENDENT_VALIDATION_2026-08-12.json": "validation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_record(source: Path, relative: str, role: str) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError({"relative": relative, "source": str(source)})
    destination = TARGET / Path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage with same-volume hard links to avoid duplicating the already
    # validated historical release. Exported archives contain ordinary files.
    destination.hardlink_to(source)
    return {
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "role": role,
    }


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    base_manifest_path = BASE / MANIFEST_NAME
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    TARGET.mkdir(parents=True)
    records: dict[str, dict[str, object]] = {}
    for relative, old_record in sorted(base_manifest["files"].items()):
        source = ROOT / Path(relative) if relative.startswith("paper_draft/") or relative in OVERRIDES else BASE / Path(relative)
        records[relative] = copy_record(source, relative, old_record["role"])
    for relative, role in sorted(ADDITIONS.items()):
        records[relative] = copy_record(ROOT / Path(relative), relative, role)

    manifest = {
        **base_manifest,
        "schema": "step90.anonymous_validation_closure_release.v6",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "refresh": {
            "reason": "step90_sealed_raw_input_response_adapter_transfer",
            "numerical_evidence_changed": True,
            "base_manifest_sha256": sha256(base_manifest_path),
            "derived_step89_step90_packages_embedded": True,
            "public_parent_weights_embedded": False,
            "third_party_repositories_embedded": False,
        },
        "bindings": {
            **base_manifest["bindings"],
            "paper_pdf_sha256": sha256(TARGET / "paper_draft/main.pdf"),
            "step89_retained_negative": {
                "results": sha256(TARGET / "STEP89_QNLI_EXECUTABLE_ADAPTER_RESULTS_2026-08-12.json"),
                "raw": sha256(TARGET / "STEP89_QNLI_EXECUTABLE_ADAPTER_RAW_2026-08-12.npz"),
            },
            "step90": {
                "stage0_manifest": sha256(TARGET / "STEP90_HIGH_CONFIDENCE_ADAPTER_STAGE0_MANIFEST_2026-08-12.json"),
                "frozen_config": sha256(TARGET / "STEP90_HIGH_CONFIDENCE_STAGEA_FROZEN_CONFIG_2026-08-12.json"),
                "preregistration": sha256(TARGET / "STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_PREREGISTRATION_2026-08-12.md"),
                "execution_lock": sha256(TARGET / "STEP90_HIGH_CONFIDENCE_CONFIRMATORY_EXECUTION_LOCK_2026-08-12.json"),
                "results": sha256(TARGET / "STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_RESULTS_2026-08-12.json"),
                "raw": sha256(TARGET / "STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_RAW_2026-08-12.npz"),
                "independent_validator": sha256(TARGET / "validate_step90_independent.py"),
                "independent_receipt": sha256(TARGET / "STEP90_HIGH_CONFIDENCE_INDEPENDENT_VALIDATION_2026-08-12.json"),
                "manuscript_validator": sha256(TARGET / "validate_step90_manuscript_integration.py"),
            },
        },
        "summary": {
            **base_manifest["summary"],
            "files": len(records),
            "bytes": sum(int(record["bytes"]) for record in records.values()),
            "step89_retained_negative_tasks": 1,
            "step90_confirmatory_tasks": 2,
            "step90_passing_tasks": 2,
            "step90_paired_active_trajectories": 4000,
            "step90_fixed_query_replays": 2000,
            "step90_independent_checks": 12077,
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
    }, indent=2))


if __name__ == "__main__":
    main()
