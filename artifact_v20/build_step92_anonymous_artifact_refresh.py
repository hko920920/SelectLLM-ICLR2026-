"""Build the isolated V13 anonymous artifact with Step 92 evidence closure."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "STEP90_V12_CLEAN_BASE_FROM_ZIP_2026-08-12"
TARGET = ROOT / "STEP92_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V13"
MANIFEST_NAME = "STEP82_RELEASE_MANIFEST.json"

ADDITIONS = {
    "README_STEP92_ARTIFACT.md": "release_control",
    "build_step92_anonymous_artifact_refresh.py": "release_builder",
    "fetch_step92_public_dependencies.py": "dependency_fetcher",
    "validate_step92_anonymous_release.py": "release_validator",
    "validate_step92_manuscript_integration.py": "manuscript_validator",
    "paper_draft/tables/step92_learned_adapter_results.tex": "manuscript_source",
    "STEP92_LEARNED_CONTEXTUAL_ADAPTER_PREREGISTRATION_2026-08-12.md": "preregistration",
    "STEP92_STAGE0_MANIFEST_2026-08-12.json": "manifest",
    "STEP92_AGNEWS_CALIBRATION_PACKAGE_2026-08-12.json": "derived_input",
    "STEP92_AGNEWS_HOLDOUT_INPUTS_2026-08-12.json": "derived_input",
    "external_data/step92_sealed/STEP92_AGNEWS_SEALED_OUTCOMES_2026-08-12.npz": "sealed_outcome",
    "STEP92_STAGEA_CALIBRATION_OUTPUTS_2026-08-12.npz": "development_output",
    "STEP92_STAGEA_COMPLETE_SEARCH_LEDGER_2026-08-12.json": "development_ledger",
    "STEP92_STAGEA_FROZEN_CONFIG_2026-08-12.json": "frozen_config",
    "STEP92_CONFIRMATORY_EXECUTION_LOCK_2026-08-12.json": "execution_lock",
    "step92_common.py": "execution_code",
    "step92_stage0_prepare.py": "execution_code",
    "step92_stagea_train_and_develop.py": "execution_code",
    "step92_stageb_confirmatory.py": "execution_code",
    "lock_step92_confirmatory.py": "execution_code",
    "STEP92_LEARNED_ADAPTER_CONFIRMATORY_RAW_2026-08-12.npz": "raw_result",
    "STEP92_LEARNED_ADAPTER_CONFIRMATORY_RESULTS_2026-08-12.json": "result",
    "validate_step92_independent.py": "independent_validator",
    "STEP92_INDEPENDENT_VALIDATION_2026-08-12.json": "validation",
    "audit_step92_numeric_tie_robustness.py": "robustness_validator",
    "STEP92_NUMERIC_TIE_ROBUSTNESS_2026-08-12.json": "validation",
    "STEP92_NUMERIC_TIE_ROBUSTNESS_RAW_2026-08-12.npz": "raw_result",
}

for model_path in sorted((ROOT / "step92_models").glob("*.safetensors")):
    relative = model_path.relative_to(ROOT).as_posix()
    role = "clean_root_adapter" if "clean_root" in model_path.name else "learned_contextual_adapter"
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
    # V12 used space-saving hard links; V13 intentionally copies bytes so that
    # later worktree edits cannot mutate a sealed release candidate.
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
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("schema") != "step90.anonymous_validation_closure_release.v6":
        raise AssertionError("clean V12 base has the wrong schema")

    TARGET.mkdir(parents=True)
    records: dict[str, dict[str, object]] = {}
    for relative, old_record in sorted(base_manifest["files"].items()):
        # Current manuscript files replace their historical V12 counterparts;
        # all non-manuscript evidence comes only from the hash-verified V12 ZIP.
        source = ROOT / Path(relative) if relative.startswith("paper_draft/") else BASE / Path(relative)
        records[relative] = copy_record(source, relative, old_record["role"])
    for relative, role in sorted(ADDITIONS.items()):
        records[relative] = copy_record(ROOT / Path(relative), relative, role)

    model_records = {
        relative: record["sha256"]
        for relative, record in records.items()
        if relative.startswith("step92_models/")
    }
    selected = {
        relative: digest
        for relative, digest in model_records.items()
        if "parent_08_learned_adapter" in relative
    }
    manifest = {
        **base_manifest,
        "schema": "step92.anonymous_validation_closure_release.v7",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "refresh": {
            "reason": "step92_preregistered_learned_contextual_adapter_bridge",
            "numerical_evidence_changed": True,
            "base_manifest_sha256": sha256(base_manifest_path),
            "base_v12_zip_sha256": "5035f2856f7bc38efb9db191836820542aac0e9d41753febaeeb8f68f5c7d4a9",
            "derived_step92_packages_embedded": True,
            "all_stagea_learned_weights_embedded": True,
            "public_base_encoder_weights_embedded": False,
            "third_party_repositories_embedded": False,
            "isolated_from_worktree_by_byte_copy": True,
        },
        "bindings": {
            **base_manifest["bindings"],
            "paper_pdf_sha256": sha256(TARGET / "paper_draft/main.pdf"),
            "step92": {
                "preregistration": sha256(TARGET / "STEP92_LEARNED_CONTEXTUAL_ADAPTER_PREREGISTRATION_2026-08-12.md"),
                "stage0_manifest": sha256(TARGET / "STEP92_STAGE0_MANIFEST_2026-08-12.json"),
                "complete_search_ledger": sha256(TARGET / "STEP92_STAGEA_COMPLETE_SEARCH_LEDGER_2026-08-12.json"),
                "calibration_outputs": sha256(TARGET / "STEP92_STAGEA_CALIBRATION_OUTPUTS_2026-08-12.npz"),
                "frozen_config": sha256(TARGET / "STEP92_STAGEA_FROZEN_CONFIG_2026-08-12.json"),
                "execution_lock": sha256(TARGET / "STEP92_CONFIRMATORY_EXECUTION_LOCK_2026-08-12.json"),
                "results": sha256(TARGET / "STEP92_LEARNED_ADAPTER_CONFIRMATORY_RESULTS_2026-08-12.json"),
                "raw": sha256(TARGET / "STEP92_LEARNED_ADAPTER_CONFIRMATORY_RAW_2026-08-12.npz"),
                "independent_validator": sha256(TARGET / "validate_step92_independent.py"),
                "independent_receipt": sha256(TARGET / "STEP92_INDEPENDENT_VALIDATION_2026-08-12.json"),
                "numeric_robustness_validator": sha256(TARGET / "audit_step92_numeric_tie_robustness.py"),
                "numeric_robustness_receipt": sha256(TARGET / "STEP92_NUMERIC_TIE_ROBUSTNESS_2026-08-12.json"),
                "numeric_robustness_raw": sha256(TARGET / "STEP92_NUMERIC_TIE_ROBUSTNESS_RAW_2026-08-12.npz"),
                "manuscript_validator": sha256(TARGET / "validate_step92_manuscript_integration.py"),
                "model_weights": model_records,
                "selected_adapter_weights": selected,
            },
        },
        "summary": {
            **base_manifest["summary"],
            "files": len(records),
            "bytes": sum(int(record["bytes"]) for record in records.values()),
            "step92_task": "ag_news",
            "step92_clean_root_weight_files": 12,
            "step92_learned_adapter_weight_files": 20,
            "step92_selected_learned_adapters": 4,
            "step92_parameters_per_selected_adapter": 50817,
            "step92_paired_active_trajectories": 2000,
            "step92_fixed_query_replays": 2000,
            "step92_independent_checks": 4170,
            "step92_raw_input_endpoint_samples": 256,
            "step92_terminal_harm_pp": 0.54675,
            "step92_fixed_query_exact_zero": True,
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
                "model_weight_files": len(model_records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
