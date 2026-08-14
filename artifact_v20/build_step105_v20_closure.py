"""Build the isolated Step 105 / V20 anonymous review artifact."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
PARENT_ZIP = ROOT / "STEP104_ANONYMOUS_REVIEW_ARTIFACT_V19_2026-08-13.zip"
PARENT_MANIFEST_NAME = "STEP82_RELEASE_MANIFEST.json"
PARENT_COPY_NAME = "STEP105_V19_PARENT_MANIFEST.json"
TARGET = ROOT / "STEP105_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V20"
MANIFEST_NAME = "STEP105_RELEASE_MANIFEST.json"

ROOT_OVERLAYS = {
    "README_STEP105_ARTIFACT.md": "release_control",
    "STEP105_PROSPECTIVE_ONE_SHOT_CLOSURE_REPORT_2026-08-13.md": "review_closure_report",
    "STEP105_YELP_PROSPECTIVE_CROSS_TASK_ONE_SHOT_PROTOCOL_2026-08-13.md": "prospective_protocol",
    "STEP105_PUBLIC_METADATA_SNAPSHOT_2026-08-13.json": "public_metadata_lock",
    "STEP105_PREDATA_LOCK_2026-08-13.json": "predata_lock",
    "STEP105_STAGE0_DATA_AND_SEAL_MANIFEST_2026-08-13.json": "data_seal_manifest",
    "STEP105_YELP_DEVELOPMENT_ROWS_2026-08-13.json": "development_rows",
    "STEP105_TRAIN_AND_PREOUTCOME_PREDICTION_LEDGER_2026-08-13.json": "training_ledger",
    "STEP105_PREOUTCOME_PREDICTIONS_2026-08-13.npz": "preoutcome_predictions",
    "STEP105_PREOUTCOME_LOCK_2026-08-13.json": "preoutcome_lock",
    "STEP105_PRIMARY_CONFIRMATORY_ARRAYS_2026-08-13.npz": "primary_arrays",
    "STEP105_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json": "primary_ledger",
    "STEP105_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json": "raw_input_validation_receipt",
    "STEP105_ARTIFACT_RECONSTRUCTION_2026-08-13.json": "runner_independent_receipt",
    "step105_common.py": "experiment_code",
    "step105_executable_adapter_endpoint.py": "runtime_endpoint",
    "step105_create_predata_lock.py": "lock_code",
    "step105_stage0_prepare_and_seal.py": "data_seal_code",
    "step105_train_lock_predictions.py": "training_code",
    "step105_create_preoutcome_lock.py": "lock_code",
    "step105_confirm_sealed_one_shot.py": "confirmatory_code",
    "step105_validate_independent.py": "raw_input_validator",
    "validate_step105_artifact_reconstruction.py": "runner_independent_validator",
    "validate_step105_anonymous_release.py": "portable_release_validator",
    "build_step105_v20_closure.py": "release_builder",
    "package_step105_v20.py": "release_packager",
}

DIR_OVERLAYS = {
    "step105_models": "learned_adapter_or_audit",
    "step105_test_inputs": "label_free_test_input",
    "step105_preoutcome_work": "preoutcome_auxiliary",
    "external_data/step105_sealed": "sealed_outcome",
}

PAPER_EXCLUDE = {
    "main.aux", "main.bbl", "main.blg", "main.fdb_latexmk", "main.fls",
    "main.log", "main.out", "main.synctex.gz",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(relative: str) -> None:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise AssertionError({"unsafe_relative_path": relative})


def reset_target() -> None:
    resolved_root = ROOT.resolve()
    resolved_target = TARGET.resolve()
    if resolved_target.parent != resolved_root or resolved_target.name != "STEP105_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V20":
        raise AssertionError(resolved_target)
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir()


def copy_file(relative: str) -> None:
    source = ROOT / relative
    target = TARGET / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(relative: str) -> None:
    source = ROOT / relative
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if path.is_file():
            copy_file(path.relative_to(ROOT).as_posix())


def copy_paper() -> None:
    paper = ROOT / "paper_draft"
    for path in sorted(paper.rglob("*")):
        if not path.is_file() or path.name in PAPER_EXCLUDE:
            continue
        copy_file(path.relative_to(ROOT).as_posix())


def extract_parent() -> tuple[dict, bytes]:
    if not PARENT_ZIP.is_file():
        raise FileNotFoundError(PARENT_ZIP)
    with ZipFile(PARENT_ZIP) as bundle:
        manifest_bytes = bundle.read(PARENT_MANIFEST_NAME)
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != "step104.anonymous_validation_closure_release.v13":
            raise AssertionError("unexpected V19 parent schema")
        names = set(bundle.namelist())
        for relative, record in sorted(manifest["files"].items()):
            safe_relative(relative)
            if relative not in names:
                raise AssertionError({"missing_parent_member": relative})
            payload = bundle.read(relative)
            if len(payload) != record["bytes"] or sha256_bytes(payload) != record["sha256"]:
                raise AssertionError({"parent_member_drift": relative})
            if relative.startswith("paper_draft/"):
                continue
            target = TARGET / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    (TARGET / PARENT_COPY_NAME).write_bytes(manifest_bytes)
    return manifest, manifest_bytes


def role_for(relative: str, parent: dict) -> str:
    if relative == PARENT_COPY_NAME:
        return "parent_release_manifest"
    if relative.startswith("paper_draft/"):
        return "manuscript_or_validation_source"
    if relative in ROOT_OVERLAYS:
        return ROOT_OVERLAYS[relative]
    for directory, role in DIR_OVERLAYS.items():
        if relative.startswith(directory + "/"):
            return role
    return parent.get("files", {}).get(relative, {}).get("role", "inherited_validated_artifact")


def main() -> None:
    reset_target()
    parent, parent_bytes = extract_parent()
    copy_paper()
    for relative in ROOT_OVERLAYS:
        copy_file(relative)
    for relative in DIR_OVERLAYS:
        copy_tree(relative)

    primary104 = json.loads((TARGET / "STEP104_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json").read_text(encoding="utf-8"))
    primary105 = json.loads((TARGET / "STEP105_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json").read_text(encoding="utf-8"))
    raw105 = json.loads((TARGET / "STEP105_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json").read_text(encoding="utf-8"))
    recon105 = json.loads((TARGET / "STEP105_ARTIFACT_RECONSTRUCTION_2026-08-13.json").read_text(encoding="utf-8"))
    if primary104["decision"] != "GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY":
        raise AssertionError("Step 104 primary drift")
    if primary105["decision"] != "NO_GO_RETAIN_STEP105_PROSPECTIVE_NEGATIVE":
        raise AssertionError("Step 105 decision drift")
    if primary105["failed_gates"] != ["terminal_mean_at_least_half_point", "active_minus_fixed_mean_at_least_half_point"]:
        raise AssertionError("Step 105 failed-gate drift")
    if raw105["verdict"] != "PASS_STEP105_PRIMARY_INDEPENDENT_RECONSTRUCTION" or raw105["failed_checks"]:
        raise AssertionError("Step 105 raw-input validation failed")
    if recon105["decision"] != "PASS_STEP105_ARTIFACT_RECONSTRUCTION" or recon105["failed_checks"]:
        raise AssertionError("Step 105 reconstruction failed")

    pdf = TARGET / "paper_draft/main.pdf"
    reader = PdfReader(str(pdf))
    if len(reader.pages) != 28:
        raise AssertionError({"paper_pages": len(reader.pages)})
    if "LIMITATIONS" not in (reader.pages[8].extract_text() or "").upper():
        raise AssertionError("main scientific text does not close on page 9")
    if "doi:" not in (reader.pages[9].extract_text() or "").lower():
        raise AssertionError("references do not begin on page 10")

    records: dict[str, dict[str, object]] = {}
    for path in sorted(TARGET.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(TARGET).as_posix()
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": role_for(relative, parent),
        }

    effect = primary105["effect"]
    manifest = {
        "schema": "step105.anonymous_validation_closure_release.v14",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "claims": {
            "step104_primary_retained": True,
            "step105_literal_no_go_retained": True,
            "step105_target_selector_tuning": False,
            "step105_positive_directional_inference": True,
            "step105_materiality_gate_passed": False,
            "step105_replacement_outcome_authorized": False,
            "production_or_live_registry_claim": False,
        },
        "bindings": {
            "parent_v19_zip_sha256": sha256(PARENT_ZIP),
            "parent_v19_manifest_sha256": sha256_bytes(parent_bytes),
            "paper_pdf_sha256": sha256(pdf),
            "step105_predata_lock_sha256": sha256(TARGET / "STEP105_PREDATA_LOCK_2026-08-13.json"),
            "step105_preoutcome_lock_sha256": sha256(TARGET / "STEP105_PREOUTCOME_LOCK_2026-08-13.json"),
            "step105_primary_arrays_sha256": sha256(TARGET / "STEP105_PRIMARY_CONFIRMATORY_ARRAYS_2026-08-13.npz"),
            "step105_primary_ledger_sha256": sha256(TARGET / "STEP105_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json"),
            "step105_raw_input_validation_sha256": sha256(TARGET / "STEP105_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json"),
            "step105_artifact_reconstruction_sha256": sha256(TARGET / "STEP105_ARTIFACT_RECONSTRUCTION_2026-08-13.json"),
        },
        "summary": {
            "files": len(records),
            "bytes": sum(record["bytes"] for record in records.values()),
            "paper_pages": 28,
            "main_scientific_text_pages": 9,
            "references_begin_page": 10,
            "step104_decision": primary104["decision"],
            "step105_task": "yelp_polarity_test",
            "step105_rows": primary105["rows"],
            "step105_runs": primary105["paired_runs"],
            "step105_target_selector_grid_cells": primary105["target_selector_grid_cells"],
            "step105_terminal_harm_pp": 100 * effect["terminal"]["mean"],
            "step105_terminal_ci_pp": [100 * value for value in effect["terminal"]["bootstrap_95"]],
            "step105_cumulative_regret_delta": effect["cumulative"]["mean"],
            "step105_path_change_rate": effect["path_change_rate"],
            "step105_final_root_change_rate": effect["final_root_change_rate"],
            "step105_fixed_query_exact_zero": effect["max_abs_fixed_terminal_delta"] == 0 and effect["max_abs_fixed_cumulative_delta"] == 0,
            "step105_decision": primary105["decision"],
            "step105_failed_gates": primary105["failed_gates"],
            "step105_scope": "prospective_cross_benchmark_same_task_family_materiality_near_miss",
        },
        "files": records,
    }
    manifest_path = TARGET / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": "PASS_STEP105_V20_BUILD",
        "target": TARGET.name,
        "files": len(records),
        "bytes": manifest["summary"]["bytes"],
        "paper_pdf_sha256": manifest["bindings"]["paper_pdf_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "step105_decision": primary105["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
