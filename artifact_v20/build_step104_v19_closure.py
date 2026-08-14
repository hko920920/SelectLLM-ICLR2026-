"""Build the isolated V19 anonymous artifact with Steps 98--104 bound."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARENT_ZIP = ROOT / "STEP97_ANONYMOUS_REVIEW_ARTIFACT_V18_2026-08-13.zip"
TARGET = ROOT / "STEP104_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V19"
MANIFEST_NAME = "STEP82_RELEASE_MANIFEST.json"
PARENT_MANIFEST_NAME = "STEP104_V18_PARENT_MANIFEST.json"
EXPECTED_PARENT_ZIP_SHA256 = "d6ccd3ebdb336225f3b46123bc770e435c388bf8a98f875e7169f01df9aab41a"
EXPECTED_PARENT_MANIFEST_SHA256 = "d921ba8f1cf0759820f1bddc473425ea17752ca5caf85fc8e9a57d9ec3a44429"


EXPLICIT_ADDITIONS: dict[str, str] = {
    "README_STEP104_ARTIFACT.md": "release_control",
    "STEP104_REVIEW_CLOSURE_REPORT_2026-08-13.md": "review_closure_report",
    "build_step104_v19_closure.py": "release_builder",
    "package_step104_v19.py": "release_packager",
    "validate_step104_anonymous_release.py": "release_validator",
    "validate_step104_artifact_reconstruction.py": "artifact_reconstruction_validator",
    "paper_draft/tables/step104_imdb_heldout_primary.tex": "manuscript_source",
}


TREE_ROLES = {
    "step98_models": "retained_anli_learned_adapter",
    "step98_selection_work": "retained_anli_selection_cache",
    "step98_stagea_work": "retained_anli_development_cache",
    "step98_test_inputs": "retained_anli_test_input",
    "external_data/step98_sealed": "retained_anli_sealed_outcome",
    "step100_stagea_work": "imdb_development_cache",
    "step100_test_inputs": "imdb_heldout_input",
    "step101_candidate_predictions": "imdb_parent_screen_predictions",
    "step102_models": "imdb_learned_adapter",
    "external_data/step100_sealed": "imdb_sealed_outcome",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_for_top_level(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".safetensors"):
        return "learned_adapter_weights"
    if "preregistration" in lowered or "protocol" in lowered:
        return "protocol_or_preregistration"
    if "validation" in lowered or lowered.startswith("validate_"):
        return "independent_validation"
    if "lock" in lowered or "frozen_config" in lowered:
        return "preoutcome_lock_or_config"
    if lowered.endswith(".npz"):
        return "machine_readable_array"
    if lowered.endswith(".json"):
        return "machine_readable_ledger"
    if lowered.endswith(".py"):
        return "execution_code"
    return "supporting_record"


def collect_additions() -> dict[str, str]:
    additions = dict(EXPLICIT_ADDITIONS)
    prefix = re.compile(r"^(?:STEP|step)(?:98|99|100|101|102|103|104)_")
    for path in sorted(ROOT.iterdir()):
        if path.is_file() and prefix.match(path.name):
            additions[path.name] = role_for_top_level(path.name)
    for directory, default_role in TREE_ROLES.items():
        base = ROOT / Path(directory)
        if not base.is_dir():
            raise FileNotFoundError(base)
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if path.suffix == ".safetensors":
                additions[relative] = "learned_error_adapter_weights"
            elif path.name.endswith(".audit.json"):
                additions[relative] = "learned_error_adapter_audit"
            else:
                additions[relative] = default_role
    return additions


def copy_local(relative: str, role: str) -> dict[str, object]:
    source = ROOT / Path(relative)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = TARGET / Path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"bytes": destination.stat().st_size, "sha256": sha256(destination), "role": role}


def write_parent_bytes(relative: str, data: bytes, role: str) -> dict[str, object]:
    destination = TARGET / Path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {"bytes": len(data), "sha256": sha256_bytes(data), "role": role}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    if sha256(PARENT_ZIP) != EXPECTED_PARENT_ZIP_SHA256:
        raise AssertionError("V18 parent ZIP drift")
    additions = collect_additions()
    TARGET.mkdir(parents=True)
    with zipfile.ZipFile(PARENT_ZIP, "r") as bundle:
        parent_manifest_bytes = bundle.read(MANIFEST_NAME)
        if sha256_bytes(parent_manifest_bytes) != EXPECTED_PARENT_MANIFEST_SHA256:
            raise AssertionError("V18 parent manifest drift")
        parent = json.loads(parent_manifest_bytes)
        if parent.get("schema") != "step97.anonymous_validation_closure_release.v12":
            raise AssertionError("unexpected V18 parent schema")
        records: dict[str, dict[str, object]] = {}
        for relative, old_record in sorted(parent["files"].items()):
            if relative.startswith("paper_draft/"):
                records[relative] = copy_local(relative, str(old_record["role"]))
                continue
            data = bundle.read(relative)
            if len(data) != old_record["bytes"] or sha256_bytes(data) != old_record["sha256"]:
                raise AssertionError(f"parent file drift: {relative}")
            records[relative] = write_parent_bytes(relative, data, str(old_record["role"]))
        records[PARENT_MANIFEST_NAME] = write_parent_bytes(
            PARENT_MANIFEST_NAME, parent_manifest_bytes, "parent_release_manifest"
        )
    for relative, role in sorted(additions.items()):
        records[relative] = copy_local(relative, role)

    step98 = load(TARGET / "STEP98_STAGEA_COMPLETE_LEDGER_2026-08-13.json")
    step99 = load(TARGET / "STEP99_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json")
    step100 = load(TARGET / "STEP100_STAGEA_COMPLETE_LEDGER_2026-08-13.json")
    step101 = load(TARGET / "STEP101_IMDB_DEVELOPMENT_PARENT_ROSTER_SELECTION_2026-08-13.json")
    step102 = load(TARGET / "STEP102_STAGEA_COMPLETE_LEDGER_2026-08-13.json")
    step103 = load(TARGET / "STEP103_ROBUST_DUAL_DEVELOPMENT_LEDGER_2026-08-13.json")
    step103_validation = load(TARGET / "STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION_2026-08-13.json")
    lock = load(TARGET / "STEP104_PREOUTCOME_LOCK_2026-08-13.json")
    primary = load(TARGET / "STEP104_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json")
    independent = load(TARGET / "STEP104_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json")
    reconstruction = load(TARGET / "STEP104_ARTIFACT_RECONSTRUCTION_2026-08-13.json")

    expected_decisions = {
        "step98": (step98["decision"], "NO_GO_STEP98_DEVELOPMENT_STOP"),
        "step99": (step99["decision"], "NO_GO_RETAIN_STEP99_NEGATIVE"),
        "step100": (step100["decision"], "NO_GO_STEP100_GEOMETRY_STOP"),
        "step101": (step101["decision"], "NO_GO_STEP101_PARENT_GEOMETRY_STOP"),
        "step102": (step102["decision"], "NO_GO_STEP102_LEARNED_DEVELOPMENT_STOP"),
        "step103": (step103["decision"], "GO_STEP103_TO_HELDOUT_LOCK"),
        "step104": (primary["decision"], "GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY"),
    }
    for name, (observed, expected) in expected_decisions.items():
        if observed != expected:
            raise AssertionError({name: observed})
    if step100.get("sealed_outcome_opened") or step101.get("sealed_outcome_opened") or step102.get("sealed_outcome_opened") or step103.get("sealed_outcome_opened"):
        raise AssertionError("IMDB held-out outcome opened during development")
    if step103_validation["decision"] != "PASS_STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION" or not all(step103_validation["checks"].values()):
        raise AssertionError("Step 103 independent development validation drift")
    if primary["failed_gates"] or not all(primary["gates"].values()):
        raise AssertionError("Step 104 primary gate drift")
    if independent["decision"] != "PASS_STEP104_PRIMARY_INDEPENDENT_RECONSTRUCTION" or independent["failed_checks"] or not all(independent["checks"].values()):
        raise AssertionError("Step 104 raw-input independent validation drift")
    if reconstruction["decision"] != "PASS_STEP104_ARTIFACT_RECONSTRUCTION" or reconstruction["failed_checks"] or not all(reconstruction["checks"].values()):
        raise AssertionError("Step 104 artifact reconstruction drift")
    if sha256(TARGET / lock["input_file"]) != lock["input_sha256"] or sha256(TARGET / lock["sealed_file"]) != lock["sealed_sha256"]:
        raise AssertionError("Step 104 input/outcome binding drift")
    for group in ("authority_sha256", "code_sha256", "model_sha256", "model_audit_sha256", "preoutcome_sha256"):
        for relative, digest in lock[group].items():
            if records.get(relative, {}).get("sha256") != digest:
                raise AssertionError({"lock_group": group, "relative": relative})
    if len(lock["model_sha256"]) != 4 or len(set(lock["model_sha256"].values())) != 4:
        raise AssertionError("Step 104 adapter map drift")

    step98_104_files = {
        relative: str(record["sha256"])
        for relative, record in records.items()
        if re.match(r"^(?:STEP|step)(?:98|99|100|101|102|103|104)_", relative)
        or relative.startswith("step98_models/")
        or relative.startswith("step102_models/")
        or relative.startswith("step100_test_inputs/")
        or relative.startswith("external_data/step100_sealed/")
    }
    manifest = {
        **parent,
        "schema": "step104.anonymous_validation_closure_release.v13",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "refresh": {
            "reason": "development_informed_imdb_heldout_learned_same_similarity_primary",
            "numerical_evidence_changed": True,
            "parent_v18_zip_sha256": EXPECTED_PARENT_ZIP_SHA256,
            "parent_v18_manifest_sha256": EXPECTED_PARENT_MANIFEST_SHA256,
            "retained_step98_102_failures_embedded": True,
            "step103_robust_selection_embedded": True,
            "step104_preoutcome_lock_embedded": True,
            "step104_primary_go_embedded": True,
            "step104_raw_input_independent_receipt_embedded": True,
            "step104_runner_independent_reconstruction_embedded": True,
            "public_base_model_weights_embedded": False,
            "third_party_repositories_embedded": False,
            "isolated_from_parent_zip_and_worktree": True,
        },
        "bindings": {
            **parent["bindings"],
            "paper_pdf_sha256": sha256(TARGET / "paper_draft/main.pdf"),
            "parent_v18_manifest_sha256": sha256(TARGET / PARENT_MANIFEST_NAME),
            "step104": {
                "files": step98_104_files,
                "lock_sha256": sha256(TARGET / "STEP104_PREOUTCOME_LOCK_2026-08-13.json"),
                "model_weights": lock["model_sha256"],
                "model_audits": lock["model_audit_sha256"],
                "input_sha256": lock["input_sha256"],
                "sealed_outcome_sha256": lock["sealed_sha256"],
                "primary_arrays_sha256": primary["arrays_sha256"],
                "raw_input_validation_sha256": sha256(TARGET / "STEP104_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json"),
                "artifact_reconstruction_sha256": sha256(TARGET / "STEP104_ARTIFACT_RECONSTRUCTION_2026-08-13.json"),
            },
        },
        "summary": {
            **parent["summary"],
            "files": len(records),
            "bytes": sum(int(record["bytes"]) for record in records.values()),
            "paper_pages": 26,
            "main_scientific_text_pages": 9,
            "references_begin_page": 9,
            "step104_task": "imdb_test_heldout",
            "step104_rows": primary["rows"],
            "step104_runs": primary["runs"],
            "step104_terminal_harm_pp": 100.0 * primary["effects"]["terminal"]["mean"],
            "step104_terminal_ci_pp": [100.0 * value for value in primary["effects"]["terminal"]["bootstrap_95"]],
            "step104_active_minus_fixed_terminal_harm_pp": 100.0 * primary["effects"]["active_minus_fixed_terminal"]["mean"],
            "step104_cumulative_regret_delta": primary["effects"]["cumulative"]["mean"],
            "step104_path_change_rate": primary["mechanism"]["path_change_rate"],
            "step104_final_root_change_rate": primary["mechanism"]["final_root_change_rate"],
            "step104_quality_loss_counts": primary["quality"]["loss_counts"],
            "step104_quality_allowed_loss_count": primary["quality"]["allowed_loss_count"],
            "step104_fixed_query_exact_zero": primary["gates"]["fixed_query_exact_zero"],
            "step104_decision": primary["decision"],
            "step104_failed_gates": primary["failed_gates"],
            "step104_provenance": "development_informed_disjoint_heldout_not_before_all_data_preregistered",
            "review_closure": "learned_raw_input_same_similarity_terminal_primary_passes_all_frozen_gates",
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
        "step104_adapter_weights": len(lock["model_sha256"]),
        "step104_decision": primary["decision"],
        "step104_independent": independent["decision"],
        "step104_artifact_reconstruction": reconstruction["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
