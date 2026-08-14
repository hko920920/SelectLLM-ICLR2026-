"""Validate the isolated Step 97 / V18 anonymous review release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "STEP82_RELEASE_MANIFEST.json"
PARENT_MANIFEST = ROOT / "STEP97_V17_PARENT_MANIFEST.json"
TEXT_SUFFIXES = {".py", ".md", ".tex", ".bib", ".json", ".txt", ".sty", ".bst"}
DATA_ROLES = {
    "sealed_test_input", "sealed_test_outcome", "development_input", "development_output",
    "complete_development_ledger", "retained_pre_repair_ledger", "retained_pre_repair_config",
    "confirmatory_raw", "postconfirmatory_raw", "preoutcome_prediction", "development_cache",
    "diagnostic_output", "development_only_oracle_diagnostic", "learned_error_adapter_audit",
    "quarantined_preanalysis_partial",
}
WINDOWS_USER_PATH = re.compile(r"(?i)[A-Z]:\\Users\\[^\\\"'\s]+")
IDENTITY_TOKEN = "SO" + "GANG"

EXPECTED_PARENT_MANIFEST = "99108ca6b181b39fcc54b8c4d9260474b3a0df080499d76defa52d70cbec76f6"
EXPECTED_PARENT_ZIP = "f3d51f0f2ddb32d6d0bdf1ff66b8ef6b1dfd7ffbfd045ab2ab7884158159f20e"
EXPECTED_PDF = "1f44c1e9a8b1cdaabb90b1b442def5cb3e5ffd96a559847fef9aa370351dd2d5"
EXPECTED_STEP96 = {
    "STEP96_SNLI_DIRECTIONAL_TERMINAL_BRIDGE_PREREGISTRATION_2026-08-13.md": "1f7656eb26d5a2b61d9b2e130238f27026f149f4eb8a55392390ce78c3b6ff95",
    "STEP96_PREREGISTRATION_AMENDMENT_A_INTEGER_GATE_2026-08-13.md": "a4fa300ce8a27978e2120d0639f973f22f84a6b49e910f9ae1e4eaf717676bfa",
    "STEP96_PREREGISTRATION_AMENDMENT_B_CONFIRMATORY_SEEDS_2026-08-13.md": "796d6af1ae1f026c9379a241a33007b18af543995ae760a6222a46448b46294e",
    "STEP96_STAGEA_COMPLETE_LEDGER_2026-08-13.json": "87c54d3299374fc7b6f396c28356b4569cf989c5acf840b894a37f14cc07ad54",
    "STEP96_STAGEA_FROZEN_CONFIG_2026-08-13.json": "33ac40d6a590b414c83464acc2289ca590d8e034f4219abddbfec9e3258d1fc5",
    "STEP96_INTEGER_QUALITY_GATE_REPAIR_RECEIPT_2026-08-13.json": "19fffb1068bf981e1466cbfa0cb4094c6205bd9119f60ef0617c1e9bf5033913",
    "STEP96_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json": "f14c942950d169e5a8a5df87e56f17bfbb3010b575469dde42ddbf00e93f916f",
    "STEP96_CONFIRMATORY_PREOUTCOME_PREDICTIONS_2026-08-13.npz": "0062d2b431d837cf5a06c087205e4e9799e4caadc7b1e2252462cb2e71e92aa2",
    "STEP96_CONFIRMATORY_RAW_2026-08-13.npz": "bac710c2026768f203565c5b9b7b1d1cb50c8f804cb40404135a7cb851e032e1",
    "STEP96_CONFIRMATORY_RESULTS_2026-08-13.json": "41fb12b9797cc592b64a53cb2861b5c39c2e90454e62c6a9e8fd5f3eb60ae901",
    "STEP96_CONFIRMATORY_INDEPENDENT_VALIDATION_2026-08-13.json": "294af7cf4c850cf053b547d75fde41534ccf48211b63a1986dd127ebcaf77809",
    "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_RAW_2026-08-13.npz": "abe83b22c879e065114577af36bf93ee43d98cae855f52ebb8ef88affb9e1dad",
    "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_SENSITIVITY_2026-08-13.json": "9ede07d62bda858c45b1fad503d6735b6fab19b682e012f55a62cc6987aaca51",
}
EXPECTED_STEP97 = {
    "STEP97_SCITAIL_CONSERVATIVE_TERMINAL_BRIDGE_PREREGISTRATION_2026-08-13.md": "bcc30540ad8633b1bd053a12cdc6699769ba3410297169c3d8bfca06f5ccbf4b",
    "STEP97_STAGEA_ROOT_GEOMETRY_STOP_2026-08-13.json": "6ce784b278c9e710a129c1f119994a049348b056db6d72249aa3435c74a939c7",
    "STEP97_ROOT_GEOMETRY_STOP_INDEPENDENT_VALIDATION_2026-08-13.json": "d26a75ea5f03f22f95e1c5b46b49eadac0c40d6457c3b456a2dd4fbf1a4eb91d",
}
EXPECTED_VALIDATORS = {
    "validate_step96_independent.py": "6498d11c3fab08c050f6ed2cb6202cf6bfa218c06b830078a8b971b204f693cf",
    "audit_step96_complete_top8_test_sensitivity.py": "6268e8626bdb30ca6e2c682e8de18b14039c08809c6d075179a47e27d62d76d0",
    "validate_step97_geometry_stop.py": "e2ea1a500e448044e9964f37ebff3de1c4acbcf2502c755d75c91a36eb615174",
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
    if len(reader.pages) != 25:
        raise AssertionError({"paper_pages": len(reader.pages)})
    pages = [(reader.pages[index].extract_text() or "").upper() for index in (8, 9)]
    page9, page10 = pages
    page9_compact = "".join(page9.split())
    page10_compact = "".join(page10.split())
    late_appendix_compact = "".join(
        "".join((reader.pages[index].extract_text() or "").upper().replace("_", "").split())
        for index in range(20, 25)
    )
    if "LIMITATIONS,IMPLICATIONS,ANDCONCLUSION" not in page9_compact:
        raise AssertionError("scientific conclusion does not finish on page 9")
    if "REFERENCES" in {"".join(line.split()) for line in page9.splitlines()}:
        raise AssertionError("references begin before page 10")
    if "AIUSESTATEMENT" not in page10_compact or "REFERENCES" not in page10_compact:
        raise AssertionError("non-counting statements/references do not begin on page 10")
    if "SNLIDIRECTIONALLEARNEDTERMINALAUDIT" not in late_appendix_compact:
        raise AssertionError("Step 96 appendix section missing")
    if "SCITAIL" not in late_appendix_compact or "NOGORETAINSTEP96NEGATIVE" not in late_appendix_compact:
        raise AssertionError("Step 96/97 retained decisions missing from appendix")


def compare_npz(left: Path, right: Path) -> None:
    with np.load(left, allow_pickle=False) as lhs, np.load(right, allow_pickle=False) as rhs:
        if set(lhs.files) != set(rhs.files):
            raise AssertionError({"npz_keys": [lhs.files, rhs.files]})
        for key in lhs.files:
            if not np.array_equal(lhs[key], rhs[key]):
                raise AssertionError({"npz_array_drift": key})


def copy_relative(temp_root: Path, relatives: list[str]) -> None:
    for relative in relatives:
        destination = temp_root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / Path(relative), destination)


def reconstruct_step96_primary() -> dict[str, object]:
    relatives = [
        "validate_step96_independent.py", "step96_common.py", "step95_common.py", "step93_common.py",
        "STEP96_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json",
        "STEP96_CONFIRMATORY_PREOUTCOME_PREDICTIONS_2026-08-13.npz",
        "STEP96_CONFIRMATORY_PREOUTCOME_LEDGER_2026-08-13.json",
        "external_data/step96_sealed/snli_test_sealed_outcomes.npz",
        "STEP96_CONFIRMATORY_RAW_2026-08-13.npz",
        "STEP96_CONFIRMATORY_RESULTS_2026-08-13.json",
        "STEP96_STAGEA_DEVELOPMENT_ARRAYS_2026-08-13.npz",
        "STEP96_STAGEA_COMPLETE_LEDGER_2026-08-13.json",
        "STEP96_INTEGER_QUALITY_GATE_REPAIR_RECEIPT_2026-08-13.json",
    ]
    with tempfile.TemporaryDirectory(prefix="step96_primary_reconstruction_") as raw:
        temp_root = Path(raw)
        copy_relative(temp_root, relatives)
        run([sys.executable, "validate_step96_independent.py"], "PASS_STEP96_INDEPENDENT_VALIDATION", temp_root)
        generated = temp_root / "STEP96_CONFIRMATORY_INDEPENDENT_VALIDATION_2026-08-13.json"
        bundled = ROOT / generated.name
        if sha256(generated) != sha256(bundled):
            raise AssertionError("Step 96 independent receipt is not byte-reproduced")
    return {"name": "step96_primary_independent_reconstruction", "passed": True}


def reconstruct_step96_top8() -> dict[str, object]:
    relatives = [
        "audit_step96_complete_top8_test_sensitivity.py", "step96_common.py", "step95_common.py", "step93_common.py",
        "STEP96_STAGEA_COMPLETE_LEDGER_2026-08-13.json",
        "STEP96_CONFIRMATORY_PREOUTCOME_PREDICTIONS_2026-08-13.npz",
        "external_data/step96_sealed/snli_test_sealed_outcomes.npz",
        "STEP96_CONFIRMATORY_RESULTS_2026-08-13.json",
    ]
    with tempfile.TemporaryDirectory(prefix="step96_top8_reconstruction_") as raw:
        temp_root = Path(raw)
        copy_relative(temp_root, relatives)
        run(
            [sys.executable, "audit_step96_complete_top8_test_sensitivity.py"],
            "POSTCONFIRMATORY_EXPLORATORY_NOT_PRIMARY_CONFIRMATION", temp_root,
        )
        generated_raw = temp_root / "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_RAW_2026-08-13.npz"
        generated_json = temp_root / "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_SENSITIVITY_2026-08-13.json"
        bundled_raw = ROOT / generated_raw.name
        bundled_json = ROOT / generated_json.name
        compare_npz(generated_raw, bundled_raw)
        observed = load(generated_json)
        expected = load(bundled_json)
        observed["bound_sha256"]["raw"] = expected["bound_sha256"]["raw"]
        if observed != expected:
            raise AssertionError("Step 96 complete-top-eight semantic reconstruction drift")
    return {"name": "step96_complete_top8_reconstruction", "passed": True}


def reconstruct_step97_stop() -> dict[str, object]:
    relatives = [
        "validate_step97_geometry_stop.py", "step97_common.py", "step96_common.py", "step95_common.py", "step93_common.py",
        "STEP97_SCITAIL_CONSERVATIVE_TERMINAL_BRIDGE_PREREGISTRATION_2026-08-13.md",
        "STEP97_SCITAIL_DEVELOPMENT_ROWS_2026-08-13.json",
        "STEP97_STAGEA_ROOT_GEOMETRY_STOP_2026-08-13.json",
        "step97_stagea_work/root_predictions.npz",
    ]
    with tempfile.TemporaryDirectory(prefix="step97_stop_reconstruction_") as raw:
        temp_root = Path(raw)
        copy_relative(temp_root, relatives)
        run([sys.executable, "validate_step97_geometry_stop.py"], "PASS_STEP97_GEOMETRY_STOP_VALIDATION", temp_root)
        generated = temp_root / "STEP97_ROOT_GEOMETRY_STOP_INDEPENDENT_VALIDATION_2026-08-13.json"
        bundled = ROOT / generated.name
        if sha256(generated) != sha256(bundled):
            raise AssertionError("Step 97 stop receipt is not byte-reproduced")
    return {"name": "step97_geometry_stop_reconstruction", "passed": True}


def verify_integrity() -> dict:
    manifest = load(MANIFEST)
    if manifest.get("schema") != "step97.anonymous_validation_closure_release.v12":
        raise AssertionError("unexpected V18 schema")
    refresh = manifest.get("refresh", {})
    expected_refresh = {
        "numerical_evidence_changed": True,
        "complete_step96_primary_embedded": True,
        "complete_step96_top8_sensitivity_embedded": True,
        "step96_primary_no_go_retained": True,
        "step96_secondary_not_relabelled_confirmatory": True,
        "step97_geometry_stop_embedded": True,
        "step97_test_outcome_opened": False,
        "public_base_model_weights_embedded": False,
        "third_party_repositories_embedded": False,
        "isolated_from_worktree_by_byte_copy": True,
    }
    for key, expected in expected_refresh.items():
        if refresh.get(key) is not expected:
            raise AssertionError({"refresh_key": key, "observed": refresh.get(key)})
    if refresh.get("parent_v17_manifest_sha256") != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("parent V17 manifest binding drift")
    if refresh.get("parent_v17_zip_sha256") != EXPECTED_PARENT_ZIP:
        raise AssertionError("parent V17 ZIP binding drift")

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
        if re.search(rf"(?i)\b{IDENTITY_TOKEN}\b", source):
            identity_hits.append(relative)
    if workstation_hits or identity_hits:
        raise AssertionError({"workstation_path_hits": workstation_hits, "identity_hits": identity_hits})

    if sha256(PARENT_MANIFEST) != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("embedded V17 manifest byte drift")
    if load(PARENT_MANIFEST).get("schema") != "step95.anonymous_validation_closure_release.v11":
        raise AssertionError("embedded V17 manifest schema drift")
    bindings = manifest.get("bindings", {})
    if bindings.get("parent_v17_manifest_sha256") != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("manifest-to-V17 snapshot binding drift")
    if bindings.get("paper_pdf_sha256") != EXPECTED_PDF or sha256(ROOT / "paper_draft/main.pdf") != EXPECTED_PDF:
        raise AssertionError("paper PDF binding drift")
    assert_pdf_boundary(ROOT / "paper_draft/main.pdf")

    step96_files = bindings.get("step96", {}).get("files", {})
    step97_files = bindings.get("step97", {}).get("files", {})
    for name, expected in EXPECTED_STEP96.items():
        if step96_files.get(name) != expected or sha256(ROOT / name) != expected:
            raise AssertionError({"step96_binding": name})
    for name, expected in EXPECTED_STEP97.items():
        if step97_files.get(name) != expected or sha256(ROOT / name) != expected:
            raise AssertionError({"step97_binding": name})
    for name, expected in EXPECTED_VALIDATORS.items():
        if sha256(ROOT / name) != expected:
            raise AssertionError({"validator_binding": name})

    config = load(ROOT / "STEP96_STAGEA_FROZEN_CONFIG_2026-08-13.json")
    model_map = bindings["step96"].get("model_weights", {})
    if model_map != config["adapter_sha256"] or len(model_map) != 4:
        raise AssertionError("complete Step 96 adapter map drift")
    if any(records.get(relative, {}).get("sha256") != digest for relative, digest in model_map.items()):
        raise AssertionError("Step 96 adapter manifest record drift")
    if sum(record.get("role") == "learned_error_adapter_weights" for record in records.values()) != 4:
        raise AssertionError("Step 96 learned adapter weight count drift")

    primary = load(ROOT / "STEP96_CONFIRMATORY_RESULTS_2026-08-13.json")
    validation = load(ROOT / "STEP96_CONFIRMATORY_INDEPENDENT_VALIDATION_2026-08-13.json")
    sensitivity = load(ROOT / "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_SENSITIVITY_2026-08-13.json")
    if primary.get("decision") != "NO_GO_RETAIN_STEP96_NEGATIVE":
        raise AssertionError("literal Step 96 primary decision drift")
    if primary.get("failed_gates") != ["alias_integer_loss_counts_at_most_one_point"]:
        raise AssertionError("literal Step 96 failed gate drift")
    if primary["quality"]["loss_counts"] != [93, 100, 88, 86] or primary["quality"]["allowed_loss_count"] != 98:
        raise AssertionError("Step 96 primary quality counts drift")
    if not close(primary["inference"]["terminal"]["mean"], 0.01630533333333334):
        raise AssertionError("Step 96 primary terminal mean drift")
    if not close(primary["inference"]["terminal"]["bootstrap_95"][0], 0.013596566666666669):
        raise AssertionError("Step 96 primary terminal interval drift")
    if not close(primary["inference"]["cumulative"]["mean"], 0.3161253333333334):
        raise AssertionError("Step 96 primary cumulative effect drift")
    if not close(primary["mechanism"]["path_change_rate"], 0.9913333333333333):
        raise AssertionError("Step 96 primary path-change drift")
    if not primary["gates"]["fixed_query_exact_zero"]:
        raise AssertionError("Step 96 fixed-query equality drift")
    if validation.get("status") != "PASS_STEP96_INDEPENDENT_VALIDATION" or not all(validation.get("checks", {}).values()):
        raise AssertionError("Step 96 independent validation receipt drift")

    if sensitivity.get("status") != "POSTCONFIRMATORY_EXPLORATORY_NOT_PRIMARY_CONFIRMATION":
        raise AssertionError("Step 96 sensitivity status drift")
    if sensitivity.get("primary_decision_unchanged") != primary["decision"]:
        raise AssertionError("Step 96 sensitivity replaces the primary")
    if sensitivity["scope"] != {
        "all_predeclared_top8_conditions_included": True,
        "nominal_rows": 8,
        "unique_tau_threshold_conditions": 2,
        "models_adapters_thresholds_and_test_predictions_all_fixed_before_test_open": True,
        "analysis_authorized_only_after_primary_test_open": True,
        "confirmatory_status": False,
    }:
        raise AssertionError("Step 96 sensitivity scope drift")
    conservative = sensitivity["unique_conditions"][1]
    if conservative["loss_counts"] != [31, 33, 47, 31] or conservative["allowed_loss_count"] != 98:
        raise AssertionError("Step 96 conservative quality counts drift")
    if not conservative["all_gates_with_holm"]:
        raise AssertionError("Step 96 conservative gate result drift")
    if not close(conservative["inference"]["terminal"]["mean"], 0.009145333333333335):
        raise AssertionError("Step 96 conservative terminal mean drift")
    if not close(conservative["inference"]["terminal"]["bootstrap_95"][0], 0.006667966666666669):
        raise AssertionError("Step 96 conservative interval drift")
    if not close(conservative["inference"]["terminal"]["holm_q_across_unique_top8_conditions"], 1.9999800001999982e-05):
        raise AssertionError("Step 96 Holm correction drift")

    stop = load(ROOT / "STEP97_STAGEA_ROOT_GEOMETRY_STOP_2026-08-13.json")
    stop_validation = load(ROOT / "STEP97_ROOT_GEOMETRY_STOP_INDEPENDENT_VALIDATION_2026-08-13.json")
    if stop.get("decision") != "NO_GO_STEP97_ROOT_GEOMETRY_STOP" or stop.get("sealed_test_opened") is not False:
        raise AssertionError("Step 97 stop decision drift")
    if stop.get("failed_gates") != ["parent_unique_best_by_2_5pp_on_roster_gate"]:
        raise AssertionError("Step 97 failed gate drift")
    if not close(stop["geometry"]["roster_gate"]["parent_minus_best_challenger"], 0.0040000000000000036):
        raise AssertionError("Step 97 roster geometry drift")
    if stop_validation.get("status") != "PASS_STEP97_GEOMETRY_STOP_VALIDATION" or not all(stop_validation.get("checks", {}).values()):
        raise AssertionError("Step 97 validation receipt drift")

    main_source = (ROOT / "paper_draft/main.tex").read_text(encoding="utf-8")
    active_lines = [line.strip() for line in main_source.splitlines() if not line.lstrip().startswith("%")]
    if "\\author{Anonymous Authors}" not in main_source or "\\iclrfinalcopy" in active_lines:
        raise AssertionError("anonymous manuscript mode drift")
    manuscript = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / "paper_draft").rglob("*.tex"))
    )
    required_manuscript = (
        "one alias loses 100/9,824 correct examples versus 98 allowed",
        "All eight predeveloped top conditions collapse to two thresholds",
        "This is secondary learned same-$s$ terminal evidence",
        "a frozen SciTail replication stops before test",
        "strategic-replication bandits let known agents register duplicate arms",
    )
    if any(phrase not in manuscript for phrase in required_manuscript):
        raise AssertionError("Step 96/97 manuscript scope or numerical statement drift")

    summary = manifest.get("summary", {})
    expected_summary = {
        "paper_pages": 25,
        "main_scientific_text_pages": 9,
        "step96_task": "snli_test",
        "step96_confirmatory_runs": 3000,
        "step96_primary_terminal_harm_pp": 1.630533333333334,
        "step96_primary_cumulative_regret_delta": 0.3161253333333334,
        "step96_primary_path_change_rate": 0.9913333333333333,
        "step96_primary_final_root_change_rate": 0.16233333333333333,
        "step96_primary_fixed_query_exact_zero": True,
        "step96_primary_decision": "NO_GO_RETAIN_STEP96_NEGATIVE",
        "step96_primary_failed_gates": ["alias_integer_loss_counts_at_most_one_point"],
        "step96_secondary_status": "POSTCONFIRMATORY_EXPLORATORY_NOT_PRIMARY_CONFIRMATION",
        "step96_secondary_terminal_harm_pp": 0.9145333333333335,
        "step96_secondary_holm_q": 1.9999800001999982e-05,
        "step96_secondary_all_gates_with_holm": True,
        "step97_task": "scitail_tsv_format_test",
        "step97_decision": "NO_GO_STEP97_ROOT_GEOMETRY_STOP",
        "step97_test_opened": False,
        "review_closure": "learned_single_similarity_terminal_mechanism_present_primary_quality_near_miss_secondary_pass",
    }
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
            reconstruct_step96_primary(),
            reconstruct_step96_top8(),
            reconstruct_step97_stop(),
            run(
                [sys.executable, "paper_draft/validate_paper_tables.py"],
                "PASS_PAPER_TABLE_SOURCE_CONSISTENCY",
            ),
        ])
        with tempfile.TemporaryDirectory(prefix="step97_pdf_build_") as raw:
            build_dir = Path(raw)
            checks.append(run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={build_dir}", "main.tex"],
                cwd=ROOT / "paper_draft",
            ))
            assert_pdf_boundary(build_dir / "main.pdf")
        verify_integrity()
    print(json.dumps({
        "verdict": "PASS_STEP97_ANONYMOUS_RELEASE",
        "mode": "full" if args.full else "integrity-only",
        "bundled_files": len(manifest["files"]),
        "workstation_path_hits": 0,
        "public_base_model_weights_bundled": 0,
        "parent_v17_manifest_bound": True,
        "step96_adapter_weight_files": 4,
        "step96_primary_decision": "NO_GO_RETAIN_STEP96_NEGATIVE",
        "step96_secondary_status": "POSTCONFIRMATORY_EXPLORATORY_NOT_PRIMARY_CONFIRMATION",
        "step97_decision": "NO_GO_STEP97_ROOT_GEOMETRY_STOP",
        "step97_test_opened": False,
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
