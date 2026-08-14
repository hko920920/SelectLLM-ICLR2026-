"""Validate the isolated Step 93 / V15 anonymous review release."""

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
PARENT_MANIFEST = ROOT / "STEP93_V14_PARENT_MANIFEST.json"
TEXT_SUFFIXES = {".py", ".md", ".tex", ".bib", ".json", ".txt", ".sty", ".bst"}
WINDOWS_USER_PATH = re.compile(r"(?i)[A-Z]:\\Users\\[^\\\"'\s]+")

EXPECTED_PARENT_MANIFEST = "134ecd195ee3b8d125109c2a423b47199f503c0d7c06888afe2784912eef089e"
EXPECTED_PARENT_ZIP = "1eeb19f4d07689283d57725b88811ba2485b95b593e696e08f97eac3853bff8c"
EXPECTED_PDF = "dbb68bdf581c9b95b651d59708b6a18a76a943307f09f811e94a2aa9662983ba"
EXPECTED_STEP93 = {
    "preregistration": "5b91bbac3a3eac126dba44b13959c85d25a018fde1db78386662d976708f4bf0",
    "amendment_a": "20ea03822505f74521276d816e83980815439f401bdf60ebdf639b0ec8573fe9",
    "amendment_b": "950049a20ae699505ef55a0231cd5a25023311adedcf9b996113c705338be4e7",
    "stage0_manifest": "2c6759fb0a33cae9c2b96f76c734a52d13c7636ca6d3e2db7458404a372fd761",
    "complete_search_ledger": "646674f795a0b9ab9597b725e8a6e7f512434e89214ac2fad2607f4e278781f0",
    "calibration_outputs": "6fe65bd70d540a671c2fae66911de3b6c403cd815a9b9cac8652211dc1dbf319",
    "frozen_config": "8de9e66fa32ac0c3e2b7fe4af9b0ea514d41d55cb7f7cbd496bcf72a6783963a",
    "execution_lock": "f389d4be70abc4cf57f10fb7a4685ca4ee3a67a9ed9e3c7a75e0e8272d66ef62",
    "literal_results": "bf6fe6dd3d07be47c9aafa534d0bfa1efd8a7bdbef9a702b61f3f9246d229698",
    "raw": "4b0ad4c7e321924c618f23429d84201d3787d57f7fb478f35624a9066030ab8d",
    "independent_validator": "465ce7cd6cfd81fce717e77ae5994a7c35667cb88f44cc01dc46dfbd1d1abfa7",
    "independent_adjudication": "5ae09c1e3f6801940183e1e17ef06da4f683b3f2cdd8fed080dbca573b3be1e0",
    "manuscript_validator": "952ca60139867885d45443b4ac356d3d466f75b953e38e802dc8df1baecec8c2",
    "closure_report": "2ed55bfd6c2f9f140bafa3744202aaaadd136d04f451346b5719cc8eab2de0d3",
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


def run(command: list[str], marker: str) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or marker not in output:
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
    page9_lines = {line.replace(" ", "").strip() for line in page9.splitlines()}
    page10_lines = {line.replace(" ", "").strip() for line in page10.splitlines()}
    if (
        "LIMITATIONS" not in page9
        or "CANDIDATE LISTS ARE THEREFORE EXPERIMENTAL PRIORS" not in page9
    ):
        raise AssertionError("main scientific text does not finish on page 9")
    if "AIUSESTATEMENT" in page9_lines or "REFERENCES" in page9_lines:
        raise AssertionError("non-scientific statements intrude into page 9")
    if "AIUSESTATEMENT" not in page10_lines or "REFERENCES" not in page10_lines:
        raise AssertionError("statements and references do not begin on page 10")


def verify_integrity() -> dict:
    manifest = load(MANIFEST)
    if manifest.get("schema") != "step93.anonymous_validation_closure_release.v9":
        raise AssertionError("unexpected V15 schema")

    refresh = manifest.get("refresh", {})
    expected_refresh = {
        "numerical_evidence_changed": False,
        "derived_step93_packages_embedded": True,
        "all_stagea_step93_weights_embedded": True,
        "literal_negative_and_adjudication_both_embedded": True,
        "public_base_encoder_weights_embedded": False,
        "third_party_repositories_embedded": False,
        "isolated_from_worktree_by_byte_copy": True,
    }
    for key, expected in expected_refresh.items():
        if refresh.get(key) is not expected:
            raise AssertionError({"refresh_key": key, "observed": refresh.get(key)})
    if refresh.get("parent_v14_manifest_sha256") != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("parent V14 manifest binding drift")
    if refresh.get("parent_v14_zip_sha256") != EXPECTED_PARENT_ZIP:
        raise AssertionError("parent V14 ZIP binding drift")

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
    for relative in records:
        path = ROOT / Path(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES:
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
        raise AssertionError(
            {"workstation_path_hits": workstation_hits, "identity_hits": identity_hits}
        )

    if sha256(PARENT_MANIFEST) != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("embedded parent manifest byte drift")
    parent = load(PARENT_MANIFEST)
    if parent.get("schema") != "step93.anonymous_validation_closure_release.v8":
        raise AssertionError("embedded parent manifest schema drift")
    bindings = manifest.get("bindings", {})
    if bindings.get("parent_v14_manifest_sha256") != EXPECTED_PARENT_MANIFEST:
        raise AssertionError("manifest-to-parent snapshot binding drift")

    step93 = bindings.get("step93", {})
    for key, expected in EXPECTED_STEP93.items():
        if step93.get(key) != expected:
            raise AssertionError({"step93_binding": key, "observed": step93.get(key)})

    config = load(ROOT / "STEP93_STAGEA_FROZEN_CONFIG_2026-08-13.json")
    model_map = step93.get("model_weights", {})
    clean = {path: digest for path, digest in model_map.items() if "clean_root" in path}
    learned = {path: digest for path, digest in model_map.items() if "abstention_adapter" in path}
    if len(model_map) != 40 or len(clean) != 12 or len(learned) != 28:
        raise AssertionError(
            {"step93_models": len(model_map), "clean_roots": len(clean), "learned": len(learned)}
        )
    expected_models = {
        f"step93_models/{name}": digest
        for name, digest in {
            **config["clean_root_sha256"],
            **config["all_learned_adapter_sha256"],
        }.items()
    }
    if model_map != expected_models:
        raise AssertionError("complete Step 93 model-weight map drift")
    for relative, digest in model_map.items():
        if records.get(relative, {}).get("sha256") != digest:
            raise AssertionError({"model_record_mismatch": relative})
    selected = step93.get("selected_adapter_weights", {})
    expected_selected = {
        f"step93_models/{name}": digest
        for name, digest in config["selected_learned_adapter_sha256"].items()
    }
    if selected != expected_selected or len(set(selected.values())) != 4:
        raise AssertionError("selected Step 93 adapter binding drift")
    if config["selected"].get("parent_root") != 10:
        raise AssertionError("selected parent drift")
    audits = config["selected_learned_adapter_audits"]
    if any(audit.get("parameter_count") != 50817 for audit in audits.values()):
        raise AssertionError("selected adapter parameter-count drift")

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

    literal = load(ROOT / "STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RESULTS_2026-08-13.json")
    receipt = load(ROOT / "STEP93_INDEPENDENT_VALIDATION_AND_ADJUDICATION_2026-08-13.json")
    if literal.get("decision") != "INVALID_STEP93_IMPLEMENTATION":
        raise AssertionError("literal machine decision must remain unedited")
    if receipt.get("status") != "PASS_STEP93_INDEPENDENT_RECONSTRUCTION":
        raise AssertionError("independent reconstruction receipt drift")
    if receipt.get("adjudicated_decision") != "NO_GO_RETAIN_SOURCE_FAITHFUL_NEGATIVE":
        raise AssertionError("retained-negative adjudication drift")
    if receipt.get("original_machine_decision") != literal.get("decision"):
        raise AssertionError("literal/adjudication decision linkage drift")
    if receipt.get("original_result_sha256") != EXPECTED_STEP93["literal_results"]:
        raise AssertionError("literal result hash linkage drift")
    numeric = receipt.get("numeric_validator_adjudication", {})
    if (
        not numeric.get("feedback_equality_is_bitwise_exact")
        or not numeric.get("source_hash_assertions_match")
        or not numeric.get("corrected_same_s_gate")
        or not numeric.get("scientific_effect_gates_unchanged")
        or numeric.get("observed_max_abs_difference") != numeric.get("binary64_ulp_at_one")
    ):
        raise AssertionError("one-ULP same-s adjudication drift")
    gates = receipt.get("corrected_gates", {})
    required_true = {
        "all_locked_hashes_match",
        "same_exact_similarity_for_acquisition_and_posterior",
        "four_genuine_distinct_50817_parameter_adapters",
        "raw_text_endpoint_exact_replay_and_runtime_exclusions",
        "coordinate_wise_nonimproving",
        "each_alias_accuracy_loss_at_most_1pp",
        "path_change_at_least_half",
        "fixed_query_terminal_and_cumulative_exact_zero",
    }
    required_false = {
        "mean_terminal_delta_at_least_0_5pp",
        "terminal_bootstrap_lower_above_zero",
        "terminal_one_sided_signflip_p_at_most_0_05",
        "active_minus_fixed_at_least_0_5pp_and_lower_above_zero",
    }
    if any(gates.get(key) is not True for key in required_true):
        raise AssertionError("corrected implementation/fidelity gate drift")
    if any(gates.get(key) is not False for key in required_false):
        raise AssertionError("scientific no-go gate drift")
    effect = receipt.get("confirmatory_effect", {})
    literal_task = literal["task"]
    if not close(effect.get("terminal_delta_mean"), 0.00032):
        raise AssertionError("terminal effect drift")
    if not close(literal_task["cumulative_regret_delta"]["mean"], 0.040945):
        raise AssertionError("cumulative effect drift")
    if effect.get("path_change_rate") != 0.998 or not effect.get("fixed_query_exact_zero"):
        raise AssertionError("path/fixed-query result drift")
    if receipt.get("checks_count") != 82 or receipt["independent_endpoint_probe"].get("samples") != 128:
        raise AssertionError("independent validation coverage drift")

    step92_readme = (ROOT / "README_STEP92_ARTIFACT.md").read_text(encoding="utf-8")
    required_scope = (
        "Scope correction retained in V14",
        "decoupled-view audit",
        "not\n> an equation-faithful single-similarity Select-LLM reproduction",
        "null-terminal Banking77 result",
    )
    if any(phrase not in step92_readme for phrase in required_scope):
        raise AssertionError("historical Step 92 scope correction drift")

    review_closure = (ROOT / "STEP93_REVIEW_CLOSURE_REPORT_2026-08-13.md").read_text(
        encoding="utf-8"
    )
    required_review_closure = (
        "Strategic-replication boundary",
        "one acquired reference label scores every candidate entry",
        "does not transitively rerun the Step 80/87/88/90/92 validators",
        "Numerical evidence changed: no",
    )
    normalized_review_closure = " ".join(review_closure.split())
    if any(phrase not in normalized_review_closure for phrase in required_review_closure):
        raise AssertionError("V15 review-closure report drift")

    expected_summary = {
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
        "review_closure": "strategic_replication_and_reproducibility_precision",
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
                    [sys.executable, "fetch_step93_public_dependencies.py"],
                    "PASS_STEP93_PUBLIC_DEPENDENCY_FETCH",
                ),
                run(
                    [sys.executable, "validate_step93_independent.py", "--verify-existing"],
                    "PASS_STEP93_INDEPENDENT_RECONSTRUCTION",
                ),
                run(
                    [sys.executable, "validate_step93_manuscript_integration.py"],
                    "PASS_STEP93_MANUSCRIPT_INTEGRATION",
                ),
            ]
        )
        verify_integrity()
    print(
        json.dumps(
            {
                "verdict": "PASS_STEP93_ANONYMOUS_RELEASE",
                "mode": "full" if args.full else "integrity-only",
                "bundled_files": len(manifest["files"]),
                "workstation_path_hits": 0,
                "public_base_encoder_weights_bundled": 0,
                "parent_v14_manifest_bound": True,
                "step93_model_weight_files": 40,
                "step93_independently_reconstructed": bool(args.full),
                "step93_decision": "NO_GO_RETAIN_SOURCE_FAITHFUL_NEGATIVE",
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
