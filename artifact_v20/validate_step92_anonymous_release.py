"""Validate V13 integrity, anonymity, learned weights, and replay closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "STEP82_RELEASE_MANIFEST.json"
TEXT_SUFFIXES = {".py", ".md", ".tex", ".bib", ".json", ".txt", ".sty", ".bst"}
WINDOWS_USER_PATH = re.compile(r"(?i)[A-Z]:\\Users\\[^\\\"'\s]+")
EXPECTED_V12_MANIFEST = "f283b2a353126279731e738c88dcf09903cd1b78ad7056854773f5ca66dda110"
EXPECTED_PDF = "cc24e713387f215663466d7f0f42a76233ee4e0ae026f3e31a518541b501c499"
EXPECTED_MODEL_MAP_AGGREGATE = "9c54bec90964ee6fa20cf5e2d37a684d12dd120092593955f8dba2307cf0cf85"

EXPECTED_STEP90 = {
    "stage0_manifest": "4aa672d6657d1324636c74a29510c8a06a5ea804c6f5db0fd51d3272c9df87cb",
    "frozen_config": "3dfda0e5deb6db51bba510fcf074360f4a8764a95271c16f427bbec1b585ecb4",
    "preregistration": "70147e0b5b1c70024e95492f23d31dbe9b739866468a407237bc0cb661b05c55",
    "execution_lock": "b73730c970f6e8d78a59978fe1e379168ebed498ca4bbc1123c39b8bf3ff42b3",
    "results": "f699eee78c4fd84139933e469d1c1c2bfbefb9ffe4a6cc3f278a2996f55d5654",
    "raw": "5ef206b57d2204c128e621f07c83f8a621d1e94ab6856cbddc05d8e3bbd44d6d",
    "independent_validator": "ee3aa0b1377733c25fcf3d6c356faf8d0a6696eee368127b83ebdceadc5aa918",
    "independent_receipt": "0226d76e493905bfed74b62647729f4ed3e83d37bc127c57d81d20a41bf4aa79",
    "manuscript_validator": "77beaa671e314a89e7be0e861c7e1ab7ddec7021cd61b15113317aa6f3cf9d8e",
}
EXPECTED_STEP92 = {
    "preregistration": "47c63712082f5b119b7495195f6134bdfa9dae1947931aae93d6cc13cca7451c",
    "stage0_manifest": "9a6bc9d60a1e339fc94eddf62c0fea16ae611ec64515f0e95695bb8bd8208147",
    "complete_search_ledger": "562fbc63ab223c2cff5cbebcb013a1eb9745398d45b98e50f213b33f56f0d525",
    "calibration_outputs": "813a9f6e4fcf1af1afb6f81c18c73b7eab4ff9d21acdb2b8d3b79aa612c59a32",
    "frozen_config": "2c2d51aa81c4a99d942c0a129fed99785149315adfc20092ec62b869e442e9f1",
    "execution_lock": "c219349a46e4a326f4d3cf8853ac69093014d0bdfe73ae44f0359f4259d426cc",
    "results": "8e20db3d7bbeb6fd4dff550ba8a24e2e992d88eb7bcaeb3d1233428deca011e4",
    "raw": "fdcaf9b09c648dcf13ae21b4a28e5358afe6df85a5f97b497bc866f3f07816ff",
    "independent_validator": "9fc7c1e90fe9c003167de87c8dcd4f08f99daf73f5af8338aded0c2dd7c88bdf",
    "independent_receipt": "8ac808aaa8b68686b1d4190c92a90035207c5d3de3bbb7d1d099f0bcc167015b",
    "numeric_robustness_validator": "dbf2a9c3ce9e23fb7bde151fbadd0958882603235bc2e9eae3932cc0ef24605a",
    "numeric_robustness_receipt": "e78c017dd34cc425022dcf3aaf4e9c503b0a428bd6b840b836692478649ff32a",
    "numeric_robustness_raw": "ea98945cfae3216ca9a9efd905c91e23035e2afb61652ca30f63ef754b47ce2f",
    "manuscript_validator": "5566d0df1f150ba8600276cb764d770a9f8a8fa9b0a3534c7e1643910c6a40f1",
}
EXPECTED_SELECTED = {
    "step92_models/step92_parent_08_learned_adapter_0.safetensors": "29fbc038d39093c69e1e75811bf14f579fcf69f73169764d8f55e7f3a7ad98fb",
    "step92_models/step92_parent_08_learned_adapter_1.safetensors": "ec46f442071830e0b4ccb13d9879bf0c52e4b856fc8eb891fa4afb93b3bb79ad",
    "step92_models/step92_parent_08_learned_adapter_2.safetensors": "9892f0bd36ac9d1a940c2207bac989922d81c979f5ff449664ce254086a748c4",
    "step92_models/step92_parent_08_learned_adapter_3.safetensors": "7a161beb3e1c1961fd749117d55a1f09ee47405c81a90451f111dbce97c3a93e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def model_map_aggregate(model_map: dict[str, str]) -> str:
    payload = "\n".join(f"{path}={model_map[path]}" for path in sorted(model_map))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_integrity() -> dict:
    manifest = load(MANIFEST)
    if manifest.get("schema") != "step92.anonymous_validation_closure_release.v7":
        raise AssertionError("unexpected V13 schema")
    refresh = manifest.get("refresh", {})
    expected_refresh = {
        "numerical_evidence_changed": True,
        "derived_step92_packages_embedded": True,
        "all_stagea_learned_weights_embedded": True,
        "public_base_encoder_weights_embedded": False,
        "third_party_repositories_embedded": False,
        "isolated_from_worktree_by_byte_copy": True,
    }
    for key, expected in expected_refresh.items():
        if refresh.get(key) is not expected:
            raise AssertionError({"refresh_key": key, "observed": refresh.get(key)})
    if refresh.get("base_manifest_sha256") != EXPECTED_V12_MANIFEST:
        raise AssertionError("inherited clean V12 manifest binding drift")

    records = manifest["files"]
    for relative, record in records.items():
        path = ROOT / Path(relative)
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise AssertionError(f"missing or drifting bundled file: {relative}")

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
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if WINDOWS_USER_PATH.search(text):
            workstation_hits.append(relative)
        if re.search(r"(?i)\bSOGANG\b", text):
            identity_hits.append(relative)
    if workstation_hits or identity_hits:
        raise AssertionError(
            {"workstation_path_hits": workstation_hits, "identity_hits": identity_hits}
        )

    if manifest["bindings"].get("step90") != EXPECTED_STEP90:
        raise AssertionError("inherited Step 90 binding drift")
    step92 = manifest["bindings"].get("step92", {})
    for key, expected in EXPECTED_STEP92.items():
        if step92.get(key) != expected:
            raise AssertionError({"step92_binding": key, "observed": step92.get(key)})
    if step92.get("selected_adapter_weights") != EXPECTED_SELECTED:
        raise AssertionError("selected adapter binding drift")

    model_map = step92.get("model_weights", {})
    clean = [path for path in model_map if "clean_root" in path]
    learned = [path for path in model_map if "learned_adapter" in path]
    if len(model_map) != 32 or len(clean) != 12 or len(learned) != 20:
        raise AssertionError(
            {"model_files": len(model_map), "clean_roots": len(clean), "learned": len(learned)}
        )
    if model_map_aggregate(model_map) != EXPECTED_MODEL_MAP_AGGREGATE:
        raise AssertionError("complete Step 92 model map drift")
    for relative, digest in model_map.items():
        if records.get(relative, {}).get("sha256") != digest:
            raise AssertionError(f"model/record mismatch: {relative}")

    if manifest["bindings"].get("paper_pdf_sha256") != EXPECTED_PDF:
        raise AssertionError("paper PDF manifest binding drift")
    if sha256(ROOT / "paper_draft/main.pdf") != EXPECTED_PDF:
        raise AssertionError("paper PDF byte drift")
    main_source = (ROOT / "paper_draft/main.tex").read_text(encoding="utf-8")
    if "\\author{Anonymous Authors}" not in main_source or "\\iclrfinalcopy" in main_source.splitlines():
        raise AssertionError("anonymous manuscript mode drift")

    result = load(ROOT / "STEP92_LEARNED_ADAPTER_CONFIRMATORY_RESULTS_2026-08-12.json")
    receipt = load(ROOT / "STEP92_INDEPENDENT_VALIDATION_2026-08-12.json")
    numeric = load(ROOT / "STEP92_NUMERIC_TIE_ROBUSTNESS_2026-08-12.json")
    if result.get("decision") != "GO_SCORE_CHANGING_LEARNED_ADAPTER_BRIDGE":
        raise AssertionError("Step 92 confirmatory decision drift")
    if not all(result["task"]["gates"].values()):
        raise AssertionError("Step 92 gate drift")
    if receipt.get("verdict") != "PASS_STEP92_INDEPENDENT_VALIDATION":
        raise AssertionError("Step 92 independent receipt drift")
    if numeric.get("decision") != "PASS_NUMERIC_TIE_ROBUSTNESS":
        raise AssertionError("Step 92 numeric robustness drift")

    expected_summary = {
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
    }
    for key, value in expected_summary.items():
        if manifest["summary"].get(key) != value:
            raise AssertionError({"summary_key": key, "observed": manifest["summary"].get(key)})
    if manifest["summary"].get("files") != len(records):
        raise AssertionError("manifest file count drift")
    if manifest["summary"].get("bytes") != sum(record["bytes"] for record in records.values()):
        raise AssertionError("manifest byte count drift")
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
                run([sys.executable, "fetch_step92_public_dependencies.py"], "PASS_STEP92_PUBLIC_DEPENDENCY_FETCH"),
                run([sys.executable, "validate_step90_independent.py"], "PASS_STEP90_INDEPENDENT_VALIDATION"),
                run([sys.executable, "validate_step92_independent.py"], "PASS_STEP92_INDEPENDENT_VALIDATION"),
                run([sys.executable, "audit_step92_numeric_tie_robustness.py"], "PASS_NUMERIC_TIE_ROBUSTNESS"),
                run([sys.executable, "validate_step92_manuscript_integration.py"], "PASS_STEP92_MANUSCRIPT_INTEGRATION"),
            ]
        )
        verify_integrity()
    print(
        json.dumps(
            {
                "verdict": "PASS_STEP92_ANONYMOUS_RELEASE",
                "mode": "full" if args.full else "integrity-only",
                "bundled_files": len(manifest["files"]),
                "workstation_path_hits": 0,
                "public_base_encoder_weights_bundled": 0,
                "inherited_v12_manifest_bound": True,
                "step92_model_weight_files": 32,
                "step92_independently_reconstructed": bool(args.full),
                "step92_decision": "GO_SCORE_CHANGING_LEARNED_ADAPTER_BRIDGE",
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
