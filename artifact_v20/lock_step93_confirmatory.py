from __future__ import annotations

import importlib.metadata
import inspect
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

import step93_common as common
from step93_stagea_train_and_develop import source_text_sha256


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
PROTOCOL = ROOT / f"STEP93_SOURCE_FAITHFUL_LEARNED_ABSTENTION_PREREGISTRATION_{DATE}.md"
AMENDMENT_A = ROOT / f"STEP93_PREREGISTRATION_AMENDMENT_A_{DATE}.md"
AMENDMENT_B = ROOT / f"STEP93_PREREGISTRATION_AMENDMENT_B_{DATE}.md"
MANIFEST = ROOT / f"STEP93_STAGE0_MANIFEST_{DATE}.json"
CONFIG = ROOT / f"STEP93_STAGEA_FROZEN_CONFIG_{DATE}.json"
LEDGER = ROOT / f"STEP93_STAGEA_COMPLETE_SEARCH_LEDGER_{DATE}.json"
CALIBRATION_OUTPUTS = ROOT / f"STEP93_STAGEA_CALIBRATION_OUTPUTS_{DATE}.npz"
HOLDOUT_INPUTS = ROOT / f"STEP93_BANKING77_HOLDOUT_INPUTS_{DATE}.json"
SEALED = ROOT / "external_data" / "step93_sealed" / f"STEP93_BANKING77_SEALED_OUTCOMES_{DATE}.npz"
STAGE0 = ROOT / "step93_stage0_prepare.py"
STAGEA = ROOT / "step93_stagea_train_and_develop.py"
STAGEB = ROOT / "step93_stageb_confirmatory.py"
COMMON = ROOT / "step93_common.py"
ENDPOINT = ROOT / "step93_executable_abstention_endpoint.py"
LOCK = ROOT / f"STEP93_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json"
RESULT = ROOT / f"STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RESULTS_{DATE}.json"
RAW = ROOT / f"STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RAW_{DATE}.npz"
MODEL_DIR = ROOT / "step93_models"


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    if LOCK.exists():
        raise RuntimeError("refusing to overwrite Step 93 execution lock")
    for required in (
        PROTOCOL,
        AMENDMENT_A,
        AMENDMENT_B,
        MANIFEST,
        CONFIG,
        LEDGER,
        CALIBRATION_OUTPUTS,
        HOLDOUT_INPUTS,
        SEALED,
        STAGE0,
        STAGEA,
        STAGEB,
        COMMON,
        ENDPOINT,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if RESULT.exists() or RAW.exists():
        raise RuntimeError("confirmatory outputs already exist before lock")

    stagea_text = STAGEA.read_text(encoding="utf-8")
    forbidden = (
        "STEP93_BANKING77_HOLDOUT_INPUTS",
        "STEP93_BANKING77_SEALED_OUTCOMES",
        "external_data/step93_sealed",
        "external_data\\step93_sealed",
        'split="test"',
        "split='test'",
        '["test"]',
        "['test']",
    )
    hits = [token for token in forbidden if token in stagea_text]
    if hits:
        raise AssertionError({"stagea_forbidden_tokens": hits})

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    authority_paths = {
        "protocol_sha256": PROTOCOL,
        "amendment_a_sha256": AMENDMENT_A,
        "amendment_b_sha256": AMENDMENT_B,
    }
    for key, path in authority_paths.items():
        if manifest[key] != common.sha256_path(path):
            raise AssertionError(f"manifest authority mismatch: {key}")
        if config["authority_sha256"][key] != common.sha256_path(path):
            raise AssertionError(f"config authority mismatch: {key}")
    if config["stage0_manifest_sha256"] != common.sha256_path(MANIFEST):
        raise AssertionError("config/manifest binding mismatch")
    if config["stagea_ledger_sha256"] != common.sha256_path(LEDGER):
        raise AssertionError("config/ledger binding mismatch")
    if config["calibration_outputs_sha256"] != common.sha256_path(CALIBRATION_OUTPUTS):
        raise AssertionError("config/calibration-output binding mismatch")
    if manifest["outputs"][HOLDOUT_INPUTS.name] != common.sha256_path(HOLDOUT_INPUTS):
        raise AssertionError("input-only holdout hash mismatch")
    sealed_key = relative(SEALED)
    if manifest["outputs"][sealed_key] != common.sha256_path(SEALED):
        raise AssertionError("sealed outcome hash mismatch")

    model_files: list[Path] = []
    for name, expected in config["clean_root_sha256"].items():
        path = MODEL_DIR / name
        if common.sha256_path(path) != expected:
            raise AssertionError(f"clean root hash mismatch: {name}")
        audit = common.tensor_audit(path)
        if audit["parameter_count"] < 109_000 or audit["variance"] <= 0.0:
            raise AssertionError({"invalid_clean_root": name, "audit": audit})
        model_files.append(path)
    for name, expected in config["all_learned_adapter_sha256"].items():
        path = MODEL_DIR / name
        if common.sha256_path(path) != expected:
            raise AssertionError(f"learned adapter hash mismatch: {name}")
        audit = common.tensor_audit(path)
        if (
            audit["parameter_count"] != 50_817
            or audit["variance"] <= 0.0
            or audit["nonzero_fraction"] < 0.95
        ):
            raise AssertionError({"invalid_learned_adapter": name, "audit": audit})
        model_files.append(path)
    if len(model_files) != 40:
        raise AssertionError({"model_files": len(model_files)})
    selected_hashes = list(config["selected_learned_adapter_sha256"].values())
    if len(selected_hashes) != 4 or len(set(selected_hashes)) != 4:
        raise AssertionError("selected learned adapter hashes are not four distinct values")

    base_path = Path(
        hf_hub_download(
            common.BASE_MODEL,
            "model.safetensors",
            revision=common.BASE_REVISION,
            local_files_only=True,
        )
    )
    if common.sha256_path(base_path) != common.BASE_MODEL_SHA256:
        raise AssertionError("base model hash mismatch")

    source_assertion = config["single_similarity_assertion"]
    observed_source_assertion = {
        "function_source_sha256": source_text_sha256(common.exact_match_similarity),
        "feedback_function_source_sha256": source_text_sha256(common.exact_match_feedback),
        "group_builder_source_sha256": source_text_sha256(common.build_source_faithful_groups),
        "run_active_source_sha256": source_text_sha256(common.run_active),
        "run_active_accepts_external_feedback": "feedback" in inspect.signature(
            common.run_active
        ).parameters,
    }
    for key, observed in observed_source_assertion.items():
        if source_assertion[key] != observed:
            raise AssertionError(
                {"same_similarity_assertion": key, "expected": source_assertion[key], "observed": observed}
            )
    if observed_source_assertion["run_active_accepts_external_feedback"]:
        raise AssertionError("run_active unexpectedly accepts a decoupled feedback matrix")

    locked_paths = [
        PROTOCOL,
        AMENDMENT_A,
        AMENDMENT_B,
        MANIFEST,
        CONFIG,
        LEDGER,
        CALIBRATION_OUTPUTS,
        HOLDOUT_INPUTS,
        SEALED,
        STAGE0,
        STAGEA,
        STAGEB,
        COMMON,
        ENDPOINT,
        *model_files,
    ]
    lock = {
        "lock_id": "STEP93_SOURCE_FAITHFUL_LEARNED_ABSTENTION_CONFIRMATORY_LOCK_V1",
        "date": DATE,
        "locked_sha256": {relative(path): common.sha256_path(path) for path in locked_paths},
        "base_model": {
            "id": common.BASE_MODEL,
            "revision": common.BASE_REVISION,
            "model_safetensors_sha256": common.BASE_MODEL_SHA256,
        },
        "confirmatory_design": {
            "seeds": [931000, 931999],
            "paired_runs": 1000,
            "pool_size": common.POOL_SIZE,
            "budget": common.BUDGET,
            "bootstrap_seed": 93500,
            "signflip_seed": 93501,
            "single_similarity": "s(a,b)=1[a=b]",
        },
        "same_similarity_source_assertion": observed_source_assertion,
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "datasets", "numpy", "scipy", "safetensors")
        },
        "stagea_forbidden_token_hits": hits,
        "confirmatory_outputs_absent_at_lock": True,
        "test_outcome_access_before_lock": False,
        "item_or_prompt_lookup": False,
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "lock": LOCK.name,
                "lock_sha256": common.sha256_path(LOCK),
                "locked_files": len(lock["locked_sha256"]),
                "locked_model_files": len(model_files),
                "stagea_forbidden_token_hits": hits,
                "same_similarity_source_assertion": observed_source_assertion,
                "confirmatory_outputs_absent_at_lock": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
