from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

import step92_common as common


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
PROTOCOL = ROOT / f"STEP92_LEARNED_CONTEXTUAL_ADAPTER_PREREGISTRATION_{DATE}.md"
MANIFEST = ROOT / f"STEP92_STAGE0_MANIFEST_{DATE}.json"
CONFIG = ROOT / f"STEP92_STAGEA_FROZEN_CONFIG_{DATE}.json"
LEDGER = ROOT / f"STEP92_STAGEA_COMPLETE_SEARCH_LEDGER_{DATE}.json"
CALIBRATION_OUTPUTS = ROOT / f"STEP92_STAGEA_CALIBRATION_OUTPUTS_{DATE}.npz"
HOLDOUT_INPUTS = ROOT / f"STEP92_AGNEWS_HOLDOUT_INPUTS_{DATE}.json"
SEALED = ROOT / "external_data" / "step92_sealed" / f"STEP92_AGNEWS_SEALED_OUTCOMES_{DATE}.npz"
STAGEA = ROOT / "step92_stagea_train_and_develop.py"
STAGEB = ROOT / "step92_stageb_confirmatory.py"
COMMON = ROOT / "step92_common.py"
LOCK = ROOT / f"STEP92_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json"
RESULT = ROOT / f"STEP92_LEARNED_ADAPTER_CONFIRMATORY_RESULTS_{DATE}.json"
RAW = ROOT / f"STEP92_LEARNED_ADAPTER_CONFIRMATORY_RAW_{DATE}.npz"


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    for required in (
        PROTOCOL,
        MANIFEST,
        CONFIG,
        LEDGER,
        CALIBRATION_OUTPUTS,
        HOLDOUT_INPUTS,
        SEALED,
        STAGEA,
        STAGEB,
        COMMON,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if RESULT.exists() or RAW.exists():
        raise RuntimeError("confirmatory outputs already exist before lock")

    stagea_text = STAGEA.read_text(encoding="utf-8")
    forbidden = (
        "STEP92_AGNEWS_HOLDOUT_INPUTS",
        "STEP92_AGNEWS_SEALED_OUTCOMES",
        "external_data/step92_sealed",
        "external_data\\step92_sealed",
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
    if manifest["protocol_sha256"] != common.sha256_path(PROTOCOL):
        raise AssertionError("protocol was modified after Stage 0")
    if config["stage0_manifest_sha256"] != common.sha256_path(MANIFEST):
        raise AssertionError("config/manifest binding mismatch")
    if config["stagea_ledger_sha256"] != common.sha256_path(LEDGER):
        raise AssertionError("config/ledger binding mismatch")
    if config["calibration_outputs_sha256"] != common.sha256_path(CALIBRATION_OUTPUTS):
        raise AssertionError("config/calibration-output binding mismatch")

    model_files: list[Path] = []
    for name, expected in config["clean_root_sha256"].items():
        path = ROOT / "step92_models" / name
        if common.sha256_path(path) != expected:
            raise AssertionError(f"clean root hash mismatch: {name}")
        model_files.append(path)
    for name, expected in config["learned_adapter_sha256"].items():
        path = ROOT / "step92_models" / name
        if common.sha256_path(path) != expected:
            raise AssertionError(f"learned adapter hash mismatch: {name}")
        audit = common.adapter_tensor_audit(path)
        if audit["parameter_count"] < 49_000 or audit["variance"] <= 0.0:
            raise AssertionError({"invalid_learned_adapter": name, "audit": audit})
        model_files.append(path)
    if len(set(config["learned_adapter_sha256"].values())) != 4:
        raise AssertionError("selected learned adapter hashes are not distinct")

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

    locked_paths = [
        PROTOCOL,
        MANIFEST,
        CONFIG,
        LEDGER,
        CALIBRATION_OUTPUTS,
        HOLDOUT_INPUTS,
        SEALED,
        STAGEA,
        STAGEB,
        COMMON,
        *model_files,
    ]
    lock = {
        "lock_id": "STEP92_LEARNED_CONTEXTUAL_ADAPTER_CONFIRMATORY_LOCK_V1",
        "date": DATE,
        "locked_sha256": {relative(path): common.sha256_path(path) for path in locked_paths},
        "base_model": {
            "id": common.BASE_MODEL,
            "revision": common.BASE_REVISION,
            "model_safetensors_sha256": common.BASE_MODEL_SHA256,
        },
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
                "stagea_forbidden_token_hits": hits,
                "confirmatory_outputs_absent_at_lock": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
