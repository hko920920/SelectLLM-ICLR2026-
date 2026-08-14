from __future__ import annotations

import json
from pathlib import Path

import step98_common as common


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
PROTOCOL = ROOT / f"STEP99_ANLI_DEVELOPMENT_INFORMED_SEALED_PRIMARY_PROTOCOL_{DATE}.md"
CONFIG = ROOT / f"STEP98_STAGEA_FROZEN_CONFIG_{DATE}.json"
SEAL = ROOT / f"STEP98_SELECTED_DEV_SEAL_MANIFEST_{DATE}.json"
STAGEA = ROOT / f"STEP98_STAGEA_COMPLETE_LEDGER_{DATE}.json"
VALIDATION = ROOT / f"STEP98_STAGEA_INDEPENDENT_VALIDATION_{DATE}.json"
PREOUTCOME = ROOT / f"STEP99_PREOUTCOME_PREDICTIONS_{DATE}.npz"
PREOUTCOME_LEDGER = ROOT / f"STEP99_PREOUTCOME_PREDICTION_LEDGER_{DATE}.json"
OUTPUT = ROOT / f"STEP99_PREOUTCOME_LOCK_{DATE}.json"


AUTHORITY_FILES = (
    f"STEP98_ANLI_FRESH_PRIMARY_PREREGISTRATION_{DATE}.md",
    f"STEP98_PREREGISTRATION_AMENDMENT_A_STAGE0_ELIGIBILITY_{DATE}.md",
    f"STEP98_STAGE0_TRAIN_ONLY_MANIFEST_{DATE}.json",
    f"STEP98_ANLI_TRAIN_ONLY_ROWS_{DATE}.json",
    f"STEP98_TRAIN_ONLY_ROUND_SELECTION_{DATE}.json",
    f"STEP98_SELECTED_DEV_SEAL_MANIFEST_{DATE}.json",
    f"STEP98_STAGEA_DEVELOPMENT_ARRAYS_{DATE}.npz",
    f"STEP98_STAGEA_COMPLETE_LEDGER_{DATE}.json",
    f"STEP98_STAGEA_FROZEN_CONFIG_{DATE}.json",
    f"STEP98_STAGEA_INDEPENDENT_VALIDATION_{DATE}.json",
    f"STEP99_ANLI_DEVELOPMENT_INFORMED_SEALED_PRIMARY_PROTOCOL_{DATE}.md",
)

CODE_FILES = (
    "step93_common.py",
    "step95_common.py",
    "step96_common.py",
    "step98_common.py",
    "step98_executable_adapter_endpoint.py",
    "step98_stagea_develop.py",
    "step98_validate_stagea_independent.py",
    "step99_prepare_preoutcome_predictions.py",
    "step99_create_preoutcome_lock.py",
    "step99_confirm_sealed_primary.py",
    "step99_validate_primary_independent.py",
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    stagea = json.loads(STAGEA.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    preoutcome = json.loads(PREOUTCOME_LEDGER.read_text(encoding="utf-8"))
    if stagea["decision"] != "NO_GO_STEP98_DEVELOPMENT_STOP":
        raise AssertionError("Step 98 non-relabeling drift")
    if stagea["failed_gates"] != ["clean_parent_terminal_rate_at_least_60pct"]:
        raise AssertionError("Step 98 failure provenance drift")
    if validation["decision"] != "PASS_STEP98_STAGEA_RECONSTRUCTION_RETAIN_ORIGINAL_NO_GO":
        raise AssertionError("Step 98 independent reconstruction did not pass")
    if preoutcome["sealed_outcome_loaded"] is not False:
        raise AssertionError("pre-outcome construction claims outcome access")
    if preoutcome["prediction_sha256"] != common.sha256_path(PREOUTCOME):
        raise AssertionError("prediction binding drift")
    input_path = ROOT / seal["input_file"]
    sealed_path = ROOT / seal["sealed_file"]
    if common.sha256_path(input_path) != seal["input_sha256"]:
        raise AssertionError("input drift")
    if common.sha256_path(sealed_path) != seal["sealed_sha256"]:
        raise AssertionError("sealed byte drift")
    missing = [name for name in (*AUTHORITY_FILES, *CODE_FILES) if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    model_sha256 = {
        relative: common.sha256_path(ROOT / relative)
        for relative in config["adapter_paths"]
    }
    if model_sha256 != config["adapter_sha256"]:
        raise AssertionError("adapter hash drift")
    lock = {
        "lock_id": "STEP99_PREOUTCOME_LOCK_V1",
        "date": DATE,
        "protocol_sha256": common.sha256_path(PROTOCOL),
        "authority_sha256": {
            name: common.sha256_path(ROOT / name) for name in AUTHORITY_FILES
        },
        "code_sha256": {
            name: common.sha256_path(ROOT / name) for name in CODE_FILES
        },
        "model_sha256": model_sha256,
        "model_audit_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): common.sha256_path(path)
            for path in sorted((ROOT / "step98_models").glob("*.audit.json"))
        },
        "preoutcome_sha256": {
            PREOUTCOME.name: common.sha256_path(PREOUTCOME),
            PREOUTCOME_LEDGER.name: common.sha256_path(PREOUTCOME_LEDGER),
        },
        "input_file": seal["input_file"],
        "input_sha256": seal["input_sha256"],
        "sealed_file": seal["sealed_file"],
        "sealed_sha256": seal["sealed_sha256"],
        "selected_round": config["selected_round"],
        "adapter_paths": config["adapter_paths"],
        "thresholds": config["thresholds"],
        "roster": [spec.key for spec in common.ROOT_SPECS],
        "root_revisions": {spec.key: spec.revision for spec in common.ROOT_SPECS},
        "pool_size": common.POOL_SIZE,
        "budget": common.BUDGET,
        "tau": common.TAU,
        "test_seed_start": common.TEST_SEEDS[0],
        "test_seed_stop_exclusive": common.TEST_SEEDS[-1] + 1,
        "test_seed_count": len(common.TEST_SEEDS),
        "inference": {
            "bootstrap_repetitions": 10000,
            "signflip_repetitions": 100000,
            "terminal_seeds": [98900, 98901],
            "active_minus_fixed_terminal_seeds": [98901, 98902],
            "cumulative_seeds": [98902, 98903],
        },
        "step98_original_decision": stagea["decision"],
        "step98_only_failed_gate": stagea["failed_gates"][0],
        "success_label": "GO_STEP99_ANLI_DEVELOPMENT_INFORMED_SEALED_PRIMARY",
        "failure_label": "NO_GO_RETAIN_STEP99_NEGATIVE",
        "sealed_outcome_opened": False,
    }
    common.json_dump(OUTPUT, lock)
    print(json.dumps({
        "lock_file": OUTPUT.name,
        "lock_sha256": common.sha256_path(OUTPUT),
        "prediction_sha256": lock["preoutcome_sha256"][PREOUTCOME.name],
        "sealed_sha256": lock["sealed_sha256"],
        "test_seed_count": lock["test_seed_count"],
        "sealed_outcome_opened": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
