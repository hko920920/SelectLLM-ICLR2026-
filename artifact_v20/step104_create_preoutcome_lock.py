from __future__ import annotations

import json
from pathlib import Path

import step102_common as common


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
STAGE0 = ROOT / f"STEP100_STAGE0_DATA_AND_SEAL_MANIFEST_{DATE}.json"
STEP100_LEDGER = ROOT / f"STEP100_STAGEA_COMPLETE_LEDGER_{DATE}.json"
STEP101_LEDGER = ROOT / f"STEP101_IMDB_DEVELOPMENT_PARENT_ROSTER_SELECTION_{DATE}.json"
STEP102_LEDGER = ROOT / f"STEP102_STAGEA_COMPLETE_LEDGER_{DATE}.json"
STEP103_LEDGER = ROOT / f"STEP103_ROBUST_DUAL_DEVELOPMENT_LEDGER_{DATE}.json"
STEP103_VALIDATION = ROOT / f"STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION_{DATE}.json"
CONFIG = ROOT / f"STEP103_ROBUST_HELDOUT_FROZEN_CONFIG_{DATE}.json"
PREOUTCOME = ROOT / f"STEP104_PREOUTCOME_PREDICTIONS_{DATE}.npz"
PREOUTCOME_LEDGER = ROOT / f"STEP104_PREOUTCOME_PREDICTION_LEDGER_{DATE}.json"
OUTPUT = ROOT / f"STEP104_PREOUTCOME_LOCK_{DATE}.json"


AUTHORITY_FILES = (
    f"STEP100_IMDB_FRESH_LEARNED_SAME_S_PRIMARY_PREREGISTRATION_{DATE}.md",
    f"STEP100_PREREGISTRATION_AMENDMENT_A_GEOMETRY_AND_QUALITY_{DATE}.md",
    f"STEP100_IMDB_DEVELOPMENT_ROWS_{DATE}.json",
    f"STEP100_STAGE0_DATA_AND_SEAL_MANIFEST_{DATE}.json",
    f"STEP100_STAGEA_COMPLETE_LEDGER_{DATE}.json",
    f"STEP101_IMDB_DEVELOPMENT_PARENT_ROSTER_SELECTION_PROTOCOL_{DATE}.md",
    f"STEP101_IMDB_DEVELOPMENT_PARENT_ROSTER_SELECTION_{DATE}.json",
    f"STEP102_IMDB_WRMURRAY_LEARNED_DEVELOPMENT_PROTOCOL_{DATE}.md",
    f"STEP102_STAGEA_DEVELOPMENT_ARRAYS_{DATE}.npz",
    f"STEP102_STAGEA_COMPLETE_GRID_{DATE}.json",
    f"STEP102_STAGEA_COMPLETE_LEDGER_{DATE}.json",
    f"STEP102_STAGEA_FROZEN_CONFIG_{DATE}.json",
    f"STEP103_IMDB_ROBUST_DUAL_DEVELOPMENT_SELECTION_PROTOCOL_{DATE}.md",
    f"STEP103_COMPLETE_DUAL_DEVELOPMENT_GRID_{DATE}.json",
    f"STEP103_SELECTED_DEVELOPMENT_ARRAYS_{DATE}.npz",
    f"STEP103_ROBUST_DUAL_DEVELOPMENT_LEDGER_{DATE}.json",
    f"STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION_{DATE}.json",
    f"STEP103_ROBUST_HELDOUT_FROZEN_CONFIG_{DATE}.json",
    f"STEP104_IMDB_HELDOUT_LEARNED_SAME_S_PRIMARY_PROTOCOL_{DATE}.md",
)

CANDIDATE_FILES = tuple(
    f"step101_candidate_predictions/{name}{suffix}"
    for name in ("aychang_roberta", "wrmurray_roberta", "dfurman_deberta")
    for suffix in (".npz", ".audit.json")
)

CODE_FILES = (
    "step93_common.py",
    "step100_common.py",
    "step100_stage0_prepare_and_seal.py",
    "step100_stagea_develop.py",
    "step101_infer_candidate.py",
    "step101_select_parent_roster.py",
    "step102_common.py",
    "step102_executable_adapter_endpoint.py",
    "step102_stagea_develop.py",
    "step103_robust_dual_development.py",
    "step103_validate_robust_development.py",
    "step103_freeze_heldout_config.py",
    "step104_prepare_preoutcome_predictions.py",
    "step104_create_preoutcome_lock.py",
    "step104_confirm_sealed_primary.py",
    "step104_validate_primary_independent.py",
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    step100 = json.loads(STEP100_LEDGER.read_text(encoding="utf-8"))
    step101 = json.loads(STEP101_LEDGER.read_text(encoding="utf-8"))
    step102 = json.loads(STEP102_LEDGER.read_text(encoding="utf-8"))
    step103 = json.loads(STEP103_LEDGER.read_text(encoding="utf-8"))
    validation = json.loads(STEP103_VALIDATION.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    preoutcome = json.loads(PREOUTCOME_LEDGER.read_text(encoding="utf-8"))
    expected_decisions = (
        step100["decision"] == "NO_GO_STEP100_GEOMETRY_STOP",
        step101["decision"] == "NO_GO_STEP101_PARENT_GEOMETRY_STOP",
        step102["decision"] == "NO_GO_STEP102_LEARNED_DEVELOPMENT_STOP",
        step103["decision"] == "GO_STEP103_TO_HELDOUT_LOCK",
        validation["decision"] == "PASS_STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION",
        config["decision"] == "GO_STEP103_TO_HELDOUT_LOCK",
    )
    if not all(expected_decisions):
        raise AssertionError("development provenance or validation drift")
    if config["selected_cell"] != "l0.0075_t0.025_b5_tau0.050":
        raise AssertionError("selected cell drift")
    if preoutcome["sealed_outcome_loaded"] is not False:
        raise AssertionError("pre-outcome construction claims outcome access")
    if preoutcome["prediction_sha256"] != common.sha256_path(PREOUTCOME):
        raise AssertionError("prediction binding drift")
    input_path = ROOT / stage0["input_file"]
    sealed_path = ROOT / stage0["sealed_file"]
    if common.sha256_path(input_path) != stage0["input_sha256"]:
        raise AssertionError("input drift")
    if common.sha256_path(sealed_path) != stage0["sealed_sha256"]:
        raise AssertionError("sealed byte drift")
    missing = [
        name for name in (*AUTHORITY_FILES, *CANDIDATE_FILES, *CODE_FILES)
        if not (ROOT / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(missing)
    model_sha256 = {
        relative: common.sha256_path(ROOT / relative)
        for relative in config["adapter_paths"]
    }
    if model_sha256 != config["adapter_sha256"]:
        raise AssertionError("adapter hash drift")
    lock = {
        "lock_id": "STEP104_PREOUTCOME_LOCK_V1",
        "date": DATE,
        "protocol_sha256": common.sha256_path(
            ROOT / f"STEP104_IMDB_HELDOUT_LEARNED_SAME_S_PRIMARY_PROTOCOL_{DATE}.md"
        ),
        "authority_sha256": {
            name: common.sha256_path(ROOT / name)
            for name in (*AUTHORITY_FILES, *CANDIDATE_FILES)
        },
        "code_sha256": {
            name: common.sha256_path(ROOT / name) for name in CODE_FILES
        },
        "model_sha256": model_sha256,
        "model_audit_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): common.sha256_path(path)
            for path in sorted((ROOT / "step102_models").glob("*.audit.json"))
        },
        "preoutcome_sha256": {
            PREOUTCOME.name: common.sha256_path(PREOUTCOME),
            PREOUTCOME_LEDGER.name: common.sha256_path(PREOUTCOME_LEDGER),
        },
        "input_file": stage0["input_file"],
        "input_sha256": stage0["input_sha256"],
        "sealed_file": stage0["sealed_file"],
        "sealed_sha256": stage0["sealed_sha256"],
        "adapter_paths": config["adapter_paths"],
        "thresholds": config["thresholds"],
        "roster": [spec.key for spec in common.ROOT_SPECS],
        "root_revisions": {spec.key: spec.revision for spec in common.ROOT_SPECS},
        "selected_cell": config["selected_cell"],
        "pool_size": common.POOL_SIZE,
        "budget": 5,
        "tau": 0.05,
        "test_seed_start": common.TEST_SEEDS[0],
        "test_seed_stop_exclusive": common.TEST_SEEDS[-1] + 1,
        "test_seed_count": len(common.TEST_SEEDS),
        "inference": {
            "bootstrap_repetitions": 10000,
            "signflip_repetitions": 100000,
            "terminal_seeds": [100800, 100801],
            "active_minus_fixed_terminal_seeds": [100801, 100802],
            "cumulative_seeds": [100802, 100803],
        },
        "step100_decision": step100["decision"],
        "step101_decision": step101["decision"],
        "step102_decision": step102["decision"],
        "step103_decision": step103["decision"],
        "success_label": "GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY",
        "failure_label": "NO_GO_RETAIN_STEP104_NEGATIVE",
        "sealed_outcome_opened": False,
    }
    common.json_dump(OUTPUT, lock)
    print(json.dumps({
        "lock_file": OUTPUT.name,
        "lock_sha256": common.sha256_path(OUTPUT),
        "prediction_sha256": lock["preoutcome_sha256"][PREOUTCOME.name],
        "sealed_sha256": lock["sealed_sha256"],
        "selected_cell": lock["selected_cell"],
        "test_seed_count": lock["test_seed_count"],
        "sealed_outcome_opened": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
