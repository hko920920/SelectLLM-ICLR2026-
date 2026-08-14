from __future__ import annotations

import json
from pathlib import Path

import step105_common as common


ROOT = Path(__file__).resolve().parent
STAGE0 = ROOT / f"STEP105_STAGE0_DATA_AND_SEAL_MANIFEST_{common.DATE}.json"
TRAIN_LEDGER = ROOT / f"STEP105_TRAIN_AND_PREOUTCOME_PREDICTION_LEDGER_{common.DATE}.json"
OUTPUT = ROOT / f"STEP105_PREOUTCOME_LOCK_{common.DATE}.json"
BOUND_CODE = (
    "step93_common.py",
    "step105_common.py",
    "step105_executable_adapter_endpoint.py",
    "step105_create_predata_lock.py",
    "step105_stage0_prepare_and_seal.py",
    "step105_train_lock_predictions.py",
    "step105_create_preoutcome_lock.py",
    "step105_confirm_sealed_one_shot.py",
    "step105_validate_independent.py",
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    train = json.loads(TRAIN_LEDGER.read_text(encoding="utf-8"))
    predata = json.loads(common.PREDATA_LOCK.read_text(encoding="utf-8"))
    if train["test_outcome_opened"] or train["active_selector_run_on_target"]:
        raise AssertionError("pre-outcome provenance violation")
    if train["target_selector_grid_cells"] != 0 or train["target_selector_search_rows"] != 0:
        raise AssertionError("target-task selector tuning found")
    if stage0["sealed_outcome_opened_for_analysis"]:
        raise AssertionError("sealed outcome was opened")
    for name in BOUND_CODE:
        expected = predata["code_sha256"][name]
        observed = common.sha256_path(ROOT / name)
        if observed != expected:
            raise AssertionError({"predata_code_drift": name, "expected": expected, "observed": observed})
    prediction_path = ROOT / train["test_predictions_file"]
    sealed_path = ROOT / stage0["sealed_file"]
    input_path = ROOT / stage0["test_input_file"]
    adapters = {row["weight_file"]: row["weight_audit"]["sha256"] for row in train["adapters"]}
    audits = {row["audit_file"]: row["audit_sha256"] for row in train["adapters"]}
    payload = {
        "schema": "step105.preoutcome_lock.v1",
        "date": common.DATE,
        "status": "LOCKED_BEFORE_SINGLE_YELP_TEST_OUTCOME_OPEN",
        "predata_lock_file": common.PREDATA_LOCK.name,
        "predata_lock_sha256": common.sha256_path(common.PREDATA_LOCK),
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "metadata_sha256": common.sha256_path(common.METADATA),
        "stage0_manifest_file": STAGE0.name,
        "stage0_manifest_sha256": common.sha256_path(STAGE0),
        "train_ledger_file": TRAIN_LEDGER.name,
        "train_ledger_sha256": common.sha256_path(TRAIN_LEDGER),
        "development_file": stage0["development_file"],
        "development_sha256": stage0["development_sha256"],
        "test_input_file": stage0["test_input_file"],
        "test_input_sha256": common.sha256_path(input_path),
        "sealed_file": stage0["sealed_file"],
        "sealed_sha256": common.sha256_path(sealed_path),
        "prediction_file": train["test_predictions_file"],
        "prediction_sha256": common.sha256_path(prediction_path),
        "adapter_sha256": adapters,
        "adapter_audit_sha256": audits,
        "thresholds": train["thresholds"],
        "root_specs": [spec.__dict__ for spec in common.ROOT_SPECS],
        "selector": predata["selector"],
        "test_seed_first": common.TEST_SEEDS[0],
        "test_seed_last": common.TEST_SEEDS[-1],
        "test_seed_count": len(common.TEST_SEEDS),
        "bootstrap_seeds": list(common.BOOTSTRAP_SEEDS),
        "signflip_seeds": list(common.SIGNFLIP_SEEDS),
        "success_label": predata["success_label"],
        "failure_label": predata["failure_label"],
        "code_sha256": {name: common.sha256_path(ROOT / name) for name in BOUND_CODE},
        "target_selector_search_rows": 0,
        "target_selector_verify_rows": 0,
        "target_selector_grid_cells": 0,
        "sealed_outcome_opened": False,
    }
    common.json_dump(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "predata_lock_sha256": payload["predata_lock_sha256"],
        "prediction_sha256": payload["prediction_sha256"],
        "sealed_sha256": payload["sealed_sha256"],
        "adapter_count": len(adapters),
        "target_selector_grid_cells": 0,
        "output": OUTPUT.name,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
