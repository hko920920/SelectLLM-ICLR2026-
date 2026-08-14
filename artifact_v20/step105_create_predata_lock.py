from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import step105_common as common
import step105_executable_adapter_endpoint as endpoint


ROOT = Path(__file__).resolve().parent
OUTPUT = common.PREDATA_LOCK
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


def assert_endpoint_signature() -> dict[str, object]:
    signature = inspect.signature(endpoint.execute_aliases_from_raw_text)
    allowed = ["texts", "parent_predictions", "adapter_paths", "thresholds"]
    observed = list(signature.parameters)
    if observed != allowed:
        raise AssertionError({"endpoint_signature": observed})
    source = inspect.getsource(endpoint.execute_aliases_from_raw_text)
    forbidden = ["reference", "label", "item_id", "lookup", "peer", "pool", "posterior", "trajectory", "selector_state", "dataset_id"]
    hits = [token for token in forbidden if token in source.lower()]
    if hits:
        raise AssertionError({"forbidden_endpoint_source_tokens": hits})
    return {"parameters": observed, "forbidden_source_hits": hits}


def assert_no_target_search_code() -> dict[str, object]:
    source = (ROOT / "step105_common.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = {
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    forbidden_constants = sorted(assigned & {"LOSS_CAPS", "TRIGGER_CAPS", "BUDGETS", "TAUS", "SEARCH_SEEDS", "VERIFY_SEEDS"})
    if forbidden_constants:
        raise AssertionError({"target_search_constants": forbidden_constants})
    return {
        "forbidden_grid_constants": forbidden_constants,
        "single_condition": {
            "loss_cap": common.LOSS_CAP,
            "trigger_cap": common.TRIGGER_CAP,
            "budget": common.BUDGET,
            "temperature": common.TAU,
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    metadata = json.loads(common.METADATA.read_text(encoding="utf-8"))
    if not metadata["outcome_blind"] or metadata["dataset_rows_loaded"] or metadata["model_weights_run"]:
        raise AssertionError("metadata snapshot is not pre-data")
    missing = [name for name in BOUND_CODE if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    payload = {
        "schema": "step105.predata_lock.v1",
        "date": common.DATE,
        "status": "LOCKED_BEFORE_ANY_YELP_ROW_OR_MODEL_WEIGHT_EXECUTION",
        "protocol_file": common.PROTOCOL.name,
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "metadata_file": common.METADATA.name,
        "metadata_sha256": common.sha256_path(common.METADATA),
        "dataset_id": common.DATASET_ID,
        "dataset_config": common.DATASET_CONFIG,
        "dataset_revision": common.DATASET_REVISION,
        "root_specs": [spec.__dict__ for spec in common.ROOT_SPECS],
        "development_quotas_per_label": common.DEV_QUOTAS_PER_LABEL,
        "adapter_seeds": list(common.ADAPTER_SEEDS),
        "test_seeds": [common.TEST_SEEDS[0], common.TEST_SEEDS[-1]],
        "test_seed_count": len(common.TEST_SEEDS),
        "bootstrap_seeds": list(common.BOOTSTRAP_SEEDS),
        "signflip_seeds": list(common.SIGNFLIP_SEEDS),
        "selector": {
            "pool_size": common.POOL_SIZE,
            "budget": common.BUDGET,
            "temperature": common.TAU,
            "loss_cap": common.LOSS_CAP,
            "trigger_cap": common.TRIGGER_CAP,
            "same_literal_exact_match": True,
        },
        "success_label": "GO_STEP105_YELP_PROSPECTIVE_CROSS_TASK_ONE_SHOT",
        "failure_label": "NO_GO_RETAIN_STEP105_PROSPECTIVE_NEGATIVE",
        "endpoint_audit": assert_endpoint_signature(),
        "search_audit": assert_no_target_search_code(),
        "code_sha256": {name: common.sha256_path(ROOT / name) for name in BOUND_CODE},
        "dataset_rows_loaded_before_lock": False,
        "model_weights_run_before_lock": False,
        "prior_outcomes_modified": False,
    }
    common.json_dump(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "protocol_sha256": payload["protocol_sha256"],
        "metadata_sha256": payload["metadata_sha256"],
        "code_files": len(payload["code_sha256"]),
        "target_search_grid_present": False,
        "output": OUTPUT.name,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
