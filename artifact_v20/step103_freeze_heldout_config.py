from __future__ import annotations

import json
from pathlib import Path

import step102_common as common


ROOT = Path(__file__).resolve().parent
STEP102_LEDGER = ROOT / f"STEP102_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
STEP102_CONFIG = ROOT / f"STEP102_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
STEP103_LEDGER = ROOT / f"STEP103_ROBUST_DUAL_DEVELOPMENT_LEDGER_{common.DATE}.json"
STAGE0 = ROOT / f"STEP100_STAGE0_DATA_AND_SEAL_MANIFEST_{common.DATE}.json"
OUTPUT = ROOT / f"STEP103_ROBUST_HELDOUT_FROZEN_CONFIG_{common.DATE}.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    step102 = json.loads(STEP102_LEDGER.read_text(encoding="utf-8"))
    config102 = json.loads(STEP102_CONFIG.read_text(encoding="utf-8"))
    step103 = json.loads(STEP103_LEDGER.read_text(encoding="utf-8"))
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    if step102["decision"] != "NO_GO_STEP102_LEARNED_DEVELOPMENT_STOP":
        raise AssertionError("Step 102 non-relabeling drift")
    if step103["decision"] != "GO_STEP103_TO_HELDOUT_LOCK":
        raise AssertionError("Step 103 did not authorize held-out lock")
    selected = step103["selected"]
    if selected["cell_id"] != "l0.0075_t0.025_b5_tau0.050":
        raise AssertionError("robust selected-cell drift")
    thresholds = step102["threshold_pairs"]["l0.0075_t0.025"]["thresholds"]
    expected = [
        0.9515677094459534, 0.9398181438446045,
        0.9488589763641357, 0.9394304752349854,
    ]
    if thresholds != expected:
        raise AssertionError("robust threshold drift")
    output = {
        "config_id": "STEP103_ROBUST_HELDOUT_FROZEN_CONFIG_V1",
        "date": common.DATE,
        "decision": "GO_STEP103_TO_HELDOUT_LOCK",
        "selected_cell": selected["cell_id"],
        "loss_cap": selected["loss_cap"],
        "trigger_cap_label": selected["trigger_cap"],
        "budget": selected["budget"],
        "tau": selected["tau"],
        "pool_size": common.POOL_SIZE,
        "thresholds": thresholds,
        "adapter_paths": config102["adapter_paths"],
        "adapter_sha256": config102["adapter_sha256"],
        "root_specs": [spec.__dict__ for spec in common.ROOT_SPECS],
        "test_input": stage0["input_file"],
        "test_input_sha256": stage0["input_sha256"],
        "sealed_outcome": stage0["sealed_file"],
        "sealed_outcome_sha256": stage0["sealed_sha256"],
        "test_seed_start": common.TEST_SEEDS[0],
        "test_seed_stop_exclusive": common.TEST_SEEDS[-1] + 1,
        "test_seed_count": len(common.TEST_SEEDS),
        "authority_sha256": {
            STEP102_LEDGER.name: common.sha256_path(STEP102_LEDGER),
            STEP102_CONFIG.name: common.sha256_path(STEP102_CONFIG),
            STEP103_LEDGER.name: common.sha256_path(STEP103_LEDGER),
            STAGE0.name: common.sha256_path(STAGE0),
        },
        "sealed_outcome_opened": False,
        "code_sha256": common.sha256_path(Path(__file__)),
    }
    common.json_dump(OUTPUT, output)
    print(json.dumps({
        "config_file": OUTPUT.name,
        "config_sha256": common.sha256_path(OUTPUT),
        "selected_cell": output["selected_cell"],
        "thresholds": thresholds,
        "sealed_outcome_opened": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
