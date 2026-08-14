from __future__ import annotations

import json
from pathlib import Path

import step95_common as common


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
OUTPUT = ROOT / f"STEP95_DEVELOPMENT_STOP_RECEIPT_{DATE}.json"


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite closure receipt: {OUTPUT}")
    names = {
        "protocol": f"STEP95_MNLI_FULL_ADAPTER_PREREGISTRATION_{DATE}.md",
        "amendment_a": f"STEP95_PREREGISTRATION_AMENDMENT_A_{DATE}.md",
        "incident_b": f"STEP95_PREREGISTRATION_AMENDMENT_B_{DATE}.md",
        "manifest": f"STEP95_STAGE0_MANIFEST_{DATE}.json",
        "development": f"STEP95_MNLI_DEVELOPMENT_ROWS_{DATE}.json",
        "arrays": f"STEP95_STAGEA_DEVELOPMENT_ARRAYS_{DATE}.npz",
        "ledger": f"STEP95_STAGEA_COMPLETE_LEDGER_{DATE}.json",
        "config": f"STEP95_STAGEA_FROZEN_CONFIG_{DATE}.json",
        "validation": f"STEP95_STAGEA_INDEPENDENT_VALIDATION_{DATE}.json",
        "closure_report": f"STEP95_REVIEW_CLOSURE_REPORT_{DATE}.md",
    }
    paths = {key: ROOT / name for key, name in names.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    if ledger["decision"] != "NO_GO_DEVELOPMENT_GATE_STOP":
        raise AssertionError(ledger["decision"])
    if config["decision"] != ledger["decision"] or validation["decision"] != ledger["decision"]:
        raise AssertionError("closure decision mismatch")
    confirmatory_names = (
        f"STEP95_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json",
        f"STEP95_CONFIRMATORY_RAW_{DATE}.npz",
        f"STEP95_CONFIRMATORY_RESULTS_{DATE}.json",
        f"STEP95_INDEPENDENT_VALIDATION_{DATE}.json",
    )
    if any((ROOT / name).exists() for name in confirmatory_names):
        raise AssertionError("confirmatory output exists after development stop")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    sealed = ROOT / manifest["sealed_outcome"]["file"]
    if common.sha256_path(sealed) != manifest["sealed_outcome"]["sha256"]:
        raise AssertionError("sealed outcome hash drift")
    selected = ledger["selected_verification"]
    receipt = {
        "receipt_id": "STEP95_DEVELOPMENT_STOP_CLOSURE_V1",
        "date": DATE,
        "decision": ledger["decision"],
        "file_sha256": {key: common.sha256_path(path) for key, path in paths.items()},
        "adapter_sha256": config["adapter_sha256"],
        "sealed_outcome": {
            "file": manifest["sealed_outcome"]["file"],
            "sha256": manifest["sealed_outcome"]["sha256"],
            "opened_for_step95_confirmation": False,
        },
        "confirmatory_outputs_absent": list(confirmatory_names),
        "failed_gates": ledger["failed_gates"],
        "verification": {
            "runs": selected["runs"],
            "path_change_rate": selected["path_change_rate"],
            "query_set_change_rate": selected["query_set_change_rate"],
            "final_root_change_rate": selected["final_root_change_rate"],
            "terminal": selected["inference"]["terminal"],
            "active_minus_fixed_terminal": selected["inference"]["active_minus_fixed_terminal"],
            "cumulative": selected["inference"]["cumulative"],
            "fixed_query_exact": selected["fixed_root_history_exact"]
            and selected["max_abs_fixed_terminal_delta"] == 0
            and selected["max_abs_fixed_cumulative_delta"] == 0,
        },
        "independent_validation_checks": validation["checks"],
    }
    common.json_dump(OUTPUT, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
