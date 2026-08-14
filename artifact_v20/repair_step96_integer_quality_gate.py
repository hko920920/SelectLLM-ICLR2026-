from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import step96_common as common


ROOT = Path(__file__).resolve().parent
AMENDMENT = ROOT / f"STEP96_PREREGISTRATION_AMENDMENT_A_INTEGER_GATE_{common.DATE}.md"
ARRAYS = ROOT / f"STEP96_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
LEDGER = ROOT / f"STEP96_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
CONFIG = ROOT / f"STEP96_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
LEDGER_BACKUP = ROOT / f"STEP96_STAGEA_COMPLETE_LEDGER_PRE_INTEGER_FIX_{common.DATE}.json"
CONFIG_BACKUP = ROOT / f"STEP96_STAGEA_FROZEN_CONFIG_PRE_INTEGER_FIX_{common.DATE}.json"
RECEIPT = ROOT / f"STEP96_INTEGER_QUALITY_GATE_REPAIR_RECEIPT_{common.DATE}.json"


def main() -> None:
    for path in (AMENDMENT, ARRAYS, LEDGER, CONFIG):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (LEDGER_BACKUP, CONFIG_BACKUP, RECEIPT):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite repair output: {path}")
    original_ledger_bytes = LEDGER.read_bytes()
    original_config_bytes = CONFIG.read_bytes()
    original_ledger_hash = common.sha256_path(LEDGER)
    original_config_hash = common.sha256_path(CONFIG)
    ledger = json.loads(original_ledger_bytes)
    config = json.loads(original_config_bytes)
    if ledger["decision"] != "NO_GO_DEVELOPMENT_GATE_STOP":
        raise AssertionError(ledger["decision"])
    if ledger["failed_gates"] != ["alias_losses_at_most_one_point_threshold_search_validation"]:
        raise AssertionError(ledger["failed_gates"])
    if any(not value for name, value in ledger["gates"].items() if name != ledger["failed_gates"][0]):
        raise AssertionError("a nonnumeric gate was false")
    selected = ledger["selected_verification"]
    arrays = np.load(ARRAYS, allow_pickle=False)
    labels = arrays["labels"].astype(np.int64)
    parent = arrays["root_predictions"][:, 0].astype(np.int64)
    scores = arrays["error_scores"].astype(np.float64)
    thresholds = np.asarray(selected["thresholds"], dtype=np.float64)
    triggers = scores > thresholds[None, :]
    partition_keys = {
        "threshold": "threshold_indices",
        "selector_search": "search_indices",
        "selector_verify": "verify_indices",
    }
    counts = {}
    all_pass = True
    for name, key in partition_keys.items():
        indices = arrays[key].astype(np.int64)
        loss_counts = np.sum(
            triggers[indices] & (parent[indices, None] == labels[indices, None]), axis=0
        ).astype(np.int64)
        allowed = int(math.floor(0.01 * len(indices) + 1e-12))
        passed = bool(np.all(loss_counts <= allowed))
        all_pass = all_pass and passed
        counts[name] = {
            "rows": len(indices),
            "loss_counts": loss_counts.tolist(),
            "allowed_count": allowed,
            "fractions": (loss_counts / len(indices)).tolist(),
            "pass": passed,
        }
    if not all_pass:
        raise AssertionError(counts)
    LEDGER_BACKUP.write_bytes(original_ledger_bytes)
    CONFIG_BACKUP.write_bytes(original_config_bytes)
    gate_name = "alias_losses_at_most_one_point_threshold_search_validation"
    ledger["authority_sha256"][AMENDMENT.name] = common.sha256_path(AMENDMENT)
    ledger["gates"][gate_name] = True
    ledger["failed_gates"] = []
    ledger["decision"] = "GO_TO_STEP96_ONE_TIME_CONFIRMATORY_LOCK"
    ledger["integer_quality_gate_repair"] = {
        "amendment": AMENDMENT.name,
        "amendment_sha256": common.sha256_path(AMENDMENT),
        "original_ledger_sha256": original_ledger_hash,
        "original_config_sha256": original_config_hash,
        "counts": counts,
        "selected_condition_unchanged": True,
        "effect_values_recomputed": False,
        "test_opened_before_repair": False,
    }
    common.json_dump(LEDGER, ledger)
    config["authority_sha256"][AMENDMENT.name] = common.sha256_path(AMENDMENT)
    config["decision"] = ledger["decision"]
    config["development_gates"] = ledger["gates"]
    config["stagea_ledger_sha256"] = common.sha256_path(LEDGER)
    config["integer_quality_gate_repair"] = ledger["integer_quality_gate_repair"]
    config["code_sha256"]["step96_stagea_develop.py"] = common.sha256_path(
        ROOT / "step96_stagea_develop.py"
    )
    config["code_sha256"]["repair_step96_integer_quality_gate.py"] = common.sha256_path(
        ROOT / "repair_step96_integer_quality_gate.py"
    )
    common.json_dump(CONFIG, config)
    receipt = {
        "receipt_id": "STEP96_INTEGER_QUALITY_GATE_REPAIR_V1",
        "date": common.DATE,
        "decision": ledger["decision"],
        "original_ledger_sha256": original_ledger_hash,
        "original_config_sha256": original_config_hash,
        "backup_ledger_sha256": common.sha256_path(LEDGER_BACKUP),
        "backup_config_sha256": common.sha256_path(CONFIG_BACKUP),
        "repaired_ledger_sha256": common.sha256_path(LEDGER),
        "repaired_config_sha256": common.sha256_path(CONFIG),
        "arrays_sha256": common.sha256_path(ARRAYS),
        "amendment_sha256": common.sha256_path(AMENDMENT),
        "patched_stagea_sha256": common.sha256_path(ROOT / "step96_stagea_develop.py"),
        "repair_code_sha256": common.sha256_path(ROOT / "repair_step96_integer_quality_gate.py"),
        "counts": counts,
        "selected_condition_unchanged": True,
        "test_opened_before_repair": False,
    }
    common.json_dump(RECEIPT, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
