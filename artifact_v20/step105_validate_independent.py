from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

import step105_common as common
import step105_executable_adapter_endpoint as endpoint


ROOT = Path(__file__).resolve().parent
LOCK = ROOT / f"STEP105_PREOUTCOME_LOCK_{common.DATE}.json"
LEDGER = ROOT / f"STEP105_PRIMARY_CONFIRMATORY_LEDGER_{common.DATE}.json"
ARRAYS = ROOT / f"STEP105_PRIMARY_CONFIRMATORY_ARRAYS_{common.DATE}.npz"
OUTPUT = ROOT / f"STEP105_PRIMARY_INDEPENDENT_VALIDATION_{common.DATE}.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    arrays = np.load(ARRAYS, allow_pickle=False)
    test_payload = json.loads((ROOT / lock["test_input_file"]).read_text(encoding="utf-8"))
    texts = [str(row["text"]) for row in test_payload["rows"]]
    root_columns: list[np.ndarray] = []
    root_audits: list[dict[str, object]] = []
    for spec in common.ROOT_SPECS:
        predictions, audit = common.infer_root_predictions(spec, texts)
        root_columns.append(predictions)
        root_audits.append(audit)
    roots = np.stack(root_columns, axis=1).astype(np.int64)
    adapter_paths = list(lock["adapter_sha256"])
    aliases = endpoint.execute_aliases_from_raw_text(
        texts, roots[:, 0], adapter_paths, [float(value) for value in lock["thresholds"]],
    )
    sealed = np.load(ROOT / lock["sealed_file"], allow_pickle=False)
    labels = sealed["labels"].astype(np.int64)
    summary, reconstructed = common.compute_effects(roots, aliases, labels)
    checks = {
        "preoutcome_lock_hash": common.sha256_path(LOCK) == ledger["preoutcome_lock_sha256"],
        "primary_arrays_hash": common.sha256_path(ARRAYS) == ledger["arrays_sha256"],
        "root_predictions_bitwise": np.array_equal(roots, arrays["root_predictions"]),
        "aliases_bitwise": np.array_equal(aliases, arrays["aliases"]),
        "labels_bitwise": np.array_equal(labels, arrays["labels"]),
        "clean_queries_bitwise": np.array_equal(reconstructed["clean_queries"], arrays["clean_queries"]),
        "refined_queries_bitwise": np.array_equal(reconstructed["refined_queries"], arrays["refined_queries"]),
        "clean_roots_bitwise": np.array_equal(reconstructed["clean_roots"], arrays["clean_roots"]),
        "refined_roots_bitwise": np.array_equal(reconstructed["refined_roots"], arrays["refined_roots"]),
        "fixed_roots_bitwise": np.array_equal(reconstructed["fixed_roots"], arrays["fixed_roots"]),
        "terminal_summary_exact": summary["terminal"] == ledger["effect"]["terminal"],
        "active_minus_fixed_summary_exact": summary["active_minus_fixed_terminal"] == ledger["effect"]["active_minus_fixed_terminal"],
        "cumulative_summary_exact": summary["cumulative"] == ledger["effect"]["cumulative"],
        "path_summary_exact": all(summary[key] == ledger["effect"][key] for key in (
            "path_change_rate", "query_set_change_rate", "final_root_change_rate",
            "parent_to_challenger_rate", "challenger_to_parent_rate",
            "max_abs_fixed_terminal_delta", "max_abs_fixed_cumulative_delta",
            "fixed_root_history_exact",
        )),
        "endpoint_signature": list(inspect.signature(endpoint.execute_aliases_from_raw_text).parameters)
        == ["texts", "parent_predictions", "adapter_paths", "thresholds"],
        "decision_consistent": ledger["decision"] == (
            lock["success_label"] if all(ledger["gates"].values()) else lock["failure_label"]
        ),
    }
    failed = [key for key, value in checks.items() if not value]
    payload = {
        "schema": "step105.primary_independent_raw_input_validation.v1",
        "date": common.DATE,
        "verdict": "PASS_STEP105_PRIMARY_INDEPENDENT_RECONSTRUCTION" if not failed else "FAIL_STEP105_PRIMARY_INDEPENDENT_RECONSTRUCTION",
        "failed_checks": failed,
        "checks": checks,
        "primary_decision": ledger["decision"],
        "primary_failed_gates": ledger["failed_gates"],
        "root_reinference_audits": root_audits,
        "test_rows": len(texts),
        "adapter_paths": adapter_paths,
        "test_reference_passed_to_endpoint": False,
        "item_lookup_passed_to_endpoint": False,
    }
    common.json_dump(OUTPUT, payload)
    print(json.dumps({
        "verdict": payload["verdict"],
        "failed_checks": failed,
        "primary_decision": ledger["decision"],
        "primary_failed_gates": ledger["failed_gates"],
        "output": OUTPUT.name,
    }, indent=2), flush=True)
    if failed:
        raise AssertionError(failed)


if __name__ == "__main__":
    main()
