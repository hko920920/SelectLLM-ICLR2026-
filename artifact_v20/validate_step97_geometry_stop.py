from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import step97_common as common


ROOT = Path(__file__).resolve().parent
RECEIPT = ROOT / f"STEP97_STAGEA_ROOT_GEOMETRY_STOP_{common.DATE}.json"
DEVELOPMENT = ROOT / f"STEP97_SCITAIL_DEVELOPMENT_ROWS_{common.DATE}.json"
ROOT_CACHE = ROOT / "step97_stagea_work" / "root_predictions.npz"
OUTPUT = ROOT / f"STEP97_ROOT_GEOMETRY_STOP_INDEPENDENT_VALIDATION_{common.DATE}.json"


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    payload = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    rows = [*payload["train_rows"], *payload["validation_rows"]]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    predictions = np.load(ROOT_CACHE, allow_pickle=False)["root_predictions"].astype(np.int64)
    partitions = {name: [] for name in ("roster_gate", "selector_search", "selector_verify")}
    for index, row in enumerate(rows):
        if row["partition"] in partitions:
            partitions[row["partition"]].append(index)
    reconstructed = {}
    for name, values in partitions.items():
        indices = np.asarray(values, dtype=np.int64)
        codes = predictions[indices]
        references = labels[indices]
        accuracies = np.mean(codes == references[:, None], axis=0)
        parent_wrong = codes[:, 0] != references
        complementarity = [
            float(np.mean((codes[:, root] == references)[parent_wrong]))
            for root in range(1, common.ROSTER_SIZE)
        ]
        reconstructed[name] = {
            "rows": len(indices),
            "root_accuracies": accuracies.tolist(),
            "parent_minus_best_challenger": float(accuracies[0] - np.max(accuracies[1:])),
            "parent_error_prevalence": float(np.mean(parent_wrong)),
            "max_challenger_correct_given_parent_wrong": max(complementarity),
        }
    gate = reconstructed["roster_gate"]
    gates = {
        "parent_unique_best_by_2_5pp_on_roster_gate": gate["parent_minus_best_challenger"] >= .025,
        "parent_error_prevalence_at_least_3pct": gate["parent_error_prevalence"] >= .03,
        "challenger_correct_on_at_least_15pct_parent_errors": gate["max_challenger_correct_given_parent_wrong"] >= .15,
    }
    checks = {
        "protocol_hash": receipt["authority_sha256"][common.PROTOCOL.name] == common.sha256_path(common.PROTOCOL),
        "development_hash": receipt["development_sha256"] == common.sha256_path(DEVELOPMENT),
        "root_cache_hash": receipt["root_cache_sha256"] == common.sha256_path(ROOT_CACHE),
        "gates": gates == receipt["gates"],
        "decision": receipt["decision"] == "NO_GO_STEP97_ROOT_GEOMETRY_STOP" and receipt["failed_gates"] == ["parent_unique_best_by_2_5pp_on_roster_gate"],
        "test_unopened": receipt["sealed_test_opened"] is False,
        "no_later_outputs": not any(
            (ROOT / name).exists()
            for name in (
                f"STEP97_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz",
                f"STEP97_STAGEA_COMPLETE_LEDGER_{common.DATE}.json",
                f"STEP97_CONFIRMATORY_EXECUTION_LOCK_{common.DATE}.json",
                f"STEP97_CONFIRMATORY_RESULTS_{common.DATE}.json",
            )
        ),
    }
    if not all(checks.values()):
        raise AssertionError({name: value for name, value in checks.items() if not value})
    value = {
        "validation_id": "STEP97_ROOT_GEOMETRY_STOP_INDEPENDENT_V1",
        "date": common.DATE,
        "status": "PASS_STEP97_GEOMETRY_STOP_VALIDATION",
        "checks": checks,
        "reconstructed_geometry": reconstructed,
        "failed_gates": receipt["failed_gates"],
        "sealed_test_opened": False,
        "bound_sha256": {
            "receipt": common.sha256_path(RECEIPT),
            "development": common.sha256_path(DEVELOPMENT),
            "root_cache": common.sha256_path(ROOT_CACHE),
            "validator": common.sha256_path(Path(__file__)),
        },
    }
    common.json_dump(OUTPUT, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
