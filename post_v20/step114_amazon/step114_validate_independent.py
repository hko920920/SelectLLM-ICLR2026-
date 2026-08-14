from __future__ import annotations

import json

import numpy as np

import step114_common as common


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    predata = load_json(common.PREDATA_LOCK)
    stage = load_json(common.STAGE0_MANIFEST)
    development = load_json(common.DEVELOPMENT_LEDGER)
    go = load_json(common.DEVELOPMENT_GO_LOCK)
    prediction = load_json(common.PREOUTCOME_LEDGER)
    lock = load_json(common.PREOUTCOME_LOCK)
    reported = load_json(common.PRIMARY_LEDGER)
    bindings = {
        "protocol": common.sha256_path(common.PROTOCOL) == lock["protocol_sha256"],
        "metadata": common.sha256_path(common.METADATA) == lock["metadata_sha256"],
        "predata": common.sha256_path(common.PREDATA_LOCK) == lock["predata_lock_sha256"],
        "stage0": common.sha256_path(common.STAGE0_MANIFEST) == lock["stage0_manifest_sha256"],
        "development_go": common.sha256_path(common.DEVELOPMENT_GO_LOCK)
        == lock["development_go_lock_sha256"],
        "prediction_ledger": common.sha256_path(common.PREOUTCOME_LEDGER)
        == lock["preoutcome_ledger_sha256"],
        "predictions": common.sha256_path(common.ROOT / lock["predictions_file"])
        == lock["predictions_sha256"],
        "outcome": common.sha256_path(common.ROOT / lock["sealed_outcome_file"])
        == lock["sealed_outcome_sha256"],
        "primary_arrays": common.sha256_path(common.PRIMARY_ARRAYS)
        == reported["arrays_sha256"],
    }
    for path, expected in predata["code_sha256"].items():
        bindings[f"code:{path}"] = common.sha256_path(common.ROOT / path) == expected
    for path, expected in lock["head_sha256"].items():
        bindings[f"head:{path}"] = common.sha256_path(common.ROOT / path) == expected
    with np.load(common.ROOT / lock["predictions_file"], allow_pickle=False) as package:
        predictions = {key: package[key] for key in package.files}
    prediction_hashes = {
        name: common.array_sha256(predictions[name]) == expected
        for name, expected in lock["prediction_hashes"].items()
    }
    with np.load(common.ROOT / lock["sealed_outcome_file"], allow_pickle=False) as package:
        labels = package["labels"].astype(np.int64)
        uids = package["uids"]
    roots = predictions["roots"].astype(np.int64)
    aliases = predictions["aliases"].astype(np.int64)
    reconstructed_aliases, reconstructed_triggers = common.make_aliases(
        roots[:, 0], predictions["scores"], predictions["thresholds"]
    )
    quality = common.alias_quality_audit(roots[:, 0], aliases, labels)
    distinct = len(
        {common.array_sha256(aliases[:, column]) for column in range(common.N_ALIASES)}
    ) == common.N_ALIASES
    raw, arrays = common.compute_raw_effects(roots, aliases, labels)
    summary = common.add_inference(raw, arrays)
    gates = common.primary_gates(summary, quality, distinct)
    expected_decision = (
        "GO_STEP114_PRIMARY_TERMINAL_CONFIRMATION"
        if all(gates.values())
        else "NO_GO_STEP114_PRIMARY_TERMINAL_CONFIRMATION"
    )
    with np.load(common.PRIMARY_ARRAYS, allow_pickle=False) as package:
        saved = {key: package[key] for key in package.files}
    array_replay = {name: np.array_equal(value, saved[name]) for name, value in arrays.items()}
    trigger_counts = np.sum(predictions["triggers"], axis=0).astype(int)
    checks = {
        "all_bindings": all(bindings.values()),
        "all_prediction_hashes": all(prediction_hashes.values()),
        "uid_alignment": np.array_equal(uids, predictions["uids"]),
        "aliases_reconstruct": np.array_equal(aliases, reconstructed_aliases),
        "triggers_reconstruct": np.array_equal(predictions["triggers"], reconstructed_triggers),
        "development_was_go": development["decision"] == "GO_STEP114_DEVELOPMENT_CERTIFIED"
        and all(development["gates"].values()),
        "preoutcome_was_go": prediction["decision"] == "GO_STEP114_PREOUTCOME_TRIGGER_CERTIFIED"
        and all(prediction["preoutcome_gates"].values()),
        "trigger_certificate_replay": bool(
            np.all(trigger_counts >= 1)
            and np.all(trigger_counts <= lock["maximum_allowed_trigger_count"])
        ),
        "quality_replay": common.canonical_json_sha256(quality)
        == common.canonical_json_sha256(reported["quality"]),
        "summary_replay": common.canonical_json_sha256(summary)
        == common.canonical_json_sha256(reported["summary"]),
        "gates_replay": gates == reported["gates"],
        "decision_replay": expected_decision == reported["decision"],
        "all_primary_arrays_replay": all(array_replay.values()),
        "single_outcome_binding": stage["files"][lock["sealed_outcome_file"]]
        == go["sealed_outcome_sha256"]
        == lock["sealed_outcome_sha256"]
        == reported["sealed_outcome_sha256"],
        "replacement_forbidden": lock["replacement_allowed"] is False
        and reported["replacement_allowed"] is False,
    }
    result = {
        "schema": "step114.primary_independent_validation.v1",
        "date": common.DATE,
        "decision": "PASS_STEP114_INDEPENDENT_VALIDATION" if all(checks.values()) else "FAIL_STEP114_INDEPENDENT_VALIDATION",
        "reported_primary_decision": reported["decision"],
        "bindings": bindings,
        "prediction_hashes": prediction_hashes,
        "array_replay": array_replay,
        "checks": checks,
        "recomputed_quality": quality,
        "recomputed_summary": summary,
        "recomputed_gates": gates,
    }
    common.json_dump(common.INDEPENDENT_VALIDATION, result)
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise AssertionError(result)


if __name__ == "__main__":
    main()
