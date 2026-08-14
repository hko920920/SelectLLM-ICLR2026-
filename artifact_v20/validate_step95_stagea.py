from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import step95_common as common
import step95_stagea_develop as stagea


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / f"STEP95_STAGE0_MANIFEST_{common.DATE}.json"
DEVELOPMENT_PATH = ROOT / f"STEP95_MNLI_DEVELOPMENT_ROWS_{common.DATE}.json"
ARRAYS_PATH = ROOT / f"STEP95_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
LEDGER_PATH = ROOT / f"STEP95_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
CONFIG_PATH = ROOT / f"STEP95_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
VALIDATION_PATH = ROOT / f"STEP95_STAGEA_INDEPENDENT_VALIDATION_{common.DATE}.json"


def assert_close(observed: Any, expected: Any, tolerance: float = 1e-15) -> None:
    if isinstance(observed, list) or isinstance(expected, list):
        if not np.allclose(np.asarray(observed), np.asarray(expected), rtol=0, atol=tolerance):
            raise AssertionError({"observed": observed, "expected": expected})
    elif abs(float(observed) - float(expected)) > tolerance:
        raise AssertionError({"observed": observed, "expected": expected})


def compare_row(stored: dict[str, Any], replayed: dict[str, Any]) -> None:
    for key in (
        "tau", "loss_cap", "trigger_cap", "thresholds", "alias_losses",
        "trigger_fractions", "mean_terminal_delta",
        "mean_active_minus_fixed_terminal_delta", "mean_cumulative_delta",
        "mean_fixed_terminal_delta", "mean_fixed_cumulative_delta",
        "max_abs_fixed_terminal_delta", "max_abs_fixed_cumulative_delta",
        "path_change_rate", "query_set_change_rate", "final_root_change_rate",
    ):
        assert_close(stored[key], replayed[key])
    for key in ("coordinate_wise_nonimproving", "fixed_root_history_exact", "runs"):
        if stored[key] != replayed[key]:
            raise AssertionError({"row_field": key, "stored": stored[key], "replayed": replayed[key]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if VALIDATION_PATH.exists() and not args.check_only:
        raise RuntimeError(f"refusing to overwrite validation: {VALIDATION_PATH}")
    for path in (MANIFEST_PATH, DEVELOPMENT_PATH, ARRAYS_PATH, LEDGER_PATH, CONFIG_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    development = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if manifest["development"]["sha256"] != common.sha256_path(DEVELOPMENT_PATH):
        raise AssertionError("development binding")
    if config["stagea_ledger_sha256"] != common.sha256_path(LEDGER_PATH):
        raise AssertionError("ledger binding")
    if config["development_arrays_sha256"] != common.sha256_path(ARRAYS_PATH):
        raise AssertionError("arrays binding")
    for filename, expected in config["code_sha256"].items():
        if common.sha256_path(ROOT / filename) != expected:
            raise AssertionError({"code": filename, "hash": False})
    for relative, expected in config["adapter_sha256"].items():
        if common.sha256_path(ROOT / relative) != expected:
            raise AssertionError({"adapter": relative, "hash": False})
    with np.load(ARRAYS_PATH, allow_pickle=False) as arrays_file:
        arrays = {key: arrays_file[key] for key in arrays_file.files}
    labels = arrays["labels"].astype(np.int64)
    predictions = arrays["root_predictions"].astype(np.int64)
    scores = arrays["error_scores"].astype(np.float64)
    rows = development["rows"]
    expected_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    if not np.array_equal(labels, expected_labels):
        raise AssertionError("development label reconstruction")
    for index, audit in enumerate(ledger["root_prediction_audits"]):
        if audit["prediction_sha256"] != common.array_sha256(predictions[:, index]):
            raise AssertionError({"root_prediction": index})
    for index, audit in enumerate(ledger["adapter_audits"]):
        if audit["score_sha256"] != common.array_sha256(scores[:, index]):
            raise AssertionError({"adapter_score": index})
        if audit["checkpoint"]["parameter_count"] < 10_000_000:
            raise AssertionError({"adapter_parameter_floor": index})
    partitions = stagea.partition_rows(rows)
    parent_index, roster, _ = stagea.select_parent_and_roster(
        predictions,
        labels,
        np.asarray(partitions["parent_select"], dtype=np.int64),
    )
    if parent_index != ledger["selected_parent"]["bank_index"]:
        raise AssertionError("parent reconstruction")
    if list(roster) != ledger["roster_bank_indices"]:
        raise AssertionError("roster reconstruction")
    selected = ledger["selected_verification"]
    ranked_search = sorted(ledger["search_complete_48"], key=stagea.rank_key, reverse=True)
    if len(ranked_search) != 48 or len(ledger["verification_top_8"]) != 8:
        raise AssertionError("complete grid/top-eight count")
    expected_top = [
        (row["tau"], row["loss_cap"], row["trigger_cap"])
        for row in ranked_search[:8]
    ]
    observed_top = [
        (row["tau"], row["loss_cap"], row["trigger_cap"])
        for row in ledger["verification_top_8"]
    ]
    if expected_top != observed_top:
        raise AssertionError("top-eight selection reconstruction")
    best_recorded = max(ledger["verification_top_8"], key=stagea.rank_key)
    if (best_recorded["tau"], best_recorded["loss_cap"], best_recorded["trigger_cap"]) != (
        selected["tau"], selected["loss_cap"], selected["trigger_cap"]
    ):
        raise AssertionError("verification winner reconstruction")

    threshold_indices = arrays["threshold_indices"].astype(np.int64)
    search_indices = arrays["search_indices"].astype(np.int64)
    verify_indices = arrays["verify_indices"].astype(np.int64)
    threshold = stagea.construction(
        predictions[threshold_indices, parent_index],
        labels[threshold_indices],
        scores[threshold_indices],
        float(selected["loss_cap"]),
        float(selected["trigger_cap"]),
    )
    assert_close(selected["thresholds"], threshold["thresholds"])
    search_built = stagea.replay_construction(
        predictions[search_indices, parent_index],
        labels[search_indices],
        scores[search_indices],
        threshold["thresholds"],
    )
    verify_built = stagea.replay_construction(
        predictions[verify_indices, parent_index],
        labels[verify_indices],
        scores[verify_indices],
        threshold["thresholds"],
    )
    search_match = next(
        row for row in ledger["search_complete_48"]
        if row["tau"] == selected["tau"]
        and row["loss_cap"] == selected["loss_cap"]
        and row["trigger_cap"] == selected["trigger_cap"]
    )
    search_replay = stagea.evaluate(
        "selector_search",
        predictions[search_indices][:, roster],
        labels[search_indices],
        0,
        search_built,
        float(selected["tau"]),
        float(selected["loss_cap"]),
        float(selected["trigger_cap"]),
        stagea.SEARCH_SEEDS,
        {},
        {},
        True,
    )
    verify_replay = stagea.evaluate(
        "selector_verify",
        predictions[verify_indices][:, roster],
        labels[verify_indices],
        0,
        verify_built,
        float(selected["tau"]),
        float(selected["loss_cap"]),
        float(selected["trigger_cap"]),
        stagea.VERIFY_SEEDS,
        {},
        {},
        True,
    )
    compare_row(search_match, search_replay)
    compare_row(selected, verify_replay)
    terminal_summary = common.effect_summary(
        np.asarray(verify_replay["vectors"]["terminal"]), 95900, 95901
    )
    active_summary = common.effect_summary(
        np.asarray(verify_replay["vectors"]["active_minus_fixed_terminal"]), 95901, 95902
    )
    cumulative_summary = common.effect_summary(
        np.asarray(verify_replay["vectors"]["cumulative"]), 95902, 95903
    )
    for key, replayed in (
        ("terminal", terminal_summary),
        ("active_minus_fixed_terminal", active_summary),
        ("cumulative", cumulative_summary),
    ):
        for statistic in (
            "mean", "median", "bootstrap_95", "one_sided_signflip_p",
            "positive_fraction", "negative_fraction", "zero_fraction",
        ):
            assert_close(selected["inference"][key][statistic], replayed[statistic])
    search_accuracies = np.mean(
        predictions[search_indices][:, roster] == labels[search_indices, None], axis=0
    )
    verify_accuracies = np.mean(
        predictions[verify_indices][:, roster] == labels[verify_indices, None], axis=0
    )
    other_gap = float(verify_accuracies[0] - np.max(verify_accuracies[1:]))
    gates = {
        "all_four_distinct_adapters_at_least_ten_million_parameters": bool(
            len(set(config["adapter_sha256"].values())) == 4
            and all(row["checkpoint"]["parameter_count"] >= 10_000_000 for row in ledger["adapter_audits"])
        ),
        "raw_input_endpoint_has_no_forbidden_runtime_input": ledger["gates"]["raw_input_endpoint_has_no_forbidden_runtime_input"],
        "same_literal_similarity": True,
        "parent_best_on_search_and_verification": bool(
            search_accuracies[0] == np.max(search_accuracies)
            and verify_accuracies[0] == np.max(verify_accuracies)
        ),
        "another_verification_root_at_least_half_point_worse": bool(other_gap >= 0.005),
        "coordinate_wise_nonimproving_threshold_search_verify": bool(
            threshold["coordinate_wise_nonimproving"]
            and search_built["coordinate_wise_nonimproving"]
            and verify_built["coordinate_wise_nonimproving"]
        ),
        "alias_losses_at_most_one_point_threshold_search_verify": bool(
            max(threshold["alias_losses"]) <= 0.0100
            and max(search_built["alias_losses"]) <= 0.0100
            and max(verify_built["alias_losses"]) <= 0.0100
        ),
        "search_and_verification_path_change_at_least_half": bool(
            search_replay["path_change_rate"] >= 0.50 and verify_replay["path_change_rate"] >= 0.50
        ),
        "search_and_verification_terminal_mean_at_least_half_point": bool(
            search_replay["mean_terminal_delta"] >= 0.005 and verify_replay["mean_terminal_delta"] >= 0.005
        ),
        "search_and_verification_active_minus_fixed_mean_at_least_half_point": bool(
            search_replay["mean_active_minus_fixed_terminal_delta"] >= 0.005
            and verify_replay["mean_active_minus_fixed_terminal_delta"] >= 0.005
        ),
        "verification_terminal_lower_positive_and_p_at_most_point05": bool(
            terminal_summary["bootstrap_95"][0] > 0
            and terminal_summary["one_sided_signflip_p"] <= 0.05
        ),
        "verification_active_lower_positive_and_p_at_most_point05": bool(
            active_summary["bootstrap_95"][0] > 0
            and active_summary["one_sided_signflip_p"] <= 0.05
        ),
        "verification_cumulative_mean_and_lower_positive": bool(
            cumulative_summary["mean"] > 0 and cumulative_summary["bootstrap_95"][0] > 0
        ),
        "verification_fixed_query_exact_zero": bool(
            verify_replay["max_abs_fixed_terminal_delta"] == 0
            and verify_replay["max_abs_fixed_cumulative_delta"] == 0
            and verify_replay["fixed_root_history_exact"]
        ),
    }
    if gates != ledger["gates"]:
        raise AssertionError({"stored_gates": ledger["gates"], "recomputed_gates": gates})
    decision = "GO_TO_STEP95_ONE_TIME_CONFIRMATORY_LOCK" if all(gates.values()) else "NO_GO_DEVELOPMENT_GATE_STOP"
    if ledger["decision"] != decision or config["decision"] != decision:
        raise AssertionError("decision reconstruction")
    confirmatory_absent = all(
        not (ROOT / name).exists()
        for name in (
            f"STEP95_CONFIRMATORY_EXECUTION_LOCK_{common.DATE}.json",
            f"STEP95_CONFIRMATORY_RAW_{common.DATE}.npz",
            f"STEP95_CONFIRMATORY_RESULTS_{common.DATE}.json",
        )
    )
    if not confirmatory_absent:
        raise AssertionError("development stop violated by confirmatory output")
    validation = {
        "validation_id": "STEP95_STAGEA_INDEPENDENT_RECONSTRUCTION_V1",
        "date": common.DATE,
        "ledger_sha256": common.sha256_path(LEDGER_PATH),
        "config_sha256": common.sha256_path(CONFIG_PATH),
        "arrays_sha256": common.sha256_path(ARRAYS_PATH),
        "checks": {
            "hash_bindings": True,
            "parent_and_roster_reconstruction": True,
            "complete_grid_and_top_eight_selection": True,
            "selected_search_full_rerun_exact": True,
            "selected_verification_full_rerun_exact": True,
            "bootstrap_and_signflip_reconstruction": True,
            "gate_and_decision_reconstruction": True,
            "sealed_confirmation_absent": True,
        },
        "decision": decision,
        "failed_gates": [key for key, value in gates.items() if not value],
        "verification_terminal_mean": terminal_summary["mean"],
        "verification_terminal_bootstrap_95": terminal_summary["bootstrap_95"],
        "verification_cumulative_mean": cumulative_summary["mean"],
        "sealed_test_opened": False,
    }
    if args.check_only:
        if not VALIDATION_PATH.is_file():
            raise FileNotFoundError(VALIDATION_PATH)
        stored = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
        if stored != validation:
            raise AssertionError("stored Stage-A validation receipt drift")
    else:
        common.json_dump(VALIDATION_PATH, validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
