from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import step94_common as common


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
LOCK_PATH = ROOT / f"STEP94_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json"
CONFIG_PATH = ROOT / f"STEP94_STAGEA_FROZEN_CONFIG_{DATE}.json"
LEDGER_PATH = ROOT / f"STEP94_STAGEA_COMPLETE_LEDGER_{DATE}.json"
RAW_PATH = ROOT / f"STEP94_CONFIRMATORY_RAW_{DATE}.npz"
RESULT_PATH = ROOT / f"STEP94_CONFIRMATORY_RESULTS_{DATE}.json"
VALIDATION_PATH = ROOT / f"STEP94_INDEPENDENT_VALIDATION_{DATE}.json"


def assert_close(observed: Any, expected: Any, tolerance: float = 1e-15) -> None:
    if isinstance(observed, list) or isinstance(expected, list):
        if not np.allclose(
            np.asarray(observed, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=0,
            atol=tolerance,
        ):
            raise AssertionError({"observed": observed, "expected": expected})
    elif abs(float(observed) - float(expected)) > tolerance:
        raise AssertionError({"observed": observed, "expected": expected})


def main() -> None:
    if VALIDATION_PATH.exists():
        raise RuntimeError(f"refusing to overwrite validation: {VALIDATION_PATH}")
    for path in (LOCK_PATH, CONFIG_PATH, LEDGER_PATH, RAW_PATH, RESULT_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if lock["stagea_config_sha256"] != common.sha256_path(CONFIG_PATH):
        raise AssertionError("config binding")
    if lock["stagea_ledger_sha256"] != common.sha256_path(LEDGER_PATH):
        raise AssertionError("ledger binding")
    if results["execution_lock_sha256"] != common.sha256_path(LOCK_PATH):
        raise AssertionError("result lock binding")
    if results["raw_sha256"] != common.sha256_path(RAW_PATH):
        raise AssertionError("raw binding")
    for filename, expected in lock["code_sha256"].items():
        if common.sha256_path(ROOT / filename) != expected:
            raise AssertionError({"code": filename, "hash": False})
    for relative, expected in lock["model_sha256"].items():
        if common.sha256_path(ROOT / relative) != expected:
            raise AssertionError({"model": relative, "hash": False})

    with np.load(RAW_PATH, allow_pickle=False) as raw:
        arrays = {key: raw[key] for key in raw.files}
    terminal = arrays["refined_terminal"] - arrays["clean_terminal"]
    cumulative = arrays["refined_cumulative"] - arrays["clean_cumulative"]
    fixed_terminal = arrays["refined_fixed_terminal"] - arrays["clean_terminal"]
    fixed_cumulative = arrays["refined_fixed_cumulative"] - arrays["clean_cumulative"]
    active_minus_fixed = terminal - fixed_terminal
    for name, value in (
        ("terminal_delta", terminal),
        ("cumulative_delta", cumulative),
        ("fixed_terminal_delta", fixed_terminal),
        ("fixed_cumulative_delta", fixed_cumulative),
        ("active_minus_fixed_terminal_delta", active_minus_fixed),
    ):
        if not np.array_equal(arrays[name], value):
            raise AssertionError({"raw_vector": name})

    labels = arrays["labels"].astype(np.int64)
    clean_codes = arrays["clean_codes"].astype(np.int64)
    alias_codes = arrays["alias_codes"].astype(np.int64)
    parent_position = int(results["parent_position"])
    parent_feedback = common.exact_match_feedback(
        clean_codes[:, [parent_position]], labels
    )
    alias_feedback = common.exact_match_feedback(alias_codes, labels)
    coordinate_nonimproving = bool(
        np.all(
            alias_feedback
            <= np.repeat(parent_feedback, common.N_ALIASES, axis=1)
        )
    )
    alias_losses = (
        np.mean(parent_feedback, axis=0)[0] - np.mean(alias_feedback, axis=0)
    )
    assert_close(results["quality"]["alias_losses"], alias_losses.tolist())
    if coordinate_nonimproving != results["quality"]["coordinate_wise_nonimproving"]:
        raise AssertionError("coordinate non-improvement")

    terminal_summary = common.effect_summary(terminal, 94900, 94901)
    active_summary = common.effect_summary(active_minus_fixed, 94901, 94902)
    cumulative_summary = common.effect_summary(cumulative, 94902, 94903)
    for key, recomputed in (
        ("terminal", terminal_summary),
        ("active_minus_fixed_terminal", active_summary),
        ("cumulative", cumulative_summary),
    ):
        stored = results["effects"][key]
        assert_close(stored["mean"], recomputed["mean"])
        assert_close(stored["median"], recomputed["median"])
        assert_close(stored["bootstrap_95"], recomputed["bootstrap_95"])
        assert_close(
            stored["one_sided_signflip_p"], recomputed["one_sided_signflip_p"]
        )
        assert_close(stored["positive_fraction"], recomputed["positive_fraction"])
        assert_close(stored["negative_fraction"], recomputed["negative_fraction"])
        assert_close(stored["zero_fraction"], recomputed["zero_fraction"])

    path_change = float(
        np.mean(np.any(arrays["clean_queries"] != arrays["refined_queries"], axis=1))
    )
    query_set_change = float(
        np.mean(
            [
                set(left.tolist()) != set(right.tolist())
                for left, right in zip(arrays["clean_queries"], arrays["refined_queries"])
            ]
        )
    )
    assert_close(results["mechanism"]["path_change_rate"], path_change)
    assert_close(results["mechanism"]["query_set_change_rate"], query_set_change)
    fixed_root_exact = bool(
        np.array_equal(arrays["refined_fixed_roots"], arrays["clean_roots"])
    )

    recomputed_gate_truth = {
        "coordinate_wise_nonimproving": coordinate_nonimproving,
        "each_alias_loss_at_most_one_point": bool(np.max(alias_losses) <= 0.0100),
        "path_change_at_least_half": bool(path_change >= 0.50),
        "terminal_mean_at_least_half_point": bool(terminal_summary["mean"] >= 0.005),
        "terminal_bootstrap_lower_positive": bool(terminal_summary["bootstrap_95"][0] > 0),
        "terminal_signflip_at_most_point05": bool(
            terminal_summary["one_sided_signflip_p"] <= 0.05
        ),
        "active_minus_fixed_mean_at_least_half_point": bool(
            active_summary["mean"] >= 0.005
        ),
        "active_minus_fixed_bootstrap_lower_positive": bool(
            active_summary["bootstrap_95"][0] > 0
        ),
        "cumulative_mean_positive": bool(cumulative_summary["mean"] > 0),
        "cumulative_bootstrap_lower_positive": bool(
            cumulative_summary["bootstrap_95"][0] > 0
        ),
        "fixed_terminal_exact_zero": bool(np.max(np.abs(fixed_terminal)) == 0),
        "fixed_cumulative_exact_zero": bool(np.max(np.abs(fixed_cumulative)) == 0),
        "fixed_root_history_bitwise_exact": fixed_root_exact,
    }
    for key, value in recomputed_gate_truth.items():
        if results["gates"][key] != value:
            raise AssertionError({"gate": key, "stored": results["gates"][key], "recomputed": value})
    all_gates = bool(all(results["gates"].values()))
    expected_decision = (
        "GO_FRESH_SOURCE_FAITHFUL_LEARNED_VARIANT_BRIDGE"
        if all_gates
        else "NO_GO_RETAIN_FRESH_TASK_NEGATIVE"
    )
    if results["decision"] != expected_decision:
        raise AssertionError({"decision": results["decision"], "expected": expected_decision})

    validation = {
        "validation_id": "STEP94_INDEPENDENT_RECONSTRUCTION_V1",
        "date": DATE,
        "result_sha256": common.sha256_path(RESULT_PATH),
        "raw_sha256": common.sha256_path(RAW_PATH),
        "lock_sha256": common.sha256_path(LOCK_PATH),
        "checks": {
            "hash_bindings": True,
            "raw_delta_reconstruction": True,
            "coordinate_nonimprovement_reconstruction": True,
            "quality_loss_reconstruction": True,
            "bootstrap_reconstruction": True,
            "signflip_reconstruction": True,
            "path_reconstruction": True,
            "fixed_query_reconstruction": True,
            "gate_reconstruction": True,
            "decision_reconstruction": True,
        },
        "decision": results["decision"],
        "terminal_mean": terminal_summary["mean"],
        "terminal_bootstrap_95": terminal_summary["bootstrap_95"],
        "cumulative_mean": cumulative_summary["mean"],
        "fixed_root_history_exact": fixed_root_exact,
    }
    common.json_dump(VALIDATION_PATH, validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
