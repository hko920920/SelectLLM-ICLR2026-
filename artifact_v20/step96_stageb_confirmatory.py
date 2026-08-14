from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import step96_common as common


ROOT = Path(__file__).resolve().parent
LOCK = ROOT / f"STEP96_CONFIRMATORY_EXECUTION_LOCK_{common.DATE}.json"
PREOUTCOME = ROOT / f"STEP96_CONFIRMATORY_PREOUTCOME_PREDICTIONS_{common.DATE}.npz"
PREOUTCOME_LEDGER = ROOT / f"STEP96_CONFIRMATORY_PREOUTCOME_LEDGER_{common.DATE}.json"
SEALED = ROOT / "external_data" / "step96_sealed" / "snli_test_sealed_outcomes.npz"
RAW = ROOT / f"STEP96_CONFIRMATORY_RAW_{common.DATE}.npz"
RESULTS = ROOT / f"STEP96_CONFIRMATORY_RESULTS_{common.DATE}.json"


def main() -> None:
    for output in (RAW, RESULTS):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite {output}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    preledger = json.loads(PREOUTCOME_LEDGER.read_text(encoding="utf-8"))
    if lock["code_sha256"][Path(__file__).name] != common.sha256_path(Path(__file__)):
        raise AssertionError("confirmatory code drift")
    if preledger["lock_sha256"] != common.sha256_path(LOCK):
        raise AssertionError("preoutcome lock drift")
    if preledger["output_sha256"] != common.sha256_path(PREOUTCOME):
        raise AssertionError("preoutcome prediction drift")
    if lock["sealed_outcome"]["sha256"] != common.sha256_path(SEALED):
        raise AssertionError("sealed file drift")
    predictions = np.load(PREOUTCOME, allow_pickle=False)
    # This is the one authorized opening of the frozen SNLI test outcomes.
    outcomes = np.load(SEALED, allow_pickle=False)
    source_indices = predictions["source_indices"].astype(np.int64)
    if not np.array_equal(source_indices, outcomes["source_indices"].astype(np.int64)):
        raise AssertionError("sealed source-index mismatch")
    labels = outcomes["labels"].astype(np.int64)
    roots = predictions["root_predictions"].astype(np.int64)
    scores = predictions["error_scores"].astype(np.float64)
    selected = lock["selected_condition"]
    thresholds = np.asarray(selected["thresholds"], dtype=np.float64)
    alias_codes, triggers = common.make_alias_codes(roots[:, 0], scores, thresholds)
    refined_codes = np.column_stack([roots, alias_codes])
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)]
    )
    seed_info = lock["test_trajectory_seeds"]
    seeds = tuple(range(int(seed_info["start"]), int(seed_info["stop_exclusive"])))
    if len(seeds) != int(seed_info["runs"]):
        raise AssertionError("test seed count drift")
    pools = common.sample_pools(len(labels), seeds, common.POOL_SIZE)
    clean = common.run_active(
        roots, labels, clean_parents, roots, pools, seeds, common.BUDGET, float(selected["tau"])
    )
    refined = common.run_active(
        refined_codes,
        labels,
        refined_parents,
        roots,
        pools,
        seeds,
        common.BUDGET,
        float(selected["tau"]),
    )
    fixed = common.run_fixed(
        refined_codes, labels, refined_parents, roots, pools, clean.queries, seeds
    )
    terminal = refined.terminal - clean.terminal
    cumulative = refined.cumulative - clean.cumulative
    fixed_terminal = fixed.terminal - clean.terminal
    fixed_cumulative = fixed.cumulative - clean.cumulative
    active = terminal - fixed_terminal
    root_accuracies = np.mean(roots == labels[:, None], axis=0)
    parent_gap = float(root_accuracies[0] - np.max(root_accuracies[1:]))
    parent_feedback = common.exact_match_feedback(roots[:, :1], labels)
    alias_feedback = common.exact_match_feedback(alias_codes, labels)
    coordinate_nonimproving = bool(
        np.all(alias_feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
    )
    loss_counts = np.sum(
        triggers & (roots[:, :1] == labels[:, None]), axis=0
    ).astype(np.int64)
    allowed_loss_count = int(math.floor(0.01 * len(labels) + 1e-12))
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    parent_to = (clean_final == 0) & (refined_final != 0)
    to_parent = (clean_final != 0) & (refined_final == 0)
    inference_seeds = lock["inference_seeds"]
    terminal_summary = common.effect_summary(
        terminal,
        int(inference_seeds["terminal"]["bootstrap"]),
        int(inference_seeds["terminal"]["signflip"]),
    )
    active_summary = common.effect_summary(
        active,
        int(inference_seeds["active_minus_fixed_terminal"]["bootstrap"]),
        int(inference_seeds["active_minus_fixed_terminal"]["signflip"]),
    )
    cumulative_summary = common.effect_summary(
        cumulative,
        int(inference_seeds["cumulative"]["bootstrap"]),
        int(inference_seeds["cumulative"]["signflip"]),
    )
    gates = {
        "parent_unique_best_by_2_5pp": bool(parent_gap >= 0.025),
        "aliases_coordinate_wise_nonimproving": coordinate_nonimproving,
        "alias_integer_loss_counts_at_most_one_point": bool(
            np.all(loss_counts <= allowed_loss_count)
        ),
        "path_change_at_least_half": bool(
            np.mean(np.any(clean.queries != refined.queries, axis=1)) >= 0.50
        ),
        "clean_parent_terminal_rate_at_least_60pct": bool(np.mean(clean_final == 0) >= 0.60),
        "parent_to_challenger_exceeds_reverse": bool(np.mean(parent_to) > np.mean(to_parent)),
        "terminal_mean_at_least_half_point": bool(terminal_summary["mean"] >= 0.005),
        "terminal_lower_positive_and_p_at_most_point05": bool(
            terminal_summary["bootstrap_95"][0] > 0
            and terminal_summary["one_sided_signflip_p"] <= 0.05
        ),
        "active_minus_fixed_mean_at_least_half_point": bool(active_summary["mean"] >= 0.005),
        "active_lower_positive_and_p_at_most_point05": bool(
            active_summary["bootstrap_95"][0] > 0
            and active_summary["one_sided_signflip_p"] <= 0.05
        ),
        "cumulative_mean_and_lower_positive": bool(
            cumulative_summary["mean"] > 0 and cumulative_summary["bootstrap_95"][0] > 0
        ),
        "fixed_query_exact_zero": bool(
            np.max(np.abs(fixed_terminal)) == 0
            and np.max(np.abs(fixed_cumulative)) == 0
            and np.array_equal(fixed.roots, clean.roots)
        ),
    }
    decision = (
        lock["confirmatory_success_label"]
        if all(gates.values())
        else lock["confirmatory_failure_label"]
    )
    np.savez_compressed(
        RAW,
        source_indices=source_indices,
        labels=labels.astype(np.int16),
        root_predictions=roots.astype(np.int16),
        error_scores=scores.astype(np.float64),
        alias_codes=alias_codes.astype(np.int16),
        triggers=triggers.astype(np.uint8),
        pools=pools.astype(np.int32),
        clean_queries=clean.queries.astype(np.int32),
        refined_queries=refined.queries.astype(np.int32),
        clean_roots=clean.roots.astype(np.int16),
        refined_roots=refined.roots.astype(np.int16),
        fixed_roots=fixed.roots.astype(np.int16),
        terminal_delta=terminal.astype(np.float64),
        cumulative_delta=cumulative.astype(np.float64),
        fixed_terminal_delta=fixed_terminal.astype(np.float64),
        fixed_cumulative_delta=fixed_cumulative.astype(np.float64),
        active_minus_fixed_terminal_delta=active.astype(np.float64),
    )
    results = {
        "result_id": "STEP96_SNLI_CONFIRMATORY_V1",
        "date": common.DATE,
        "decision": decision,
        "lock_sha256": common.sha256_path(LOCK),
        "preoutcome_predictions_sha256": common.sha256_path(PREOUTCOME),
        "sealed_outcome_sha256": common.sha256_path(SEALED),
        "raw_file": RAW.name,
        "raw_sha256": common.sha256_path(RAW),
        "runs": len(seeds),
        "selected_condition": selected,
        "root_accuracies": {
            spec.key: float(value) for spec, value in zip(common.ROOT_SPECS, root_accuracies)
        },
        "parent_minus_best_challenger": parent_gap,
        "quality": {
            "coordinate_wise_nonimproving": coordinate_nonimproving,
            "loss_counts": loss_counts.tolist(),
            "allowed_loss_count": allowed_loss_count,
            "loss_fractions": (loss_counts / len(labels)).tolist(),
            "trigger_fractions": np.mean(triggers, axis=0).tolist(),
        },
        "mechanism": {
            "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
            "query_set_change_rate": float(
                np.mean(
                    [
                        set(left.tolist()) != set(right.tolist())
                        for left, right in zip(clean.queries, refined.queries)
                    ]
                )
            ),
            "final_root_change_rate": float(np.mean(clean_final != refined_final)),
            "clean_parent_terminal_rate": float(np.mean(clean_final == 0)),
            "refined_parent_terminal_rate": float(np.mean(refined_final == 0)),
            "parent_to_challenger_rate": float(np.mean(parent_to)),
            "challenger_to_parent_rate": float(np.mean(to_parent)),
            "fixed_root_history_exact": bool(np.array_equal(fixed.roots, clean.roots)),
            "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
            "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
        },
        "inference": {
            "terminal": terminal_summary,
            "active_minus_fixed_terminal": active_summary,
            "cumulative": cumulative_summary,
        },
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "sealed_outcome_opened_once": True,
        "code_sha256": common.sha256_path(Path(__file__)),
    }
    common.json_dump(RESULTS, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
