from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

import step93_common as selector
import step95_common as step95


ROOT = Path(__file__).resolve().parent
ARRAYS = ROOT / "STEP95_STAGEA_DEVELOPMENT_ARRAYS_2026-08-13.npz"
LEDGER = ROOT / "STEP95_STAGEA_COMPLETE_LEDGER_2026-08-13.json"
OUTPUT = ROOT / "STEP95_POSTHOC_TERMINAL_BOTTLENECK_DIAGNOSTIC_2026-08-13.json"
REPORT = ROOT / "STEP95_POSTHOC_TERMINAL_BOTTLENECK_DIAGNOSTIC_2026-08-13.md"
VERIFY_SEEDS = tuple(range(954000, 955500))


def as_float(value: np.generic | float | int) -> float:
    return float(value)


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {}
    return {
        key: as_float(value)
        for key, value in zip(
            ("min", "p10", "p25", "median", "p75", "p90", "max"),
            np.quantile(values, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)),
        )
    }


def trace_active(
    codes: np.ndarray,
    references: np.ndarray,
    parents: np.ndarray,
    core_codes: np.ndarray,
    pools: np.ndarray,
    seeds: tuple[int, ...],
    max_budget: int,
    tau: float,
) -> dict[str, np.ndarray]:
    feedback = selector.exact_match_feedback(codes, references)
    core_feedback = selector.exact_match_feedback(core_codes, references)
    runs = len(seeds)
    roots = np.full((runs, max_budget), -1, dtype=np.int64)
    regrets = np.zeros((runs, max_budget), dtype=np.float64)
    queries = np.full((runs, max_budget), -1, dtype=np.int64)
    root_max_scores = np.zeros((runs, max_budget, core_codes.shape[1]), dtype=np.float64)
    entry_margin = np.zeros((runs, max_budget), dtype=np.float64)
    for run_index, seed in enumerate(seeds):
        pool = pools[run_index]
        groups = selector.build_source_faithful_groups(codes[pool])
        active = np.ones(len(pool), dtype=bool)
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        root_quality = np.mean(core_feedback[pool], axis=0)
        best_quality = float(np.max(root_quality))
        for step in range(max_budget):
            shifted = scores / float(tau)
            shifted -= float(np.max(shifted))
            posterior = np.exp(shifted)
            posterior /= float(np.sum(posterior))
            acquisition = selector.select_source_faithful_acquisition(groups, posterior)
            acquisition[~active] = np.inf
            tied = np.flatnonzero(acquisition == float(np.min(acquisition)))
            position = selector.choose_query(tied, pool, int(seed), step)
            active[position] = False
            query = int(pool[position])
            queries[run_index, step] = query
            scores += feedback[query]
            tied_entries = np.flatnonzero(scores == float(np.max(scores)))
            root = selector.choose_root(tied_entries, parents, int(seed), step)
            roots[run_index, step] = root
            regrets[run_index, step] = best_quality - float(root_quality[root])
            maxima = np.asarray(
                [np.max(scores[parents == root_id]) for root_id in range(core_codes.shape[1])],
                dtype=np.float64,
            )
            root_max_scores[run_index, step] = maxima
            ordered = np.sort(scores)
            entry_margin[run_index, step] = float(ordered[-1] - ordered[-2])
    return {
        "roots": roots,
        "regrets": regrets,
        "queries": queries,
        "root_max_scores": root_max_scores,
        "entry_margin": entry_margin,
    }


def transition_rows(
    clean_roots: np.ndarray,
    refined_roots: np.ndarray,
    root_quality: np.ndarray,
    root_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in itertools.product(range(len(root_names)), repeat=2):
        mask = (clean_roots == left) & (refined_roots == right)
        if not np.any(mask):
            continue
        delta = root_quality[mask, left] - root_quality[mask, right]
        rows.append(
            {
                "clean_root": root_names[left],
                "refined_root": root_names[right],
                "count": int(np.sum(mask)),
                "rate": as_float(np.mean(mask)),
                "mean_terminal_delta": as_float(np.mean(delta)),
                "positive": int(np.sum(delta > 0)),
                "negative": int(np.sum(delta < 0)),
                "zero": int(np.sum(delta == 0)),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["count"]), row["clean_root"], row["refined_root"]))


def main() -> None:
    arrays = np.load(ARRAYS, allow_pickle=False)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    labels_all = arrays["labels"].astype(np.int64)
    predictions_all = arrays["root_predictions"].astype(np.int64)
    scores_all = arrays["error_scores"].astype(np.float64)
    verify_indices = arrays["verify_indices"].astype(np.int64)
    labels = labels_all[verify_indices]
    roster = tuple(int(value) for value in ledger["roster_bank_indices"])
    root_names = [step95.ROOT_SPECS[index].key for index in roster]
    clean_codes = predictions_all[verify_indices][:, roster]
    parent_position = int(ledger["selected_parent"]["position"])
    selected = ledger["selected_verification"]
    thresholds = [float(value) for value in selected["thresholds"]]
    alias_codes, triggers = step95.make_alias_codes(
        clean_codes[:, parent_position], scores_all[verify_indices], thresholds
    )
    refined_codes = np.column_stack([clean_codes, alias_codes])
    clean_parents = np.arange(len(roster), dtype=np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.full(step95.N_ALIASES, parent_position, dtype=np.int64)]
    )
    diagnostic_seeds = VERIFY_SEEDS[:400]
    pools = selector.sample_pools(
        len(labels), diagnostic_seeds, pool_size=step95.POOL_SIZE
    )
    core_feedback = selector.exact_match_feedback(clean_codes, labels)
    root_quality = np.mean(core_feedback[pools], axis=1)

    tau = float(selected["tau"])
    clean = trace_active(
        clean_codes,
        labels,
        clean_parents,
        clean_codes,
        pools,
        diagnostic_seeds,
        100,
        tau,
    )
    refined = trace_active(
        refined_codes,
        labels,
        refined_parents,
        clean_codes,
        pools,
        diagnostic_seeds,
        100,
        tau,
    )
    terminal_step = step95.BUDGET - 1
    clean_final = clean["roots"][:, terminal_step]
    refined_final = refined["roots"][:, terminal_step]
    delta = refined["regrets"][:, terminal_step] - clean["regrets"][:, terminal_step]
    changed = clean_final != refined_final

    clean_budget = clean
    refined_budget = refined
    budget_scan: list[dict[str, Any]] = []
    for budget in (5, 10, 15, 20, 30, 40, 60, 80, 100):
        step = budget - 1
        terminal_delta = refined_budget["regrets"][:, step] - clean_budget["regrets"][:, step]
        cumulative_delta = np.sum(
            refined_budget["regrets"][:, :budget] - clean_budget["regrets"][:, :budget],
            axis=1,
        )
        budget_scan.append(
            {
                "budget": budget,
                "runs": len(diagnostic_seeds),
                "mean_terminal_delta": as_float(np.mean(terminal_delta)),
                "terminal_pp": as_float(100.0 * np.mean(terminal_delta)),
                "positive_fraction": as_float(np.mean(terminal_delta > 0)),
                "negative_fraction": as_float(np.mean(terminal_delta < 0)),
                "root_change_rate": as_float(
                    np.mean(clean_budget["roots"][:, step] != refined_budget["roots"][:, step])
                ),
                "mean_cumulative_delta": as_float(np.mean(cumulative_delta)),
            }
        )

    clean_queries = clean["queries"][:, : step95.BUDGET]
    refined_queries = refined["queries"][:, : step95.BUDGET]
    parent_correct = clean_codes[:, parent_position] == labels
    root_disagreement = np.apply_along_axis(lambda row: len(set(row.tolist())), 1, clean_codes)
    query_diagnostics = {
        "clean_parent_error_fraction": as_float(np.mean(~parent_correct[clean_queries])),
        "refined_parent_error_fraction": as_float(np.mean(~parent_correct[refined_queries])),
        "clean_root_disagreement_mean": as_float(np.mean(root_disagreement[clean_queries])),
        "refined_root_disagreement_mean": as_float(np.mean(root_disagreement[refined_queries])),
        "mean_unique_clean_only": as_float(
            np.mean([len(set(left.tolist()) - set(right.tolist())) for left, right in zip(clean_queries, refined_queries)])
        ),
        "mean_unique_refined_only": as_float(
            np.mean([len(set(right.tolist()) - set(left.tolist())) for left, right in zip(clean_queries, refined_queries)])
        ),
        "alias_trigger_fraction_on_clean_queries": [
            as_float(np.mean(triggers[clean_queries, alias])) for alias in range(step95.N_ALIASES)
        ],
        "alias_trigger_fraction_on_refined_queries": [
            as_float(np.mean(triggers[refined_queries, alias])) for alias in range(step95.N_ALIASES)
        ],
    }

    parent_global = as_float(np.mean(clean_codes[:, parent_position] == labels))
    global_accuracies = np.mean(clean_codes == labels[:, None], axis=0)
    best_minus_second_pool = np.sort(root_quality, axis=1)[:, -1] - np.sort(root_quality, axis=1)[:, -2]
    counterfactual_worst = np.max(root_quality, axis=1) - np.min(root_quality, axis=1)
    changed_delta = delta[changed]
    positive = delta[delta > 0]
    negative = delta[delta < 0]
    needed_conditional = 0.005 / max(as_float(np.mean(changed)), np.finfo(float).eps)

    search_rows = ledger["search_complete_48"]
    verification_rows = ledger["verification_top_8"]
    grid_summary = {
        "search_max_terminal_pp": 100.0 * max(float(row["mean_terminal_delta"]) for row in search_rows),
        "search_min_terminal_pp": 100.0 * min(float(row["mean_terminal_delta"]) for row in search_rows),
        "search_max_cumulative": max(float(row["mean_cumulative_delta"]) for row in search_rows),
        "verification_max_terminal_pp": 100.0
        * max(float(row["mean_terminal_delta"]) for row in verification_rows),
        "verification_min_terminal_pp": 100.0
        * min(float(row["mean_terminal_delta"]) for row in verification_rows),
        "verification_rows": [
            {
                "search_rank": int(row["search_rank"]),
                "tau": float(row["tau"]),
                "loss_cap": float(row["loss_cap"]),
                "trigger_cap": float(row["trigger_cap"]),
                "terminal_pp": 100.0 * float(row["mean_terminal_delta"]),
                "cumulative": float(row["mean_cumulative_delta"]),
                "root_change_rate": float(row["final_root_change_rate"]),
            }
            for row in verification_rows
        ],
    }

    output: dict[str, Any] = {
        "diagnostic_id": "STEP95_POSTHOC_TERMINAL_BOTTLENECK_V1",
        "scope": {
            "data": "Step 95 selector_verify development partition only",
            "sealed_test_opened": False,
            "confirmatory_use": False,
            "warning": "Post-hoc mechanism diagnosis; no number here can relabel Step 95 or select its sealed confirmation.",
        },
        "selected_condition_locked_1500_runs": {
            "tau": tau,
            "thresholds": thresholds,
            "mean_terminal_delta": float(selected["mean_terminal_delta"]),
            "terminal_pp": 100.0 * float(selected["mean_terminal_delta"]),
            "root_change_rate": float(selected["final_root_change_rate"]),
            "positive_fraction": float(selected["inference"]["terminal"]["positive_fraction"]),
            "negative_fraction": float(selected["inference"]["terminal"]["negative_fraction"]),
            "zero_fraction": float(selected["inference"]["terminal"]["zero_fraction"]),
        },
        "trajectory_diagnostic_400_fixed_runs": {
            "seed_range": [diagnostic_seeds[0], diagnostic_seeds[-1]],
            "mean_terminal_delta": as_float(np.mean(delta)),
            "terminal_pp": as_float(100.0 * np.mean(delta)),
            "root_change_rate": as_float(np.mean(changed)),
        },
        "root_global_accuracy": {
            name: as_float(value) for name, value in zip(root_names, global_accuracies)
        },
        "root_global_gap_from_parent_pp": {
            name: as_float(100.0 * (parent_global - value))
            for name, value in zip(root_names, global_accuracies)
        },
        "pool_quality_geometry": {
            "best_minus_second": quantiles(best_minus_second_pool),
            "best_minus_worst": quantiles(counterfactual_worst),
            "fraction_best_second_tied": as_float(np.mean(best_minus_second_pool == 0)),
            "mean_best_minus_second_pp": as_float(100.0 * np.mean(best_minus_second_pool)),
            "mean_best_minus_worst_pp": as_float(100.0 * np.mean(counterfactual_worst)),
        },
        "terminal_delta_geometry": {
            "all_quantiles": quantiles(delta),
            "changed_quantiles": quantiles(changed_delta),
            "positive_quantiles": quantiles(positive),
            "negative_quantiles": quantiles(negative),
            "mean_positive": as_float(np.mean(positive)) if len(positive) else 0.0,
            "mean_negative": as_float(np.mean(negative)) if len(negative) else 0.0,
            "mean_given_root_change": as_float(np.mean(changed_delta)) if len(changed_delta) else 0.0,
            "conditional_drop_needed_for_half_point": needed_conditional,
            "conditional_drop_needed_pp": 100.0 * needed_conditional,
        },
        "terminal_transitions": transition_rows(clean_final, refined_final, root_quality, root_names),
        "terminal_score_geometry": {
            "clean_entry_margin_zero_fraction": as_float(np.mean(clean["entry_margin"][:, terminal_step] == 0)),
            "refined_entry_margin_zero_fraction": as_float(np.mean(refined["entry_margin"][:, terminal_step] == 0)),
            "clean_entry_margin_quantiles": quantiles(clean["entry_margin"][:, terminal_step]),
            "refined_entry_margin_quantiles": quantiles(refined["entry_margin"][:, terminal_step]),
        },
        "query_diagnostics": query_diagnostics,
        "budget_scan": budget_scan,
        "locked_grid_summary": grid_summary,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Step 95 post-hoc terminal-bottleneck diagnosis",
        "",
        "This report uses only the frozen `selector_verify` development partition. It does not open or inspect the sealed MultiNLI outcome, cannot relabel Step 95, and is not confirmatory evidence.",
        "",
        "## Reproduction",
        "",
        f"- Locked 1,500-run terminal effect: {100*float(selected['mean_terminal_delta']):+.5f} pp.",
        f"- Locked 1,500-run final-root change: {100*float(selected['final_root_change_rate']):.2f}%.",
        f"- The fixed diagnostic trajectory sample is seeds {diagnostic_seeds[0]}--{diagnostic_seeds[-1]} (400 runs).",
        "",
        "## Geometry",
        "",
        f"- Mean pool best-minus-second gap: {100*np.mean(best_minus_second_pool):.3f} pp; tied in {100*np.mean(best_minus_second_pool==0):.2f}% of pools.",
        f"- Mean pool best-minus-worst gap: {100*np.mean(counterfactual_worst):.3f} pp.",
        f"- Mean terminal delta conditional on a changed root: {100*np.mean(changed_delta):+.3f} pp.",
        f"- At the observed {100*np.mean(changed):.2f}% root-change rate, the +0.5 pp gate requires a net conditional drop of {100*needed_conditional:.3f} pp.",
        "",
        "## Query shift",
        "",
        f"- Parent-error fraction among clean/refined queries: {100*query_diagnostics['clean_parent_error_fraction']:.2f}% / {100*query_diagnostics['refined_parent_error_fraction']:.2f}%.",
        f"- Mean root-disagreement categories among clean/refined queries: {query_diagnostics['clean_root_disagreement_mean']:.3f} / {query_diagnostics['refined_root_disagreement_mean']:.3f}.",
        f"- Mean unique labels replaced per run: {query_diagnostics['mean_unique_refined_only']:.3f} of {step95.BUDGET}.",
        "",
        "## Budget diagnostic",
        "",
        "| Budget | Terminal pp | Root change | Cumulative delta |",
        "|---:|---:|---:|---:|",
    ]
    for row in budget_scan:
        lines.append(
            f"| {row['budget']} | {row['terminal_pp']:+.4f} | {100*row['root_change_rate']:.2f}% | {row['mean_cumulative_delta']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These post-hoc diagnostics identify mechanism bottlenecks only. Any follow-up must use a newly locked task/endpoint and a design justified without consulting that task's outcomes.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": OUTPUT.name, "report": REPORT.name, "terminal_pp": 100*np.mean(delta), "root_change_rate": np.mean(changed)}, indent=2))


if __name__ == "__main__":
    main()
