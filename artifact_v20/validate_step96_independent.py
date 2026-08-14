from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import step96_common as common


ROOT = Path(__file__).resolve().parent
LOCK = ROOT / f"STEP96_CONFIRMATORY_EXECUTION_LOCK_{common.DATE}.json"
PREOUTCOME = ROOT / f"STEP96_CONFIRMATORY_PREOUTCOME_PREDICTIONS_{common.DATE}.npz"
PRELEDGER = ROOT / f"STEP96_CONFIRMATORY_PREOUTCOME_LEDGER_{common.DATE}.json"
SEALED = ROOT / "external_data" / "step96_sealed" / "snli_test_sealed_outcomes.npz"
RAW = ROOT / f"STEP96_CONFIRMATORY_RAW_{common.DATE}.npz"
RESULTS = ROOT / f"STEP96_CONFIRMATORY_RESULTS_{common.DATE}.json"
STAGEA_ARRAYS = ROOT / f"STEP96_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
STAGEA_LEDGER = ROOT / f"STEP96_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
REPAIR = ROOT / f"STEP96_INTEGER_QUALITY_GATE_REPAIR_RECEIPT_{common.DATE}.json"
OUTPUT = ROOT / f"STEP96_CONFIRMATORY_INDEPENDENT_VALIDATION_{common.DATE}.json"


def close(left: Any, right: Any, atol: float = 1e-14) -> bool:
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=0.0, atol=atol))


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    preledger = json.loads(PRELEDGER.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    stagea = json.loads(STAGEA_LEDGER.read_text(encoding="utf-8"))
    repair = json.loads(REPAIR.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["lock_bound_preoutcome"] = preledger["lock_sha256"] == common.sha256_path(LOCK)
    checks["preoutcome_hash"] = preledger["output_sha256"] == common.sha256_path(PREOUTCOME)
    checks["sealed_hash"] = lock["sealed_outcome"]["sha256"] == common.sha256_path(SEALED)
    checks["raw_hash"] = results["raw_sha256"] == common.sha256_path(RAW)
    checks["stagea_repaired_go"] = (
        stagea["decision"] == "GO_TO_STEP96_ONE_TIME_CONFIRMATORY_LOCK"
        and not stagea["failed_gates"]
        and all(stagea["gates"].values())
    )
    checks["integer_repair_bound"] = (
        repair["repaired_ledger_sha256"] == common.sha256_path(STAGEA_LEDGER)
        and repair["selected_condition_unchanged"]
        and not repair["test_opened_before_repair"]
    )

    stage_arrays = np.load(STAGEA_ARRAYS, allow_pickle=False)
    selected = lock["selected_condition"]
    thresholds = np.asarray(selected["thresholds"], dtype=np.float64)
    stage_triggers = stage_arrays["error_scores"] > thresholds[None, :]
    stage_labels = stage_arrays["labels"].astype(np.int64)
    stage_parent = stage_arrays["root_predictions"][:, 0].astype(np.int64)
    reconstructed_counts = {}
    stage_quality_pass = True
    for name, key, cap in (
        ("threshold", "threshold_indices", 0.01),
        ("selector_search", "search_indices", 0.01),
        ("selector_verify", "verify_indices", 0.01),
    ):
        indices = stage_arrays[key].astype(np.int64)
        loss_counts = np.sum(
            stage_triggers[indices]
            & (stage_parent[indices, None] == stage_labels[indices, None]),
            axis=0,
        ).astype(np.int64)
        allowed = int(math.floor(cap * len(indices) + 1e-12))
        stage_quality_pass = stage_quality_pass and bool(np.all(loss_counts <= allowed))
        reconstructed_counts[name] = {
            "loss_counts": loss_counts.tolist(),
            "allowed": allowed,
        }
    checks["stagea_integer_quality_reconstruction"] = (
        stage_quality_pass and reconstructed_counts == {
            name: {
                "loss_counts": row["loss_counts"],
                "allowed": row["allowed_count"],
            }
            for name, row in repair["counts"].items()
        }
    )

    pre = np.load(PREOUTCOME, allow_pickle=False)
    sealed = np.load(SEALED, allow_pickle=False)
    raw = np.load(RAW, allow_pickle=False)
    checks["source_indices_align"] = (
        np.array_equal(pre["source_indices"], sealed["source_indices"])
        and np.array_equal(pre["source_indices"], raw["source_indices"])
    )
    checks["preoutcome_contains_no_label_array"] = "labels" not in pre.files
    checks["raw_predictions_equal_preoutcome"] = (
        np.array_equal(raw["root_predictions"], pre["root_predictions"])
        and np.array_equal(raw["error_scores"], pre["error_scores"])
    )
    checks["raw_labels_equal_sealed"] = np.array_equal(raw["labels"], sealed["labels"])

    labels = raw["labels"].astype(np.int64)
    roots = raw["root_predictions"].astype(np.int64)
    scores = raw["error_scores"].astype(np.float64)
    alias_codes, triggers = common.make_alias_codes(roots[:, 0], scores, thresholds)
    checks["alias_code_reconstruction"] = (
        np.array_equal(alias_codes, raw["alias_codes"])
        and np.array_equal(triggers.astype(np.uint8), raw["triggers"])
    )
    seed_info = lock["test_trajectory_seeds"]
    seeds = tuple(range(int(seed_info["start"]), int(seed_info["stop_exclusive"])))
    pools = common.sample_pools(len(labels), seeds, common.POOL_SIZE)
    checks["pool_reconstruction"] = np.array_equal(pools, raw["pools"])
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    refined_codes = np.column_stack([roots, alias_codes])
    refined_parents = np.concatenate(
        [clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)]
    )
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
    checks["trajectory_reconstruction"] = (
        np.array_equal(clean.queries, raw["clean_queries"])
        and np.array_equal(refined.queries, raw["refined_queries"])
        and np.array_equal(clean.roots, raw["clean_roots"])
        and np.array_equal(refined.roots, raw["refined_roots"])
        and np.array_equal(fixed.roots, raw["fixed_roots"])
    )
    checks["effect_vector_reconstruction"] = (
        np.array_equal(terminal, raw["terminal_delta"])
        and np.array_equal(cumulative, raw["cumulative_delta"])
        and np.array_equal(fixed_terminal, raw["fixed_terminal_delta"])
        and np.array_equal(fixed_cumulative, raw["fixed_cumulative_delta"])
        and np.array_equal(active, raw["active_minus_fixed_terminal_delta"])
    )
    inf = lock["inference_seeds"]
    summaries = {
        "terminal": common.effect_summary(
            terminal,
            int(inf["terminal"]["bootstrap"]),
            int(inf["terminal"]["signflip"]),
        ),
        "active_minus_fixed_terminal": common.effect_summary(
            active,
            int(inf["active_minus_fixed_terminal"]["bootstrap"]),
            int(inf["active_minus_fixed_terminal"]["signflip"]),
        ),
        "cumulative": common.effect_summary(
            cumulative,
            int(inf["cumulative"]["bootstrap"]),
            int(inf["cumulative"]["signflip"]),
        ),
    }
    checks["inference_reconstruction"] = all(
        close(summaries[name]["mean"], results["inference"][name]["mean"])
        and close(summaries[name]["median"], results["inference"][name]["median"])
        and close(summaries[name]["bootstrap_95"], results["inference"][name]["bootstrap_95"])
        and close(summaries[name]["one_sided_signflip_p"], results["inference"][name]["one_sided_signflip_p"])
        for name in summaries
    )
    root_accuracies = np.mean(roots == labels[:, None], axis=0)
    parent_gap = float(root_accuracies[0] - np.max(root_accuracies[1:]))
    loss_counts = np.sum(triggers & (roots[:, :1] == labels[:, None]), axis=0).astype(np.int64)
    allowed = int(math.floor(0.01 * len(labels) + 1e-12))
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    parent_to = (clean_final == 0) & (refined_final != 0)
    to_parent = (clean_final != 0) & (refined_final == 0)
    gates = {
        "parent_unique_best_by_2_5pp": parent_gap >= 0.025,
        "aliases_coordinate_wise_nonimproving": bool(
            np.all(
                common.exact_match_feedback(alias_codes, labels)
                <= np.repeat(common.exact_match_feedback(roots[:, :1], labels), common.N_ALIASES, axis=1)
            )
        ),
        "alias_integer_loss_counts_at_most_one_point": bool(np.all(loss_counts <= allowed)),
        "path_change_at_least_half": bool(np.mean(np.any(clean.queries != refined.queries, axis=1)) >= .5),
        "clean_parent_terminal_rate_at_least_60pct": bool(np.mean(clean_final == 0) >= .6),
        "parent_to_challenger_exceeds_reverse": bool(np.mean(parent_to) > np.mean(to_parent)),
        "terminal_mean_at_least_half_point": summaries["terminal"]["mean"] >= .005,
        "terminal_lower_positive_and_p_at_most_point05": summaries["terminal"]["bootstrap_95"][0] > 0 and summaries["terminal"]["one_sided_signflip_p"] <= .05,
        "active_minus_fixed_mean_at_least_half_point": summaries["active_minus_fixed_terminal"]["mean"] >= .005,
        "active_lower_positive_and_p_at_most_point05": summaries["active_minus_fixed_terminal"]["bootstrap_95"][0] > 0 and summaries["active_minus_fixed_terminal"]["one_sided_signflip_p"] <= .05,
        "cumulative_mean_and_lower_positive": summaries["cumulative"]["mean"] > 0 and summaries["cumulative"]["bootstrap_95"][0] > 0,
        "fixed_query_exact_zero": bool(np.max(np.abs(fixed_terminal)) == 0 and np.max(np.abs(fixed_cumulative)) == 0 and np.array_equal(fixed.roots, clean.roots)),
    }
    checks["gate_reconstruction"] = gates == results["gates"]
    failed = [name for name, passed in gates.items() if not passed]
    checks["literal_decision_reconstruction"] = (
        failed == ["alias_integer_loss_counts_at_most_one_point"]
        and results["failed_gates"] == failed
        and results["decision"] == "NO_GO_RETAIN_STEP96_NEGATIVE"
    )
    if not all(checks.values()):
        raise AssertionError({name: value for name, value in checks.items() if not value})
    validation = {
        "validation_id": "STEP96_CONFIRMATORY_INDEPENDENT_RECONSTRUCTION_V1",
        "date": common.DATE,
        "status": "PASS_STEP96_INDEPENDENT_VALIDATION",
        "checks": checks,
        "reconstructed": {
            "decision": results["decision"],
            "failed_gates": failed,
            "terminal": summaries["terminal"],
            "active_minus_fixed_terminal": summaries["active_minus_fixed_terminal"],
            "cumulative": summaries["cumulative"],
            "loss_counts": loss_counts.tolist(),
            "allowed_loss_count": allowed,
            "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
            "fixed_query_exact": True,
        },
        "bound_sha256": {
            "lock": common.sha256_path(LOCK),
            "preoutcome": common.sha256_path(PREOUTCOME),
            "sealed": common.sha256_path(SEALED),
            "raw": common.sha256_path(RAW),
            "results": common.sha256_path(RESULTS),
            "stagea_ledger": common.sha256_path(STAGEA_LEDGER),
            "repair_receipt": common.sha256_path(REPAIR),
            "validator": common.sha256_path(Path(__file__)),
        },
    }
    common.json_dump(OUTPUT, validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
