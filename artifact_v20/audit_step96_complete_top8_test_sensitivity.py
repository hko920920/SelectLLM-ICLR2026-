from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import step96_common as common


ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / f"STEP96_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
PREOUTCOME = ROOT / f"STEP96_CONFIRMATORY_PREOUTCOME_PREDICTIONS_{common.DATE}.npz"
SEALED = ROOT / "external_data" / "step96_sealed" / "snli_test_sealed_outcomes.npz"
PRIMARY = ROOT / f"STEP96_CONFIRMATORY_RESULTS_{common.DATE}.json"
RAW = ROOT / f"STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_RAW_{common.DATE}.npz"
OUTPUT = ROOT / f"STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_SENSITIVITY_{common.DATE}.json"
SEEDS = tuple(range(966000, 969000))


def canonical_key(row: dict[str, Any]) -> str:
    payload = {
        "tau": row["tau"],
        "thresholds": row["thresholds"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * float(pvalues[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def main() -> None:
    for path in (RAW, OUTPUT):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    primary = json.loads(PRIMARY.read_text(encoding="utf-8"))
    predictions = np.load(PREOUTCOME, allow_pickle=False)
    outcomes = np.load(SEALED, allow_pickle=False)
    if not np.array_equal(predictions["source_indices"], outcomes["source_indices"]):
        raise AssertionError("source-index mismatch")
    labels = outcomes["labels"].astype(np.int64)
    roots = predictions["root_predictions"].astype(np.int64)
    scores = predictions["error_scores"].astype(np.float64)
    pools = common.sample_pools(len(labels), SEEDS, common.POOL_SIZE)
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    rows = sorted(ledger["verification_top_8"], key=lambda row: int(row["search_rank"]))
    unique: dict[str, dict[str, Any]] = {}
    raw_arrays: dict[str, np.ndarray] = {
        "source_indices": predictions["source_indices"].astype(np.int64),
        "labels": labels.astype(np.int16),
        "pools": pools.astype(np.int32),
    }
    for row in rows:
        key = canonical_key(row)
        if key in unique:
            continue
        thresholds = np.asarray(row["thresholds"], dtype=np.float64)
        alias_codes, triggers = common.make_alias_codes(roots[:, 0], scores, thresholds)
        refined_codes = np.column_stack([roots, alias_codes])
        refined_parents = np.concatenate(
            [clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)]
        )
        tau = float(row["tau"])
        clean = common.run_active(
            roots, labels, clean_parents, roots, pools, SEEDS, common.BUDGET, tau
        )
        refined = common.run_active(
            refined_codes,
            labels,
            refined_parents,
            roots,
            pools,
            SEEDS,
            common.BUDGET,
            tau,
        )
        fixed = common.run_fixed(
            refined_codes, labels, refined_parents, roots, pools, clean.queries, SEEDS
        )
        terminal = refined.terminal - clean.terminal
        cumulative = refined.cumulative - clean.cumulative
        fixed_terminal = fixed.terminal - clean.terminal
        fixed_cumulative = fixed.cumulative - clean.cumulative
        active = terminal - fixed_terminal
        loss_counts = np.sum(
            triggers & (roots[:, :1] == labels[:, None]), axis=0
        ).astype(np.int64)
        allowed = int(math.floor(0.01 * len(labels) + 1e-12))
        clean_final = clean.roots[:, -1]
        refined_final = refined.roots[:, -1]
        parent_to = (clean_final == 0) & (refined_final != 0)
        to_parent = (clean_final != 0) & (refined_final == 0)
        rank_namespace = min(
            int(candidate["search_rank"])
            for candidate in rows
            if canonical_key(candidate) == key
        )
        inference = {
            "terminal": common.effect_summary(
                terminal, 98900 + rank_namespace * 10, 98901 + rank_namespace * 10
            ),
            "active_minus_fixed_terminal": common.effect_summary(
                active, 98901 + rank_namespace * 10, 98902 + rank_namespace * 10
            ),
            "cumulative": common.effect_summary(
                cumulative, 98902 + rank_namespace * 10, 98903 + rank_namespace * 10
            ),
        }
        root_accuracies = np.mean(roots == labels[:, None], axis=0)
        gates = {
            "parent_unique_best_by_2_5pp": bool(
                root_accuracies[0] - np.max(root_accuracies[1:]) >= .025
            ),
            "aliases_coordinate_wise_nonimproving": bool(
                np.all(
                    common.exact_match_feedback(alias_codes, labels)
                    <= np.repeat(
                        common.exact_match_feedback(roots[:, :1], labels),
                        common.N_ALIASES,
                        axis=1,
                    )
                )
            ),
            "quality_at_most_one_point": bool(np.all(loss_counts <= allowed)),
            "path_change_at_least_half": bool(
                np.mean(np.any(clean.queries != refined.queries, axis=1)) >= .5
            ),
            "clean_parent_terminal_at_least_60pct": bool(np.mean(clean_final == 0) >= .6),
            "parent_to_challenger_exceeds_reverse": bool(np.mean(parent_to) > np.mean(to_parent)),
            "terminal_mean_at_least_half_point": inference["terminal"]["mean"] >= .005,
            "terminal_inference": bool(
                inference["terminal"]["bootstrap_95"][0] > 0
                and inference["terminal"]["one_sided_signflip_p"] <= .05
            ),
            "active_mean_at_least_half_point": inference["active_minus_fixed_terminal"]["mean"] >= .005,
            "active_inference": bool(
                inference["active_minus_fixed_terminal"]["bootstrap_95"][0] > 0
                and inference["active_minus_fixed_terminal"]["one_sided_signflip_p"] <= .05
            ),
            "cumulative_positive": bool(
                inference["cumulative"]["mean"] > 0
                and inference["cumulative"]["bootstrap_95"][0] > 0
            ),
            "fixed_query_exact_zero": bool(
                np.max(np.abs(fixed_terminal)) == 0
                and np.max(np.abs(fixed_cumulative)) == 0
                and np.array_equal(fixed.roots, clean.roots)
            ),
        }
        unique[key] = {
            "canonical_key": key,
            "representative_search_rank": rank_namespace,
            "tau": tau,
            "loss_cap": float(row["loss_cap"]),
            "trigger_caps_with_identical_thresholds": sorted(
                float(candidate["trigger_cap"])
                for candidate in rows
                if canonical_key(candidate) == key
            ),
            "thresholds": row["thresholds"],
            "loss_counts": loss_counts.tolist(),
            "allowed_loss_count": allowed,
            "loss_fractions": (loss_counts / len(labels)).tolist(),
            "trigger_fractions": np.mean(triggers, axis=0).tolist(),
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
            "inference": inference,
            "gates_before_multiplicity": gates,
            "all_gates_before_multiplicity": all(gates.values()),
        }
        prefix = f"condition_{rank_namespace}"
        raw_arrays[f"{prefix}_terminal"] = terminal.astype(np.float64)
        raw_arrays[f"{prefix}_active"] = active.astype(np.float64)
        raw_arrays[f"{prefix}_cumulative"] = cumulative.astype(np.float64)
        raw_arrays[f"{prefix}_fixed_terminal"] = fixed_terminal.astype(np.float64)
        raw_arrays[f"{prefix}_fixed_cumulative"] = fixed_cumulative.astype(np.float64)
        raw_arrays[f"{prefix}_clean_queries"] = clean.queries.astype(np.int32)
        raw_arrays[f"{prefix}_refined_queries"] = refined.queries.astype(np.int32)
        raw_arrays[f"{prefix}_clean_roots"] = clean.roots.astype(np.int16)
        raw_arrays[f"{prefix}_refined_roots"] = refined.roots.astype(np.int16)
        raw_arrays[f"{prefix}_fixed_roots"] = fixed.roots.astype(np.int16)
        print(
            f"unique condition rank={rank_namespace} loss_cap={row['loss_cap']} "
            f"terminal={100*inference['terminal']['mean']:+.3f}pp quality={gates['quality_at_most_one_point']}",
            flush=True,
        )
    unique_rows = sorted(unique.values(), key=lambda row: row["representative_search_rank"])
    for endpoint in ("terminal", "active_minus_fixed_terminal"):
        adjusted = holm(
            [row["inference"][endpoint]["one_sided_signflip_p"] for row in unique_rows]
        )
        for row, qvalue in zip(unique_rows, adjusted):
            row["inference"][endpoint]["holm_q_across_unique_top8_conditions"] = qvalue
    for row in unique_rows:
        row["all_gates_with_holm"] = bool(
            row["all_gates_before_multiplicity"]
            and row["inference"]["terminal"]["holm_q_across_unique_top8_conditions"] <= .05
            and row["inference"]["active_minus_fixed_terminal"]["holm_q_across_unique_top8_conditions"] <= .05
        )
    np.savez_compressed(RAW, **raw_arrays)
    result = {
        "audit_id": "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_SENSITIVITY_V1",
        "date": common.DATE,
        "status": "POSTCONFIRMATORY_EXPLORATORY_NOT_PRIMARY_CONFIRMATION",
        "primary_decision_unchanged": primary["decision"],
        "primary_failed_gates_unchanged": primary["failed_gates"],
        "scope": {
            "all_predeclared_top8_conditions_included": True,
            "nominal_rows": len(rows),
            "unique_tau_threshold_conditions": len(unique_rows),
            "models_adapters_thresholds_and_test_predictions_all_fixed_before_test_open": True,
            "analysis_authorized_only_after_primary_test_open": True,
            "confirmatory_status": False,
        },
        "bound_sha256": {
            "stagea_ledger": common.sha256_path(LEDGER),
            "preoutcome_predictions": common.sha256_path(PREOUTCOME),
            "sealed_outcome": common.sha256_path(SEALED),
            "primary_results": common.sha256_path(PRIMARY),
            "raw": common.sha256_path(RAW),
            "audit_code": common.sha256_path(Path(__file__)),
        },
        "unique_conditions": unique_rows,
        "nominal_top8_mapping": [
            {
                "search_rank": int(row["search_rank"]),
                "loss_cap": float(row["loss_cap"]),
                "trigger_cap": float(row["trigger_cap"]),
                "canonical_key": canonical_key(row),
            }
            for row in rows
        ],
        "interpretation": "The locked primary remains NO-GO. This complete, multiplicity-controlled sensitivity reports whether other already-developed top-eight thresholds would satisfy the same scientific gates; it is not a replacement confirmatory selection.",
    }
    common.json_dump(OUTPUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
