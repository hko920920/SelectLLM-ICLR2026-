from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import sparse

import validate_step92_independent as independent


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-12"
RAW_PATH = ROOT / f"STEP92_LEARNED_ADAPTER_CONFIRMATORY_RAW_{DATE}.npz"
CONFIG_PATH = ROOT / f"STEP92_STAGEA_FROZEN_CONFIG_{DATE}.json"
OUT_RAW = ROOT / f"STEP92_NUMERIC_TIE_ROBUSTNESS_RAW_{DATE}.npz"
OUT_JSON = ROOT / f"STEP92_NUMERIC_TIE_ROBUSTNESS_{DATE}.json"


def group(codes: np.ndarray, ordering: str) -> independent.Grouping:
    pool_size, entries = codes.shape
    row_ids = np.empty_like(codes, dtype=np.int64)
    query_ids: list[int] = []
    offset = 0
    for query in range(pool_size):
        observed = [int(value) for value in codes[query]]
        if ordering == "sorted_code":
            unique = sorted(set(observed))
        elif ordering == "reverse_code":
            unique = sorted(set(observed), reverse=True)
        elif ordering == "first_appearance":
            unique = list(dict.fromkeys(observed))
        else:
            raise ValueError(ordering)
        mapping = {code: offset + rank for rank, code in enumerate(unique)}
        for entry, code in enumerate(observed):
            row_ids[query, entry] = mapping[code]
        query_ids.extend([query] * len(unique))
        offset += len(unique)
    rows = row_ids.reshape(-1)
    columns = np.tile(np.arange(entries), pool_size)
    membership = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(offset, entries)
    )
    return independent.Grouping(membership, np.asarray(query_ids, dtype=np.int64))


def replay(
    codes: np.ndarray,
    feedback: np.ndarray,
    parents: np.ndarray,
    core_feedback: np.ndarray,
    pools: np.ndarray,
    tau: float,
    ordering: str,
    tie_atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    queries = np.full((independent.N_RUNS, independent.BUDGET), -1, dtype=np.int64)
    roots = np.full_like(queries, -1)
    terminal = np.zeros(independent.N_RUNS, dtype=np.float64)
    cumulative = np.zeros(independent.N_RUNS, dtype=np.float64)
    for run_index, seed in enumerate(independent.SEEDS):
        pool = pools[run_index]
        grouping = group(codes[pool], ordering)
        available = np.ones(len(pool), dtype=bool)
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        quality = np.mean(core_feedback[pool], axis=0)
        best = float(np.max(quality))
        for step in range(independent.BUDGET):
            shifted = scores / tau
            shifted -= float(np.max(shifted))
            posterior = np.exp(shifted)
            posterior /= float(np.sum(posterior))
            values = independent.acquisition(grouping, posterior, len(pool))
            values[~available] = np.inf
            minimum = float(np.min(values))
            if tie_atol == 0.0:
                tied = np.flatnonzero(values == minimum)
            else:
                tied = np.flatnonzero(values <= minimum + tie_atol)
            position = independent.choose_query(tied, pool, seed, step)
            available[position] = False
            query = int(pool[position])
            queries[run_index, step] = query
            scores += feedback[query]
            root = independent.choose_root(
                np.flatnonzero(scores == float(np.max(scores))), parents, seed, step
            )
            roots[run_index, step] = root
            regret = best - float(quality[root])
            cumulative[run_index] += regret
            if step == independent.BUDGET - 1:
                terminal[run_index] = regret
    return queries, roots, terminal, cumulative


def main() -> None:
    raw = np.load(RAW_PATH, allow_pickle=False)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = int(config["selected"]["parent_root"])
    tau = float(config["selected"]["tau"])
    labels = raw["labels"].astype(np.int8)
    predictions = raw["clean_root_predictions"].astype(np.int8)
    clean_codes = predictions.astype(np.int64)
    core_feedback = (predictions == labels[:, None]).astype(np.float64)
    clean_feedback = core_feedback.copy()
    clean_parents = np.arange(independent.N_ROOTS, dtype=np.int64)
    refined_codes = np.column_stack([clean_codes, raw["adapter_response_codes"]]).astype(np.int64)
    refined_feedback = np.column_stack(
        [core_feedback, np.repeat(core_feedback[:, [parent]], independent.N_ALIASES, axis=1)]
    )
    refined_parents = np.concatenate(
        [clean_parents, np.full(independent.N_ALIASES, parent, dtype=np.int64)]
    )
    policies = {
        "canonical_sorted_exact": ("sorted_code", 0.0),
        "first_appearance_exact": ("first_appearance", 0.0),
        "reverse_code_exact": ("reverse_code", 0.0),
        "sorted_tie_atol_1e-12": ("sorted_code", 1e-12),
    }
    arrays: dict[str, np.ndarray] = {}
    rows: dict[str, dict[str, object]] = {}
    for index, (name, (ordering, tie_atol)) in enumerate(policies.items()):
        if name == "canonical_sorted_exact":
            clean_queries = raw["clean_queries"]
            refined_queries = raw["refined_queries"]
            clean_roots = raw["clean_roots"]
            refined_roots = raw["refined_roots"]
            terminal_delta = raw["terminal_delta"]
            cumulative_delta = raw["cumulative_delta"]
        else:
            clean_queries, clean_roots, clean_terminal, clean_cumulative = replay(
                clean_codes,
                clean_feedback,
                clean_parents,
                core_feedback,
                raw["pools"],
                tau,
                ordering,
                tie_atol,
            )
            refined_queries, refined_roots, refined_terminal, refined_cumulative = replay(
                refined_codes,
                refined_feedback,
                refined_parents,
                core_feedback,
                raw["pools"],
                tau,
                ordering,
                tie_atol,
            )
            terminal_delta = refined_terminal - clean_terminal
            cumulative_delta = refined_cumulative - clean_cumulative
        fixed_clean_roots, fixed_clean_terminal, fixed_clean_cumulative = independent.replay_fixed(
            clean_feedback, clean_parents, core_feedback, raw["pools"], clean_queries
        )
        fixed_refined_roots, fixed_refined_terminal, fixed_refined_cumulative = independent.replay_fixed(
            refined_feedback, refined_parents, core_feedback, raw["pools"], clean_queries
        )
        fixed_terminal = fixed_refined_terminal - fixed_clean_terminal
        fixed_cumulative = fixed_refined_cumulative - fixed_clean_cumulative
        if not (
            np.array_equal(fixed_clean_roots, fixed_refined_roots)
            and np.all(fixed_terminal == 0.0)
            and np.all(fixed_cumulative == 0.0)
        ):
            raise AssertionError(f"{name}: fixed-query invariant failed")
        ci = independent.bootstrap(terminal_delta, 923_000 + index * 10)
        pvalue = independent.signflip(terminal_delta, 923_001 + index * 10)
        mean = float(np.mean(terminal_delta))
        rows[name] = {
            "group_order": ordering,
            "query_tie_atol": tie_atol,
            "mean_terminal_delta": mean,
            "mean_terminal_delta_pp": 100.0 * mean,
            "bootstrap_95": ci,
            "bootstrap_95_pp": [100.0 * value for value in ci],
            "one_sided_signflip_p": pvalue,
            "path_change_rate": float(
                np.mean(np.any(clean_queries != refined_queries, axis=1))
            ),
            "clean_path_difference_from_canonical": float(
                np.mean(np.any(clean_queries != raw["clean_queries"], axis=1))
            ),
            "refined_path_difference_from_canonical": float(
                np.mean(np.any(refined_queries != raw["refined_queries"], axis=1))
            ),
            "fixed_query_exact_zero": True,
            "passes_original_harm_gate": bool(
                mean >= 0.005 and ci[0] > 0.0 and pvalue <= 0.05
            ),
        }
        for suffix, values in {
            "clean_queries": clean_queries,
            "refined_queries": refined_queries,
            "clean_roots": clean_roots,
            "refined_roots": refined_roots,
            "terminal_delta": terminal_delta,
            "cumulative_delta": cumulative_delta,
            "fixed_terminal_delta": fixed_terminal,
            "fixed_cumulative_delta": fixed_cumulative,
        }.items():
            arrays[f"{name}__{suffix}"] = values
        print(name, rows[name], flush=True)

    all_pass = all(bool(row["passes_original_harm_gate"]) for row in rows.values())
    np.savez_compressed(OUT_RAW, **arrays)
    result = {
        "audit_id": "STEP92_NUMERIC_TIE_ROBUSTNESS_V1",
        "status": "post-confirmatory robustness audit; not preregistered and does not replace the primary result",
        "primary_raw_sha256": independent.sha256_path(RAW_PATH),
        "config_sha256": independent.sha256_path(CONFIG_PATH),
        "decision": "PASS_NUMERIC_TIE_ROBUSTNESS" if all_pass else "NUMERIC_TIE_SENSITIVITY_DETECTED",
        "all_policies_pass_original_harm_gate": all_pass,
        "policies": rows,
        "raw_sha256": independent.sha256_path(OUT_RAW),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
