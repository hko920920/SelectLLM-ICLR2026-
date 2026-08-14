"""Independent Step 90 validator.  Imports no Step 89/90 runner module."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-12"
LOCK_PATH = ROOT / f"STEP90_HIGH_CONFIDENCE_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json"
CONFIG_PATH = ROOT / f"STEP90_HIGH_CONFIDENCE_STAGEA_FROZEN_CONFIG_{DATE}.json"
RAW_PATH = ROOT / f"STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_RAW_{DATE}.npz"
RESULT_PATH = ROOT / f"STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_RESULTS_{DATE}.json"
REPORT_PATH = ROOT / f"STEP90_HIGH_CONFIDENCE_INDEPENDENT_VALIDATION_{DATE}.json"
SEEDS = tuple(range(1000))
BOOTSTRAP_REPS = 50_000
SIGNFLIP_REPS = 100_000
MINIMUM_DELTA = 0.005
CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()[:8],
        "big",
    )


def choose_query(tied: np.ndarray, pool: np.ndarray, seed: int, step: int) -> int:
    return min(
        (int(position) for position in tied),
        key=lambda position: (
            stable_u64("step89", "query", seed, step, int(pool[position])),
            int(pool[position]),
        ),
    )


def choose_root(tied: np.ndarray, parents: np.ndarray, seed: int, step: int) -> int:
    roots = sorted(set(int(parents[int(entry)]) for entry in tied))
    return min(
        roots,
        key=lambda root: (stable_u64("step89", "root", seed, step, root), root),
    )


def group_structure(codes: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    pool_size, entries = codes.shape
    group_indices = np.empty((pool_size, entries), dtype=np.int64)
    offset = 0
    group_query: list[int] = []
    for query in range(pool_size):
        _, inverse = np.unique(codes[query], return_inverse=True)
        group_indices[query] = inverse + offset
        groups = int(np.max(inverse)) + 1
        group_query.extend([query] * groups)
        offset += groups
    rows = group_indices.ravel()
    columns = np.tile(np.arange(entries, dtype=np.int64), pool_size)
    membership = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(offset, entries),
    )
    return membership, np.asarray(group_query, dtype=np.int64)


def acquisition(
    membership: sparse.csr_matrix,
    group_query: np.ndarray,
    pool_size: int,
    posterior: np.ndarray,
) -> np.ndarray:
    mass = np.asarray(membership @ posterior).reshape(-1)
    return np.bincount(
        group_query,
        weights=mass * mass,
        minlength=pool_size,
    )


def reconstruct_active(
    codes: np.ndarray,
    feedback: np.ndarray,
    parents: np.ndarray,
    core_feedback: np.ndarray,
    pool: np.ndarray,
    seed: int,
    budget: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pool_codes = codes[pool]
    active = np.ones(len(pool), dtype=bool)
    cumulative = np.zeros(codes.shape[1], dtype=np.float64)
    quality = np.mean(core_feedback[pool], axis=0)
    best = float(np.max(quality))
    queries = np.full(budget, -1, dtype=np.int64)
    roots = np.full(budget, -1, dtype=np.int64)
    regrets = np.zeros(budget, dtype=np.float64)
    membership, group_query = group_structure(pool_codes)
    for step in range(budget):
        shifted = cumulative / tau
        shifted -= float(np.max(shifted))
        posterior = np.exp(shifted)
        posterior /= float(np.sum(posterior))
        values = acquisition(membership, group_query, len(pool), posterior)
        values[~active] = np.inf
        minimum = float(np.min(values))
        position = choose_query(np.flatnonzero(values == minimum), pool, seed, step)
        active[position] = False
        query = int(pool[position])
        queries[step] = query
        cumulative += feedback[query]
        root = choose_root(
            np.flatnonzero(cumulative == float(np.max(cumulative))),
            parents,
            seed,
            step,
        )
        roots[step] = root
        regrets[step] = best - float(quality[root])
    return queries, roots, regrets


def reconstruct_fixed(
    feedback: np.ndarray,
    parents: np.ndarray,
    core_feedback: np.ndarray,
    pool: np.ndarray,
    seed: int,
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    quality = np.mean(core_feedback[pool], axis=0)
    best = float(np.max(quality))
    cumulative = np.zeros(feedback.shape[1], dtype=np.float64)
    roots = np.full(len(queries), -1, dtype=np.int64)
    regrets = np.zeros(len(queries), dtype=np.float64)
    for step, query in enumerate(queries):
        cumulative += feedback[int(query)]
        root = choose_root(
            np.flatnonzero(cumulative == float(np.max(cumulative))),
            parents,
            seed,
            step,
        )
        roots[step] = root
        regrets[step] = best - float(quality[root])
    return roots, regrets


def bootstrap(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPS
    while remaining:
        count = min(2000, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        blocks.append(np.mean(values[indices], axis=1))
        remaining -= count
    return [float(value) for value in np.quantile(np.concatenate(blocks), [0.025, 0.975])]


def signflip(values: np.ndarray, seed: int) -> float:
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = SIGNFLIP_REPS
    while remaining:
        count = min(2000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        exceed += int(np.sum(np.mean(signs * values[None, :], axis=1) >= observed))
        remaining -= count
    return float((exceed + 1) / (SIGNFLIP_REPS + 1))


def close(left: Any, right: Any, tolerance: float = 1e-14) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda task: (pvalues[task], task))
    result: dict[str, float] = {}
    running = 0.0
    for rank, task in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[task]))
        result[task] = running
    return result


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    reported = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    raw = np.load(RAW_PATH, allow_pickle=False)
    for relative, expected in lock["locked_sha256"].items():
        check(sha256_path(ROOT / relative) == expected, f"locked hash mismatch: {relative}")
    check(sha256_path(RAW_PATH) == reported["raw_sha256"], "raw result hash mismatch")
    check(sha256_path(CONFIG_PATH) == reported["config_sha256"], "config hash mismatch")
    check(sha256_path(LOCK_PATH) == reported["execution_lock_sha256"], "execution lock hash mismatch")

    pvalues: dict[str, float] = {}
    reconstructed: dict[str, Any] = {}
    for task_index, task in enumerate(("mnli", "qqp")):
        spec = config["tasks"][task]
        selected = spec["selected"]
        labels = raw[f"{task}__labels"].astype(np.int8)
        official = raw[f"{task}__official_predictions"].astype(np.int8)
        hard = raw[f"{task}__parent_hard"].astype(np.int8)
        confidence = raw[f"{task}__parent_confidence"].astype(np.float64)
        aliases = raw[f"{task}__adapter_response_codes"].astype(np.int64)
        pools = raw[f"{task}__pools"].astype(np.int64)
        clean_queries = raw[f"{task}__clean_queries"].astype(np.int64)
        refined_queries = raw[f"{task}__refined_queries"].astype(np.int64)
        clean_roots = raw[f"{task}__clean_roots"].astype(np.int64)
        refined_roots = raw[f"{task}__refined_roots"].astype(np.int64)
        stored_terminal = raw[f"{task}__terminal_delta"].astype(np.float64)
        stored_cumulative = raw[f"{task}__cumulative_delta"].astype(np.float64)
        stored_fixed_terminal = raw[f"{task}__fixed_terminal_delta"].astype(np.float64)
        stored_fixed_cumulative = raw[f"{task}__fixed_cumulative_delta"].astype(np.float64)
        stored_active_fixed = raw[f"{task}__active_minus_fixed_terminal"].astype(np.float64)

        roster = np.asarray(selected["roster_indices"], dtype=np.int64)
        original = np.column_stack([official[:, roster], hard]).astype(np.int8)
        core_feedback = (original == labels[:, None]).astype(np.float64)
        clean_codes = original.astype(np.int64)
        clean_feedback = core_feedback.copy()
        clean_parents = np.arange(original.shape[1], dtype=np.int64)
        classes = int(spec["classes"])
        decoded = aliases.copy()
        mask = decoded >= classes
        decoded[mask] = (decoded[mask] - classes) // 4
        for alias in range(4):
            check(np.array_equal(decoded[:, alias], hard), f"{task}: adapter {alias} changes hard answer")
        triggered = confidence >= float(selected["threshold"])
        for alias in range(4):
            expected_codes = hard.astype(np.int64).copy()
            expected_codes[triggered] = classes + hard[triggered].astype(np.int64) * 4 + alias
            check(np.array_equal(aliases[:, alias], expected_codes), f"{task}: adapter code mismatch {alias}")
        refined_codes = np.column_stack([clean_codes, aliases])
        refined_feedback = np.column_stack([clean_feedback, np.repeat(core_feedback[:, [-1]], 4, axis=1)])
        parent_root = clean_codes.shape[1] - 1
        refined_parents = np.concatenate([clean_parents, np.full(4, parent_root, dtype=np.int64)])
        check(np.array_equal(refined_feedback[:, -4:], np.repeat(core_feedback[:, [-1]], 4, axis=1)), f"{task}: utility inequality")
        expected_pools = np.asarray(
            [sorted(random.Random(seed).sample(range(len(labels)), 400)) for seed in SEEDS],
            dtype=np.int64,
        )
        check(np.array_equal(pools, expected_pools), f"{task}: pool reconstruction failed")

        replay_terminal = np.zeros(len(SEEDS), dtype=np.float64)
        replay_cumulative = np.zeros(len(SEEDS), dtype=np.float64)
        replay_fixed_terminal = np.zeros(len(SEEDS), dtype=np.float64)
        replay_fixed_cumulative = np.zeros(len(SEEDS), dtype=np.float64)
        for run_index, seed in enumerate(SEEDS):
            pool = pools[run_index]
            cq, cr, cregret = reconstruct_active(
                clean_codes,
                clean_feedback,
                clean_parents,
                core_feedback,
                pool,
                seed,
                int(selected["budget"]),
                float(selected["tau"]),
            )
            rq, rr, rregret = reconstruct_active(
                refined_codes,
                refined_feedback,
                refined_parents,
                core_feedback,
                pool,
                seed,
                int(selected["budget"]),
                float(selected["tau"]),
            )
            check(np.array_equal(cq, clean_queries[run_index]), f"{task}/{seed}: clean query replay")
            check(np.array_equal(rq, refined_queries[run_index]), f"{task}/{seed}: refined query replay")
            check(np.array_equal(cr, clean_roots[run_index]), f"{task}/{seed}: clean root replay")
            check(np.array_equal(rr, refined_roots[run_index]), f"{task}/{seed}: refined root replay")
            cfr, cfregret = reconstruct_fixed(
                clean_feedback, clean_parents, core_feedback, pool, seed, cq
            )
            rfr, rfregret = reconstruct_fixed(
                refined_feedback, refined_parents, core_feedback, pool, seed, cq
            )
            check(np.array_equal(cfr, rfr), f"{task}/{seed}: fixed-query root mismatch")
            check(np.array_equal(cfregret, rfregret), f"{task}/{seed}: fixed-query regret mismatch")
            replay_terminal[run_index] = rregret[-1] - cregret[-1]
            replay_cumulative[run_index] = float(np.sum(rregret) - np.sum(cregret))
            replay_fixed_terminal[run_index] = rfregret[-1] - cfregret[-1]
            replay_fixed_cumulative[run_index] = float(np.sum(rfregret) - np.sum(cfregret))
        check(np.array_equal(replay_terminal, stored_terminal), f"{task}: terminal array")
        check(np.allclose(replay_cumulative, stored_cumulative, rtol=0.0, atol=1e-14), f"{task}: cumulative array")
        check(np.array_equal(replay_fixed_terminal, stored_fixed_terminal), f"{task}: fixed terminal array")
        check(np.array_equal(replay_fixed_cumulative, stored_fixed_cumulative), f"{task}: fixed cumulative array")
        check(np.array_equal(stored_active_fixed, stored_terminal - stored_fixed_terminal), f"{task}: active-minus-fixed array")
        check(np.array_equal(stored_fixed_terminal, np.zeros_like(stored_fixed_terminal)), f"{task}: fixed terminal nonzero")
        check(np.array_equal(stored_fixed_cumulative, np.zeros_like(stored_fixed_cumulative)), f"{task}: fixed cumulative nonzero")

        reported_task = reported["tasks"][task]
        terminal_ci = bootstrap(stored_terminal, 900_100 + 10 * task_index)
        active_ci = bootstrap(stored_active_fixed, 900_300 + 10 * task_index)
        pvalue = signflip(stored_terminal, 900_101 + 10 * task_index)
        pvalues[task] = pvalue
        check(close(np.mean(stored_terminal), reported_task["terminal_regret_delta"]["mean"]), f"{task}: terminal mean")
        check(np.allclose(terminal_ci, reported_task["terminal_regret_delta"]["bootstrap_95"], rtol=0.0, atol=1e-14), f"{task}: terminal CI")
        check(close(pvalue, reported_task["terminal_regret_delta"]["one_sided_signflip_p"]), f"{task}: pvalue")
        check(np.allclose(active_ci, reported_task["active_minus_fixed_terminal"]["bootstrap_95"], rtol=0.0, atol=1e-14), f"{task}: active CI")
        reconstructed[task] = {
            "mean_terminal_delta": float(np.mean(stored_terminal)),
            "terminal_bootstrap_95": terminal_ci,
            "one_sided_signflip_p": pvalue,
            "path_change_rate": float(np.mean(np.any(clean_queries != refined_queries, axis=1))),
            "fixed_query_exact": True,
        }

    adjusted = holm(pvalues)
    passing: list[str] = []
    for task in ("mnli", "qqp"):
        row = reconstructed[task]
        reported_task = reported["tasks"][task]
        check(close(adjusted[task], reported_task["holm_adjusted_p"]), f"{task}: Holm p")
        passed = bool(
            row["mean_terminal_delta"] >= MINIMUM_DELTA
            and row["terminal_bootstrap_95"][0] > 0.0
            and adjusted[task] <= 0.05
            and row["path_change_rate"] >= 0.5
            and reported_task["active_minus_fixed_terminal"]["mean"] >= MINIMUM_DELTA
            and reported_task["active_minus_fixed_terminal"]["bootstrap_95"][0] > 0.0
        )
        check(passed == reported_task["passed"], f"{task}: pass label")
        if passed:
            passing.append(task)
    check(passing == ["mnli", "qqp"], "passing task set")
    check(reported["decision"] == "GO_STRONG_EXECUTABLE_ADAPTER_TRANSFER", "decision")
    report = {
        "validation_id": "STEP90_HIGH_CONFIDENCE_INDEPENDENT_VALIDATION_V1",
        "validator_imports_runner_code": False,
        "checks": CHECKS,
        "literal_trajectory_replays": 4000,
        "fixed_query_replays": 2000,
        "reconstructed": reconstructed,
        "holm_adjusted_p": adjusted,
        "passing_tasks": passing,
        "decision": reported["decision"],
        "raw_sha256": sha256_path(RAW_PATH),
        "result_sha256": sha256_path(RESULT_PATH),
        "verdict": "PASS_STEP90_INDEPENDENT_VALIDATION",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
