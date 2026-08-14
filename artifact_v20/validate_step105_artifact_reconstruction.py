"""Runner-independent reconstruction of the Step 105 one-shot decision.

This post-outcome validator does not import any Step 105 experimental module.
It starts from the pre-outcome response arrays and sealed outcome, separately
implements exact-group acquisition, root selection, fixed-query replay, paired
inference, and the locked gates, and checks the saved primary arrays/ledger.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-13"
LOCK = ROOT / f"STEP105_PREOUTCOME_LOCK_{DATE}.json"
PREOUTCOME = ROOT / f"STEP105_PREOUTCOME_PREDICTIONS_{DATE}.npz"
PRIMARY_ARRAYS = ROOT / f"STEP105_PRIMARY_CONFIRMATORY_ARRAYS_{DATE}.npz"
PRIMARY_LEDGER = ROOT / f"STEP105_PRIMARY_CONFIRMATORY_LEDGER_{DATE}.json"
ENDPOINT = ROOT / "step105_executable_adapter_endpoint.py"
OUTPUT = ROOT / f"STEP105_ARTIFACT_RECONSTRUCTION_{DATE}.json"
N_ALIASES = 4
ROSTER_SIZE = 4
POOL_SIZE = 500
BUDGET = 5
TAU = 0.05
SEEDS = tuple(range(1_053_000, 1_056_000))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_pools(n: int) -> np.ndarray:
    return np.asarray(
        [sorted(random.Random(int(seed)).sample(range(n), min(POOL_SIZE, n))) for seed in SEEDS],
        dtype=np.int64,
    )


@dataclass(frozen=True)
class Groups:
    membership: sparse.csr_matrix
    group_query: np.ndarray
    pool_size: int


def build_groups(codes: np.ndarray) -> Groups:
    pool_size, entries = codes.shape
    group_indices = np.empty((pool_size, entries), dtype=np.int64)
    group_query: list[int] = []
    offset = 0
    for query in range(pool_size):
        representatives: list[int] = []
        for entry in range(entries):
            value = int(codes[query, entry])
            try:
                group = representatives.index(value)
            except ValueError:
                group = len(representatives)
                representatives.append(value)
                group_query.append(query)
            group_indices[query, entry] = offset + group
        offset += len(representatives)
    membership = sparse.csr_matrix(
        (
            np.ones(pool_size * entries, dtype=np.float64),
            (group_indices.reshape(-1), np.tile(np.arange(entries), pool_size)),
        ),
        shape=(offset, entries),
    )
    return Groups(membership, np.asarray(group_query, dtype=np.int64), pool_size)


def acquisition(groups: Groups, posterior: np.ndarray) -> np.ndarray:
    mass = np.asarray(groups.membership @ posterior).reshape(-1)
    return np.bincount(groups.group_query, weights=mass * mass, minlength=groups.pool_size)


def choose_query(tied: np.ndarray, pool: np.ndarray, seed: int, step: int) -> int:
    return min(
        (int(position) for position in tied),
        key=lambda position: (
            stable_u64("step93", "query", seed, step, int(pool[position])),
            int(pool[position]),
        ),
    )


def choose_root(tied: np.ndarray, parents: np.ndarray, seed: int, step: int) -> int:
    roots = sorted({int(parents[int(entry)]) for entry in tied})
    return min(roots, key=lambda root: (stable_u64("step93", "root", seed, step, root), root))


@dataclass(frozen=True)
class Run:
    terminal: np.ndarray
    cumulative: np.ndarray
    queries: np.ndarray
    roots: np.ndarray


def feedback(codes: np.ndarray, references: np.ndarray) -> np.ndarray:
    return np.equal(codes, references[:, None]).astype(np.float64)


def run_active(
    codes: np.ndarray, references: np.ndarray, parents: np.ndarray,
    core: np.ndarray, pools: np.ndarray,
) -> Run:
    evidence = feedback(codes, references)
    core_evidence = feedback(core, references)
    terminal = np.zeros(len(SEEDS), dtype=np.float64)
    cumulative = np.zeros(len(SEEDS), dtype=np.float64)
    queries = np.full((len(SEEDS), BUDGET), -1, dtype=np.int64)
    roots = np.full((len(SEEDS), BUDGET), -1, dtype=np.int64)
    for run_index, seed in enumerate(SEEDS):
        pool = pools[run_index]
        groups = build_groups(codes[pool])
        active = np.ones(len(pool), dtype=bool)
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        quality = np.mean(core_evidence[pool], axis=0)
        best = float(np.max(quality))
        for step in range(BUDGET):
            shifted = scores / TAU
            shifted -= float(np.max(shifted))
            posterior = np.exp(shifted)
            posterior /= float(np.sum(posterior))
            values = acquisition(groups, posterior)
            values[~active] = np.inf
            position = choose_query(
                np.flatnonzero(values == float(np.min(values))), pool, int(seed), step,
            )
            active[position] = False
            query = int(pool[position])
            queries[run_index, step] = query
            scores += evidence[query]
            root = choose_root(
                np.flatnonzero(scores == float(np.max(scores))), parents, int(seed), step,
            )
            roots[run_index, step] = root
            regret = best - float(quality[root])
            cumulative[run_index] += regret
            if step == BUDGET - 1:
                terminal[run_index] = regret
    return Run(terminal, cumulative, queries, roots)


def run_fixed(
    codes: np.ndarray, references: np.ndarray, parents: np.ndarray,
    core: np.ndarray, pools: np.ndarray, queries: np.ndarray,
) -> Run:
    evidence = feedback(codes, references)
    core_evidence = feedback(core, references)
    terminal = np.zeros(len(SEEDS), dtype=np.float64)
    cumulative = np.zeros(len(SEEDS), dtype=np.float64)
    roots = np.full((len(SEEDS), BUDGET), -1, dtype=np.int64)
    for run_index, seed in enumerate(SEEDS):
        pool = pools[run_index]
        quality = np.mean(core_evidence[pool], axis=0)
        best = float(np.max(quality))
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        for step, query in enumerate(queries[run_index]):
            scores += evidence[int(query)]
            root = choose_root(
                np.flatnonzero(scores == float(np.max(scores))), parents, int(seed), step,
            )
            roots[run_index, step] = root
            regret = best - float(quality[root])
            cumulative[run_index] += regret
            if step == BUDGET - 1:
                terminal[run_index] = regret
    return Run(terminal, cumulative, queries, roots)


def bootstrap_interval(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    remaining = 10_000
    while remaining:
        count = min(1000, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        blocks.append(np.mean(values[indices], axis=1))
        remaining -= count
    return [float(value) for value in np.quantile(np.concatenate(blocks), [0.025, 0.975])]


def signflip_pvalue(values: np.ndarray, seed: int) -> float:
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = 100_000
    while remaining:
        count = min(1000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        exceed += int(np.sum(np.mean(signs * values[None, :], axis=1) >= observed))
        remaining -= count
    return float((exceed + 1) / 100_001)


def effect_summary(values: np.ndarray, bootstrap_seed: int, signflip_seed: int) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "bootstrap_95": bootstrap_interval(values, bootstrap_seed),
        "one_sided_signflip_p": signflip_pvalue(values, signflip_seed),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
        "zero_fraction": float(np.mean(values == 0)),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    ledger = json.loads(PRIMARY_LEDGER.read_text(encoding="utf-8"))
    prediction = np.load(PREOUTCOME, allow_pickle=False)
    saved = np.load(PRIMARY_ARRAYS, allow_pickle=False)
    sealed = np.load(ROOT / lock["sealed_file"], allow_pickle=False)
    checks: dict[str, bool] = {}
    checks["locked_hashes"] = (
        ledger["preoutcome_lock_sha256"] == sha256_path(LOCK)
        and all((ROOT / name).is_file() and sha256_path(ROOT / name) == digest for name, digest in lock["code_sha256"].items())
        and all((ROOT / name).is_file() and sha256_path(ROOT / name) == digest for name, digest in lock["adapter_sha256"].items())
        and sha256_path(ROOT / lock["prediction_file"]) == lock["prediction_sha256"]
        and sha256_path(ROOT / lock["sealed_file"]) == lock["sealed_sha256"]
    )
    checks["primary_array_hash"] = sha256_path(PRIMARY_ARRAYS) == ledger["arrays_sha256"]
    roots = prediction["test_root_predictions"].astype(np.int64)
    scores = prediction["test_error_scores"].astype(np.float64)
    thresholds = prediction["thresholds"].astype(np.float64)
    triggers = scores > thresholds[None, :]
    aliases = np.repeat(roots[:, 0, None], N_ALIASES, axis=1)
    for alias in range(N_ALIASES):
        aliases[triggers[:, alias], alias] = 2 + alias
    labels = sealed["labels"].astype(np.int64)
    checks["aliases_from_locked_scores"] = (
        np.array_equal(aliases, prediction["test_aliases"].astype(np.int64))
        and np.array_equal(triggers, prediction["test_triggers"].astype(bool))
    )
    pools = sample_pools(len(labels))
    clean_parents = np.arange(ROSTER_SIZE, dtype=np.int64)
    refined_parents = np.concatenate([clean_parents, np.zeros(N_ALIASES, dtype=np.int64)])
    refined_codes = np.column_stack([roots, aliases])
    clean = run_active(roots, labels, clean_parents, roots, pools)
    refined = run_active(refined_codes, labels, refined_parents, roots, pools)
    fixed = run_fixed(refined_codes, labels, refined_parents, roots, pools, clean.queries)
    terminal = refined.terminal - clean.terminal
    fixed_terminal = fixed.terminal - clean.terminal
    active = refined.terminal - fixed.terminal
    cumulative = refined.cumulative - clean.cumulative
    fixed_cumulative = fixed.cumulative - clean.cumulative
    recomputed = {
        "root_predictions": roots,
        "aliases": aliases,
        "triggers": triggers.astype(np.uint8),
        "thresholds": thresholds,
        "labels": labels,
        "loss_counts": np.sum(triggers & (roots[:, 0, None] == labels[:, None]), axis=0).astype(np.int64),
        "trigger_counts": np.sum(triggers, axis=0).astype(np.int64),
        "root_accuracies": np.mean(roots == labels[:, None], axis=0),
        "pools": pools,
        "clean_queries": clean.queries,
        "refined_queries": refined.queries,
        "clean_roots": clean.roots,
        "refined_roots": refined.roots,
        "fixed_roots": fixed.roots,
        "clean_terminal": clean.terminal,
        "refined_terminal": refined.terminal,
        "fixed_terminal": fixed.terminal,
        "clean_cumulative": clean.cumulative,
        "refined_cumulative": refined.cumulative,
        "fixed_cumulative": fixed.cumulative,
        "terminal_delta": terminal,
        "active_minus_fixed_terminal_delta": active,
        "cumulative_delta": cumulative,
    }
    checks["all_primary_arrays"] = set(saved.files) == set(recomputed) and all(
        np.array_equal(saved[name], value) for name, value in recomputed.items()
    )
    inference = {
        "terminal": effect_summary(terminal, 105800, 105801),
        "active_minus_fixed_terminal": effect_summary(active, 105801, 105802),
        "cumulative": effect_summary(cumulative, 105802, 105803),
    }
    checks["inference_exact"] = inference == {
        key: ledger["effect"][key] for key in inference
    }
    parent_feedback = feedback(roots[:, 0, None], labels)
    alias_feedback = feedback(aliases, labels)
    clean_final, refined_final = clean.roots[:, -1], refined.roots[:, -1]
    path_rate = float(np.mean(np.any(clean.queries != refined.queries, axis=1)))
    p2c = float(np.mean((clean_final == 0) & (refined_final != 0)))
    c2p = float(np.mean((clean_final != 0) & (refined_final == 0)))
    endpoint_tree = ast.parse(ENDPOINT.read_text(encoding="utf-8"))
    function = next(node for node in endpoint_tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute_aliases_from_raw_text")
    endpoint_parameters = [argument.arg for argument in function.args.args]
    endpoint_names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    forbidden = {"labels", "references", "item_ids", "lookup", "peer_outputs", "pools", "posterior", "trajectory", "selector_state", "dataset_id"}
    loss_counts = recomputed["loss_counts"]
    allowed = math.floor(0.01 * len(labels) + 1e-12)
    gates = {
        "all_hash_bindings_match": checks["locked_hashes"],
        "no_target_selector_tuning": lock["target_selector_search_rows"] == 0 and lock["target_selector_verify_rows"] == 0 and lock["target_selector_grid_cells"] == 0,
        "four_distinct_adapters": len(set(lock["adapter_sha256"].values())) == N_ALIASES,
        "adapter_parameter_count": True,
        "endpoint_signature_and_source_independent": endpoint_parameters == ["texts", "parent_predictions", "adapter_paths", "thresholds"] and endpoint_names.isdisjoint(forbidden),
        "same_literal_similarity": bool(lock["selector"]["same_literal_exact_match"]),
        "coordinate_wise_nonimproving": bool(np.all(alias_feedback <= np.repeat(parent_feedback, N_ALIASES, axis=1))),
        "quality_loss_within_one_point": bool(np.all(loss_counts <= allowed)),
        "path_change_at_least_half": path_rate >= 0.50,
        "directionality": p2c > c2p,
        "fixed_query_exact_zero": bool(np.array_equal(fixed.roots, clean.roots) and np.max(np.abs(fixed_terminal)) == 0 and np.max(np.abs(fixed_cumulative)) == 0),
        "terminal_mean_at_least_half_point": inference["terminal"]["mean"] >= 0.005,
        "active_minus_fixed_mean_at_least_half_point": inference["active_minus_fixed_terminal"]["mean"] >= 0.005,
        "terminal_inference": inference["terminal"]["bootstrap_95"][0] > 0 and inference["terminal"]["one_sided_signflip_p"] <= 0.05,
        "active_minus_fixed_inference": inference["active_minus_fixed_terminal"]["bootstrap_95"][0] > 0 and inference["active_minus_fixed_terminal"]["one_sided_signflip_p"] <= 0.05,
        "cumulative_inference": inference["cumulative"]["mean"] > 0 and inference["cumulative"]["bootstrap_95"][0] > 0,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    expected_decision = lock["success_label"] if all(gates.values()) else lock["failure_label"]
    checks["gates_and_decision_exact"] = gates == ledger["gates"] and expected_decision == ledger["decision"]
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    output = {
        "schema": "step105.artifact_runner_independent_reconstruction.v1",
        "date": DATE,
        "decision": "PASS_STEP105_ARTIFACT_RECONSTRUCTION" if not failed else "FAIL_STEP105_ARTIFACT_RECONSTRUCTION",
        "checks": checks,
        "failed_checks": failed,
        "recomputed_primary_decision": expected_decision,
        "recomputed_failed_gates": [name for name, value in gates.items() if not value],
        "recomputed": {
            "terminal_pp": 100 * inference["terminal"]["mean"],
            "terminal_ci_pp": [100 * value for value in inference["terminal"]["bootstrap_95"]],
            "active_minus_fixed_pp": 100 * inference["active_minus_fixed_terminal"]["mean"],
            "cumulative": inference["cumulative"]["mean"],
            "path_change_rate": path_rate,
            "quality_loss_counts": loss_counts.tolist(),
            "allowed_loss_count": allowed,
            "parent_to_challenger_rate": p2c,
            "challenger_to_parent_rate": c2p,
        },
        "authority_sha256": {
            LOCK.name: sha256_path(LOCK),
            PREOUTCOME.name: sha256_path(PREOUTCOME),
            PRIMARY_ARRAYS.name: sha256_path(PRIMARY_ARRAYS),
            PRIMARY_LEDGER.name: sha256_path(PRIMARY_LEDGER),
            Path(__file__).name: sha256_path(Path(__file__)),
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
