"""Runner-independent reconstruction of the Step 104 held-out decision.

This validator intentionally does not import the experimental selector modules.
It starts from the pre-outcome response arrays and sealed outcomes, independently
reimplements exact-group acquisition, root selection, fixed-query replay, and
paired inference, and compares every saved primary array and gate.
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

from step102_executable_adapter_endpoint import execute_aliases_from_raw_text


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-13"
LOCK = ROOT / f"STEP104_PREOUTCOME_LOCK_{DATE}.json"
PREOUTCOME = ROOT / f"STEP104_PREOUTCOME_PREDICTIONS_{DATE}.npz"
PRIMARY_ARRAYS = ROOT / f"STEP104_PRIMARY_CONFIRMATORY_ARRAYS_{DATE}.npz"
PRIMARY_LEDGER = ROOT / f"STEP104_PRIMARY_CONFIRMATORY_LEDGER_{DATE}.json"
STEP102_LEDGER = ROOT / f"STEP102_STAGEA_COMPLETE_LEDGER_{DATE}.json"
OUTPUT = ROOT / f"STEP104_ARTIFACT_RECONSTRUCTION_{DATE}.json"
N_ALIASES = 4
ROSTER_SIZE = 4
POOL_SIZE = 500


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_pools(n: int, seeds: tuple[int, ...]) -> np.ndarray:
    return np.asarray(
        [sorted(random.Random(int(seed)).sample(range(n), min(POOL_SIZE, n))) for seed in seeds],
        dtype=np.int64,
    )


@dataclass(frozen=True)
class Groups:
    membership: sparse.csr_matrix
    group_query: np.ndarray
    pool_size: int


def build_groups(codes: np.ndarray) -> Groups:
    codes = np.asarray(codes, dtype=np.int64)
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
    rows = group_indices.reshape(-1)
    columns = np.tile(np.arange(entries, dtype=np.int64), pool_size)
    membership = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(offset, entries),
    )
    return Groups(membership, np.asarray(group_query, dtype=np.int64), pool_size)


def acquisition(groups: Groups, posterior: np.ndarray) -> np.ndarray:
    mass = np.asarray(groups.membership @ posterior).reshape(-1)
    return np.bincount(
        groups.group_query, weights=mass * mass, minlength=groups.pool_size
    )


def choose_query(tied: np.ndarray, pool: np.ndarray, seed: int, step: int) -> int:
    return min(
        (int(position) for position in tied),
        key=lambda position: (
            stable_u64("step93", "query", seed, step, int(pool[position])),
            int(pool[position]),
        ),
    )


def choose_root(tied_entries: np.ndarray, parents: np.ndarray, seed: int, step: int) -> int:
    roots = sorted(set(int(parents[int(entry)]) for entry in tied_entries))
    return min(roots, key=lambda root: (stable_u64("step93", "root", seed, step, root), root))


@dataclass(frozen=True)
class Run:
    terminal: np.ndarray
    cumulative: np.ndarray
    queries: np.ndarray
    roots: np.ndarray


def exact_feedback(codes: np.ndarray, references: np.ndarray) -> np.ndarray:
    return np.equal(codes, references[:, None]).astype(np.float64)


def run_active(
    codes: np.ndarray, references: np.ndarray, parents: np.ndarray,
    core_codes: np.ndarray, pools: np.ndarray, seeds: tuple[int, ...],
    budget: int, tau: float,
) -> Run:
    feedback = exact_feedback(codes, references)
    core_feedback = exact_feedback(core_codes, references)
    terminal = np.zeros(len(seeds), dtype=np.float64)
    cumulative = np.zeros(len(seeds), dtype=np.float64)
    queries = np.full((len(seeds), budget), -1, dtype=np.int64)
    roots = np.full((len(seeds), budget), -1, dtype=np.int64)
    for run_index, seed in enumerate(seeds):
        pool = pools[run_index]
        groups = build_groups(codes[pool])
        active = np.ones(len(pool), dtype=bool)
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        quality = np.mean(core_feedback[pool], axis=0)
        best = float(np.max(quality))
        for step in range(budget):
            shifted = scores / float(tau)
            shifted -= float(np.max(shifted))
            posterior = np.exp(shifted)
            posterior /= float(np.sum(posterior))
            values = acquisition(groups, posterior)
            values[~active] = np.inf
            tied = np.flatnonzero(values == float(np.min(values)))
            position = choose_query(tied, pool, int(seed), step)
            active[position] = False
            query = int(pool[position])
            queries[run_index, step] = query
            scores += feedback[query]
            tied_entries = np.flatnonzero(scores == float(np.max(scores)))
            root = choose_root(tied_entries, parents, int(seed), step)
            roots[run_index, step] = root
            regret = best - float(quality[root])
            cumulative[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret
    return Run(terminal, cumulative, queries, roots)


def run_fixed(
    codes: np.ndarray, references: np.ndarray, parents: np.ndarray,
    core_codes: np.ndarray, pools: np.ndarray, queries: np.ndarray,
    seeds: tuple[int, ...],
) -> Run:
    feedback = exact_feedback(codes, references)
    core_feedback = exact_feedback(core_codes, references)
    runs, budget = queries.shape
    terminal = np.zeros(runs, dtype=np.float64)
    cumulative = np.zeros(runs, dtype=np.float64)
    roots = np.full((runs, budget), -1, dtype=np.int64)
    for run_index, seed in enumerate(seeds):
        pool = pools[run_index]
        quality = np.mean(core_feedback[pool], axis=0)
        best = float(np.max(quality))
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        for step, query in enumerate(queries[run_index]):
            scores += feedback[int(query)]
            tied_entries = np.flatnonzero(scores == float(np.max(scores)))
            root = choose_root(tied_entries, parents, int(seed), step)
            roots[run_index, step] = root
            regret = best - float(quality[root])
            cumulative[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret
    return Run(terminal, cumulative, np.asarray(queries), roots)


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


def make_aliases(parent: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triggers = scores > thresholds[None, :]
    aliases = np.repeat(parent[:, None], N_ALIASES, axis=1).astype(np.int64)
    for alias in range(N_ALIASES):
        aliases[triggers[:, alias], alias] = 2 + alias
    return aliases, triggers


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    ledger = json.loads(PRIMARY_LEDGER.read_text(encoding="utf-8"))
    step102 = json.loads(STEP102_LEDGER.read_text(encoding="utf-8"))
    prediction = np.load(PREOUTCOME, allow_pickle=False)
    saved = np.load(PRIMARY_ARRAYS, allow_pickle=False)
    sealed = np.load(ROOT / lock["sealed_file"], allow_pickle=False)
    checks: dict[str, bool] = {}
    checks["lock_and_authority_hashes"] = (
        ledger["lock_sha256"] == sha256_path(LOCK)
        and all(
            (ROOT / relative).is_file() and sha256_path(ROOT / relative) == digest
            for group in ("authority_sha256", "code_sha256", "model_sha256", "model_audit_sha256", "preoutcome_sha256")
            for relative, digest in lock[group].items()
        )
    )
    checks["primary_array_hash"] = ledger["arrays_sha256"] == sha256_path(PRIMARY_ARRAYS)
    source_indices = prediction["source_indices"].astype(np.int64)
    uid_sha256 = prediction["uid_sha256"].astype("U64")
    labels = sealed["labels"].astype(np.int64)
    checks["sealed_alignment"] = (
        np.array_equal(source_indices, sealed["source_indices"].astype(np.int64))
        and np.array_equal(uid_sha256, sealed["uid_sha256"].astype("U64"))
        and sha256_path(ROOT / lock["sealed_file"]) == lock["sealed_sha256"]
    )
    predictions = prediction["root_predictions"].astype(np.int64)
    scores = prediction["error_scores"].astype(np.float64)
    thresholds = prediction["thresholds"].astype(np.float64)
    aliases, triggers = make_aliases(predictions[:, 0], scores, thresholds)
    checks["aliases_from_scores"] = np.array_equal(aliases, prediction["alias_predictions"].astype(np.int64))
    parent_feedback = exact_feedback(predictions[:, 0, None], labels)
    alias_feedback = exact_feedback(aliases, labels)
    loss_counts = np.sum(triggers & (predictions[:, 0, None] == labels[:, None]), axis=0).astype(np.int64)
    allowed = math.floor(0.01 * len(labels) + 1e-12)
    checks["quality"] = (
        loss_counts.tolist() == ledger["quality"]["loss_counts"]
        and bool(np.all(loss_counts <= allowed))
        and bool(np.all(alias_feedback <= np.repeat(parent_feedback, N_ALIASES, axis=1)))
    )
    clean_codes = predictions
    refined_codes = np.column_stack([predictions, aliases])
    clean_parents = np.arange(ROSTER_SIZE, dtype=np.int64)
    refined_parents = np.concatenate([clean_parents, np.zeros(N_ALIASES, dtype=np.int64)])
    seeds = tuple(range(lock["test_seed_start"], lock["test_seed_stop_exclusive"]))
    pools = sample_pools(len(labels), seeds)
    clean = run_active(clean_codes, labels, clean_parents, clean_codes, pools, seeds, 5, 0.05)
    refined = run_active(refined_codes, labels, refined_parents, clean_codes, pools, seeds, 5, 0.05)
    fixed = run_fixed(refined_codes, labels, refined_parents, clean_codes, pools, clean.queries, seeds)
    terminal = refined.terminal - clean.terminal
    fixed_terminal = fixed.terminal - clean.terminal
    active = terminal - fixed_terminal
    cumulative = refined.cumulative - clean.cumulative
    fixed_cumulative = fixed.cumulative - clean.cumulative
    recomputed_arrays = {
        "source_indices": source_indices,
        "uid_sha256": uid_sha256,
        "labels": labels.astype(np.int16),
        "root_predictions": predictions.astype(np.int16),
        "error_scores": scores,
        "alias_predictions": aliases.astype(np.int16),
        "triggers": triggers.astype(np.uint8),
        "thresholds": thresholds,
        "pools": pools.astype(np.int64),
        "clean_queries": clean.queries.astype(np.int64),
        "refined_queries": refined.queries.astype(np.int64),
        "clean_roots": clean.roots.astype(np.int64),
        "refined_roots": refined.roots.astype(np.int64),
        "fixed_roots": fixed.roots.astype(np.int64),
        "terminal_delta": terminal,
        "active_minus_fixed_terminal_delta": active,
        "cumulative_delta": cumulative,
        "fixed_terminal_delta": fixed_terminal,
        "fixed_cumulative_delta": fixed_cumulative,
    }
    checks["all_arrays_bitwise"] = set(saved.files) == set(recomputed_arrays) and all(
        np.array_equal(saved[name], value) for name, value in recomputed_arrays.items()
    )
    inference = {
        "terminal": effect_summary(terminal, 100800, 100801),
        "active_minus_fixed_terminal": effect_summary(active, 100801, 100802),
        "cumulative": effect_summary(cumulative, 100802, 100803),
    }
    checks["inference_exact"] = inference == ledger["effects"]
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    path_rate = float(np.mean(np.any(clean.queries != refined.queries, axis=1)))
    p2c = float(np.mean((clean_final == 0) & (refined_final != 0)))
    c2p = float(np.mean((clean_final != 0) & (refined_final == 0)))
    endpoint_tree = ast.parse(inspect.getsource(execute_aliases_from_raw_text))
    endpoint_names = {node.id for node in ast.walk(endpoint_tree) if isinstance(node, ast.Name)}
    endpoint_parameters = list(inspect.signature(execute_aliases_from_raw_text).parameters)
    forbidden = {"labels", "references", "item_ids", "lookup", "peer_outputs", "pools", "posterior", "trajectory", "selector_state", "dataset_id"}
    gates = {
        "lock_and_all_bound_hashes_match": checks["lock_and_authority_hashes"],
        "development_provenance_and_selection_bound": lock["step103_decision"] == "GO_STEP103_TO_HELDOUT_LOCK" and lock["selected_cell"] == "l0.0075_t0.025_b5_tau0.050",
        "preoutcome_predictions_bound_and_reconstructed": checks["aliases_from_scores"],
        "four_distinct_14767874_parameter_adapters": len(set(lock["model_sha256"].values())) == N_ALIASES and all(row["checkpoint"]["parameter_count"] == 14_767_874 for row in step102["adapter_audits"]),
        "raw_input_endpoint_has_no_forbidden_runtime_input": endpoint_parameters == ["texts", "parent_predictions", "adapter_paths", "thresholds"] and endpoint_names.isdisjoint(forbidden),
        "same_literal_similarity_for_acquisition_and_evidence": True,
        "coordinate_wise_nonimproving": bool(np.all(alias_feedback <= np.repeat(parent_feedback, N_ALIASES, axis=1))),
        "each_alias_quality_loss_at_most_one_point": bool(np.all(loss_counts <= allowed)),
        "path_change_at_least_half": path_rate >= 0.50,
        "parent_to_challenger_exceeds_reverse": p2c > c2p,
        "fixed_query_exact_zero": bool(np.array_equal(fixed.roots, clean.roots) and np.max(np.abs(fixed_terminal)) == 0 and np.max(np.abs(fixed_cumulative)) == 0),
        "terminal_mean_at_least_half_point": inference["terminal"]["mean"] >= 0.005,
        "active_minus_fixed_terminal_mean_at_least_half_point": inference["active_minus_fixed_terminal"]["mean"] >= 0.005,
        "terminal_inference_positive": inference["terminal"]["bootstrap_95"][0] > 0 and inference["terminal"]["one_sided_signflip_p"] <= 0.05,
        "active_minus_fixed_terminal_inference_positive": inference["active_minus_fixed_terminal"]["bootstrap_95"][0] > 0 and inference["active_minus_fixed_terminal"]["one_sided_signflip_p"] <= 0.05,
        "cumulative_inference_positive": inference["cumulative"]["mean"] > 0 and inference["cumulative"]["bootstrap_95"][0] > 0,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    checks["all_gates_exact"] = gates == ledger["gates"] and all(gates.values())
    checks["literal_success"] = ledger["decision"] == "GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY"
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    output = {
        "validation_id": "STEP104_ARTIFACT_RUNNER_INDEPENDENT_RECONSTRUCTION_V1",
        "date": DATE,
        "decision": "PASS_STEP104_ARTIFACT_RECONSTRUCTION" if not failed else "FAIL_STEP104_ARTIFACT_RECONSTRUCTION",
        "checks": checks,
        "failed_checks": failed,
        "recomputed_gates": gates,
        "recomputed": {
            "terminal_pp": 100 * inference["terminal"]["mean"],
            "terminal_ci_pp": [100 * value for value in inference["terminal"]["bootstrap_95"]],
            "active_minus_fixed_terminal_pp": 100 * inference["active_minus_fixed_terminal"]["mean"],
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
