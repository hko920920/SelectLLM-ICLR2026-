#!/usr/bin/env python3
"""Run the preregistered Step 88 LLM Selector registry-refinement audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "external" / "llm-selector"
OUTPUT = ROOT / "STEP88_LLM_SELECTOR_RESULTS_2026-08-12.json"

DATASETS = {
    "alpacaeval": {
        "judge_sha256": "9ee29592ab55d883cad91eee15a3a2607589c43682e624a447a09d8578d275f4",
        "weak_sha256": "84e0faaa2e81993a5ae13d14acda9ca357869d570bd41de1afa9c97baa3d96e0",
        "parent": "llama-2-13b-chat-hf",
    },
    "arena-hard": {
        "judge_sha256": "94516f0ba7eaf0467943deb12538ad5e473d0d7ef984baaaef2a2a54c90b6a03",
        "weak_sha256": "bd8852a964aef17fb95aa3141e4ba1984272183968f4a85ac3c87cf54e38b1d5",
        "parent": "gpt-4o-2024-08-06",
    },
    "bingo": {
        "judge_sha256": "d066efca199f4209b284c3c6b0917e1d9b613aa74fa2c506cf7fd2a6f23f5c45",
        "weak_sha256": "0cf1728c7dc6ddddebec71669f7eb4f20259cd0bcbd16cf5203ed8f19725d1f7",
        "parent": "HuggingFaceM4_idefics-9b-instruct",
    },
    "flickr30k": {
        "judge_sha256": "3a1a2d03a63f5f45acf0a4b9dd85c78885c231a48ce0d9393631206f752433ba",
        "weak_sha256": "1fddbc5c8a37facb5f6d9d845ebf31994b7a6a5e85e7562d45908a44a8a26e06",
        "parent": "openai_gpt-4o-2024-05-13",
    },
    "medi_qa": {
        "judge_sha256": "cd76912a786c0ed3f1081b1418b14e8b5827b8e511844a9d292338327fd1b7a5",
        "weak_sha256": "2494bc845b9c16b2d223b48858311a594f95eb304781727e2800acfd90dcaf0a",
        "parent": "openai_gpt-4o-mini-2024-07-18",
    },
    "mt-bench": {
        "judge_sha256": "8bbe2087b844032c7fd0fd04dcfde3817041b9075233c06352fd601adb20d8f0",
        "weak_sha256": "a00c992bbf4fdd487d76ae277153b315ea677be571a8bd5b50b477f59ade6574",
        "parent": "alpaca-13b",
    },
}

N_SEEDS = 1000
BUDGET = 30
MAX_POOL = 400
N_ALIASES = 4
PERTURB_RATE = 0.20
EPS_LOSS = 0.20
EPS_DRAW = 0.40
EPS_OTHER = 1.0 - EPS_LOSS - EPS_DRAW
BOOTSTRAP_REPS = 50_000
SIGNFLIP_REPS = 100_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def load_dataset(name: str) -> tuple[list[str], str, np.ndarray, np.ndarray]:
    spec = DATASETS[name]
    judge_path = REPO / name / "judge.json"
    weak_path = REPO / name / "weak_judge.json"
    if sha256_file(judge_path) != spec["judge_sha256"]:
        raise RuntimeError(f"{name}: judge hash mismatch")
    if sha256_file(weak_path) != spec["weak_sha256"]:
        raise RuntimeError(f"{name}: weak-judge hash mismatch")

    with judge_path.open("r", encoding="utf-8") as handle:
        judge_data = json.load(handle)
    with weak_path.open("r", encoding="utf-8") as handle:
        weak_data = json.load(handle)

    model_names = [key for key, value in judge_data.items() if value is not None]
    baselines = [key for key, value in judge_data.items() if value is None]
    if len(baselines) != 1:
        raise RuntimeError(f"{name}: expected one baseline, got {baselines}")
    if not model_names or model_names[0] != spec["parent"]:
        raise RuntimeError(f"{name}: frozen parent/source order mismatch")
    if list(weak_data.keys()) != list(judge_data.keys()):
        raise RuntimeError(f"{name}: judge/weak model-key order mismatch")

    judge = np.asarray([judge_data[key] for key in model_names], dtype=np.int8)
    weak = np.asarray([weak_data[key] for key in model_names], dtype=np.int8)
    if judge.ndim != 2 or weak.ndim != 3:
        raise RuntimeError(f"{name}: unexpected dimensions {judge.shape}, {weak.shape}")
    if weak.shape[:2] != judge.shape or weak.shape[2] != 10:
        raise RuntimeError(f"{name}: incompatible score shapes {judge.shape}, {weak.shape}")
    if not np.all(np.isin(judge, (-1, 0, 1))):
        raise RuntimeError(f"{name}: judge outcomes outside ternary support")
    if not np.all(np.isin(weak, (-1, 0, 1))):
        raise RuntimeError(f"{name}: weak outcomes outside ternary support")
    return model_names, baselines[0], judge, weak


def alias_mask_value(name: str, query: int, metric: int, alias: int) -> float:
    label = f"step88|{name}|{query}|{metric}|{alias}".encode("utf-8")
    raw = hashlib.sha256(label).digest()
    return int.from_bytes(raw[:8], "big") / float(1 << 64)


def make_alias_weak(name: str, parent_weak: np.ndarray) -> np.ndarray:
    n_queries, n_metrics = parent_weak.shape
    aliases = np.repeat(parent_weak[None, :, :], N_ALIASES, axis=0)
    forward = {-1: 0, 0: 1, 1: -1}
    reverse = {-1: 1, 0: -1, 1: 0}
    for alias in range(N_ALIASES):
        mapping = forward if alias in (0, 2) else reverse
        for query in range(n_queries):
            for metric in range(n_metrics):
                if alias_mask_value(name, query, metric, alias) < PERTURB_RATE:
                    aliases[alias, query, metric] = mapping[int(parent_weak[query, metric])]
    return aliases


def entropy_by_query(posterior: np.ndarray, loss_flat: np.ndarray, n_queries: int, n_metrics: int) -> np.ndarray:
    """Official posterior entropy, exploiting eps_draw == eps_win == 0.4."""
    if not math.isclose(EPS_DRAW, EPS_OTHER, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("optimized entropy requires equal draw/win factors")
    plogp = np.where(posterior > 0.0, posterior * np.log(posterior), 0.0)
    grouped = np.vstack((posterior, plogp)) @ loss_flat
    loss_mass = grouped[0]
    loss_plogp = grouped[1]
    all_plogp = float(plogp.sum())
    z = EPS_LOSS * loss_mass + EPS_OTHER * (1.0 - loss_mass)
    weighted_plogp = EPS_LOSS * loss_plogp + EPS_OTHER * (all_plogp - loss_plogp)
    weighted_log_factor = (
        EPS_LOSS * loss_mass * math.log(EPS_LOSS)
        + EPS_OTHER * (1.0 - loss_mass) * math.log(EPS_OTHER)
    )
    entropy = -(weighted_plogp + weighted_log_factor) / z + np.log(z)
    return entropy.reshape(n_queries, n_metrics).mean(axis=1)


def run_acquisition(
    judge_entries: np.ndarray,
    weak_entries: np.ndarray,
    original_judge: np.ndarray,
    pool: list[int],
) -> dict[str, Any]:
    n_entries = judge_entries.shape[0]
    n_original, _, n_metrics = original_judge.shape[0], weak_entries.shape[1], weak_entries.shape[2]
    pool_array = np.asarray(pool, dtype=np.int64)
    pool_size = len(pool)

    # The fixed baseline is the final entry, matching the upstream implementation.
    loss_cube = np.concatenate(
        (weak_entries[:, pool_array, :] < 0, np.zeros((1, pool_size, n_metrics), dtype=bool)),
        axis=0,
    )
    loss_flat = np.ascontiguousarray(loss_cube.reshape(n_entries + 1, pool_size * n_metrics), dtype=np.float64)
    posterior = np.full(n_entries + 1, 1.0 / (n_entries + 1), dtype=np.float64)
    available = np.ones(pool_size, dtype=bool)

    pool_scores = np.concatenate((original_judge[:, pool_array].mean(axis=1), np.zeros(1)))
    pool_win_rates = (pool_scores + 1.0) / 2.0
    best_pool_wr = float(pool_win_rates.max())

    path: list[int] = []
    chosen_roots: list[int] = []
    regrets: list[float] = []
    acquired_positions: list[int] = []

    for _ in range(BUDGET):
        entropy = entropy_by_query(posterior, loss_flat, pool_size, n_metrics)
        entropy[~available] = np.inf
        position = int(np.argmin(entropy))
        available[position] = False
        query = int(pool_array[position])
        acquired_positions.append(position)
        path.append(query)

        observed = np.concatenate((judge_entries[:, query], np.zeros(1, dtype=np.int8)))
        posterior *= np.where(observed < 0, EPS_LOSS, EPS_OTHER)
        posterior /= posterior.sum()

        acquired_queries = pool_array[np.asarray(acquired_positions, dtype=np.int64)]
        root_scores = np.concatenate((original_judge[:, acquired_queries].mean(axis=1), np.zeros(1)))
        chosen_root = int(np.argmax(root_scores))
        chosen_roots.append(chosen_root)
        regrets.append(best_pool_wr - float(pool_win_rates[chosen_root]))

    return {"path": path, "chosen_roots": chosen_roots, "regrets": regrets}


def forced_query_deployment(original_judge: np.ndarray, pool: list[int], path: list[int]) -> dict[str, Any]:
    pool_array = np.asarray(pool, dtype=np.int64)
    pool_scores = np.concatenate((original_judge[:, pool_array].mean(axis=1), np.zeros(1)))
    pool_win_rates = (pool_scores + 1.0) / 2.0
    best_pool_wr = float(pool_win_rates.max())
    chosen_roots: list[int] = []
    regrets: list[float] = []
    acquired: list[int] = []
    for query in path:
        acquired.append(query)
        root_scores = np.concatenate((original_judge[:, acquired].mean(axis=1), np.zeros(1)))
        chosen_root = int(np.argmax(root_scores))
        chosen_roots.append(chosen_root)
        regrets.append(best_pool_wr - float(pool_win_rates[chosen_root]))
    return {"chosen_roots": chosen_roots, "regrets": regrets}


def run_dataset(name: str) -> dict[str, Any]:
    model_names, baseline_name, judge, weak = load_dataset(name)
    aliases_weak = make_alias_weak(name, weak[0])
    aliases_judge = np.repeat(judge[0][None, :], N_ALIASES, axis=0)
    refined_judge = np.concatenate((judge, aliases_judge), axis=0)
    refined_weak = np.concatenate((weak, aliases_weak), axis=0)

    utility_max_abs = int(np.max(np.abs(aliases_judge - judge[0][None, :])))
    utility_mean_abs = float(np.mean(np.abs(aliases_judge.astype(float) - judge[0][None, :].astype(float))))
    weak_hamming = [float(np.mean(aliases_weak[a] != weak[0])) for a in range(N_ALIASES)]

    pool_size = min(MAX_POOL, judge.shape[1])
    raw: list[dict[str, Any]] = []
    for seed in range(N_SEEDS):
        pool = random.Random(seed).sample(range(judge.shape[1]), pool_size)
        clean = run_acquisition(judge, weak, judge, pool)
        refined = run_acquisition(refined_judge, refined_weak, judge, pool)
        fixed = forced_query_deployment(judge, pool, clean["path"])

        clean_regrets = np.asarray(clean["regrets"], dtype=float)
        refined_regrets = np.asarray(refined["regrets"], dtype=float)
        fixed_regrets = np.asarray(fixed["regrets"], dtype=float)
        if not np.array_equal(clean_regrets, fixed_regrets):
            raise RuntimeError(f"{name} seed {seed}: fixed-query invariant failed")

        clean_set = set(clean["path"])
        refined_set = set(refined["path"])
        union = clean_set | refined_set
        jaccard = len(clean_set & refined_set) / len(union) if union else 1.0
        raw.append(
            {
                "seed": seed,
                "pool": pool,
                "clean_path": clean["path"],
                "refined_path": refined["path"],
                "clean_chosen_roots": clean["chosen_roots"],
                "refined_chosen_roots": refined["chosen_roots"],
                "clean_regrets": clean["regrets"],
                "refined_regrets": refined["regrets"],
                "fixed_regrets": fixed["regrets"],
                "ordered_path_changed": clean["path"] != refined["path"],
                "query_set_changed": clean_set != refined_set,
                "query_set_jaccard": jaccard,
                "terminal_root_changed": clean["chosen_roots"][-1] != refined["chosen_roots"][-1],
                "terminal_delta": float(refined_regrets[-1] - clean_regrets[-1]),
                "cumulative_delta": float(refined_regrets.sum() - clean_regrets.sum()),
                "fixed_terminal_delta": float(fixed_regrets[-1] - clean_regrets[-1]),
                "fixed_cumulative_delta": float(fixed_regrets.sum() - clean_regrets.sum()),
            }
        )

    return {
        "dataset": name,
        "model_names": model_names,
        "baseline_name": baseline_name,
        "parent": model_names[0],
        "n_queries": int(judge.shape[1]),
        "n_original_candidates": int(judge.shape[0]),
        "pool_size": pool_size,
        "weak_hamming_by_alias": weak_hamming,
        "strong_utility_max_abs_difference": utility_max_abs,
        "strong_utility_mean_abs_difference": utility_mean_abs,
        "raw": raw,
    }


def bootstrap_interval(values: np.ndarray, label: str) -> list[float]:
    rng = np.random.default_rng(stable_seed(f"step88-bootstrap|{label}"))
    means = np.empty(BOOTSTRAP_REPS, dtype=float)
    batch = 500
    offset = 0
    while offset < BOOTSTRAP_REPS:
        size = min(batch, BOOTSTRAP_REPS - offset)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[offset : offset + size] = values[indices].mean(axis=1)
        offset += size
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def signflip_pvalue(values: np.ndarray, label: str) -> float:
    observed = float(values.mean())
    rng = np.random.default_rng(stable_seed(f"step88-signflip|{label}"))
    exceed = 0
    done = 0
    batch = 1000
    while done < SIGNFLIP_REPS:
        size = min(batch, SIGNFLIP_REPS - done)
        signs = rng.integers(0, 2, size=(size, len(values)), dtype=np.int8) * 2 - 1
        permuted = (signs * values[None, :]).mean(axis=1)
        exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
        done += size
    return (exceed + 1.0) / (SIGNFLIP_REPS + 1.0)


def bh_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank_from_end, (name, pvalue) in enumerate(reversed(ordered), start=1):
        rank = m - rank_from_end + 1
        running = min(running, pvalue * m / rank)
        adjusted[name] = min(1.0, running)
    return adjusted


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    raw = result["raw"]
    terminal = np.asarray([row["terminal_delta"] for row in raw], dtype=float)
    cumulative = np.asarray([row["cumulative_delta"] for row in raw], dtype=float)
    fixed_terminal = np.asarray([row["fixed_terminal_delta"] for row in raw], dtype=float)
    fixed_cumulative = np.asarray([row["fixed_cumulative_delta"] for row in raw], dtype=float)
    return {
        "mean_terminal_delta": float(terminal.mean()),
        "median_terminal_delta": float(np.median(terminal)),
        "terminal_bootstrap_95_ci": bootstrap_interval(terminal, result["dataset"]),
        "terminal_signflip_p_one_sided": signflip_pvalue(terminal, result["dataset"]),
        "mean_cumulative_delta": float(cumulative.mean()),
        "median_cumulative_delta": float(np.median(cumulative)),
        "ordered_path_change_rate": float(np.mean([row["ordered_path_changed"] for row in raw])),
        "query_set_change_rate": float(np.mean([row["query_set_changed"] for row in raw])),
        "mean_query_set_jaccard": float(np.mean([row["query_set_jaccard"] for row in raw])),
        "terminal_root_change_rate": float(np.mean([row["terminal_root_changed"] for row in raw])),
        "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
        "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
    }


def main() -> int:
    max_workers = min(len(DATASETS), max(1, os.cpu_count() or 1))
    completed: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_dataset, name): name for name in DATASETS}
        for future in as_completed(futures):
            name = futures[future]
            completed[name] = future.result()
            print(f"completed {name}", flush=True)

    ordered_results = {name: completed[name] for name in DATASETS}
    summaries = {name: summarize(ordered_results[name]) for name in DATASETS}
    qvalues = bh_adjust({name: item["terminal_signflip_p_one_sided"] for name, item in summaries.items()})

    passing: list[str] = []
    for name, item in summaries.items():
        item["terminal_signflip_q_bh"] = qvalues[name]
        source = ordered_results[name]
        checks = {
            "utility_exact": source["strong_utility_max_abs_difference"] == 0,
            "fixed_query_exact": item["max_abs_fixed_terminal_delta"] <= 1e-12
            and item["max_abs_fixed_cumulative_delta"] <= 1e-12,
            "path_change_rate_ge_0_50": item["ordered_path_change_rate"] >= 0.50,
            "terminal_delta_ge_0_0025": item["mean_terminal_delta"] >= 0.0025,
            "bootstrap_lower_gt_zero": item["terminal_bootstrap_95_ci"][0] > 0.0,
            "bh_q_le_0_05": item["terminal_signflip_q_bh"] <= 0.05,
        }
        item["gate_checks"] = checks
        item["passes_terminal_harm_gate"] = all(checks.values())
        if item["passes_terminal_harm_gate"]:
            passing.append(name)

    invalid = any(
        source["strong_utility_max_abs_difference"] != 0
        or summaries[name]["max_abs_fixed_terminal_delta"] > 1e-12
        or summaries[name]["max_abs_fixed_cumulative_delta"] > 1e-12
        for name, source in ordered_results.items()
    )
    if invalid:
        decision = "INVALID_IMPLEMENTATION"
    elif len(passing) >= 2:
        decision = "GO_STRONG_CROSS_SELECTOR"
    elif len(passing) == 1:
        decision = "GO_LIMITED_CROSS_SELECTOR"
    else:
        decision = "NO_CONFIRMATORY_CROSS_SELECTOR"

    payload = {
        "schema": "step88-llm-selector-audit-v1",
        "decision": decision,
        "passing_datasets": passing,
        "source": {
            "repository": "https://github.com/RobustML-Lab/llm-selector",
            "commit": "15faa47dab102d7a92124b13c1838b55524fc2bf",
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "protocol": {
            "n_seeds": N_SEEDS,
            "budget": BUDGET,
            "max_pool": MAX_POOL,
            "n_aliases": N_ALIASES,
            "perturb_rate": PERTURB_RATE,
            "eps_loss": EPS_LOSS,
            "eps_draw": EPS_DRAW,
            "weak_judges": 10,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "signflip_reps": SIGNFLIP_REPS,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "summaries": summaries,
        "datasets": ordered_results,
    }
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"decision": decision, "passing_datasets": passing, "summaries": summaries}, indent=2))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

