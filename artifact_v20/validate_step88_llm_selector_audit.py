#!/usr/bin/env python3
"""Independent validation of the preregistered Step 88 audit.

This file does not import the runner.  It reconstructs input arrays and aliases,
checks every stored trajectory record, recomputes all statistics, and replays a
result-independent subset of paths with the literal official entropy equation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import entropy


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "external" / "llm-selector"
RESULT_PATH = ROOT / "STEP88_LLM_SELECTOR_RESULTS_2026-08-12.json"
PREREG_PATH = ROOT / "STEP88_LLM_SELECTOR_PREREGISTRATION_2026-08-12.md"
RUNNER_PATH = ROOT / "run_step88_llm_selector_audit.py"
RECEIPT_PATH = ROOT / "STEP88_INDEPENDENT_VALIDATION_RECEIPT_2026-08-12.json"

DATASETS = {
    "alpacaeval": (
        "9ee29592ab55d883cad91eee15a3a2607589c43682e624a447a09d8578d275f4",
        "84e0faaa2e81993a5ae13d14acda9ca357869d570bd41de1afa9c97baa3d96e0",
        "llama-2-13b-chat-hf",
    ),
    "arena-hard": (
        "94516f0ba7eaf0467943deb12538ad5e473d0d7ef984baaaef2a2a54c90b6a03",
        "bd8852a964aef17fb95aa3141e4ba1984272183968f4a85ac3c87cf54e38b1d5",
        "gpt-4o-2024-08-06",
    ),
    "bingo": (
        "d066efca199f4209b284c3c6b0917e1d9b613aa74fa2c506cf7fd2a6f23f5c45",
        "0cf1728c7dc6ddddebec71669f7eb4f20259cd0bcbd16cf5203ed8f19725d1f7",
        "HuggingFaceM4_idefics-9b-instruct",
    ),
    "flickr30k": (
        "3a1a2d03a63f5f45acf0a4b9dd85c78885c231a48ce0d9393631206f752433ba",
        "1fddbc5c8a37facb5f6d9d845ebf31994b7a6a5e85e7562d45908a44a8a26e06",
        "openai_gpt-4o-2024-05-13",
    ),
    "medi_qa": (
        "cd76912a786c0ed3f1081b1418b14e8b5827b8e511844a9d292338327fd1b7a5",
        "2494bc845b9c16b2d223b48858311a594f95eb304781727e2800acfd90dcaf0a",
        "openai_gpt-4o-mini-2024-07-18",
    ),
    "mt-bench": (
        "8bbe2087b844032c7fd0fd04dcfde3817041b9075233c06352fd601adb20d8f0",
        "a00c992bbf4fdd487d76ae277153b315ea677be571a8bd5b50b477f59ade6574",
        "alpaca-13b",
    ),
}

N_SEEDS = 1000
BUDGET = 30
N_ALIASES = 4
BOOTSTRAP_REPS = 50_000
SIGNFLIP_REPS = 100_000
EPS_LOSS = 0.2
EPS_DRAW = 0.4
EPS_WIN = 0.4


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed64(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")


def assert_close(actual: float, expected: float, tol: float = 1e-12, label: str = "") -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def load_source(name: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    judge_sha, weak_sha, parent = DATASETS[name]
    judge_path = REPO / name / "judge.json"
    weak_path = REPO / name / "weak_judge.json"
    assert file_hash(judge_path) == judge_sha
    assert file_hash(weak_path) == weak_sha
    with judge_path.open("r", encoding="utf-8") as handle:
        jd = json.load(handle)
    with weak_path.open("r", encoding="utf-8") as handle:
        wd = json.load(handle)
    names = [key for key, value in jd.items() if value is not None]
    assert names[0] == parent
    judge = np.asarray([jd[key] for key in names], dtype=np.int8)
    weak = np.asarray([wd[key] for key in names], dtype=np.int8)
    return names, judge, weak


def reconstruct_aliases(name: str, parent_weak: np.ndarray) -> np.ndarray:
    out = np.repeat(parent_weak[None], N_ALIASES, axis=0)
    forward = {-1: 0, 0: 1, 1: -1}
    reverse = {-1: 1, 0: -1, 1: 0}
    for alias in range(N_ALIASES):
        mapping = forward if alias in (0, 2) else reverse
        for query in range(parent_weak.shape[0]):
            for metric in range(parent_weak.shape[1]):
                raw = hashlib.sha256(f"step88|{name}|{query}|{metric}|{alias}".encode()).digest()
                if int.from_bytes(raw[:8], "big") / float(1 << 64) < 0.20:
                    out[alias, query, metric] = mapping[int(parent_weak[query, metric])]
    return out


def replay_literal(judge_entries: np.ndarray, weak_entries: np.ndarray, original: np.ndarray, pool: list[int]) -> dict[str, Any]:
    """Literal, independently written form of upstream `_llm_selector_one_iter`."""
    remaining = list(pool)
    posterior = np.ones(judge_entries.shape[0] + 1, dtype=float)
    posterior /= posterior.size
    pool_array = np.asarray(pool, dtype=int)
    pool_scores = np.concatenate((original[:, pool_array].mean(axis=1), np.zeros(1)))
    pool_wrs = (pool_scores + 1.0) / 2.0
    best_wr = float(pool_wrs.max())
    acquired: list[int] = []
    path: list[int] = []
    roots: list[int] = []
    regrets: list[float] = []
    for _ in range(BUDGET):
        criterion = np.zeros(len(remaining), dtype=float)
        for metric in range(weak_entries.shape[2]):
            games = np.concatenate(
                (weak_entries[:, remaining, metric], np.zeros((1, len(remaining)), dtype=np.int8)), axis=0
            )
            factors = np.where(games < 0, EPS_LOSS, np.where(games == 0, EPS_DRAW, EPS_WIN))
            candidate_posteriors = posterior[:, None] * factors
            candidate_posteriors /= candidate_posteriors.sum(axis=0, keepdims=True)
            criterion += entropy(candidate_posteriors, axis=0) / weak_entries.shape[2]
        local_index = int(np.argmin(criterion))
        query = int(remaining.pop(local_index))
        acquired.append(query)
        path.append(query)
        observed = np.concatenate((judge_entries[:, query], np.zeros(1, dtype=np.int8)))
        posterior *= np.where(observed < 0, EPS_LOSS, np.where(observed == 0, EPS_DRAW, EPS_WIN))
        posterior /= posterior.sum()
        root_scores = np.concatenate((original[:, acquired].mean(axis=1), np.zeros(1)))
        root = int(np.argmax(root_scores))
        roots.append(root)
        regrets.append(best_wr - float(pool_wrs[root]))
    return {"path": path, "chosen_roots": roots, "regrets": regrets}


def audit_seed_set(name: str) -> list[int]:
    seeds = list(range(20))
    cursor = 0
    while len(seeds) < 50:
        candidate = seed64(f"step88-independent-replay|{name}|{cursor}") % N_SEEDS
        cursor += 1
        if candidate not in seeds:
            seeds.append(candidate)
    return sorted(seeds)


def bootstrap(values: np.ndarray, name: str) -> list[float]:
    rng = np.random.default_rng(seed64(f"step88-bootstrap|{name}"))
    means = np.empty(BOOTSTRAP_REPS)
    start = 0
    while start < BOOTSTRAP_REPS:
        batch = min(500, BOOTSTRAP_REPS - start)
        indices = rng.integers(0, values.size, size=(batch, values.size))
        means[start : start + batch] = values[indices].mean(axis=1)
        start += batch
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def signflip(values: np.ndarray, name: str) -> float:
    rng = np.random.default_rng(seed64(f"step88-signflip|{name}"))
    observed = float(values.mean())
    exceed = 0
    completed = 0
    while completed < SIGNFLIP_REPS:
        batch = min(1000, SIGNFLIP_REPS - completed)
        # Match the frozen Monte Carlo generator exactly.  `choice` consumes a
        # different NumPy bit stream even under the same seed.
        signs = rng.integers(0, 2, size=(batch, values.size), dtype=np.int8) * 2 - 1
        exceed += int(np.count_nonzero((signs * values).mean(axis=1) >= observed - 1e-15))
        completed += batch
    return (exceed + 1.0) / (SIGNFLIP_REPS + 1.0)


def bh(pvalues: dict[str, float]) -> dict[str, float]:
    pairs = sorted(pvalues.items(), key=lambda x: x[1])
    out: dict[str, float] = {}
    cap = 1.0
    for index in range(len(pairs) - 1, -1, -1):
        name, value = pairs[index]
        cap = min(cap, value * len(pairs) / (index + 1))
        out[name] = cap
    return out


def main() -> int:
    with RESULT_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["schema"] == "step88-llm-selector-audit-v1"
    assert payload["source"]["runner_sha256"] == file_hash(RUNNER_PATH)
    assert list(payload["datasets"]) == list(DATASETS)
    assert list(payload["summaries"]) == list(DATASETS)

    checks = 0
    replayed = 0
    recomputed: dict[str, dict[str, Any]] = {}

    for name in DATASETS:
        names, judge, weak = load_source(name)
        aliases_weak = reconstruct_aliases(name, weak[0])
        aliases_judge = np.repeat(judge[0][None], N_ALIASES, axis=0)
        refined_judge = np.concatenate((judge, aliases_judge), axis=0)
        refined_weak = np.concatenate((weak, aliases_weak), axis=0)
        stored = payload["datasets"][name]
        raw = stored["raw"]
        assert len(raw) == N_SEEDS
        assert stored["model_names"] == names
        assert stored["parent"] == names[0]
        assert stored["strong_utility_max_abs_difference"] == 0
        assert_close(stored["strong_utility_mean_abs_difference"], 0.0, label=f"{name} utility")
        for alias, value in enumerate(stored["weak_hamming_by_alias"]):
            assert_close(value, np.mean(aliases_weak[alias] != weak[0]), label=f"{name} hamming {alias}")
        checks += 8

        by_seed = {int(row["seed"]): row for row in raw}
        assert list(by_seed) == list(range(N_SEEDS))
        terminal = []
        cumulative = []
        for seed in range(N_SEEDS):
            row = by_seed[seed]
            expected_pool = random.Random(seed).sample(range(judge.shape[1]), stored["pool_size"])
            assert row["pool"] == expected_pool
            assert len(row["clean_path"]) == BUDGET == len(row["refined_path"])
            assert len(set(row["clean_path"])) == BUDGET
            assert len(set(row["refined_path"])) == BUDGET
            assert set(row["clean_path"]).issubset(set(expected_pool))
            assert set(row["refined_path"]).issubset(set(expected_pool))
            clean_regrets = np.asarray(row["clean_regrets"], dtype=float)
            refined_regrets = np.asarray(row["refined_regrets"], dtype=float)
            fixed_regrets = np.asarray(row["fixed_regrets"], dtype=float)
            assert np.array_equal(clean_regrets, fixed_regrets)
            assert_close(row["terminal_delta"], refined_regrets[-1] - clean_regrets[-1])
            assert_close(row["cumulative_delta"], refined_regrets.sum() - clean_regrets.sum())
            assert_close(row["fixed_terminal_delta"], 0.0)
            assert_close(row["fixed_cumulative_delta"], 0.0)
            assert row["ordered_path_changed"] == (row["clean_path"] != row["refined_path"])
            assert row["query_set_changed"] == (set(row["clean_path"]) != set(row["refined_path"]))
            assert row["terminal_root_changed"] == (
                row["clean_chosen_roots"][-1] != row["refined_chosen_roots"][-1]
            )
            terminal.append(float(row["terminal_delta"]))
            cumulative.append(float(row["cumulative_delta"]))
            checks += 14

        for seed in audit_seed_set(name):
            row = by_seed[seed]
            clean = replay_literal(judge, weak, judge, row["pool"])
            refined = replay_literal(refined_judge, refined_weak, judge, row["pool"])
            assert clean["path"] == row["clean_path"]
            assert refined["path"] == row["refined_path"]
            assert clean["chosen_roots"] == row["clean_chosen_roots"]
            assert refined["chosen_roots"] == row["refined_chosen_roots"]
            assert np.allclose(clean["regrets"], row["clean_regrets"], rtol=0.0, atol=1e-12)
            assert np.allclose(refined["regrets"], row["refined_regrets"], rtol=0.0, atol=1e-12)
            checks += 6
            replayed += 1

        terminal_array = np.asarray(terminal)
        cumulative_array = np.asarray(cumulative)
        recomputed[name] = {
            "mean_terminal_delta": float(terminal_array.mean()),
            "median_terminal_delta": float(np.median(terminal_array)),
            "terminal_bootstrap_95_ci": bootstrap(terminal_array, name),
            "terminal_signflip_p_one_sided": signflip(terminal_array, name),
            "mean_cumulative_delta": float(cumulative_array.mean()),
            "median_cumulative_delta": float(np.median(cumulative_array)),
            "ordered_path_change_rate": float(np.mean([row["ordered_path_changed"] for row in raw])),
            "query_set_change_rate": float(np.mean([row["query_set_changed"] for row in raw])),
            "mean_query_set_jaccard": float(np.mean([row["query_set_jaccard"] for row in raw])),
            "terminal_root_change_rate": float(np.mean([row["terminal_root_changed"] for row in raw])),
            "max_abs_fixed_terminal_delta": float(max(abs(row["fixed_terminal_delta"]) for row in raw)),
            "max_abs_fixed_cumulative_delta": float(max(abs(row["fixed_cumulative_delta"]) for row in raw)),
        }
        for key, value in recomputed[name].items():
            expected = payload["summaries"][name][key]
            if isinstance(value, list):
                assert np.allclose(value, expected, rtol=0.0, atol=1e-12), (name, key, value, expected)
            else:
                assert_close(value, expected, label=f"{name} {key}")
            checks += 1

    qvalues = bh({name: result["terminal_signflip_p_one_sided"] for name, result in recomputed.items()})
    passes: list[str] = []
    for name in DATASETS:
        stored = payload["summaries"][name]
        assert_close(qvalues[name], stored["terminal_signflip_q_bh"], label=f"{name} q")
        gate = (
            payload["datasets"][name]["strong_utility_max_abs_difference"] == 0
            and recomputed[name]["max_abs_fixed_terminal_delta"] <= 1e-12
            and recomputed[name]["max_abs_fixed_cumulative_delta"] <= 1e-12
            and recomputed[name]["ordered_path_change_rate"] >= 0.50
            and recomputed[name]["mean_terminal_delta"] >= 0.0025
            and recomputed[name]["terminal_bootstrap_95_ci"][0] > 0.0
            and qvalues[name] <= 0.05
        )
        assert gate == stored["passes_terminal_harm_gate"]
        if gate:
            passes.append(name)
        checks += 2
    decision = (
        "GO_STRONG_CROSS_SELECTOR" if len(passes) >= 2 else
        "GO_LIMITED_CROSS_SELECTOR" if len(passes) == 1 else
        "NO_CONFIRMATORY_CROSS_SELECTOR"
    )
    assert passes == payload["passing_datasets"]
    assert decision == payload["decision"]
    checks += 2

    receipt = {
        "schema": "step88-independent-validation-receipt-v1",
        "status": "PASS_STEP88_INDEPENDENT_VALIDATION",
        "decision_reconstructed": decision,
        "passing_datasets_reconstructed": passes,
        "checks": checks,
        "literal_official_path_replays": replayed,
        "literal_replay_seeds_by_dataset": {name: audit_seed_set(name) for name in DATASETS},
        "hashes": {
            "preregistration_sha256": file_hash(PREREG_PATH),
            "runner_sha256": file_hash(RUNNER_PATH),
            "results_sha256": file_hash(RESULT_PATH),
            "validator_sha256": file_hash(Path(__file__).resolve()),
        },
        "recomputed_summaries": recomputed,
    }
    with RECEIPT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
