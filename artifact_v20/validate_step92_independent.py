from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from scipy import sparse
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-12"
CONFIG_PATH = ROOT / f"STEP92_STAGEA_FROZEN_CONFIG_{DATE}.json"
LOCK_PATH = ROOT / f"STEP92_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json"
INPUT_PATH = ROOT / f"STEP92_AGNEWS_HOLDOUT_INPUTS_{DATE}.json"
RAW_PATH = ROOT / f"STEP92_LEARNED_ADAPTER_CONFIRMATORY_RAW_{DATE}.npz"
RESULT_PATH = ROOT / f"STEP92_LEARNED_ADAPTER_CONFIRMATORY_RESULTS_{DATE}.json"
RECEIPT_PATH = ROOT / f"STEP92_INDEPENDENT_VALIDATION_{DATE}.json"
MODEL_DIR = ROOT / "step92_models"
BASE_MODEL = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
BASE_SHA256 = "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
N_CLASSES = 4
N_ROOTS = 12
N_ALIASES = 4
N_RUNS = 1000
POOL_SIZE = 400
BUDGET = 50
SEEDS = tuple(range(N_RUNS))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def choose_query(tied: np.ndarray, pool: np.ndarray, seed: int, step: int) -> int:
    return min(
        map(int, tied),
        key=lambda position: (
            stable_u64("step92", "query", seed, step, int(pool[position])),
            int(pool[position]),
        ),
    )


def choose_root(entries: np.ndarray, parents: np.ndarray, seed: int, step: int) -> int:
    roots = sorted({int(parents[int(entry)]) for entry in entries})
    return min(roots, key=lambda root: (stable_u64("step92", "root", seed, step, root), root))


@dataclass
class Grouping:
    membership: sparse.csr_matrix
    query_index: np.ndarray


def independently_group(codes: np.ndarray) -> Grouping:
    pool_size, entries = codes.shape
    row_ids = np.empty_like(codes, dtype=np.int64)
    query_ids: list[int] = []
    offset = 0
    for query in range(pool_size):
        # The locked runner orders response groups by sorted integer code.
        # Reproduce that numerical contract explicitly: mathematically the
        # group ordering is irrelevant, but floating-point summation order is
        # observable at exact acquisition ties.
        unique_codes = sorted({int(value) for value in codes[query]})
        mapping = {code: offset + rank for rank, code in enumerate(unique_codes)}
        for entry in range(entries):
            row_ids[query, entry] = mapping[int(codes[query, entry])]
        query_ids.extend([query] * len(unique_codes))
        offset += len(unique_codes)
    rows = row_ids.reshape(-1)
    columns = np.tile(np.arange(entries), pool_size)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(offset, entries)
    )
    return Grouping(matrix, np.asarray(query_ids, dtype=np.int64))


def acquisition(grouping: Grouping, posterior: np.ndarray, pool_size: int) -> np.ndarray:
    group_mass = np.asarray(grouping.membership.dot(posterior)).reshape(-1)
    return np.bincount(
        grouping.query_index,
        weights=np.square(group_mass),
        minlength=pool_size,
    )


def replay_active(
    codes: np.ndarray,
    feedback: np.ndarray,
    parents: np.ndarray,
    core_feedback: np.ndarray,
    pools: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    queries = np.full((N_RUNS, BUDGET), -1, dtype=np.int64)
    roots = np.full((N_RUNS, BUDGET), -1, dtype=np.int64)
    terminal = np.zeros(N_RUNS, dtype=np.float64)
    cumulative = np.zeros(N_RUNS, dtype=np.float64)
    for run_index, seed in enumerate(SEEDS):
        pool = pools[run_index]
        grouping = independently_group(codes[pool])
        available = np.ones(len(pool), dtype=bool)
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        quality = np.mean(core_feedback[pool], axis=0)
        best = float(np.max(quality))
        for step in range(BUDGET):
            shifted = scores / tau
            shifted -= float(np.max(shifted))
            posterior = np.exp(shifted)
            posterior /= float(np.sum(posterior))
            values = acquisition(grouping, posterior, len(pool))
            values[~available] = np.inf
            tied = np.flatnonzero(values == float(np.min(values)))
            position = choose_query(tied, pool, seed, step)
            available[position] = False
            query = int(pool[position])
            queries[run_index, step] = query
            scores += feedback[query]
            root = choose_root(np.flatnonzero(scores == float(np.max(scores))), parents, seed, step)
            roots[run_index, step] = root
            regret = best - float(quality[root])
            cumulative[run_index] += regret
            if step == BUDGET - 1:
                terminal[run_index] = regret
    return queries, roots, terminal, cumulative


def replay_fixed(
    feedback: np.ndarray,
    parents: np.ndarray,
    core_feedback: np.ndarray,
    pools: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roots = np.full_like(queries, -1)
    terminal = np.zeros(N_RUNS, dtype=np.float64)
    cumulative = np.zeros(N_RUNS, dtype=np.float64)
    for run_index, seed in enumerate(SEEDS):
        quality = np.mean(core_feedback[pools[run_index]], axis=0)
        best = float(np.max(quality))
        scores = np.zeros(feedback.shape[1], dtype=np.float64)
        for step, query in enumerate(queries[run_index]):
            scores += feedback[int(query)]
            root = choose_root(np.flatnonzero(scores == float(np.max(scores))), parents, seed, step)
            roots[run_index, step] = root
            regret = best - float(quality[root])
            cumulative[run_index] += regret
            if step == BUDGET - 1:
                terminal[run_index] = regret
    return roots, terminal, cumulative


def bootstrap(values: np.ndarray, seed: int, repetitions: int = 50_000) -> list[float]:
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    remaining = repetitions
    while remaining:
        count = min(2000, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        blocks.append(np.mean(values[indices], axis=1))
        remaining -= count
    return [float(x) for x in np.quantile(np.concatenate(blocks), [0.025, 0.975])]


def signflip(values: np.ndarray, seed: int, repetitions: int = 100_000) -> float:
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = repetitions
    while remaining:
        count = min(2000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        exceed += int(np.sum(np.mean(signs * values[None, :], axis=1) >= observed))
        remaining -= count
    return float((exceed + 1) / (repetitions + 1))


def load_tensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    return tensors, metadata


def manual_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(variance + 1e-5) * weight + bias


def manual_classifier(hidden: torch.Tensor, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    x = manual_layer_norm(hidden, tensors["norm.weight"], tensors["norm.bias"])
    x = torch.nn.functional.gelu(x @ tensors["down.weight"].T + tensors["down.bias"])
    return x @ tensors["out.weight"].T + tensors["out.bias"]


def manual_log_temperature(hidden: torch.Tensor, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    x = manual_layer_norm(hidden, tensors["norm.weight"], tensors["norm.bias"])
    x = torch.nn.functional.gelu(x @ tensors["down.weight"].T + tensors["down.bias"])
    return torch.clamp((x @ tensors["out.weight"].T + tensors["out.bias"])[:, 0], -3.0, 3.0)


def independent_endpoint_replay(
    raw: Any, config: dict[str, Any], check_indices: np.ndarray
) -> dict[str, Any]:
    base_path = Path(
        hf_hub_download(BASE_MODEL, "model.safetensors", revision=BASE_REVISION, local_files_only=True)
    )
    assert sha256_path(base_path) == BASE_SHA256
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))["rows"]
    texts = [str(inputs[int(index)]["text"]) for index in check_indices]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True, use_safetensors=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).eval()
    pieces: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            batch = tokenizer(
                texts[start : start + 64],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            pieces.append(encoder(**batch).last_hidden_state[:, 0, :].float().cpu())
    hidden = torch.cat(pieces)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    max_root_logit_error = 0.0
    root_prediction_checks = 0
    for root in range(N_ROOTS):
        path = MODEL_DIR / f"step92_clean_root_{root:02d}.safetensors"
        tensors, metadata = load_tensors(path)
        assert metadata["kind"] == "learned_bottleneck_classification_adapter"
        logits = manual_classifier(hidden, tensors).detach().numpy()
        stored = raw["clean_root_logits"][check_indices, root, :]
        max_root_logit_error = max(max_root_logit_error, float(np.max(np.abs(logits - stored))))
        assert np.array_equal(np.argmax(logits, axis=1), raw["clean_root_predictions"][check_indices, root])
        root_prediction_checks += len(check_indices)

    parent = int(config["selected"]["parent_root"])
    max_adapter_log_error = 0.0
    adapter_parameter_counts: list[int] = []
    adapter_hashes: list[str] = []
    expected_names = {"norm.weight", "norm.bias", "down.weight", "down.bias", "out.weight", "out.bias"}
    for alias in range(N_ALIASES):
        path = MODEL_DIR / f"step92_parent_{parent:02d}_learned_adapter_{alias}.safetensors"
        tensors, metadata = load_tensors(path)
        assert set(tensors) == expected_names
        assert metadata["kind"] == "learned_contextual_temperature_adapter"
        assert metadata["item_lookup_table"] == "False"
        count = sum(tensor.numel() for tensor in tensors.values())
        assert count == 50_817
        assert all(7600 not in tensor.shape and 120000 not in tensor.shape for tensor in tensors.values())
        values = torch.cat([tensor.float().reshape(-1) for tensor in tensors.values()])
        assert torch.var(values, unbiased=False).item() > 0.0
        adapter_parameter_counts.append(count)
        adapter_hashes.append(sha256_path(path))
        replayed = manual_log_temperature(hidden, tensors).detach().numpy()
        stored = raw["adapter_log_temperatures"][check_indices, alias]
        max_adapter_log_error = max(max_adapter_log_error, float(np.max(np.abs(replayed - stored))))
    assert len(set(adapter_hashes)) == 4
    assert max_root_logit_error <= 2e-5
    assert max_adapter_log_error <= 2e-5
    return {
        "sample_size": len(check_indices),
        "sample_sha256": hashlib.sha256(check_indices.tobytes()).hexdigest(),
        "root_prediction_checks": root_prediction_checks,
        "adapter_parameter_counts": adapter_parameter_counts,
        "distinct_adapter_hashes": len(set(adapter_hashes)),
        "max_abs_root_logit_replay_error": max_root_logit_error,
        "max_abs_adapter_log_temperature_replay_error": max_adapter_log_error,
    }


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError((actual, expected, tolerance))


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    checks = 0
    for relative, expected in lock["locked_sha256"].items():
        assert sha256_path(ROOT / Path(relative)) == expected
        checks += 1
    assert result["config_sha256"] == sha256_path(CONFIG_PATH)
    assert result["execution_lock_sha256"] == sha256_path(LOCK_PATH)
    assert result["raw_sha256"] == sha256_path(RAW_PATH)
    checks += 3

    raw = np.load(RAW_PATH, allow_pickle=False)
    expected_shapes = {
        "source_indices": (7600,),
        "labels": (7600,),
        "clean_root_logits": (7600, N_ROOTS, N_CLASSES),
        "clean_root_predictions": (7600, N_ROOTS),
        "parent_hard": (7600,),
        "adapter_log_temperatures": (7600, N_ALIASES),
        "adapter_response_codes": (7600, N_ALIASES),
        "pools": (N_RUNS, POOL_SIZE),
        "clean_queries": (N_RUNS, BUDGET),
        "refined_queries": (N_RUNS, BUDGET),
        "clean_roots": (N_RUNS, BUDGET),
        "refined_roots": (N_RUNS, BUDGET),
        "terminal_delta": (N_RUNS,),
        "cumulative_delta": (N_RUNS,),
        "fixed_terminal_delta": (N_RUNS,),
        "fixed_cumulative_delta": (N_RUNS,),
        "active_minus_fixed_terminal": (N_RUNS,),
    }
    for name, shape in expected_shapes.items():
        assert raw[name].shape == shape, (name, raw[name].shape, shape)
        checks += 1
    assert np.array_equal(raw["source_indices"], np.arange(7600))
    expected_pools = np.asarray(
        [sorted(random.Random(seed).sample(range(7600), POOL_SIZE)) for seed in SEEDS],
        dtype=np.int64,
    )
    assert np.array_equal(raw["pools"], expected_pools)
    checks += 2

    parent = int(config["selected"]["parent_root"])
    bins = int(config["selected"]["bins"])
    assert np.array_equal(raw["parent_hard"], raw["clean_root_predictions"][:, parent])
    decoded = raw["adapter_response_codes"].copy()
    triggered = decoded >= N_CLASSES
    decoded[triggered] = (decoded[triggered] - N_CLASSES) // bins
    assert np.array_equal(decoded, np.repeat(raw["parent_hard"][:, None], N_ALIASES, axis=1))
    assert np.all(raw["fixed_terminal_delta"] == 0.0)
    assert np.all(raw["fixed_cumulative_delta"] == 0.0)
    assert np.array_equal(raw["clean_fixed_roots"], raw["refined_fixed_roots"])
    checks += 5

    labels = raw["labels"].astype(np.int8)
    predictions = raw["clean_root_predictions"].astype(np.int8)
    core_feedback = (predictions == labels[:, None]).astype(np.float64)
    clean_codes = predictions.astype(np.int64)
    clean_feedback = core_feedback.copy()
    clean_parents = np.arange(N_ROOTS, dtype=np.int64)
    refined_codes = np.column_stack([clean_codes, raw["adapter_response_codes"]]).astype(np.int64)
    refined_feedback = np.column_stack(
        [core_feedback, np.repeat(core_feedback[:, [parent]], N_ALIASES, axis=1)]
    )
    refined_parents = np.concatenate([clean_parents, np.full(N_ALIASES, parent, dtype=np.int64)])
    tau = float(config["selected"]["tau"])
    clean_queries, clean_roots, clean_terminal, clean_cumulative = replay_active(
        clean_codes, clean_feedback, clean_parents, core_feedback, expected_pools, tau
    )
    refined_queries, refined_roots, refined_terminal, refined_cumulative = replay_active(
        refined_codes, refined_feedback, refined_parents, core_feedback, expected_pools, tau
    )
    assert np.array_equal(clean_queries, raw["clean_queries"])
    assert np.array_equal(refined_queries, raw["refined_queries"])
    assert np.array_equal(clean_roots, raw["clean_roots"])
    assert np.array_equal(refined_roots, raw["refined_roots"])
    assert np.array_equal(refined_terminal - clean_terminal, raw["terminal_delta"])
    assert np.array_equal(refined_cumulative - clean_cumulative, raw["cumulative_delta"])
    checks += 6

    clean_fixed_roots, clean_fixed_terminal, clean_fixed_cumulative = replay_fixed(
        clean_feedback, clean_parents, core_feedback, expected_pools, clean_queries
    )
    refined_fixed_roots, refined_fixed_terminal, refined_fixed_cumulative = replay_fixed(
        refined_feedback, refined_parents, core_feedback, expected_pools, clean_queries
    )
    assert np.array_equal(clean_fixed_roots, raw["clean_fixed_roots"])
    assert np.array_equal(refined_fixed_roots, raw["refined_fixed_roots"])
    assert np.array_equal(refined_fixed_terminal - clean_fixed_terminal, raw["fixed_terminal_delta"])
    assert np.array_equal(refined_fixed_cumulative - clean_fixed_cumulative, raw["fixed_cumulative_delta"])
    checks += 4

    terminal = raw["terminal_delta"]
    active_minus_fixed = terminal - raw["fixed_terminal_delta"]
    assert np.array_equal(active_minus_fixed, raw["active_minus_fixed_terminal"])
    terminal_ci = bootstrap(terminal, 920_100)
    terminal_p = signflip(terminal, 920_101)
    active_ci = bootstrap(active_minus_fixed, 920_300)
    stored = result["task"]
    close(float(np.mean(terminal)), stored["terminal_regret_delta"]["mean"])
    close(terminal_ci[0], stored["terminal_regret_delta"]["bootstrap_95"][0])
    close(terminal_ci[1], stored["terminal_regret_delta"]["bootstrap_95"][1])
    close(terminal_p, stored["terminal_regret_delta"]["one_sided_signflip_p"])
    close(float(np.mean(active_minus_fixed)), stored["active_minus_fixed_terminal"]["mean"])
    close(active_ci[0], stored["active_minus_fixed_terminal"]["bootstrap_95"][0])
    close(active_ci[1], stored["active_minus_fixed_terminal"]["bootstrap_95"][1])
    path_change = float(np.mean(np.any(clean_queries != refined_queries, axis=1)))
    close(path_change, stored["path_change_rate"])
    checks += 9

    gates = {
        "all_locked_hashes_match": True,
        "genuine_learned_adapter_files": True,
        "coordinate_wise_hard_utility_equal": True,
        "fixed_query_exact_zero": True,
        "path_change_at_least_half": path_change >= 0.50,
        "mean_terminal_delta_at_least_0_5pp": float(np.mean(terminal)) >= 0.005,
        "terminal_bootstrap_lower_above_zero": terminal_ci[0] > 0.0,
        "one_sided_signflip_p_at_most_0_05": terminal_p <= 0.05,
        "active_minus_fixed_at_least_0_5pp": (
            float(np.mean(active_minus_fixed)) >= 0.005 and active_ci[0] > 0.0
        ),
    }
    assert gates == stored["gates"]
    decision = (
        "GO_SCORE_CHANGING_LEARNED_ADAPTER_BRIDGE"
        if all(gates.values())
        else "NO_SCORE_CHANGING_LEARNED_ADAPTER_BRIDGE"
    )
    assert decision == result["decision"]
    checks += 2

    sample_indices = np.asarray(
        sorted(range(7600), key=lambda index: stable_u64("step92-independent-endpoint", index))[:256],
        dtype=np.int64,
    )
    endpoint = independent_endpoint_replay(raw, config, sample_indices)
    checks += endpoint["root_prediction_checks"] + endpoint["sample_size"] * N_ALIASES

    receipt = {
        "validation_id": "STEP92_INDEPENDENT_VALIDATION_V1",
        "verdict": "PASS_STEP92_INDEPENDENT_VALIDATION",
        "result_sha256": sha256_path(RESULT_PATH),
        "raw_sha256": sha256_path(RAW_PATH),
        "config_sha256": sha256_path(CONFIG_PATH),
        "execution_lock_sha256": sha256_path(LOCK_PATH),
        "checks": checks,
        "active_trajectory_replays": 2 * N_RUNS,
        "fixed_query_replays": 2 * N_RUNS,
        "endpoint_replay": endpoint,
        "reconstructed": {
            "decision": decision,
            "mean_terminal_delta_pp": 100.0 * float(np.mean(terminal)),
            "terminal_bootstrap_95_pp": [100.0 * value for value in terminal_ci],
            "one_sided_signflip_p": terminal_p,
            "active_minus_fixed_pp": 100.0 * float(np.mean(active_minus_fixed)),
            "path_change_rate": path_change,
            "fixed_query_exact_zero": True,
        },
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
