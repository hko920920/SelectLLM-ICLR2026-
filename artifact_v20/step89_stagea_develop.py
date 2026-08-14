from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from step61a_blind_common import build_group_structure, select_llm_acquisition


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-12"
MODEL_NAME = "cross-encoder/qnli-distilroberta-base"
MODEL_REVISION = "7dd04ee0a6040c06fb381ad7edcb8585f4d937fd"
CALIBRATION_PATH = ROOT / f"STEP89_QNLI_CALIBRATION_PACKAGE_{DATE}.json"
STAGE0_MANIFEST = ROOT / f"STEP89_QNLI_STAGE0_SPLIT_MANIFEST_{DATE}.json"
PREDICTIONS_PATH = ROOT / f"STEP89_QNLI_CALIBRATION_PARENT_OUTPUTS_{DATE}.npz"
SEARCH_PATH = ROOT / f"STEP89_QNLI_STAGEA_SEARCH_LEDGER_{DATE}.json"
CONFIG_PATH = ROOT / f"STEP89_QNLI_STAGEA_FROZEN_CONFIG_{DATE}.json"
MODEL_DIR = ROOT / "step89_models"

PILOT_SEEDS = tuple(range(20))
VERIFY_SEEDS = tuple(range(300))
ROSTER_SIZES = (8, 12, 20)
POOL_SIZE = 400
BUDGETS = (20, 30, 50)
TAUS = (0.25, 1.0, 4.0)
BINS = (4, 8, 16)
SCALE_SETS = {
    "owner_only_symmetric": (-2.0, -0.5, 0.5, 2.0),
    "owner_only_mild": (-1.0, -0.5, 0.5, 1.0),
    "owner_only_extreme": (-2.0, -1.5, 1.5, 2.0),
    "owner_only_cool": (-2.0, -1.0, -0.5, 0.5),
    "owner_only_sharp": (-0.5, 0.5, 1.0, 2.0),
}
TOP_CONFIGS_TO_VERIFY = 12

_POOL_CACHE: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
_GROUP_CACHE: dict[tuple[str, str], list[Any]] = {}
_RUN_CACHE: dict[tuple[object, ...], RunSummary] = {}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def infer_logits(rows: list[dict[str, Any]]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, use_safetensors=True
    )
    if model.config.num_labels != 1:
        raise AssertionError(f"expected one-logit cross encoder, found {model.config.num_labels}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            batch = rows[start : start + 64]
            encoded = tokenizer(
                [str(row["question"]) for row in batch],
                [str(row["sentence"]) for row in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits.append(model(**encoded).logits[:, 0].float().cpu().numpy())
    return np.concatenate(logits).astype(np.float64)


def infer_parent(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    values = infer_logits(rows)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    positive_to_zero = (values < 0).astype(np.int8)
    positive_to_one = (values >= 0).astype(np.int8)
    acc_zero = float(np.mean(positive_to_zero == labels))
    acc_one = float(np.mean(positive_to_one == labels))
    if acc_zero <= acc_one:
        raise AssertionError({"positive_to_label0": acc_zero, "positive_to_label1": acc_one})
    return values, positive_to_zero


def response_codes(parent_logits: np.ndarray, log_scales: tuple[float, ...], bins: int) -> np.ndarray:
    hard = (parent_logits < 0).astype(np.int64)
    magnitude = np.abs(parent_logits)
    aliases = []
    for log_scale in log_scales:
        confidence = 1.0 / (1.0 + np.exp(-magnitude * math.exp(log_scale)))
        bucket = np.minimum(bins - 1, np.floor(confidence * bins).astype(np.int64))
        aliases.append(2 + hard * bins + bucket)
    return np.stack(aliases, axis=1)


def top_roster(matrix: np.ndarray, labels: np.ndarray, size: int) -> np.ndarray:
    accuracy = np.mean(matrix == labels[:, None], axis=0)
    order = sorted(range(matrix.shape[1]), key=lambda index: (-float(accuracy[index]), index))
    return np.asarray(order[:size], dtype=np.int64)


def sample_pools(n: int, seeds: tuple[int, ...]) -> np.ndarray:
    key = (n, seeds)
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]
    pools = np.asarray(
        [sorted(random.Random(seed).sample(range(n), min(POOL_SIZE, n))) for seed in seeds],
        dtype=np.int64,
    )
    _POOL_CACHE[key] = pools
    return pools


def choose_query(tied: np.ndarray, pool: np.ndarray, seed: int, step: int) -> int:
    return min(
        (int(position) for position in tied),
        key=lambda position: (
            stable_u64("step89", "query", seed, step, int(pool[position])),
            int(pool[position]),
        ),
    )


def choose_root(tied_entries: np.ndarray, parents: np.ndarray, seed: int, step: int) -> int:
    roots = sorted(set(int(parents[int(entry)]) for entry in tied_entries))
    return min(
        roots,
        key=lambda root: (stable_u64("step89", "root", seed, step, root), root),
    )


@dataclass(frozen=True)
class RunSummary:
    terminal: np.ndarray
    cumulative: np.ndarray
    queries: np.ndarray
    roots: np.ndarray


def run_active(
    codes: np.ndarray,
    feedback: np.ndarray,
    parents: np.ndarray,
    core_feedback: np.ndarray,
    pools: np.ndarray,
    seeds: tuple[int, ...],
    budget: int,
    tau: float,
) -> RunSummary:
    codes_digest = hashlib.sha256(np.ascontiguousarray(codes).tobytes()).hexdigest()
    feedback_digest = hashlib.sha256(np.ascontiguousarray(feedback).tobytes()).hexdigest()
    parents_digest = hashlib.sha256(np.ascontiguousarray(parents).tobytes()).hexdigest()
    core_digest = hashlib.sha256(np.ascontiguousarray(core_feedback).tobytes()).hexdigest()
    pools_digest = hashlib.sha256(np.ascontiguousarray(pools).tobytes()).hexdigest()
    cache_key = (
        codes_digest,
        feedback_digest,
        parents_digest,
        core_digest,
        pools_digest,
        seeds,
        budget,
        tau,
    )
    if cache_key in _RUN_CACHE:
        return _RUN_CACHE[cache_key]
    group_key = (codes_digest, pools_digest)
    if group_key not in _GROUP_CACHE:
        _GROUP_CACHE[group_key] = [build_group_structure(codes[pool]) for pool in pools]
    cached_groups = _GROUP_CACHE[group_key]
    runs = len(seeds)
    terminal = np.zeros(runs, dtype=np.float64)
    cumulative_regret = np.zeros(runs, dtype=np.float64)
    queries = np.full((runs, budget), -1, dtype=np.int64)
    roots = np.full((runs, budget), -1, dtype=np.int64)
    entries = codes.shape[1]
    for run_index, seed in enumerate(seeds):
        pool = pools[run_index]
        pool_size = len(pool)
        groups = cached_groups[run_index]
        active = np.ones(pool_size, dtype=bool)
        cumulative_score = np.zeros(entries, dtype=np.float64)
        root_quality = np.mean(core_feedback[pool], axis=0)
        best_quality = float(np.max(root_quality))
        for step in range(budget):
            shifted = cumulative_score / tau
            shifted -= float(np.max(shifted))
            posterior = np.exp(shifted)
            posterior /= float(np.sum(posterior))
            acquisition = select_llm_acquisition(groups, posterior)
            acquisition[~active] = np.inf
            minimum = float(np.min(acquisition))
            tied = np.flatnonzero(acquisition == minimum)
            position = choose_query(tied, pool, int(seed), step)
            active[position] = False
            query = int(pool[position])
            queries[run_index, step] = query
            cumulative_score += feedback[query]
            maximum = float(np.max(cumulative_score))
            root = choose_root(np.flatnonzero(cumulative_score == maximum), parents, int(seed), step)
            roots[run_index, step] = root
            regret = best_quality - float(root_quality[root])
            cumulative_regret[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret
    summary = RunSummary(terminal=terminal, cumulative=cumulative_regret, queries=queries, roots=roots)
    _RUN_CACHE[cache_key] = summary
    return summary


def evaluate_config(
    official: np.ndarray,
    labels: np.ndarray,
    parent_logits: np.ndarray,
    parent_hard: np.ndarray,
    config: dict[str, Any],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    roster = top_roster(official, labels, int(config["roster_size"]))
    original_predictions = np.column_stack([official[:, roster], parent_hard]).astype(np.int8)
    core_feedback = (original_predictions == labels[:, None]).astype(np.float64)
    clean_codes = original_predictions.astype(np.int64)
    clean_feedback = core_feedback.copy()
    clean_parents = np.arange(clean_codes.shape[1], dtype=np.int64)

    scale_set = SCALE_SETS[str(config["scale_set"])]
    alias_codes = response_codes(parent_logits, scale_set, int(config["bins"]))
    parent_feedback = core_feedback[:, [-1]]
    refined_codes = np.column_stack([clean_codes, alias_codes])
    refined_feedback = np.column_stack([clean_feedback, np.repeat(parent_feedback, 4, axis=1)])
    parent_root = clean_codes.shape[1] - 1
    refined_parents = np.concatenate([clean_parents, np.full(4, parent_root, dtype=np.int64)])

    pools = sample_pools(len(labels), seeds)
    clean = run_active(
        clean_codes,
        clean_feedback,
        clean_parents,
        core_feedback,
        pools,
        seeds,
        int(config["budget"]),
        float(config["tau"]),
    )
    refined = run_active(
        refined_codes,
        refined_feedback,
        refined_parents,
        core_feedback,
        pools,
        seeds,
        int(config["budget"]),
        float(config["tau"]),
    )
    terminal_delta = refined.terminal - clean.terminal
    cumulative_delta = refined.cumulative - clean.cumulative
    path_changed = np.any(clean.queries != refined.queries, axis=1)
    final_root_changed = clean.roots[:, -1] != refined.roots[:, -1]
    return {
        **config,
        "seeds": len(seeds),
        "roster_indices": roster.tolist(),
        "mean_terminal_delta": float(np.mean(terminal_delta)),
        "mean_cumulative_delta": float(np.mean(cumulative_delta)),
        "path_change_rate": float(np.mean(path_changed)),
        "final_root_change_rate": float(np.mean(final_root_changed)),
        "positive_terminal_fraction": float(np.mean(terminal_delta > 0)),
        "negative_terminal_fraction": float(np.mean(terminal_delta < 0)),
        "parent_calibration_accuracy": float(np.mean(parent_hard == labels)),
        "best_roster_calibration_accuracy": float(np.max(np.mean(core_feedback, axis=0))),
        "terminal_delta_values": terminal_delta.tolist() if len(seeds) >= 300 else None,
        "query_digest_clean": hashlib.sha256(clean.queries.tobytes()).hexdigest(),
        "query_digest_refined": hashlib.sha256(refined.queries.tobytes()).hexdigest(),
    }


def config_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    serial = json.dumps(
        {key: row[key] for key in ("roster_size", "budget", "tau", "bins", "scale_set")},
        sort_keys=True,
    )
    return (
        float(row["mean_terminal_delta"]),
        float(row["path_change_rate"]),
        float(row["mean_cumulative_delta"]),
        serial,
    )


def main() -> None:
    stage0 = json.loads(STAGE0_MANIFEST.read_text(encoding="utf-8"))
    package = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    rows = package["rows"]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    official = np.asarray([row["official_predictions"] for row in rows], dtype=np.int8)
    parent_logits, parent_hard = infer_parent(rows)
    np.savez_compressed(
        PREDICTIONS_PATH,
        positions=np.asarray([row["position"] for row in rows], dtype=np.int64),
        logits=parent_logits,
        hard_predictions=parent_hard,
        labels=labels,
    )

    pilot: list[dict[str, Any]] = []
    grid = list(itertools.product(ROSTER_SIZES, BUDGETS, TAUS, BINS, SCALE_SETS))
    for grid_index, (roster_size, budget, tau, bins, scale_set) in enumerate(grid, start=1):
        config = {
            "roster_size": roster_size,
            "pool_size": POOL_SIZE,
            "budget": budget,
            "tau": tau,
            "bins": bins,
            "scale_set": scale_set,
        }
        pilot.append(evaluate_config(official, labels, parent_logits, parent_hard, config, PILOT_SEEDS))
        if grid_index % 25 == 0 or grid_index == len(grid):
            print(f"pilot {grid_index}/{len(grid)}", flush=True)
    ranked = sorted(pilot, key=config_key, reverse=True)
    verified: list[dict[str, Any]] = []
    for verify_index, row in enumerate(ranked[:TOP_CONFIGS_TO_VERIFY], start=1):
        verified.append(evaluate_config(
            official,
            labels,
            parent_logits,
            parent_hard,
            {key: row[key] for key in ("roster_size", "pool_size", "budget", "tau", "bins", "scale_set")},
            VERIFY_SEEDS,
        ))
        print(f"verify {verify_index}/{TOP_CONFIGS_TO_VERIFY}", flush=True)
    selected = max(verified, key=config_key)

    MODEL_DIR.mkdir(exist_ok=True)
    adapter_hashes: dict[str, str] = {}
    selected_scales = SCALE_SETS[str(selected["scale_set"])]
    for index, log_scale in enumerate(selected_scales):
        path = MODEL_DIR / f"step89_adapter_{index}.safetensors"
        save_file(
            {
                "log_scale": torch.tensor([log_scale], dtype=torch.float32),
                "confidence_bins": torch.tensor([int(selected["bins"])], dtype=torch.int64),
            },
            str(path),
            metadata={
                "base_model": MODEL_NAME,
                "base_revision": MODEL_REVISION,
                "hard_label_invariant": "true",
            },
        )
        adapter_hashes[path.name] = sha256_path(path)

    ledger = {
        "ledger_id": "STEP89_QNLI_STAGEA_DEVELOPMENT_V1",
        "stage0_manifest_sha256": sha256_path(STAGE0_MANIFEST),
        "calibration_package_sha256": sha256_path(CALIBRATION_PATH),
        "parent_output_sha256": sha256_path(PREDICTIONS_PATH),
        "search_space": {
            "pilot_seeds": list(PILOT_SEEDS),
            "verification_seeds": [VERIFY_SEEDS[0], VERIFY_SEEDS[-1]],
            "roster_sizes": list(ROSTER_SIZES),
            "pool_size": POOL_SIZE,
            "budgets": list(BUDGETS),
            "taus": list(TAUS),
            "bins": list(BINS),
            "scale_sets": {key: list(value) for key, value in SCALE_SETS.items()},
            "pilot_configurations": len(pilot),
            "verified_configurations": len(verified),
        },
        "pilot_results": pilot,
        "verification_results": verified,
        "selection_rule": "maximum verified mean terminal delta, then path change, cumulative delta, serialized config",
        "selected": selected,
    }
    SEARCH_PATH.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    frozen = {
        "config_id": "STEP89_QNLI_EXECUTABLE_ADAPTER_CONFIRMATORY_V1",
        "dataset_revision": stage0["dataset_revision"],
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "positive_logit_maps_to_label": 0,
        "selected": {key: selected[key] for key in (
            "roster_size", "pool_size", "budget", "tau", "bins", "scale_set", "roster_indices"
        )},
        "adapter_log_scales": list(selected_scales),
        "adapter_sha256": adapter_hashes,
        "stagea_search_ledger_sha256": sha256_path(SEARCH_PATH),
        "calibration_parent_accuracy": selected["parent_calibration_accuracy"],
        "calibration_mean_terminal_delta": selected["mean_terminal_delta"],
        "calibration_path_change_rate": selected["path_change_rate"],
        "construction_access": {
            "train_or_calibration_labels": True,
            "calibration_peer_matrix_for_condition_selection": True,
            "holdout_labels": False,
            "holdout_peer_matrix": False,
            "holdout_parent_or_adapter_outputs": False,
            "item_lookup_table": False,
        },
    }
    CONFIG_PATH.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pilot_configurations": len(pilot),
        "verified_configurations": len(verified),
        "selected": frozen["selected"],
        "calibration_mean_terminal_delta": selected["mean_terminal_delta"],
        "calibration_mean_cumulative_delta": selected["mean_cumulative_delta"],
        "calibration_path_change_rate": selected["path_change_rate"],
        "config_sha256": sha256_path(CONFIG_PATH),
        "search_ledger_sha256": sha256_path(SEARCH_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
