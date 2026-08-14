from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import step89_stagea_develop as selector
import step90_stagea_develop as development


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-12"
CONFIG_PATH = ROOT / f"STEP90_HIGH_CONFIDENCE_STAGEA_FROZEN_CONFIG_{DATE}.json"
LOCK_PATH = ROOT / f"STEP90_HIGH_CONFIDENCE_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json"
RAW_PATH = ROOT / f"STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_RAW_{DATE}.npz"
RESULT_PATH = ROOT / f"STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_RESULTS_{DATE}.json"
SEALED_DIR = ROOT / "external_data" / "step90_sealed"
SEEDS = tuple(range(1000))
BOOTSTRAP_REPS = 50_000
SIGNFLIP_REPS = 100_000
MIN_TERMINAL_DELTA = 0.005
MIN_PATH_CHANGE = 0.50
MODEL_HASHES = {
    "mnli": "9df3eb5d37118f952f4ba4fb46fde6889e3a9ccedeee0bad09b0110fc64c5c29",
    "qqp": "73fd14ad7d08f3ef30eb25841c8f4ba89e91230f48159279233b37015ccb33fb",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_mapping(logits: np.ndarray, mapping: list[int]) -> tuple[np.ndarray, np.ndarray]:
    mapping_array = np.asarray(mapping, dtype=np.int64)
    if logits.shape[1] == 1:
        raw = (logits[:, 0] >= 0).astype(np.int64)
        hard = mapping_array[raw]
        confidence = 1.0 / (1.0 + np.exp(-np.abs(logits[:, 0])))
        return hard.astype(np.int8), confidence
    raw = np.argmax(logits, axis=1)
    hard = mapping_array[raw]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= np.sum(probability, axis=1, keepdims=True)
    return hard.astype(np.int8), np.max(probability, axis=1)


def run_fixed(
    feedback: np.ndarray,
    parents: np.ndarray,
    core_feedback: np.ndarray,
    pools: np.ndarray,
    queries: np.ndarray,
) -> selector.RunSummary:
    runs, budget = queries.shape
    terminal = np.zeros(runs, dtype=np.float64)
    cumulative = np.zeros(runs, dtype=np.float64)
    roots = np.full((runs, budget), -1, dtype=np.int64)
    for run_index, seed in enumerate(SEEDS):
        pool = pools[run_index]
        quality = np.mean(core_feedback[pool], axis=0)
        best = float(np.max(quality))
        scores = np.zeros(feedback.shape[1], dtype=np.float64)
        for step, query in enumerate(queries[run_index]):
            scores += feedback[int(query)]
            root = selector.choose_root(
                np.flatnonzero(scores == float(np.max(scores))), parents, int(seed), step
            )
            roots[run_index, step] = root
            regret = best - float(quality[root])
            cumulative[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret
    return selector.RunSummary(terminal, cumulative, np.asarray(queries), roots)


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


def summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "bootstrap_95": bootstrap(values, seed),
        "one_sided_signflip_p": signflip(values, seed + 1),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
        "zero_fraction": float(np.mean(values == 0)),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda task: (pvalues[task], task))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, task in enumerate(ordered):
        candidate = min(1.0, (count - rank) * pvalues[task])
        running = max(running, candidate)
        adjusted[task] = running
    return adjusted


def main() -> None:
    if not LOCK_PATH.exists():
        raise FileNotFoundError("execution lock missing")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    for relative, expected in lock["locked_sha256"].items():
        observed = sha256_path(ROOT / relative)
        if observed != expected:
            raise AssertionError({"path": relative, "expected": expected, "observed": observed})
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    task_rows: dict[str, dict[str, Any]] = {}
    task_work: dict[str, dict[str, Any]] = {}

    for task in ("mnli", "qqp"):
        spec = config["tasks"][task]
        model_path = Path(hf_hub_download(spec["model"], "model.safetensors", revision=spec["model_revision"]))
        if sha256_path(model_path) != MODEL_HASHES[task]:
            raise AssertionError(f"{task}: model hash mismatch")
        selected = spec["selected"]
        for alias in range(4):
            adapter_path = ROOT / "step90_models" / f"step90_{task}_adapter_{alias}.safetensors"
            tensors = load_file(str(adapter_path))
            if not math.isclose(float(tensors["confidence_threshold"].item()), float(selected["threshold"]), rel_tol=0.0, abs_tol=1e-14):
                raise AssertionError(f"{task}: adapter threshold mismatch")
            if int(tensors["style_id"].item()) != alias:
                raise AssertionError(f"{task}: adapter style mismatch")

        inputs = json.loads((ROOT / f"STEP90_{task.upper()}_HOLDOUT_INPUTS_{DATE}.json").read_text(encoding="utf-8"))["rows"]
        sealed = np.load(SEALED_DIR / f"STEP90_{task.upper()}_SEALED_OUTCOMES_{DATE}.npz", allow_pickle=False)
        positions = np.asarray([row["position"] for row in inputs], dtype=np.int64)
        if not np.array_equal(positions, sealed["positions"]):
            raise AssertionError(f"{task}: position mismatch")
        labels = sealed["labels"].astype(np.int8)
        official = sealed["official_predictions"].astype(np.int8)
        logits = development.infer_logits(task, inputs)
        hard, confidence = apply_mapping(logits, list(spec["label_mapping_raw_to_glue"]))
        roster = np.asarray(selected["roster_indices"], dtype=np.int64)
        original = np.column_stack([official[:, roster], hard]).astype(np.int8)
        core_feedback = (original == labels[:, None]).astype(np.float64)
        clean_codes = original.astype(np.int64)
        clean_feedback = core_feedback.copy()
        clean_parents = np.arange(original.shape[1], dtype=np.int64)
        alias_codes = development.adapter_codes(
            hard,
            confidence,
            float(selected["threshold"]),
            int(spec["classes"]),
        )
        refined_codes = np.column_stack([clean_codes, alias_codes])
        refined_feedback = np.column_stack(
            [clean_feedback, np.repeat(core_feedback[:, [-1]], 4, axis=1)]
        )
        parent_root = clean_codes.shape[1] - 1
        refined_parents = np.concatenate([clean_parents, np.full(4, parent_root, dtype=np.int64)])
        if not np.array_equal(refined_feedback[:, -4:], np.repeat(core_feedback[:, [-1]], 4, axis=1)):
            raise AssertionError(f"{task}: hard utility mismatch")
        pools = selector.sample_pools(len(labels), SEEDS)
        clean = selector.run_active(
            clean_codes,
            clean_feedback,
            clean_parents,
            core_feedback,
            pools,
            SEEDS,
            int(selected["budget"]),
            float(selected["tau"]),
        )
        refined = selector.run_active(
            refined_codes,
            refined_feedback,
            refined_parents,
            core_feedback,
            pools,
            SEEDS,
            int(selected["budget"]),
            float(selected["tau"]),
        )
        clean_fixed = run_fixed(clean_feedback, clean_parents, core_feedback, pools, clean.queries)
        refined_fixed = run_fixed(refined_feedback, refined_parents, core_feedback, pools, clean.queries)
        terminal_delta = refined.terminal - clean.terminal
        cumulative_delta = refined.cumulative - clean.cumulative
        fixed_terminal = refined_fixed.terminal - clean_fixed.terminal
        fixed_cumulative = refined_fixed.cumulative - clean_fixed.cumulative
        active_minus_fixed = terminal_delta - fixed_terminal
        fixed_exact = bool(
            np.array_equal(fixed_terminal, np.zeros_like(fixed_terminal))
            and np.array_equal(fixed_cumulative, np.zeros_like(fixed_cumulative))
            and np.array_equal(clean_fixed.roots, refined_fixed.roots)
        )
        terminal_summary = summary(terminal_delta, 900_100 + (0 if task == "mnli" else 10))
        cumulative_summary = summary(cumulative_delta, 900_200 + (0 if task == "mnli" else 10))
        active_minus_fixed_summary = summary(active_minus_fixed, 900_300 + (0 if task == "mnli" else 10))
        path_change = float(np.mean(np.any(clean.queries != refined.queries, axis=1)))
        task_work[task] = {
            "terminal": terminal_summary,
            "cumulative": cumulative_summary,
            "active_minus_fixed": active_minus_fixed_summary,
            "fixed_exact": fixed_exact,
            "path_change": path_change,
            "hard_utility_equal": True,
        }
        task_rows[task] = {
            "n_holdout": len(labels),
            "paired_runs": len(SEEDS),
            "selected": selected,
            "parent_holdout_accuracy": float(np.mean(hard == labels)),
            "best_original_holdout_accuracy": float(np.max(np.mean(core_feedback, axis=0))),
            "realized_trigger_fraction": float(np.mean(confidence >= float(selected["threshold"]))),
            "adapter_parent_accuracy_gaps": [0.0, 0.0, 0.0, 0.0],
            "path_change_rate": path_change,
            "final_root_change_rate": float(np.mean(clean.roots[:, -1] != refined.roots[:, -1])),
            "terminal_regret_delta": terminal_summary,
            "cumulative_regret_delta": cumulative_summary,
            "active_minus_fixed_terminal": active_minus_fixed_summary,
            "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
            "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
        }
        for name, value in {
            "positions": positions,
            "labels": labels,
            "official_predictions": official,
            "parent_logits": logits,
            "parent_hard": hard,
            "parent_confidence": confidence,
            "adapter_response_codes": alias_codes,
            "pools": pools,
            "clean_queries": clean.queries,
            "refined_queries": refined.queries,
            "clean_roots": clean.roots,
            "refined_roots": refined.roots,
            "terminal_delta": terminal_delta,
            "cumulative_delta": cumulative_delta,
            "fixed_terminal_delta": fixed_terminal,
            "fixed_cumulative_delta": fixed_cumulative,
            "active_minus_fixed_terminal": active_minus_fixed,
        }.items():
            arrays[f"{task}__{name}"] = value

    adjusted = holm({task: row["terminal"]["one_sided_signflip_p"] for task, row in task_work.items()})
    passing: list[str] = []
    for task, work in task_work.items():
        gates = {
            "coordinate_wise_hard_utility_equal": bool(work["hard_utility_equal"]),
            "fixed_query_exact_zero": bool(work["fixed_exact"]),
            "path_change_at_least_half": work["path_change"] >= MIN_PATH_CHANGE,
            "mean_terminal_delta_at_least_0_5pp": work["terminal"]["mean"] >= MIN_TERMINAL_DELTA,
            "terminal_bootstrap_lower_above_zero": work["terminal"]["bootstrap_95"][0] > 0.0,
            "holm_adjusted_p_at_most_0_05": adjusted[task] <= 0.05,
            "active_minus_fixed_at_least_0_5pp": (
                work["active_minus_fixed"]["mean"] >= MIN_TERMINAL_DELTA
                and work["active_minus_fixed"]["bootstrap_95"][0] > 0.0
            ),
        }
        task_rows[task]["holm_adjusted_p"] = adjusted[task]
        task_rows[task]["gate"] = gates
        task_rows[task]["passed"] = all(gates.values())
        if task_rows[task]["passed"]:
            passing.append(task)

    if len(passing) == 2:
        decision = "GO_STRONG_EXECUTABLE_ADAPTER_TRANSFER"
    elif len(passing) == 1:
        decision = "GO_LIMITED_EXECUTABLE_ADAPTER_TRANSFER"
    else:
        decision = "NO_CONFIRMATORY_EXECUTABLE_ADAPTER_TRANSFER"
    np.savez_compressed(RAW_PATH, **arrays)
    results = {
        "result_id": "STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_CONFIRMATORY_V1",
        "decision": decision,
        "passing_tasks": passing,
        "config_sha256": sha256_path(CONFIG_PATH),
        "execution_lock_sha256": sha256_path(LOCK_PATH),
        "raw_sha256": sha256_path(RAW_PATH),
        "tasks": task_rows,
        "prior_negative_result_retained": {
            "task": "qnli",
            "decision": "NO_CONFIRMATORY_EXECUTABLE_ADAPTER_BRIDGE",
            "terminal_delta": -0.00068,
        },
        "claim_boundary": (
            "High-confidence answer-preserving adapters on two public GLUE tasks; not independent full checkpoints, hidden-task compromise, live registry admission, or universal harm."
        ),
    }
    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "passing_tasks": passing,
        "tasks": {
            task: {
                "mean_terminal_delta_pp": 100.0 * row["terminal_regret_delta"]["mean"],
                "bootstrap_95_pp": [100.0 * value for value in row["terminal_regret_delta"]["bootstrap_95"]],
                "holm_p": row["holm_adjusted_p"],
                "path_change": row["path_change_rate"],
                "fixed_terminal_max_abs": row["max_abs_fixed_terminal_delta"],
                "passed": row["passed"],
            }
            for task, row in task_rows.items()
        },
        "raw_sha256": results["raw_sha256"],
        "result_sha256": sha256_path(RESULT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
