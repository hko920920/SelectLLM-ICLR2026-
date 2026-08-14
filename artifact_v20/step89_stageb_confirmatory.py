from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import step89_stagea_develop as common


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-12"
CONFIG_PATH = ROOT / f"STEP89_QNLI_STAGEA_FROZEN_CONFIG_{DATE}.json"
LOCK_PATH = ROOT / f"STEP89_QNLI_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json"
INPUT_PATH = ROOT / f"STEP89_QNLI_HOLDOUT_INPUTS_{DATE}.json"
SEALED_PATH = ROOT / "external_data" / "step89_sealed" / f"STEP89_QNLI_SEALED_HOLDOUT_OUTCOMES_{DATE}.npz"
RAW_PATH = ROOT / f"STEP89_QNLI_EXECUTABLE_ADAPTER_RAW_{DATE}.npz"
RESULT_PATH = ROOT / f"STEP89_QNLI_EXECUTABLE_ADAPTER_RESULTS_{DATE}.json"

SEEDS = tuple(range(1000))
BOOTSTRAP_REPS = 50_000
SIGNFLIP_REPS = 100_000
MIN_TERMINAL_DELTA = 0.0015
MIN_PATH_CHANGE = 0.50
MODEL_SAFETENSORS_SHA256 = "8b0cfff5547faac6a89c58eca8e0d26b3c4df0022937d59accb417957747ce6c"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_fixed(
    feedback: np.ndarray,
    parents: np.ndarray,
    core_feedback: np.ndarray,
    pools: np.ndarray,
    queries: np.ndarray,
) -> common.RunSummary:
    runs, budget = queries.shape
    terminal = np.zeros(runs, dtype=np.float64)
    cumulative_regret = np.zeros(runs, dtype=np.float64)
    roots = np.full((runs, budget), -1, dtype=np.int64)
    for run_index, seed in enumerate(SEEDS):
        pool = pools[run_index]
        root_quality = np.mean(core_feedback[pool], axis=0)
        best_quality = float(np.max(root_quality))
        cumulative_score = np.zeros(feedback.shape[1], dtype=np.float64)
        for step, query in enumerate(queries[run_index]):
            cumulative_score += feedback[int(query)]
            maximum = float(np.max(cumulative_score))
            root = common.choose_root(
                np.flatnonzero(cumulative_score == maximum), parents, int(seed), step
            )
            roots[run_index, step] = root
            regret = best_quality - float(root_quality[root])
            cumulative_regret[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret
    return common.RunSummary(
        terminal=terminal,
        cumulative=cumulative_regret,
        queries=np.asarray(queries, dtype=np.int64),
        roots=roots,
    )


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPS
    while remaining:
        size = min(2_000, remaining)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means.append(np.mean(values[indices], axis=1))
        remaining -= size
    distribution = np.concatenate(means)
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    return float(lower), float(upper)


def signflip_pvalue(values: np.ndarray, seed: int) -> float:
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = SIGNFLIP_REPS
    while remaining:
        size = min(2_000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(values)))
        permuted = np.mean(signs * values[None, :], axis=1)
        exceed += int(np.sum(permuted >= observed))
        remaining -= size
    return float((exceed + 1) / (SIGNFLIP_REPS + 1))


def summarize(values: np.ndarray, seed: int) -> dict[str, Any]:
    lower, upper = bootstrap_interval(values, seed)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "bootstrap_95": [lower, upper],
        "one_sided_signflip_p": signflip_pvalue(values, seed + 1),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
        "zero_fraction": float(np.mean(values == 0)),
    }


def main() -> None:
    if not LOCK_PATH.exists():
        raise FileNotFoundError("confirmatory execution lock must exist before outcomes are opened")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    expected_inputs = lock["locked_sha256"]
    for relative, expected in expected_inputs.items():
        observed = sha256_path(ROOT / relative)
        if observed != expected:
            raise AssertionError({"path": relative, "expected": expected, "observed": observed})

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model_path = Path(
        hf_hub_download(
            common.MODEL_NAME,
            filename="model.safetensors",
            revision=common.MODEL_REVISION,
        )
    )
    if sha256_path(model_path) != MODEL_SAFETENSORS_SHA256:
        raise AssertionError("frozen parent safetensors hash mismatch")
    for index, expected in enumerate(config["adapter_log_scales"]):
        adapter_path = ROOT / "step89_models" / f"step89_adapter_{index}.safetensors"
        tensors = load_file(str(adapter_path))
        observed = float(tensors["log_scale"].item())
        if not math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1e-7):
            raise AssertionError((index, observed, expected))
        if int(tensors["confidence_bins"].item()) != int(config["selected"]["bins"]):
            raise AssertionError("adapter bin mismatch")

    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))["rows"]
    sealed = np.load(SEALED_PATH, allow_pickle=False)
    positions = np.asarray([row["position"] for row in inputs], dtype=np.int64)
    if not np.array_equal(positions, sealed["positions"]):
        raise AssertionError("holdout input/outcome position mismatch")
    labels = sealed["labels"].astype(np.int8)
    official = sealed["official_predictions"].astype(np.int8)
    parent_logits = common.infer_logits(inputs)
    parent_hard = (parent_logits < 0).astype(np.int8)

    selected = config["selected"]
    roster = np.asarray(selected["roster_indices"], dtype=np.int64)
    original_predictions = np.column_stack([official[:, roster], parent_hard]).astype(np.int8)
    core_feedback = (original_predictions == labels[:, None]).astype(np.float64)
    clean_codes = original_predictions.astype(np.int64)
    clean_feedback = core_feedback.copy()
    clean_parents = np.arange(clean_codes.shape[1], dtype=np.int64)

    log_scales = tuple(float(value) for value in config["adapter_log_scales"])
    alias_codes = common.response_codes(parent_logits, log_scales, int(selected["bins"]))
    refined_codes = np.column_stack([clean_codes, alias_codes])
    refined_feedback = np.column_stack(
        [clean_feedback, np.repeat(core_feedback[:, [-1]], len(log_scales), axis=1)]
    )
    parent_root = clean_codes.shape[1] - 1
    refined_parents = np.concatenate(
        [clean_parents, np.full(len(log_scales), parent_root, dtype=np.int64)]
    )

    if not np.array_equal(refined_feedback[:, -len(log_scales):], np.repeat(core_feedback[:, [-1]], len(log_scales), axis=1)):
        raise AssertionError("coordinate-wise utility equality failed")
    if any(np.array_equal(alias_codes[:, left], alias_codes[:, right]) for left in range(len(log_scales)) for right in range(left)):
        raise AssertionError("two adapter response views are identical on the complete holdout")

    pools = common.sample_pools(len(labels), SEEDS)
    clean = common.run_active(
        clean_codes,
        clean_feedback,
        clean_parents,
        core_feedback,
        pools,
        SEEDS,
        int(selected["budget"]),
        float(selected["tau"]),
    )
    refined = common.run_active(
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
    fixed_terminal_delta = refined_fixed.terminal - clean_fixed.terminal
    fixed_cumulative_delta = refined_fixed.cumulative - clean_fixed.cumulative
    active_minus_fixed_terminal = terminal_delta - fixed_terminal_delta
    active_minus_fixed_cumulative = cumulative_delta - fixed_cumulative_delta
    path_changed = np.any(clean.queries != refined.queries, axis=1)
    final_root_changed = clean.roots[:, -1] != refined.roots[:, -1]

    exact_utility = bool(
        np.array_equal(
            refined_feedback[:, -len(log_scales):],
            np.repeat(core_feedback[:, [-1]], len(log_scales), axis=1),
        )
    )
    exact_fixed = bool(
        np.array_equal(fixed_terminal_delta, np.zeros_like(fixed_terminal_delta))
        and np.array_equal(fixed_cumulative_delta, np.zeros_like(fixed_cumulative_delta))
        and np.array_equal(clean_fixed.roots, refined_fixed.roots)
    )
    terminal_summary = summarize(terminal_delta, 890_100)
    cumulative_summary = summarize(cumulative_delta, 890_200)
    active_minus_fixed_summary = summarize(active_minus_fixed_terminal, 890_300)
    path_change_rate = float(np.mean(path_changed))

    gate_parts = {
        "model_and_adapter_hashes": True,
        "coordinate_wise_hard_utility_equal": exact_utility,
        "fixed_query_terminal_and_cumulative_exact_zero": exact_fixed,
        "ordered_path_change_at_least_half": path_change_rate >= MIN_PATH_CHANGE,
        "mean_terminal_delta_at_least_0_15pp": terminal_summary["mean"] >= MIN_TERMINAL_DELTA,
        "terminal_bootstrap_lower_above_zero": terminal_summary["bootstrap_95"][0] > 0.0,
        "terminal_one_sided_p_at_most_0_05": terminal_summary["one_sided_signflip_p"] <= 0.05,
        "active_minus_fixed_terminal_positive": (
            active_minus_fixed_summary["mean"] >= MIN_TERMINAL_DELTA
            and active_minus_fixed_summary["bootstrap_95"][0] > 0.0
        ),
    }
    passed = all(gate_parts.values())
    decision = "GO_EXECUTABLE_ADAPTER_BRIDGE" if passed else "NO_CONFIRMATORY_EXECUTABLE_ADAPTER_BRIDGE"

    np.savez_compressed(
        RAW_PATH,
        positions=positions,
        labels=labels,
        official_predictions=official,
        parent_logits=parent_logits,
        parent_hard=parent_hard,
        adapter_response_codes=alias_codes,
        pools=pools,
        clean_queries=clean.queries,
        refined_queries=refined.queries,
        clean_roots=clean.roots,
        refined_roots=refined.roots,
        clean_terminal=clean.terminal,
        refined_terminal=refined.terminal,
        terminal_delta=terminal_delta,
        cumulative_delta=cumulative_delta,
        fixed_terminal_delta=fixed_terminal_delta,
        fixed_cumulative_delta=fixed_cumulative_delta,
        active_minus_fixed_terminal=active_minus_fixed_terminal,
        active_minus_fixed_cumulative=active_minus_fixed_cumulative,
    )
    results = {
        "result_id": "STEP89_QNLI_EXECUTABLE_ADAPTER_CONFIRMATORY_RESULT_V1",
        "decision": decision,
        "config_sha256": sha256_path(CONFIG_PATH),
        "execution_lock_sha256": sha256_path(LOCK_PATH),
        "raw_sha256": sha256_path(RAW_PATH),
        "model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
        "n_holdout": len(labels),
        "paired_runs": len(SEEDS),
        "pool_size": int(selected["pool_size"]),
        "budget": int(selected["budget"]),
        "tau": float(selected["tau"]),
        "roster_indices": roster.tolist(),
        "parent_holdout_accuracy": float(np.mean(parent_hard == labels)),
        "best_original_holdout_accuracy": float(np.max(np.mean(core_feedback, axis=0))),
        "adapter_parent_accuracy_gaps": [0.0 for _ in log_scales],
        "adapter_pairwise_response_hamming": [
            {
                "left": left,
                "right": right,
                "hamming": float(np.mean(alias_codes[:, left] != alias_codes[:, right])),
            }
            for left in range(len(log_scales))
            for right in range(left)
        ],
        "path_change_rate": path_change_rate,
        "final_root_change_rate": float(np.mean(final_root_changed)),
        "terminal_regret_delta": terminal_summary,
        "cumulative_regret_delta": cumulative_summary,
        "active_minus_fixed_terminal": active_minus_fixed_summary,
        "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal_delta))),
        "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative_delta))),
        "gate": gate_parts,
        "claim_boundary": (
            "One untouched QNLI task with a pinned public parent and four executable calibration-adapter files; not live registry admission, hidden-task compromise, or universal harm."
        ),
    }
    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "mean_terminal_delta_pp": 100.0 * terminal_summary["mean"],
        "terminal_bootstrap_95_pp": [100.0 * value for value in terminal_summary["bootstrap_95"]],
        "terminal_p": terminal_summary["one_sided_signflip_p"],
        "path_change_rate": path_change_rate,
        "final_root_change_rate": float(np.mean(final_root_changed)),
        "fixed_terminal_max_abs": results["max_abs_fixed_terminal_delta"],
        "fixed_cumulative_max_abs": results["max_abs_fixed_cumulative_delta"],
        "raw_sha256": results["raw_sha256"],
        "result_sha256": sha256_path(RESULT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
