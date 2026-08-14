from __future__ import annotations

import ast
import argparse
import hashlib
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from scipy import sparse
from torch import nn
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-13"
LOCK_PATH = ROOT / f"STEP93_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json"
CONFIG_PATH = ROOT / f"STEP93_STAGEA_FROZEN_CONFIG_{DATE}.json"
INPUT_PATH = ROOT / f"STEP93_BANKING77_HOLDOUT_INPUTS_{DATE}.json"
RAW_PATH = ROOT / f"STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RAW_{DATE}.npz"
RESULT_PATH = ROOT / f"STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RESULTS_{DATE}.json"
COMMON_PATH = ROOT / "step93_common.py"
ENDPOINT_PATH = ROOT / "step93_executable_abstention_endpoint.py"
MODEL_DIR = ROOT / "step93_models"
RECEIPT_PATH = ROOT / f"STEP93_INDEPENDENT_VALIDATION_AND_ADJUDICATION_{DATE}.json"

HIDDEN_SIZE = 768
N_CLASSES = 77
N_ROOTS = 12
N_ALIASES = 4
BOOTSTRAP_SEED = 93500
SIGNFLIP_SEED = 93501


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def independent_exact_match(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(left) == np.asarray(right)


def sample_pools(n: int, seeds: tuple[int, ...], pool_size: int) -> np.ndarray:
    return np.asarray(
        [sorted(random.Random(int(seed)).sample(range(n), pool_size)) for seed in seeds],
        dtype=np.int64,
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
class Summary:
    terminal: np.ndarray
    cumulative: np.ndarray
    queries: np.ndarray
    roots: np.ndarray


def acquisition_from_same_s(codes: np.ndarray, posterior: np.ndarray) -> np.ndarray:
    agreement = independent_exact_match(codes[:, :, None], codes[:, None, :]).astype(np.float64)
    return np.einsum("nij,i,j->n", agreement, posterior, posterior, optimize=True)


@dataclass(frozen=True)
class IndependentGroups:
    membership: sparse.csr_matrix
    group_query: np.ndarray
    pool_size: int


def independent_build_groups(codes: np.ndarray) -> IndependentGroups:
    pool_size, entries = codes.shape
    group_indices = np.empty((pool_size, entries), dtype=np.int64)
    group_query: list[int] = []
    offset = 0
    for query in range(pool_size):
        representatives: list[int] = []
        for entry in range(entries):
            value = int(codes[query, entry])
            match = next(
                (
                    group
                    for group, representative in enumerate(representatives)
                    if bool(independent_exact_match(np.asarray(representative), np.asarray(value)))
                ),
                None,
            )
            if match is None:
                match = len(representatives)
                representatives.append(value)
                group_query.append(query)
            group_indices[query, entry] = offset + int(match)
        offset += len(representatives)
    rows = group_indices.reshape(-1)
    columns = np.tile(np.arange(entries, dtype=np.int64), pool_size)
    membership = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(offset, entries),
    )
    return IndependentGroups(
        membership=membership,
        group_query=np.asarray(group_query, dtype=np.int64),
        pool_size=pool_size,
    )


def independent_group_acquisition(
    groups: IndependentGroups, posterior: np.ndarray
) -> np.ndarray:
    mass = np.asarray(groups.membership @ posterior).reshape(-1)
    return np.bincount(
        groups.group_query,
        weights=mass * mass,
        minlength=groups.pool_size,
    )


def run_active(
    codes: np.ndarray,
    labels: np.ndarray,
    parents: np.ndarray,
    core_codes: np.ndarray,
    pools: np.ndarray,
    seeds: tuple[int, ...],
    budget: int,
    tau: float,
) -> Summary:
    feedback = independent_exact_match(codes, labels[:, None]).astype(np.float64)
    core_feedback = independent_exact_match(core_codes, labels[:, None]).astype(np.float64)
    terminal = np.zeros(len(seeds), dtype=np.float64)
    cumulative = np.zeros(len(seeds), dtype=np.float64)
    queries = np.full((len(seeds), budget), -1, dtype=np.int64)
    roots = np.full((len(seeds), budget), -1, dtype=np.int64)
    for run_index, seed in enumerate(seeds):
        pool = pools[run_index]
        groups = independent_build_groups(codes[pool])
        active = np.ones(len(pool), dtype=bool)
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        root_quality = np.mean(core_feedback[pool], axis=0)
        best_quality = float(np.max(root_quality))
        for step in range(budget):
            shifted = scores / float(tau)
            shifted -= float(np.max(shifted))
            posterior = np.exp(shifted)
            posterior /= float(np.sum(posterior))
            acquisition = independent_group_acquisition(groups, posterior)
            acquisition[~active] = np.inf
            tied = np.flatnonzero(acquisition == float(np.min(acquisition)))
            position = choose_query(tied, pool, int(seed), step)
            active[position] = False
            query = int(pool[position])
            queries[run_index, step] = query
            scores += feedback[query]
            root = choose_root(
                np.flatnonzero(scores == float(np.max(scores))), parents, int(seed), step
            )
            roots[run_index, step] = root
            regret = best_quality - float(root_quality[root])
            cumulative[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret
    return Summary(terminal, cumulative, queries, roots)


def run_fixed(
    codes: np.ndarray,
    labels: np.ndarray,
    parents: np.ndarray,
    core_codes: np.ndarray,
    pools: np.ndarray,
    queries: np.ndarray,
    seeds: tuple[int, ...],
) -> Summary:
    feedback = independent_exact_match(codes, labels[:, None]).astype(np.float64)
    core_feedback = independent_exact_match(core_codes, labels[:, None]).astype(np.float64)
    terminal = np.zeros(len(seeds), dtype=np.float64)
    cumulative = np.zeros(len(seeds), dtype=np.float64)
    roots = np.full(queries.shape, -1, dtype=np.int64)
    for run_index, seed in enumerate(seeds):
        pool = pools[run_index]
        root_quality = np.mean(core_feedback[pool], axis=0)
        best_quality = float(np.max(root_quality))
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        for step, query in enumerate(queries[run_index]):
            scores += feedback[int(query)]
            root = choose_root(
                np.flatnonzero(scores == float(np.max(scores))), parents, int(seed), step
            )
            roots[run_index, step] = root
            regret = best_quality - float(root_quality[root])
            cumulative[run_index] += regret
            if step == queries.shape[1] - 1:
                terminal[run_index] = regret
    return Summary(terminal, cumulative, queries.copy(), roots)


def bootstrap_interval(values: np.ndarray, seed: int, repetitions: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    remaining = repetitions
    while remaining:
        count = min(1000, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        blocks.append(np.mean(values[indices], axis=1))
        remaining -= count
    return [float(value) for value in np.quantile(np.concatenate(blocks), [0.025, 0.975])]


def signflip_pvalue(values: np.ndarray, seed: int, repetitions: int = 100_000) -> float:
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = repetitions
    while remaining:
        count = min(1000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        exceed += int(np.sum(np.mean(signs * values[None, :], axis=1) >= observed))
        remaining -= count
    return float((exceed + 1) / (repetitions + 1))


def assert_array(name: str, observed: np.ndarray, expected: np.ndarray, checks: list[str]) -> None:
    if not np.array_equal(observed, expected):
        maximum = float(np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64))))
        raise AssertionError({"array": name, "max_abs": maximum})
    checks.append(f"array_exact:{name}")


class IndependentClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.down = nn.Linear(HIDDEN_SIZE, 128)
        self.out = nn.Linear(128, N_CLASSES)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(torch.nn.functional.gelu(self.down(self.norm(hidden))))


class IndependentAbstention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.down = nn.Linear(HIDDEN_SIZE, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(torch.nn.functional.gelu(self.down(self.norm(hidden))))[:, 0]


def independent_endpoint_probe(
    inputs: list[dict[str, Any]],
    raw: Any,
    config: dict[str, Any],
    checks: list[str],
) -> dict[str, Any]:
    probe_indices = np.asarray(
        sorted(range(len(inputs)), key=lambda index: stable_u64("step93-independent-endpoint", index))[:128],
        dtype=np.int64,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased",
        revision="12040accade4e8a0f71eabdb258fecc2e7e948be",
        local_files_only=True,
    )
    encoder = AutoModel.from_pretrained(
        "distilbert/distilbert-base-uncased",
        revision="12040accade4e8a0f71eabdb258fecc2e7e948be",
        local_files_only=True,
        use_safetensors=True,
    ).eval()
    texts = [str(inputs[int(index)]["text"]) for index in probe_indices]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.inference_mode():
        hidden = encoder(**encoded).last_hidden_state[:, 0, :].float()
    parent = int(config["selected"]["parent_root"])
    parent_path = MODEL_DIR / f"step93_clean_root_{parent:02d}.safetensors"
    classifier = IndependentClassifier().eval()
    classifier.load_state_dict(load_file(str(parent_path)))
    with torch.inference_mode():
        parent_logits = classifier(hidden).double().numpy()
    parent_predictions = np.argmax(parent_logits, axis=1).astype(np.int64)
    scores = []
    for alias in range(N_ALIASES):
        path = MODEL_DIR / f"step93_parent_{parent:02d}_abstention_adapter_{alias}.safetensors"
        adapter = IndependentAbstention().eval()
        adapter.load_state_dict(load_file(str(path)))
        with torch.inference_mode():
            scores.append(torch.sigmoid(adapter(hidden)).double().numpy())
    score_matrix = np.stack(scores, axis=1)
    thresholds = np.asarray(config["selected"]["thresholds"], dtype=np.float64)
    triggers = score_matrix > thresholds[None, :]
    codes = np.repeat(parent_predictions[:, None], N_ALIASES, axis=1)
    for alias in range(N_ALIASES):
        codes[triggers[:, alias], alias] = N_CLASSES + alias
    assert_array(
        "endpoint_parent_predictions_128",
        parent_predictions,
        raw["parent_predictions"][probe_indices].astype(np.int64),
        checks,
    )
    score_max_abs = float(
        np.max(np.abs(score_matrix - raw["learned_error_scores"][probe_indices]))
    )
    if not np.allclose(
        score_matrix,
        raw["learned_error_scores"][probe_indices],
        rtol=0.0,
        atol=1e-4,
    ):
        raise AssertionError({"independent_endpoint_score_max_abs": score_max_abs})
    checks.append("endpoint_scores_128_cross_device_atol_1e-4")
    assert_array(
        "endpoint_response_codes_128",
        codes,
        raw["adapter_response_codes"][probe_indices].astype(np.int64),
        checks,
    )
    return {
        "samples": len(probe_indices),
        "source_index_sha256": hashlib.sha256(probe_indices.tobytes()).hexdigest(),
        "max_abs_score_difference": score_max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if RECEIPT_PATH.exists() and not args.verify_existing:
        raise RuntimeError("refusing to overwrite independent Step 93 receipt")
    if args.verify_existing and not RECEIPT_PATH.exists():
        raise FileNotFoundError(RECEIPT_PATH)
    checks: list[str] = []
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    raw = np.load(RAW_PATH, allow_pickle=False)
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))["rows"]

    for relative, expected in lock["locked_sha256"].items():
        observed = sha256_path(ROOT / Path(relative))
        if observed != expected:
            raise AssertionError({"locked_hash": relative, "expected": expected, "observed": observed})
        checks.append(f"locked_hash:{relative}")
    if result["raw_sha256"] != sha256_path(RAW_PATH):
        raise AssertionError("result/raw hash mismatch")
    checks.append("result_raw_hash")

    common_source = COMMON_PATH.read_text(encoding="utf-8")
    parsed = ast.parse(common_source)
    run_active_node = next(
        node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "run_active"
    )
    run_active_arguments = [argument.arg for argument in run_active_node.args.args]
    if "feedback" in run_active_arguments:
        raise AssertionError("run_active accepts external feedback")
    feedback_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "exact_match_feedback"
    )
    group_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_source_faithful_groups"
    )
    feedback_calls = {
        node.func.id
        for node in ast.walk(feedback_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    group_calls = {
        node.func.id
        for node in ast.walk(group_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    if "exact_match_similarity" not in feedback_calls or "exact_match_similarity" not in group_calls:
        raise AssertionError({"feedback_calls": feedback_calls, "group_calls": group_calls})
    checks.extend(
        [
            "ast_run_active_has_no_external_feedback",
            "ast_feedback_calls_exact_match_similarity",
            "ast_grouping_calls_exact_match_similarity",
        ]
    )

    labels = raw["labels"].astype(np.int64)
    clean_codes = raw["clean_root_predictions"].astype(np.int64)
    alias_codes = raw["adapter_response_codes"].astype(np.int64)
    refined_codes = np.column_stack([clean_codes, alias_codes])
    clean_parents = np.arange(N_ROOTS, dtype=np.int64)
    parent = int(config["selected"]["parent_root"])
    refined_parents = np.concatenate([clean_parents, np.full(N_ALIASES, parent, dtype=np.int64)])
    seeds = tuple(range(931000, 932000))
    pools = sample_pools(len(labels), seeds, int(config["selected"]["pool_size"]))
    assert_array("pools", pools, raw["pools"], checks)

    clean = run_active(
        clean_codes,
        labels,
        clean_parents,
        clean_codes,
        pools,
        seeds,
        int(config["selected"]["budget"]),
        float(config["selected"]["tau"]),
    )
    refined = run_active(
        refined_codes,
        labels,
        refined_parents,
        clean_codes,
        pools,
        seeds,
        int(config["selected"]["budget"]),
        float(config["selected"]["tau"]),
    )
    clean_fixed = run_fixed(
        clean_codes, labels, clean_parents, clean_codes, pools, clean.queries, seeds
    )
    refined_fixed = run_fixed(
        refined_codes, labels, refined_parents, clean_codes, pools, clean.queries, seeds
    )
    for name, observed, expected in (
        ("clean_queries", clean.queries, raw["clean_queries"]),
        ("refined_queries", refined.queries, raw["refined_queries"]),
        ("clean_roots", clean.roots, raw["clean_roots"]),
        ("refined_roots", refined.roots, raw["refined_roots"]),
        ("clean_fixed_roots", clean_fixed.roots, raw["clean_fixed_roots"]),
        ("refined_fixed_roots", refined_fixed.roots, raw["refined_fixed_roots"]),
        ("clean_terminal", clean.terminal, raw["clean_terminal"]),
        ("refined_terminal", refined.terminal, raw["refined_terminal"]),
    ):
        assert_array(name, observed, expected, checks)

    terminal_delta = refined.terminal - clean.terminal
    cumulative_delta = refined.cumulative - clean.cumulative
    fixed_terminal_delta = refined_fixed.terminal - clean_fixed.terminal
    fixed_cumulative_delta = refined_fixed.cumulative - clean_fixed.cumulative
    active_minus_fixed = terminal_delta - fixed_terminal_delta
    for name, observed, expected in (
        ("terminal_delta", terminal_delta, raw["terminal_delta"]),
        ("cumulative_delta", cumulative_delta, raw["cumulative_delta"]),
        ("fixed_terminal_delta", fixed_terminal_delta, raw["fixed_terminal_delta"]),
        ("fixed_cumulative_delta", fixed_cumulative_delta, raw["fixed_cumulative_delta"]),
        ("active_minus_fixed_terminal", active_minus_fixed, raw["active_minus_fixed_terminal"]),
    ):
        assert_array(name, observed, expected, checks)

    terminal_ci = bootstrap_interval(terminal_delta, BOOTSTRAP_SEED)
    terminal_p = signflip_pvalue(terminal_delta, SIGNFLIP_SEED)
    task = result["task"]
    if terminal_ci != task["terminal_regret_delta"]["bootstrap_95"]:
        raise AssertionError({"terminal_ci": terminal_ci})
    if terminal_p != task["terminal_regret_delta"]["one_sided_signflip_p"]:
        raise AssertionError({"terminal_p": terminal_p})
    checks.extend(["bootstrap_terminal_exact", "signflip_terminal_exact"])

    parent_feedback = independent_exact_match(clean_codes[:, parent], labels)
    alias_feedback = independent_exact_match(alias_codes, labels[:, None])
    if not np.all(alias_feedback <= parent_feedback[:, None]):
        raise AssertionError("alias coordinate-wise improvement found")
    alias_losses = np.mean(parent_feedback) - np.mean(alias_feedback, axis=0)
    if not np.all(alias_losses <= 0.010 + 1e-15):
        raise AssertionError({"alias_losses": alias_losses.tolist()})
    checks.extend(["coordinate_nonimprovement", "alias_accuracy_loss_within_1pp"])

    probe_posterior = np.arange(1, refined_codes.shape[1] + 1, dtype=np.float64)
    probe_posterior /= np.sum(probe_posterior)
    dense = acquisition_from_same_s(refined_codes[:64], probe_posterior)
    sparse = independent_group_acquisition(
        independent_build_groups(refined_codes[:64]), probe_posterior
    )
    max_numeric_difference = float(np.max(np.abs(dense - sparse)))
    one_ulp_at_one = float(np.spacing(1.0))
    if max_numeric_difference > one_ulp_at_one:
        raise AssertionError(
            {"same_s_numeric_difference": max_numeric_difference, "one_ulp_at_one": one_ulp_at_one}
        )
    checks.append("same_s_sparse_dense_within_one_binary64_ulp")

    endpoint_probe = independent_endpoint_probe(inputs, raw, config, checks)
    fixed_exact = bool(
        np.array_equal(fixed_terminal_delta, np.zeros_like(fixed_terminal_delta))
        and np.array_equal(fixed_cumulative_delta, np.zeros_like(fixed_cumulative_delta))
        and np.array_equal(clean_fixed.roots, refined_fixed.roots)
    )
    corrected_same_s_gate = True
    corrected_gates = dict(task["gates"])
    corrected_gates["same_exact_similarity_for_acquisition_and_posterior"] = True
    expected_pass = {
        "all_locked_hashes_match": True,
        "same_exact_similarity_for_acquisition_and_posterior": corrected_same_s_gate,
        "four_genuine_distinct_50817_parameter_adapters": True,
        "raw_text_endpoint_exact_replay_and_runtime_exclusions": True,
        "coordinate_wise_nonimproving": True,
        "each_alias_accuracy_loss_at_most_1pp": True,
        "path_change_at_least_half": float(
            np.mean(np.any(clean.queries != refined.queries, axis=1))
        )
        >= 0.50,
        "mean_terminal_delta_at_least_0_5pp": float(np.mean(terminal_delta)) >= 0.005,
        "terminal_bootstrap_lower_above_zero": terminal_ci[0] > 0.0,
        "terminal_one_sided_signflip_p_at_most_0_05": terminal_p <= 0.05,
        "active_minus_fixed_at_least_0_5pp_and_lower_above_zero": bool(
            float(np.mean(active_minus_fixed)) >= 0.005
            and bootstrap_interval(active_minus_fixed, BOOTSTRAP_SEED)[0] > 0.0
        ),
        "fixed_query_terminal_and_cumulative_exact_zero": fixed_exact,
    }
    if corrected_gates != expected_pass:
        raise AssertionError({"corrected": corrected_gates, "expected": expected_pass})
    adjudicated_decision = (
        "GO_SOURCE_FAITHFUL_LEARNED_ABSTENTION_BRIDGE"
        if all(corrected_gates.values())
        else "NO_GO_RETAIN_SOURCE_FAITHFUL_NEGATIVE"
    )
    if adjudicated_decision != "NO_GO_RETAIN_SOURCE_FAITHFUL_NEGATIVE":
        raise AssertionError(adjudicated_decision)
    checks.extend(["corrected_gate_reconstruction", "adjudicated_retained_negative"])

    receipt = {
        "receipt_id": "STEP93_INDEPENDENT_VALIDATION_AND_ADJUDICATION_V1",
        "status": "PASS_STEP93_INDEPENDENT_RECONSTRUCTION",
        "adjudicated_decision": adjudicated_decision,
        "original_machine_decision": result["decision"],
        "original_result_sha256": sha256_path(RESULT_PATH),
        "raw_sha256": sha256_path(RAW_PATH),
        "execution_lock_sha256": sha256_path(LOCK_PATH),
        "numeric_validator_adjudication": {
            "original_sparse_dense_atol": 2e-16,
            "observed_max_abs_difference": max_numeric_difference,
            "binary64_ulp_at_one": one_ulp_at_one,
            "cause": "equivalent summation orders differ by one binary64 ULP",
            "feedback_equality_is_bitwise_exact": True,
            "source_hash_assertions_match": True,
            "corrected_same_s_gate": True,
            "scientific_effect_gates_unchanged": True,
        },
        "confirmatory_effect": {
            "terminal_delta_mean": float(np.mean(terminal_delta)),
            "terminal_delta_bootstrap_95": terminal_ci,
            "terminal_one_sided_signflip_p": terminal_p,
            "active_minus_fixed_mean": float(np.mean(active_minus_fixed)),
            "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
            "alias_accuracy_losses": alias_losses.tolist(),
            "fixed_query_exact_zero": fixed_exact,
        },
        "corrected_gates": corrected_gates,
        "independent_endpoint_probe": endpoint_probe,
        "checks_count": len(checks),
        "checks": checks,
        "claim": (
            "The Step 93 implementation is source-faithful to the frozen single exact-match "
            "similarity, but the preregistered fresh-task harm gate fails. The complete negative "
            "result is retained; no task or gate is replaced."
        ),
    }
    if args.verify_existing:
        expected_receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        if receipt != expected_receipt:
            raise AssertionError("reconstructed Step 93 receipt differs from bundled receipt")
    else:
        RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "adjudicated_decision": adjudicated_decision,
                "checks": len(checks),
                "max_numeric_difference": max_numeric_difference,
                "terminal_delta_pp": 100.0 * float(np.mean(terminal_delta)),
                "terminal_bootstrap_95_pp": [100.0 * value for value in terminal_ci],
                "receipt": RECEIPT_PATH.name,
                "receipt_sha256": sha256_path(RECEIPT_PATH),
                "verified_existing_receipt": bool(args.verify_existing),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
