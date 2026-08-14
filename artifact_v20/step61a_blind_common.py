"""Outcome-free common machinery for the Step 61 ERFA prospective test.

The module accepts only candidate responses and response-derived fractional feedback.
It contains no path to a MODEL SELECTOR oracle file or HELM per-instance statistics.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.stats import rankdata


MANIFEST_ID = "STEP61_ERFA_FRESH_CROSS_SELECTOR_V1"
MISSING_BOXED = "__MISSING_BOXED__"
EMPTY_BOXED = "__EMPTY_BOXED__"
ROOT_QUANTILES = (0.10, 0.40, 0.70, 0.90)
LN2 = math.log(2.0)
QUERY_RNG_OFFSET = 0
SELECTION_RNG_OFFSET = 10_000_000


def digest_text(*parts: Any) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


# HELM v0.4.0 MATH answer extraction and normalization, reduced to the exact
# response-only functions used by math_equiv_chain_of_thought.
def remove_boxed(string: str) -> str | None:
    left = "\\boxed{"
    try:
        assert string[: len(left)] == left
        assert string[-1] == "}"
        return string[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if not substr:
                return string
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a_str, b_str = string.split("/")
    try:
        a, b = int(a_str), int(b_str)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " not in string:
        return string
    splits = string.split("\\text{ ")
    if len(splits) != 2:
        return string
    return splits[0]


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            return string
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_math_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


def canonical_math_response(text: str | None) -> str:
    if text is None:
        return MISSING_BOXED
    boxed = last_boxed_only_string(str(text))
    if boxed is None:
        return MISSING_BOXED
    answer = remove_boxed(boxed)
    if answer is None:
        return MISSING_BOXED
    normalized = strip_math_string(answer)
    return normalized if normalized else EMPTY_BOXED


def stable_candidate_partition(task_id: str, candidate_ids: list[str]) -> dict[str, Any]:
    h = len(candidate_ids)
    if h < 8:
        raise ValueError(f"{task_id}: need at least eight candidates, found {h}")
    order = sorted(
        range(h),
        key=lambda index: digest_text(MANIFEST_ID, task_id, candidate_ids[index]),
    )
    core_count = min(h - 2, max(6, math.floor(0.8 * h)))
    return {
        "hash_order": order,
        "core": order[:core_count],
        "reserve": order[core_count:],
    }


def mean_core_disagreement(core_responses: np.ndarray) -> np.ndarray:
    n, h = core_responses.shape
    values = np.zeros(h, dtype=np.float64)
    for candidate in range(h):
        values[candidate] = float(
            np.mean(core_responses != core_responses[:, [candidate]])
        )
    return values


def select_stress_roots(
    task_id: str, core_responses: np.ndarray, core_candidate_ids: list[str]
) -> tuple[list[int], np.ndarray]:
    disagreement = mean_core_disagreement(core_responses)
    order = sorted(
        range(len(core_candidate_ids)),
        key=lambda index: (
            float(disagreement[index]),
            digest_text(MANIFEST_ID, task_id, core_candidate_ids[index], "root-tie"),
        ),
    )
    chosen_ranks: list[int] = []
    for quantile in ROOT_QUANTILES:
        target = quantile * (len(order) - 1)
        available = [rank for rank in range(len(order)) if rank not in chosen_ranks]
        rank = min(available, key=lambda value: (abs(value - target), value))
        chosen_ranks.append(rank)
    roots = [order[rank] for rank in chosen_ranks]
    if len(set(roots)) != 4:
        raise AssertionError("root quantiles did not produce four unique candidates")
    return roots, disagreement


def core_support_rewards(core_responses: np.ndarray) -> np.ndarray:
    n, h = core_responses.shape
    rewards = np.empty((n, h), dtype=np.float32)
    for candidate in range(h):
        rewards[:, candidate] = np.mean(
            core_responses == core_responses[:, [candidate]], axis=1, dtype=np.float32
        )
    return rewards


def support_for_additions(
    core_responses: np.ndarray, additions: np.ndarray
) -> np.ndarray:
    if additions.ndim == 1:
        additions = additions[:, None]
    n, added = additions.shape
    rewards = np.empty((n, added), dtype=np.float32)
    for candidate in range(added):
        rewards[:, candidate] = np.mean(
            core_responses == additions[:, [candidate]], axis=1, dtype=np.float32
        )
    return rewards


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    x = values.astype(np.uint64, copy=True) + np.uint64(seed)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _modal_nonroot(row: np.ndarray, root_value: Any) -> Any:
    alternatives = row[row != root_value]
    if len(alternatives) == 0:
        raise ValueError("no non-root response exists at selected near-refinement position")
    values, counts = np.unique(alternatives, return_counts=True)
    return values[int(np.argmax(counts))]


def near_additions(
    task_id: str,
    core_responses: np.ndarray,
    root_local: int,
    root_id: str,
    alias_count: int,
    rho: float,
    root_support: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = core_responses.shape[0]
    k = max(1, int(round(rho * n)))
    root = core_responses[:, root_local]
    eligible = np.flatnonzero(root_support < 1.0 - 1e-12)
    if len(eligible) < k:
        raise ValueError(
            f"{task_id}:{root_id}: only {len(eligible)} mutable positions for k={k}"
        )
    seed = int(digest_text(MANIFEST_ID, task_id, root_id, "near-order")[:16], 16)
    tie_hash = _splitmix64(eligible.astype(np.uint64), seed)
    local_order = np.lexsort((tie_hash, root_support[eligible]))
    ordered = eligible[local_order]
    stride = max(1, len(ordered) // 7)
    aliases = np.repeat(root[:, None], alias_count, axis=1)
    changed_sets: list[list[int]] = []
    for alias in range(alias_count):
        positions = np.roll(ordered, -(alias * stride))[:k]
        for position in positions:
            aliases[position, alias] = _modal_nonroot(
                core_responses[position], root[position]
            )
        changed = np.flatnonzero(aliases[:, alias] != root)
        if len(changed) != k:
            raise AssertionError("near-refinement realized distance mismatch")
        changed_sets.append([int(value) for value in changed])
    hashes = [digest_array(aliases[:, index]) for index in range(alias_count)]
    if len(set(hashes)) != alias_count:
        raise AssertionError("near aliases are not response-distinct")
    return aliases, {
        "changed": k,
        "realized_rho": k / n,
        "alias_hashes": hashes,
        "changed_set_hashes": [
            digest_text(task_id, root_id, *positions) for positions in changed_sets
        ],
    }


def nearest_core_distances(
    core_responses: np.ndarray, additions: np.ndarray
) -> list[float]:
    if additions.ndim == 1:
        additions = additions[:, None]
    distances: list[float] = []
    for added in range(additions.shape[1]):
        per_core = np.mean(core_responses != additions[:, [added]], axis=0)
        distances.append(float(np.min(per_core)))
    return distances


def scenario_id(task_id: str, kind: str, **parameters: Any) -> str:
    suffix = "|".join(f"{key}={parameters[key]}" for key in sorted(parameters))
    return f"{task_id}::{kind}" + (f"::{suffix}" if suffix else "")


def generate_registry_scenarios(
    task_id: str,
    core_responses: np.ndarray,
    core_candidate_ids: list[str],
    reserve_responses: np.ndarray,
    reserve_candidate_ids: list[str],
    roots: list[int],
    clean_rewards: np.ndarray,
) -> Iterable[dict[str, Any]]:
    for root_slot, root_local in enumerate(roots):
        root_id = core_candidate_ids[root_local]
        root = core_responses[:, [root_local]]
        root_reward = clean_rewards[:, [root_local]]
        for aliases in (1, 2, 4):
            additions = np.repeat(root, aliases, axis=1)
            rewards = np.repeat(root_reward, aliases, axis=1)
            sid = scenario_id(
                task_id, "exact", root_slot=root_slot, aliases=aliases, rho="0"
            )
            distances = nearest_core_distances(core_responses, additions)
            yield {
                "scenario_id": sid,
                "scenario_sha256": digest_text(sid),
                "kind": "exact",
                "root_slot": root_slot,
                "root_local": root_local,
                "root_id": root_id,
                "aliases": aliases,
                "nominal_rho": 0.0,
                "responses": np.concatenate([core_responses, additions], axis=1),
                "rewards": np.concatenate([clean_rewards, rewards], axis=1),
                "addition_hashes": [digest_array(additions[:, i]) for i in range(aliases)],
                "nearest_core_distances": distances,
                "count_distance": float(aliases * np.mean(distances)),
            }
        for aliases in (1, 4):
            for rho in (0.02, 0.10):
                additions, metadata = near_additions(
                    task_id,
                    core_responses,
                    root_local,
                    root_id,
                    aliases,
                    rho,
                    clean_rewards[:, root_local],
                )
                rewards = support_for_additions(core_responses, additions)
                sid = scenario_id(
                    task_id,
                    "near",
                    root_slot=root_slot,
                    aliases=aliases,
                    rho=f"{rho:.2f}",
                )
                distances = nearest_core_distances(core_responses, additions)
                yield {
                    "scenario_id": sid,
                    "scenario_sha256": digest_text(sid),
                    "kind": "near",
                    "root_slot": root_slot,
                    "root_local": root_local,
                    "root_id": root_id,
                    "aliases": aliases,
                    "nominal_rho": rho,
                    "responses": np.concatenate([core_responses, additions], axis=1),
                    "rewards": np.concatenate([clean_rewards, rewards], axis=1),
                    "addition_hashes": metadata["alias_hashes"],
                    "changed_set_hashes": metadata["changed_set_hashes"],
                    "realized_rho": metadata["realized_rho"],
                    "nearest_core_distances": distances,
                    "count_distance": float(aliases * np.mean(distances)),
                }
    if len(reserve_candidate_ids) < 2:
        raise AssertionError("two reserve controls are required")
    for reserve_slot in range(2):
        additions = reserve_responses[:, [reserve_slot]]
        rewards = support_for_additions(core_responses, additions)
        sid = scenario_id(task_id, "diverse", reserve_slot=reserve_slot)
        distances = nearest_core_distances(core_responses, additions)
        yield {
            "scenario_id": sid,
            "scenario_sha256": digest_text(sid),
            "kind": "diverse",
            "reserve_slot": reserve_slot,
            "reserve_id": reserve_candidate_ids[reserve_slot],
            "aliases": 1,
            "nominal_rho": None,
            "responses": np.concatenate([core_responses, additions], axis=1),
            "rewards": np.concatenate([clean_rewards, rewards], axis=1),
            "addition_hashes": [digest_array(additions[:, 0])],
            "nearest_core_distances": distances,
            "count_distance": float(np.mean(distances)),
        }


def encode_responses(responses: np.ndarray) -> tuple[np.ndarray, int]:
    flat = responses.reshape(-1)
    _, inverse = np.unique(flat, return_inverse=True)
    codes = inverse.reshape(responses.shape)
    dtype = np.int16 if int(codes.max(initial=0)) < np.iinfo(np.int16).max else np.int32
    return codes.astype(dtype), int(codes.max(initial=-1) + 1)


@dataclass
class GroupStructure:
    membership: sparse.csr_matrix
    group_query: np.ndarray
    codes: np.ndarray


def build_group_structure(pool_codes: np.ndarray) -> GroupStructure:
    pool_size, entries = pool_codes.shape
    group_indices = np.empty((pool_size, entries), dtype=np.int64)
    group_query: list[int] = []
    offset = 0
    for query in range(pool_size):
        _, inverse = np.unique(pool_codes[query], return_inverse=True)
        group_indices[query] = inverse + offset
        groups = int(inverse.max(initial=-1) + 1)
        group_query.extend([query] * groups)
        offset += groups
    rows = group_indices.reshape(-1)
    columns = np.tile(np.arange(entries, dtype=np.int64), pool_size)
    membership = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(offset, entries),
    )
    return GroupStructure(
        membership=membership,
        group_query=np.asarray(group_query, dtype=np.int64),
        codes=pool_codes,
    )


def _entropy(probability: np.ndarray) -> float:
    positive = probability > 0
    return float(-np.sum(probability[positive] * np.log(probability[positive])) / LN2)


def select_llm_acquisition(
    groups: GroupStructure, posterior: np.ndarray
) -> np.ndarray:
    mass = np.asarray(groups.membership @ posterior).reshape(-1)
    return np.bincount(
        groups.group_query,
        weights=mass * mass,
        minlength=groups.codes.shape[0],
    )


def model_picker_acquisition(
    groups: GroupStructure, posterior: np.ndarray, gamma: float, class_count: int
) -> np.ndarray:
    positive = posterior > 0
    p_log_p = np.zeros_like(posterior, dtype=np.float64)
    p_log_p[positive] = posterior[positive] * np.log(posterior[positive])
    base_a = float(np.sum(p_log_p))
    base_entropy = -base_a / LN2
    mass = np.asarray(groups.membership @ posterior).reshape(-1)
    group_a = np.asarray(groups.membership @ p_log_p).reshape(-1)
    z = 1.0 + (gamma - 1.0) * mass
    sum_w_log_w = (
        base_a
        + (gamma - 1.0) * group_a
        + gamma * mass * math.log(gamma)
    )
    conditioned = (np.log(z) - sum_w_log_w / z) / LN2
    delta = np.bincount(
        groups.group_query,
        weights=conditioned - base_entropy,
        minlength=groups.codes.shape[0],
    )
    # This reconstructs the official uniform-over-class conditional entropy.
    return base_entropy + delta / class_count


def sample_pool(seed: int, n: int, requested_size: int) -> np.ndarray:
    size = min(n, requested_size)
    return np.asarray(sorted(random.Random(seed).sample(range(n), size)), dtype=np.int64)


def _uniform_choice(rng: np.random.Generator, candidates: np.ndarray) -> int:
    if len(candidates) == 0:
        raise ValueError("empty tie set")
    return int(candidates[int(rng.integers(0, len(candidates)))])


def run_active_selector(
    codes: np.ndarray,
    feedback: np.ndarray,
    core_count: int,
    pool: np.ndarray,
    budget: int,
    seed: int,
    selector_family: str,
    *,
    epsilon: float | None = None,
    tau: float = 1.0,
    class_count: int | None = None,
) -> dict[str, Any]:
    pool_codes = codes[pool]
    pool_feedback = np.asarray(feedback[pool], dtype=np.float64)
    groups = build_group_structure(pool_codes)
    pool_size, entries = pool_codes.shape
    steps = min(budget, pool_size - 1)
    if steps < 1:
        raise ValueError("pool too small")
    # Purpose-separated paired streams prevent a changed candidate tie-set from
    # consuming a different number of random bits and thereby perturbing a later
    # query tie.  Both conditions receive the same query stream and the same
    # deployment-selection stream.
    query_rng = np.random.Generator(np.random.PCG64(seed + QUERY_RNG_OFFSET))
    selection_rng = np.random.Generator(np.random.PCG64(seed + SELECTION_RNG_OFFSET))
    cumulative = np.zeros(entries, dtype=np.float64)
    posterior = np.full(entries, 1.0 / entries, dtype=np.float64)
    available = np.ones(pool_size, dtype=bool)
    disagreement = np.any(pool_codes != pool_codes[:, [0]], axis=1)
    query_path = np.full(steps, -1, dtype=np.int64)
    selected_path = np.full(steps, -1, dtype=np.int64)
    regret_path = np.zeros(steps, dtype=np.float64)
    candidate_pool_quality = pool_feedback.mean(axis=0)
    best_core = float(np.max(candidate_pool_quality[:core_count]))
    gamma = None
    if selector_family == "model_selector":
        if epsilon is None or class_count is None:
            raise ValueError("MODEL SELECTOR requires epsilon and class_count")
        gamma = (1.0 - epsilon) / epsilon

    initial_scores: np.ndarray | None = None
    for step in range(steps):
        if selector_family == "select_llm":
            shifted = cumulative / tau
            shifted -= shifted.max()
            posterior = np.exp(shifted)
            posterior /= posterior.sum()
            acquisition = select_llm_acquisition(groups, posterior)
            eligible = available
        elif selector_family == "model_selector":
            acquisition = model_picker_acquisition(
                groups, posterior, float(gamma), int(class_count)
            )
            eligible = available & disagreement
        else:
            raise ValueError(f"unknown selector family: {selector_family}")

        if initial_scores is None:
            initial_scores = acquisition.copy()
            if selector_family == "model_selector":
                initial_scores[~disagreement] = np.inf

        if np.any(eligible):
            minimum = float(np.min(acquisition[eligible]))
            tied = np.flatnonzero(eligible & (acquisition == minimum))
            local_query = _uniform_choice(query_rng, tied)
            reward = pool_feedback[local_query]
            cumulative += reward
            if selector_family == "model_selector":
                posterior *= np.power(float(gamma), reward)
                posterior /= posterior.sum()
            maximum = float(np.max(cumulative))
            selected = _uniform_choice(
                selection_rng, np.flatnonzero(cumulative == maximum)
            )
        else:
            # Mirrors the official post-disagreement fill: query without changing
            # model evidence and repeat the current selected model.
            tied = np.flatnonzero(available)
            local_query = _uniform_choice(query_rng, tied)
            if step == 0:
                selected = _uniform_choice(
                    selection_rng, np.arange(entries, dtype=np.int64)
                )
            else:
                selected = int(selected_path[step - 1])

        available[local_query] = False
        query_path[step] = int(pool[local_query])
        selected_path[step] = selected
        regret_path[step] = best_core - float(candidate_pool_quality[selected])

    assert initial_scores is not None
    return {
        "query_path": query_path,
        "selected_path": selected_path,
        "regret_path": regret_path,
        "cumulative_regret": float(np.sum(regret_path)),
        "final_regret": float(regret_path[-1]),
        "initial_scores": initial_scores,
        "disagreement_items": int(np.sum(disagreement)),
    }


def static_rank_displacement(clean: np.ndarray, proposed: np.ndarray) -> float:
    if len(clean) != len(proposed):
        raise ValueError("initial-score vectors must align")
    n = len(clean)
    if n <= 1:
        return 0.0
    clean_rank = rankdata(clean, method="average")
    proposed_rank = rankdata(proposed, method="average")
    return float(np.mean(np.abs(clean_rank - proposed_rank)) / (n - 1))


def path_metrics(clean: np.ndarray, proposed: np.ndarray) -> tuple[float, float]:
    hamming = float(np.mean(clean != proposed))
    left = set(int(value) for value in clean)
    right = set(int(value) for value in proposed)
    union = left | right
    jaccard_distance = 0.0 if not union else 1.0 - len(left & right) / len(union)
    return hamming, float(jaccard_distance)


def slow_model_picker_acquisition(
    responses: np.ndarray, posterior: np.ndarray, gamma: float, classes: np.ndarray
) -> np.ndarray:
    scores = np.zeros(responses.shape[0], dtype=np.float64)
    for query in range(responses.shape[0]):
        total = 0.0
        for label in classes:
            updated = posterior * np.power(gamma, responses[query] == label)
            updated /= updated.sum()
            total += _entropy(updated)
        scores[query] = total / len(classes)
    return scores


def self_test() -> dict[str, Any]:
    canonical_cases = {
        r"work \boxed{\frac12}": r"\frac{1}{2}",
        r"work \boxed{0.5}": r"\frac{1}{2}",
        r"work \boxed{\sqrt3}": r"\sqrt{3}",
        "no final box": MISSING_BOXED,
    }
    observed_cases = {key: canonical_math_response(key) for key in canonical_cases}
    if observed_cases != canonical_cases:
        raise AssertionError({"expected": canonical_cases, "observed": observed_cases})

    rng = np.random.default_rng(61000)
    responses = rng.integers(0, 7, size=(19, 6), dtype=np.int16)
    posterior = rng.random(6)
    posterior /= posterior.sum()
    gamma = 1.17
    groups = build_group_structure(responses)
    fast_model = model_picker_acquisition(groups, posterior, gamma, 7)
    slow_model = slow_model_picker_acquisition(
        responses, posterior, gamma, np.arange(7)
    )
    model_error = float(np.max(np.abs(fast_model - slow_model)))
    if model_error > 1e-12:
        raise AssertionError(f"MODEL SELECTOR acquisition parity error {model_error}")

    fast_llm = select_llm_acquisition(groups, posterior)
    slow_llm = np.zeros(len(responses), dtype=np.float64)
    for query, row in enumerate(responses):
        slow_llm[query] = sum(
            float(np.sum(posterior[row == value])) ** 2 for value in np.unique(row)
        )
    llm_error = float(np.max(np.abs(fast_llm - slow_llm)))
    if llm_error > 1e-12:
        raise AssertionError(f"Select-LLM acquisition parity error {llm_error}")

    task = "toy"
    candidate_ids = [f"candidate_{i:03d}" for i in range(8)]
    partition = stable_candidate_partition(task, candidate_ids)
    toy = rng.integers(0, 4, size=(100, 8), dtype=np.int16)
    core = toy[:, partition["core"]]
    roots, _ = select_stress_roots(
        task, core, [candidate_ids[index] for index in partition["core"]]
    )
    rewards = core_support_rewards(core)
    scenarios = list(
        generate_registry_scenarios(
            task,
            core,
            [candidate_ids[index] for index in partition["core"]],
            toy[:, partition["reserve"]],
            [candidate_ids[index] for index in partition["reserve"]],
            roots,
            rewards,
        )
    )
    if len(scenarios) != 30:
        raise AssertionError(f"expected 30 nonclean scenarios, found {len(scenarios)}")
    kinds = {kind: sum(row["kind"] == kind for row in scenarios) for kind in ("exact", "near", "diverse")}
    if kinds != {"exact": 12, "near": 16, "diverse": 2}:
        raise AssertionError(kinds)

    return {
        "canonical_cases": len(canonical_cases),
        "model_picker_max_abs_score_error": model_error,
        "select_llm_max_abs_score_error": llm_error,
        "toy_scenarios": len(scenarios),
        "toy_kinds": kinds,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2))
