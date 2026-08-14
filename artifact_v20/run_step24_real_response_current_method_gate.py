"""Step 24: current-method, real-response gate for sampling-frame sensitivity.

This audit is deliberately independent of the older judge-tensor implementation
used in Steps 15--23.  It instantiates Equations (5) and (7) of Select-LLM
(arXiv:2605.24981v1) on public HELM-Lite v1.0.0 MedQA model responses.

The script evaluates three increasingly strong perturbations:

1. exact registry replication (one or more copied candidates),
2. natural same-provider candidate pairs already present in HELM, and
3. response-level wrapper aliases whose actual multiple-choice answers differ
   on a frozen fraction of queries.  The oracle-aware wrapper is an existence /
   adaptive stress test and is reported separately from the label-blind wrapper.

All selection regret is evaluated against the same unmodified 31-model oracle.
No tensor-only aliases are counted as real-response evidence in this gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "external_data" / "helm_lite_medqa_v1_0_0_raw"
OUT = ROOT / "STEP24_REAL_RESPONSE_CURRENT_METHOD_GATE_2026-08-06.json"

HELM_BASE = (
    "https://storage.googleapis.com/crfm-helm-public/"
    "lite/benchmark_output/runs/v1.0.0/"
)
PAPER_URL = "https://arxiv.org/abs/2605.24981"
HELM_REPO = "https://github.com/stanford-crfm/helm"
HELM_VERSION = "v1.0.0"

MODELS = """01-ai_yi-34b
01-ai_yi-6b
AlephAlpha_luminous-base
AlephAlpha_luminous-extended
AlephAlpha_luminous-supreme
ai21_j2-grande
ai21_j2-jumbo
anthropic_claude-2.0
anthropic_claude-2.1
anthropic_claude-instant-1.2
anthropic_claude-instant-v1
anthropic_claude-v1.3
cohere_command-light
cohere_command
google_text-bison@001
google_text-unicorn@001
meta_llama-2-13b
meta_llama-2-70b
meta_llama-2-7b
meta_llama-65b
mistralai_mistral-7b-v0.1
mistralai_mixtral-8x7b-32kseqlen
openai_gpt-3.5-turbo-0613
openai_gpt-4-0613
openai_gpt-4-1106-preview
openai_text-davinci-002
openai_text-davinci-003
tiiuae_falcon-40b
tiiuae_falcon-7b
writer_palmyra-x-v2
writer_palmyra-x-v3""".splitlines()

POOL_SIZE = 500
BUDGET = 50
K_ALIASES = 4
TAUS = [0.1, 0.5, 1.0, 3.0, 5.0]
SCREEN_SEEDS = list(range(0, 500))
ONE_STEP_TEST_SEEDS = list(range(1000, 2000))
FULL_CLONE_SEEDS = list(range(2000, 2100))
FULL_WRAPPER_SEEDS = list(range(3000, 3100))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def cache_json(relative_url: str, filename: str) -> tuple[object, dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / filename
    url = HELM_BASE + urllib.parse.quote(relative_url, safe="/:,@=-._")
    if path.exists():
        payload = path.read_bytes()
        source = "cache"
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        path.write_bytes(payload)
        source = "download"
    return json.loads(payload), {
        "url": url,
        "cache_path": str(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "source": source,
    }


def load_helm_medqa() -> tuple[list[str], np.ndarray, np.ndarray, list[list[str]], np.ndarray, dict]:
    def load_model(model: str) -> tuple[str, object, dict]:
        relative = f"med_qa:model={model}/display_predictions.json"
        rows, provenance = cache_json(relative, f"{model}__display_predictions.json")
        return model, rows, provenance

    rows_by_model: dict[str, list[dict]] = {}
    provenance: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_model, model) for model in MODELS]
        for future in as_completed(futures):
            model, rows, source = future.result()
            rows_by_model[model] = rows
            provenance[model] = source

    instance_rows, instance_source = cache_json(
        "med_qa:model=openai_gpt-4-0613/instances.json",
        "openai_gpt-4-0613__instances.json",
    )
    provenance["instances"] = instance_source
    instances = {row["id"]: row for row in instance_rows}

    ids = [row["instance_id"] for row in rows_by_model[MODELS[0]]]
    if len(ids) != 1000 or len(set(ids)) != len(ids):
        raise AssertionError("unexpected HELM MedQA instance IDs")

    responses = []
    oracle = []
    for model in MODELS:
        keyed = {row["instance_id"]: row for row in rows_by_model[model]}
        if set(keyed) != set(ids):
            raise AssertionError(f"instance mismatch for {model}")
        model_responses = []
        model_oracle = []
        for instance_id in ids:
            row = keyed[instance_id]
            # HELM's mapped_output is the canonical option text.  This avoids
            # comparing A/B/C/D labels under any option-order variation.
            mapped = row.get("mapped_output")
            if mapped is None:
                mapped = "INVALID:" + str(row.get("predicted_text", ""))
            model_responses.append(str(mapped).strip())
            model_oracle.append(float(row["stats"].get("quasi_exact_match", 0.0)))
        responses.append(model_responses)
        oracle.append(model_oracle)

    options: list[list[str]] = []
    correct: list[str] = []
    for instance_id in ids:
        refs = instances[instance_id]["references"]
        options.append([str(ref["output"]["text"]).strip() for ref in refs])
        tagged = [
            str(ref["output"]["text"]).strip()
            for ref in refs
            if "correct" in ref.get("tags", [])
        ]
        if len(tagged) != 1:
            raise AssertionError("MedQA instance lacks one canonical correct option")
        correct.append(tagged[0])

    response_array = np.asarray(responses, dtype=object)
    oracle_array = np.asarray(oracle, dtype=np.float64)
    if response_array.shape != (len(MODELS), 1000):
        raise AssertionError("unexpected response shape")
    if oracle_array.shape != response_array.shape:
        raise AssertionError("oracle/response shape mismatch")
    if not set(np.unique(oracle_array)).issubset({0.0, 1.0}):
        raise AssertionError("MedQA oracle must be binary")

    return ids, response_array, oracle_array, options, np.asarray(correct, dtype=object), provenance


def provider(model: str) -> str:
    return model.split("_", 1)[0]


def response_one_hot(responses: np.ndarray) -> np.ndarray:
    """Encode per-query categorical responses without a global text vocabulary."""
    models, queries = responses.shape
    codes = np.zeros((models, queries), dtype=np.int16)
    maximum = 0
    for query in range(queries):
        mapping: dict[str, int] = {}
        for model in range(models):
            value = str(responses[model, query])
            if value not in mapping:
                mapping[value] = len(mapping)
            codes[model, query] = mapping[value]
        maximum = max(maximum, len(mapping))
    return np.stack(
        [(codes.T == category).astype(np.float32) for category in range(maximum)]
    )


def make_scenario(
    responses: np.ndarray,
    oracle: np.ndarray,
    labels: list[str],
    original_indices: list[int],
) -> dict:
    return {
        "responses": responses,
        "oracle": oracle,
        "labels": labels,
        "original_indices": np.asarray(original_indices, dtype=np.int64),
        "one_hot": response_one_hot(responses),
    }


def initial_acquisition(one_hot: np.ndarray) -> np.ndarray:
    probability = np.ones(one_hot.shape[2], dtype=np.float64) / one_hot.shape[2]
    mass = np.einsum("lqm,m->lq", one_hot, probability, optimize=True)
    return np.sum(mass * mass, axis=0)


def pool(seed: int, queries: int) -> np.ndarray:
    return np.asarray(
        sorted(random.Random(seed).sample(range(queries), POOL_SIZE)), dtype=np.int64
    )


def first_queries(acquisition: np.ndarray, seeds: list[int]) -> np.ndarray:
    selected = []
    for seed in seeds:
        sampled = pool(seed, len(acquisition))
        selected.append(int(sampled[int(np.argmin(acquisition[sampled]))]))
    return np.asarray(selected, dtype=np.int64)


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    out = np.exp(shifted)
    return out / out.sum()


def run_select_llm(
    scenario: dict,
    *,
    common_oracle: np.ndarray,
    seed: int,
    tau: float,
    budget: int = BUDGET,
) -> dict:
    sampled = pool(seed, common_oracle.shape[1])
    unlabeled = sampled.tolist()
    cumulative = np.zeros(len(scenario["oracle"]), dtype=np.float64)
    queried: list[int] = []
    selected_original: list[int] = []
    regrets: list[float] = []

    common_scores = common_oracle[:, sampled].mean(axis=1)
    common_best = float(common_scores.max())
    best_indices = np.flatnonzero(common_scores == common_best).tolist()

    for _ in range(budget):
        posterior = stable_softmax(cumulative / tau)
        remaining = np.asarray(unlabeled, dtype=np.int64)
        masses = np.einsum(
            "lqm,m->lq",
            scenario["one_hot"][:, remaining, :],
            posterior,
            optimize=True,
        )
        scores = np.sum(masses * masses, axis=0)
        location = int(np.argmin(scores))
        query = int(unlabeled.pop(location))
        queried.append(query)

        cumulative += scenario["oracle"][:, query]
        selected_local = int(np.argmax(cumulative))
        selected = int(scenario["original_indices"][selected_local])
        selected_original.append(selected)
        regrets.append(common_best - float(common_scores[selected]))

    regret = np.asarray(regrets, dtype=np.float64)
    return {
        "seed": seed,
        "query_digest": hashlib.sha256(
            json.dumps(queried, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "queried": queried,
        "final_selected_original": selected_original[-1],
        "final_selected_label": MODELS[selected_original[-1]],
        "common_best_indices": best_indices,
        "common_best_score": common_best,
        "cumulative_common_regret": float(regret.sum()),
        "final_common_regret": float(regret[-1]),
        "steps_within_1pct_common_oracle": int(np.sum(regret <= 0.01 + 1e-12)),
        "selected_original": selected_original,
    }


def bootstrap_ci(values: np.ndarray, seed: int, repetitions: int = 10000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [math.nan, math.nan]
    if np.allclose(values, values[0]):
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    chunk = 1000
    cursor = 0
    while cursor < repetitions:
        size = min(chunk, repetitions - cursor)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[cursor : cursor + size] = values[indices].mean(axis=1)
        cursor += size
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def compare_runs(clean: list[dict], changed: list[dict], seed: int) -> dict:
    if len(clean) != len(changed):
        raise ValueError("paired run lengths differ")
    cumulative = np.asarray(
        [
            right["cumulative_common_regret"] - left["cumulative_common_regret"]
            for left, right in zip(clean, changed)
        ],
        dtype=np.float64,
    )
    final = np.asarray(
        [
            right["final_common_regret"] - left["final_common_regret"]
            for left, right in zip(clean, changed)
        ],
        dtype=np.float64,
    )
    positions = np.asarray(
        [
            sum(a != b for a, b in zip(left["queried"], right["queried"]))
            for left, right in zip(clean, changed)
        ],
        dtype=np.float64,
    )
    return {
        "runs": len(clean),
        "any_query_path_change_fraction": float(
            np.mean([left["queried"] != right["queried"] for left, right in zip(clean, changed)])
        ),
        "mean_changed_query_positions": float(positions.mean()),
        "final_selection_change_fraction": float(
            np.mean(
                [
                    left["final_selected_original"] != right["final_selected_original"]
                    for left, right in zip(clean, changed)
                ]
            )
        ),
        "mean_delta_cumulative_common_regret": float(cumulative.mean()),
        "delta_cumulative_common_regret_95ci": bootstrap_ci(cumulative, seed),
        "mean_delta_final_common_regret": float(final.mean()),
        "delta_final_common_regret_95ci": bootstrap_ci(final, seed + 1),
        "harmful_fraction": float(np.mean(cumulative > 1e-12)),
        "improved_fraction": float(np.mean(cumulative < -1e-12)),
        "tied_fraction": float(np.mean(np.abs(cumulative) <= 1e-12)),
        "paired_deltas": [float(x) for x in cumulative],
    }


def compact_runs(runs: list[dict]) -> list[dict]:
    return [
        {
            "seed": row["seed"],
            "query_digest": row["query_digest"],
            "final_selected_label": row["final_selected_label"],
            "cumulative_common_regret": row["cumulative_common_regret"],
            "final_common_regret": row["final_common_regret"],
            "steps_within_1pct_common_oracle": row[
                "steps_within_1pct_common_oracle"
            ],
        }
        for row in runs
    ]


def pair_bootstrap_gap(
    oracle: np.ndarray, left: int, right: int, seed: int
) -> tuple[float, list[float], float]:
    difference = oracle[right] - oracle[left]
    rng = np.random.default_rng(seed)
    means = np.empty(5000, dtype=np.float64)
    for start in range(0, 5000, 500):
        indices = rng.integers(0, oracle.shape[1], size=(500, oracle.shape[1]))
        means[start : start + 500] = difference[indices].mean(axis=1)
    return (
        float(difference.mean()),
        [float(x) for x in np.quantile(means, [0.025, 0.975])],
        float(np.mean(means > 0)),
    )


def natural_pairs(responses: np.ndarray, oracle: np.ndarray) -> list[dict]:
    rows = []
    seed = 7000
    for left in range(len(MODELS)):
        for right in range(left + 1, len(MODELS)):
            if provider(MODELS[left]) != provider(MODELS[right]):
                continue
            gap, ci, probability = pair_bootstrap_gap(oracle, left, right, seed)
            seed += 1
            rows.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "left": MODELS[left],
                    "right": MODELS[right],
                    "provider": provider(MODELS[left]),
                    "response_disagreement": float(
                        np.mean(responses[left] != responses[right])
                    ),
                    "left_score": float(oracle[left].mean()),
                    "right_score": float(oracle[right].mean()),
                    "right_minus_left_score": gap,
                    "right_minus_left_95ci": ci,
                    "bootstrap_probability_right_better": probability,
                }
            )
    return sorted(rows, key=lambda row: row["response_disagreement"])


def provider_components(
    responses: np.ndarray, oracle: np.ndarray, rho: float
) -> dict:
    count = len(MODELS)
    adjacency = [[] for _ in range(count)]
    merged_edges = []
    for left in range(count):
        for right in range(left + 1, count):
            if provider(MODELS[left]) != provider(MODELS[right]):
                continue
            distance = float(np.mean(responses[left] != responses[right]))
            if distance <= rho + 1e-12:
                adjacency[left].append(right)
                adjacency[right].append(left)
                merged_edges.append(
                    {"left": MODELS[left], "right": MODELS[right], "distance": distance}
                )

    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(count):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))

    score = oracle.mean(axis=1)
    nontrivial = [component for component in components if len(component) > 1]
    details = []
    for component in nontrivial:
        values = score[component]
        details.append(
            {
                "models": [MODELS[index] for index in component],
                "scores": [float(score[index]) for index in component],
                "score_span": float(values.max() - values.min()),
                "earliest_representative": MODELS[min(component)],
            }
        )
    return {
        "rho": rho,
        "components": len(components),
        "models_removed_by_one_representative_per_component": count - len(components),
        "nontrivial_components": len(nontrivial),
        "maximum_within_component_score_span": float(
            max((row["score_span"] for row in details), default=0.0)
        ),
        "merged_edges": merged_edges,
        "component_details": details,
    }


def exact_clone_scenario(
    responses: np.ndarray, oracle: np.ndarray, target: int, copies: int
) -> dict:
    expanded_responses = np.concatenate(
        [responses, np.repeat(responses[[target]], copies, axis=0)], axis=0
    )
    expanded_oracle = np.concatenate(
        [oracle, np.repeat(oracle[[target]], copies, axis=0)], axis=0
    )
    labels = MODELS + [f"CLONE:{MODELS[target]}:{index}" for index in range(copies)]
    originals = list(range(len(MODELS))) + [target] * copies
    return make_scenario(expanded_responses, expanded_oracle, labels, originals)


def best_alternative(
    query: int,
    target_response: str,
    responses: np.ndarray,
    options: list[list[str]],
    *,
    forbidden: set[str],
) -> tuple[str, float]:
    candidates = [
        option
        for option in options[query]
        if option != target_response and option not in forbidden
    ]
    if not candidates:
        candidates = [
            "INVALID:camouflage"
        ]  # only used for malformed/non-four-option edge cases
    # Lower agreement with the clean registry creates a stronger response-only decoy.
    frequencies = [float(np.mean(responses[:, query] == option)) for option in candidates]
    location = int(np.argmin(frequencies))
    return candidates[location], frequencies[location]


def wrapper_alias_scenario(
    responses: np.ndarray,
    oracle: np.ndarray,
    options: list[list[str]],
    correct: np.ndarray,
    *,
    target: int,
    distance_queries: int,
    oracle_aware: bool,
    aliases: int = K_ALIASES,
) -> tuple[dict, dict]:
    if oracle_aware:
        valid = np.flatnonzero(oracle[target] == 0)
    else:
        valid = np.arange(responses.shape[1], dtype=np.int64)
    if len(valid) < distance_queries:
        raise ValueError("requested camouflage exceeds eligible queries")

    ranking = []
    alternatives: dict[int, str] = {}
    for query in valid:
        forbidden = {str(correct[query])} if oracle_aware else set()
        alternative, frequency = best_alternative(
            int(query),
            str(responses[target, query]),
            responses,
            options,
            forbidden=forbidden,
        )
        alternatives[int(query)] = alternative
        ranking.append((frequency, int(query)))
    ordered = np.asarray([query for _, query in sorted(ranking)], dtype=np.int64)

    alias_responses = []
    alias_oracle = []
    changed_sets = []
    stride = max(1, len(ordered) // 7)
    for alias in range(aliases):
        changed = np.roll(ordered, alias * stride)[:distance_queries]
        row = responses[target].copy()
        score = oracle[target].copy()
        for query in changed:
            query = int(query)
            candidates = [
                option
                for option in options[query]
                if option != row[query]
                and (not oracle_aware or option != correct[query])
            ]
            if not candidates:
                candidates = [alternatives[query]]
            # Vary the actual string across aliases where multiple wrong options exist.
            row[query] = candidates[(alias + query) % len(candidates)]
            score[query] = float(row[query] == correct[query])
        observed = float(np.mean(row != responses[target]))
        expected = distance_queries / responses.shape[1]
        if not math.isclose(observed, expected, abs_tol=1e-12):
            raise AssertionError("wrapper distance does not match frozen dose")
        if oracle_aware and not np.array_equal(score, oracle[target]):
            raise AssertionError("oracle-aware wrong-to-wrong wrapper changed quality")
        alias_responses.append(row)
        alias_oracle.append(score)
        changed_sets.append([int(x) for x in changed])

    hashes = [
        hashlib.sha256(
            json.dumps(row.tolist(), ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        for row in alias_responses
    ]
    if len(set(hashes)) != aliases:
        raise AssertionError("wrapper aliases are not pairwise distinct")

    expanded_responses = np.concatenate(
        [responses, np.asarray(alias_responses, dtype=object)], axis=0
    )
    expanded_oracle = np.concatenate(
        [oracle, np.asarray(alias_oracle, dtype=np.float64)], axis=0
    )
    labels = MODELS + [
        f"WRAPPER:{MODELS[target]}:{'aware' if oracle_aware else 'blind'}:{index}"
        for index in range(aliases)
    ]
    originals = list(range(len(MODELS))) + [target] * aliases
    metadata = {
        "target": MODELS[target],
        "target_index": target,
        "aliases": aliases,
        "distance_queries": distance_queries,
        "response_disagreement_from_parent": distance_queries / responses.shape[1],
        "oracle_aware": oracle_aware,
        "threat_model_note": (
            "adaptive existence stress test; changed responses are restricted to "
            "parent-wrong queries and remain wrong"
            if oracle_aware
            else "label-blind response-only wrapper; oracle labels are not used to choose queries"
        ),
        "alias_full_dataset_score_gaps": [
            float(row.mean() - oracle[target].mean()) for row in alias_oracle
        ],
        "alias_response_hashes": hashes,
        "changed_query_digests": [
            hashlib.sha256(
                json.dumps(changed, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for changed in changed_sets
        ],
        "exact_raw_response_dedup_survivors": aliases,
    }
    return make_scenario(expanded_responses, expanded_oracle, labels, originals), metadata


def run_many(
    scenario: dict,
    common_oracle: np.ndarray,
    seeds: list[int],
    tau: float,
) -> list[dict]:
    return [
        run_select_llm(
            scenario, common_oracle=common_oracle, seed=seed, tau=tau, budget=BUDGET
        )
        for seed in seeds
    ]


def main() -> None:
    started = time.time()
    ids, responses, oracle, options, correct, provenance = load_helm_medqa()
    clean = make_scenario(responses, oracle, MODELS.copy(), list(range(len(MODELS))))
    clean_initial = initial_acquisition(clean["one_hot"])
    clean_screen_queries = first_queries(clean_initial, SCREEN_SEEDS)

    # Response-only, label-free target screening for the minimal one-clone attack.
    screen = []
    for target in range(len(MODELS)):
        scenario = exact_clone_scenario(responses, oracle, target, 1)
        queries = first_queries(initial_acquisition(scenario["one_hot"]), SCREEN_SEEDS)
        screen.append(
            {
                "target_index": target,
                "target": MODELS[target],
                "screen_first_query_change_fraction": float(
                    np.mean(queries != clean_screen_queries)
                ),
            }
        )
    selected = max(
        screen,
        key=lambda row: (row["screen_first_query_change_fraction"], row["target"]),
    )
    target = int(selected["target_index"])

    clean_test_queries = first_queries(clean_initial, ONE_STEP_TEST_SEEDS)
    one_step = []
    for copies in [1, 2, 4, 8]:
        scenario = exact_clone_scenario(responses, oracle, target, copies)
        attacked = first_queries(initial_acquisition(scenario["one_hot"]), ONE_STEP_TEST_SEEDS)
        one_step.append(
            {
                "copies": copies,
                "test_pools": len(ONE_STEP_TEST_SEEDS),
                "first_query_change_fraction": float(
                    np.mean(attacked != clean_test_queries)
                ),
                "clean_query_digest": hashlib.sha256(
                    clean_test_queries.tobytes()
                ).hexdigest(),
                "attacked_query_digest": hashlib.sha256(attacked.tobytes()).hexdigest(),
            }
        )

    # Full sequential exact-clone audit across the paper's sensitivity grid.
    full_clone = {}
    clone_one = exact_clone_scenario(responses, oracle, target, 1)
    for tau in TAUS:
        clean_runs = run_many(clean, oracle, FULL_CLONE_SEEDS, tau)
        clone_runs = run_many(clone_one, oracle, FULL_CLONE_SEEDS, tau)
        full_clone[str(tau)] = {
            "comparison": compare_runs(clean_runs, clone_runs, 10000 + int(tau * 100)),
            "clean_runs": compact_runs(clean_runs),
            "clone_runs": compact_runs(clone_runs),
        }

    pairs = natural_pairs(responses, oracle)
    thresholds = sorted(
        set(
            [0.0, 0.001, 0.008, 0.032, 0.075, 0.076, 0.132, 0.133, 0.183, 0.184, 0.269, 0.270, 0.331, 0.341, 0.4, 0.5]
            + [float(row["response_disagreement"]) for row in pairs]
        )
    )
    threshold_audit = [provider_components(responses, oracle, rho) for rho in thresholds]

    # Current-method response-level wrapper audit at tau=1 (the paper's GSM8K
    # exact-match setting); the common clean run is reused for every scenario.
    wrapper_clean_runs = run_many(clean, oracle, FULL_WRAPPER_SEEDS, 1.0)
    wrappers = {"label_blind": {}, "oracle_aware": {}, "wide_oracle_aware": {}}
    for aware, doses in [(False, [1, 8, 32, 76]), (True, [1, 8, 32, 76, 133, 183])]:
        bucket = "oracle_aware" if aware else "label_blind"
        for dose in doses:
            scenario, metadata = wrapper_alias_scenario(
                responses,
                oracle,
                options,
                correct,
                target=target,
                distance_queries=dose,
                oracle_aware=aware,
            )
            runs = run_many(scenario, oracle, FULL_WRAPPER_SEEDS, 1.0)
            wrappers[bucket][str(dose)] = {
                "metadata": metadata,
                "comparison": compare_runs(
                    wrapper_clean_runs, runs, 20000 + dose + (1000 if aware else 0)
                ),
                "runs": compact_runs(runs),
            }

    # Wide camouflage crosses natural-family thresholds.  These are adaptive
    # existence tests, kept separate from the selected high-quality target.
    for target_name, dose in [
        ("openai_gpt-3.5-turbo-0613", 341),
        ("meta_llama-65b", 400),
    ]:
        wide_target = MODELS.index(target_name)
        scenario, metadata = wrapper_alias_scenario(
            responses,
            oracle,
            options,
            correct,
            target=wide_target,
            distance_queries=dose,
            oracle_aware=True,
        )
        runs = run_many(scenario, oracle, FULL_WRAPPER_SEEDS, 1.0)
        wrappers["wide_oracle_aware"][target_name] = {
            "metadata": metadata,
            "comparison": compare_runs(wrapper_clean_runs, runs, 30000 + dose),
            "runs": compact_runs(runs),
        }

    result = {
        "gate": "STEP24_REAL_RESPONSE_CURRENT_METHOD",
        "created_local_date": "2026-08-06",
        "elapsed_seconds": time.time() - started,
        "method_contract": {
            "paper": PAPER_URL,
            "algorithm": "Equation (5) acquisition plus Equation (7) posterior update",
            "pairwise_similarity": "exact equality of HELM canonical mapped_output",
            "oracle": "HELM quasi_exact_match for MedQA",
            "temperature_grid": TAUS,
            "pool_size": POOL_SIZE,
            "budget": BUDGET,
            "important_scope": (
                "independent equation-faithful instantiation on public HELM-Lite MedQA; "
                "not the authors' unreleased 23-dataset artifact"
            ),
        },
        "data": {
            "source": HELM_BASE,
            "helm_repo": HELM_REPO,
            "helm_version": HELM_VERSION,
            "models": len(MODELS),
            "queries": len(ids),
            "model_names": MODELS,
            "score_min": float(oracle.mean(axis=1).min()),
            "score_max": float(oracle.mean(axis=1).max()),
            "provenance": provenance,
            "response_digest": hashlib.sha256(
                json.dumps(
                    responses.tolist(), ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "oracle_digest": hashlib.sha256(oracle.tobytes()).hexdigest(),
        },
        "target_screen": {
            "uses_oracle_labels": False,
            "screen_seeds": SCREEN_SEEDS,
            "selected": selected,
            "all_models": sorted(
                screen,
                key=lambda row: row["screen_first_query_change_fraction"],
                reverse=True,
            ),
        },
        "one_step_holdout": one_step,
        "full_exact_clone": full_clone,
        "natural_same_provider_pairs": pairs,
        "provider_threshold_audit": threshold_audit,
        "response_level_wrappers_tau_1": wrappers,
        "wrapper_clean_runs": compact_runs(wrapper_clean_runs),
        "interpretation_rules": {
            "exact_clone": "real HELM responses copied as an additional registry entry",
            "label_blind_wrapper": (
                "actual canonical answer strings are changed using only response frequencies; "
                "quality is evaluated post hoc"
            ),
            "oracle_aware_wrapper": (
                "adaptive existence stress test using wrong-to-wrong substitutions; do not "
                "present as label-free deployment attack"
            ),
            "go_condition": (
                "real-response current-method loss plus a nontrivial defense trade-off; "
                "path change alone is insufficient"
            ),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "elapsed_seconds": result["elapsed_seconds"],
                "selected_target": selected,
                "one_step": one_step,
                "tau_1_clone": full_clone["1.0"]["comparison"],
                "blind_76": wrappers["label_blind"]["76"]["comparison"],
                "aware_76": wrappers["oracle_aware"]["76"]["comparison"],
                "wide_gpt35": wrappers["wide_oracle_aware"][
                    "openai_gpt-3.5-turbo-0613"
                ]["comparison"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
