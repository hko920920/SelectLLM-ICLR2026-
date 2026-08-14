"""External real-response generality gate for current Select-LLM equations.

The Step 24 MedQA result is repeated on two independent public HELM-Lite tasks:
GSM8K (the exact task family and tau reported in arXiv:2605.24981) and
OpenBookQA.  The script uses actual canonicalized model responses, a disjoint
response-only target screen, and a common unmodified oracle for regret.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "external_data" / "helm_lite_step25_raw"
OUT = ROOT / "STEP25_CURRENT_METHOD_EXTERNAL_GENERALITY_GATE_2026-08-06.json"
BASE = (
    "https://storage.googleapis.com/crfm-helm-public/"
    "lite/benchmark_output/runs/v1.0.0/"
)
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

SCREEN_SEEDS = list(range(0, 500))
ONE_STEP_TEST_SEEDS = list(range(1000, 2000))
FULL_SEEDS = list(range(2000, 2050))
BUDGET = 30
ALIASES = 4

TASKS = {
    "gsm8k": {
        "prefix": "gsm:",
        "oracle_key": "final_number_exact_match",
        "pool_size": 500,
        "tau": 1.0,
        "camouflage_fraction": 0.05,
        "canonicalizer": "final_number",
    },
    "openbookqa": {
        "prefix": "commonsense:dataset=openbookqa,method=multiple_choice_joint,",
        "oracle_key": "exact_match",
        "pool_size": 400,
        "tau": 1.0,
        "camouflage_fraction": 0.076,
        "canonicalizer": "mapped_output",
    },
}


def cache_json(relative: str, filename: str) -> tuple[object, dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / filename
    url = BASE + urllib.parse.quote(relative, safe="/:,@=-._")
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
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "source": source,
    }


def final_number(text: str) -> str:
    # Exact HELM v0.4.0 final_number_exact_match extraction pattern.
    matches = re.findall(r"-?[\d,]+(?:.\d+)?", str(text))
    return matches[-1].replace(",", "") if matches else ""


def provider(model: str) -> str:
    return model.split("_", 1)[0]


def load_task(name: str, config: dict) -> dict:
    def load_model(model: str) -> tuple[str, list[dict], dict]:
        relative = f"{config['prefix']}model={model}/display_predictions.json"
        rows, provenance = cache_json(
            relative, f"{name}__{model}__display_predictions.json"
        )
        return model, rows, provenance

    data: dict[str, list[dict]] = {}
    provenance: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_model, model) for model in MODELS]
        for future in as_completed(futures):
            model, rows, source = future.result()
            data[model] = rows
            provenance[model] = source

    instance_relative = f"{config['prefix']}model=openai_gpt-4-0613/instances.json"
    instance_rows, instance_source = cache_json(
        instance_relative, f"{name}__openai_gpt-4-0613__instances.json"
    )
    provenance["instances"] = instance_source
    instances = {row["id"]: row for row in instance_rows}

    ids = [row["instance_id"] for row in data[MODELS[0]]]
    responses = []
    oracle = []
    for model in MODELS:
        keyed = {row["instance_id"]: row for row in data[model]}
        if set(keyed) != set(ids):
            raise AssertionError(f"{name}: instance mismatch for {model}")
        model_response = []
        model_oracle = []
        for instance_id in ids:
            row = keyed[instance_id]
            if config["canonicalizer"] == "mapped_output":
                value = row.get("mapped_output")
                if value is None:
                    value = "INVALID:" + str(row.get("predicted_text", ""))
            else:
                value = final_number(row.get("predicted_text", ""))
            model_response.append(str(value).strip())
            model_oracle.append(float(row["stats"].get(config["oracle_key"], 0.0)))
        responses.append(model_response)
        oracle.append(model_oracle)

    alternatives = []
    correct = []
    for instance_id in ids:
        refs = instances[instance_id]["references"]
        raw_options = [str(ref["output"]["text"]).strip() for ref in refs]
        tagged = [
            str(ref["output"]["text"]).strip()
            for ref in refs
            if "correct" in ref.get("tags", [])
        ]
        if len(tagged) != 1:
            raise AssertionError(f"{name}: instance lacks one correct reference")
        if config["canonicalizer"] == "final_number":
            correct_value = final_number(tagged[0])
            option_values = sorted(
                {
                    final_number(value)
                    for value in raw_options
                    if final_number(value) != ""
                }
            )
        else:
            correct_value = tagged[0]
            option_values = raw_options
        correct.append(correct_value)
        alternatives.append(option_values)

    response_array = np.asarray(responses, dtype=object)
    oracle_array = np.asarray(oracle, dtype=np.float64)
    if response_array.shape != oracle_array.shape:
        raise AssertionError(f"{name}: response/oracle shape mismatch")
    return {
        "ids": ids,
        "responses": response_array,
        "oracle": oracle_array,
        "alternatives": alternatives,
        "correct": np.asarray(correct, dtype=object),
        "provenance": provenance,
    }


def one_hot(responses: np.ndarray) -> np.ndarray:
    models, queries = responses.shape
    codes = np.zeros((models, queries), dtype=np.int16)
    maximum = 0
    for query in range(queries):
        mapping = {}
        for model in range(models):
            value = str(responses[model, query])
            if value not in mapping:
                mapping[value] = len(mapping)
            codes[model, query] = mapping[value]
        maximum = max(maximum, len(mapping))
    return np.stack(
        [(codes.T == category).astype(np.float32) for category in range(maximum)]
    )


def scenario(responses: np.ndarray, oracle: np.ndarray, originals: list[int]) -> dict:
    return {
        "responses": responses,
        "oracle": oracle,
        "originals": np.asarray(originals, dtype=np.int64),
        "one_hot": one_hot(responses),
    }


def acquisition(current: dict) -> np.ndarray:
    probability = np.ones(len(current["oracle"]), dtype=np.float64)
    probability /= probability.sum()
    mass = np.einsum("lqm,m->lq", current["one_hot"], probability, optimize=True)
    return np.sum(mass * mass, axis=0)


def sampled_pool(seed: int, queries: int, pool_size: int) -> np.ndarray:
    return np.asarray(
        sorted(random.Random(seed).sample(range(queries), pool_size)), dtype=np.int64
    )


def first_queries(scores: np.ndarray, seeds: list[int], pool_size: int) -> np.ndarray:
    out = []
    for seed in seeds:
        indices = sampled_pool(seed, len(scores), pool_size)
        out.append(int(indices[int(np.argmin(scores[indices]))]))
    return np.asarray(out, dtype=np.int64)


def clone_scenario(responses: np.ndarray, oracle: np.ndarray, target: int, copies: int) -> dict:
    return scenario(
        np.concatenate([responses, np.repeat(responses[[target]], copies, axis=0)]),
        np.concatenate([oracle, np.repeat(oracle[[target]], copies, axis=0)]),
        list(range(len(MODELS))) + [target] * copies,
    )


def run(
    current: dict,
    common_oracle: np.ndarray,
    *,
    seed: int,
    pool_size: int,
    tau: float,
) -> dict:
    indices = sampled_pool(seed, common_oracle.shape[1], pool_size)
    unlabeled = indices.tolist()
    cumulative = np.zeros(len(current["oracle"]), dtype=np.float64)
    common_scores = common_oracle[:, indices].mean(axis=1)
    best = float(common_scores.max())
    queried = []
    regrets = []
    selected = []
    for _ in range(BUDGET):
        shifted = cumulative / tau
        shifted -= shifted.max()
        posterior = np.exp(shifted)
        posterior /= posterior.sum()
        remaining = np.asarray(unlabeled, dtype=np.int64)
        masses = np.einsum(
            "lqm,m->lq",
            current["one_hot"][:, remaining, :],
            posterior,
            optimize=True,
        )
        location = int(np.argmin(np.sum(masses * masses, axis=0)))
        query = int(unlabeled.pop(location))
        queried.append(query)
        cumulative += current["oracle"][:, query]
        local = int(np.argmax(cumulative))
        original = int(current["originals"][local])
        selected.append(original)
        regrets.append(best - float(common_scores[original]))
    return {
        "seed": seed,
        "queried": queried,
        "selected": selected[-1],
        "cumulative_regret": float(np.sum(regrets)),
        "final_regret": float(regrets[-1]),
    }


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, values[0]):
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def compare(clean: list[dict], changed: list[dict], seed: int) -> dict:
    cumulative = np.asarray(
        [b["cumulative_regret"] - a["cumulative_regret"] for a, b in zip(clean, changed)]
    )
    final = np.asarray(
        [b["final_regret"] - a["final_regret"] for a, b in zip(clean, changed)]
    )
    return {
        "runs": len(clean),
        "path_change_fraction": float(
            np.mean([a["queried"] != b["queried"] for a, b in zip(clean, changed)])
        ),
        "mean_changed_positions": float(
            np.mean(
                [
                    sum(x != y for x, y in zip(a["queried"], b["queried"]))
                    for a, b in zip(clean, changed)
                ]
            )
        ),
        "final_selection_change_fraction": float(
            np.mean([a["selected"] != b["selected"] for a, b in zip(clean, changed)])
        ),
        "mean_delta_cumulative_regret": float(cumulative.mean()),
        "delta_cumulative_regret_95ci": bootstrap_ci(cumulative, seed),
        "mean_delta_final_regret": float(final.mean()),
        "delta_final_regret_95ci": bootstrap_ci(final, seed + 1),
        "harmful_fraction": float(np.mean(cumulative > 1e-12)),
        "improved_fraction": float(np.mean(cumulative < -1e-12)),
    }


def wrapper(
    data: dict,
    *,
    target: int,
    distance_queries: int,
    oracle_aware: bool,
) -> tuple[dict, dict]:
    responses = data["responses"]
    oracle = data["oracle"]
    alternatives = data["alternatives"]
    correct = data["correct"]
    eligible = np.flatnonzero(oracle[target] == 0) if oracle_aware else np.arange(responses.shape[1])
    if len(eligible) < distance_queries:
        raise ValueError("insufficient eligible queries")

    ranked = []
    candidate_options: dict[int, list[str]] = {}
    for query in eligible:
        query = int(query)
        observed = list(dict.fromkeys(str(x) for x in responses[:, query]))
        choices = list(dict.fromkeys([*alternatives[query], *observed]))
        choices = [value for value in choices if value != responses[target, query]]
        if oracle_aware:
            choices = [value for value in choices if value != correct[query]]
        if not choices:
            choices = [f"CAMOUFLAGE_WRONG_{query}"]
        choices.sort(key=lambda value: float(np.mean(responses[:, query] == value)))
        candidate_options[query] = choices
        ranked.append((float(np.mean(responses[:, query] == choices[0])), query))
    order = np.asarray([query for _, query in sorted(ranked)], dtype=np.int64)

    alias_responses = []
    alias_oracle = []
    digests = []
    stride = max(1, len(order) // 7)
    for alias in range(ALIASES):
        changed = np.roll(order, alias * stride)[:distance_queries]
        row = responses[target].copy()
        score = oracle[target].copy()
        for query in changed:
            query = int(query)
            choices = candidate_options[query]
            row[query] = choices[(alias + query) % len(choices)]
            score[query] = float(row[query] == correct[query])
        if oracle_aware and not np.array_equal(score, oracle[target]):
            raise AssertionError("wrong-to-wrong wrapper changed oracle quality")
        if not math.isclose(
            float(np.mean(row != responses[target])),
            distance_queries / responses.shape[1],
            abs_tol=1e-12,
        ):
            raise AssertionError("wrapper response distance mismatch")
        alias_responses.append(row)
        alias_oracle.append(score)
        digests.append(
            hashlib.sha256(
                json.dumps(row.tolist(), ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        )
    if len(set(digests)) != ALIASES:
        raise AssertionError("aliases are not distinct")
    current = scenario(
        np.concatenate([responses, np.asarray(alias_responses, dtype=object)]),
        np.concatenate([oracle, np.asarray(alias_oracle, dtype=np.float64)]),
        list(range(len(MODELS))) + [target] * ALIASES,
    )
    return current, {
        "target": MODELS[target],
        "target_score": float(oracle[target].mean()),
        "distance_queries": distance_queries,
        "response_disagreement": distance_queries / responses.shape[1],
        "oracle_aware": oracle_aware,
        "alias_score_gaps": [
            float(row.mean() - oracle[target].mean()) for row in alias_oracle
        ],
        "exact_dedup_survivors": ALIASES,
        "alias_response_digests": digests,
    }


def natural_pairs(data: dict) -> list[dict]:
    responses = data["responses"]
    oracle = data["oracle"]
    rows = []
    for left in range(len(MODELS)):
        for right in range(left + 1, len(MODELS)):
            if provider(MODELS[left]) != provider(MODELS[right]):
                continue
            rows.append(
                {
                    "left": MODELS[left],
                    "right": MODELS[right],
                    "provider": provider(MODELS[left]),
                    "response_disagreement": float(
                        np.mean(responses[left] != responses[right])
                    ),
                    "left_score": float(oracle[left].mean()),
                    "right_score": float(oracle[right].mean()),
                    "absolute_score_gap": float(
                        abs(oracle[left].mean() - oracle[right].mean())
                    ),
                }
            )
    return sorted(rows, key=lambda row: row["response_disagreement"])


def audit_task(name: str, config: dict) -> dict:
    data = load_task(name, config)
    responses = data["responses"]
    oracle = data["oracle"]
    clean = scenario(responses, oracle, list(range(len(MODELS))))
    clean_scores = acquisition(clean)
    clean_screen = first_queries(clean_scores, SCREEN_SEEDS, config["pool_size"])

    screens = []
    for target in range(len(MODELS)):
        cloned = clone_scenario(responses, oracle, target, 1)
        selected = first_queries(acquisition(cloned), SCREEN_SEEDS, config["pool_size"])
        screens.append(
            {
                "target_index": target,
                "target": MODELS[target],
                "change_fraction": float(np.mean(selected != clean_screen)),
            }
        )
    chosen = max(screens, key=lambda row: (row["change_fraction"], row["target"]))
    target = int(chosen["target_index"])

    clean_test = first_queries(clean_scores, ONE_STEP_TEST_SEEDS, config["pool_size"])
    one_step = []
    for copies in [1, 2, 4]:
        current = clone_scenario(responses, oracle, target, copies)
        attacked = first_queries(acquisition(current), ONE_STEP_TEST_SEEDS, config["pool_size"])
        one_step.append(
            {
                "copies": copies,
                "change_fraction": float(np.mean(attacked != clean_test)),
            }
        )

    clean_runs = [
        run(
            clean,
            oracle,
            seed=seed,
            pool_size=config["pool_size"],
            tau=config["tau"],
        )
        for seed in FULL_SEEDS
    ]
    clone_one = clone_scenario(responses, oracle, target, 1)
    clone_runs = [
        run(
            clone_one,
            oracle,
            seed=seed,
            pool_size=config["pool_size"],
            tau=config["tau"],
        )
        for seed in FULL_SEEDS
    ]

    dose = max(1, int(round(config["camouflage_fraction"] * responses.shape[1])))
    blind, blind_meta = wrapper(
        data, target=target, distance_queries=dose, oracle_aware=False
    )
    blind_runs = [
        run(
            blind,
            oracle,
            seed=seed,
            pool_size=config["pool_size"],
            tau=config["tau"],
        )
        for seed in FULL_SEEDS
    ]

    eligible = np.sum(oracle == 0, axis=1)
    aware_candidates = np.flatnonzero(eligible >= dose)
    aware_target = int(aware_candidates[np.argmax(oracle[aware_candidates].mean(axis=1))])
    aware, aware_meta = wrapper(
        data, target=aware_target, distance_queries=dose, oracle_aware=True
    )
    aware_runs = [
        run(
            aware,
            oracle,
            seed=seed,
            pool_size=config["pool_size"],
            tau=config["tau"],
        )
        for seed in FULL_SEEDS
    ]

    pairs = natural_pairs(data)
    return {
        "queries": responses.shape[1],
        "models": responses.shape[0],
        "score_range": [float(oracle.mean(axis=1).min()), float(oracle.mean(axis=1).max())],
        "pool_size": config["pool_size"],
        "budget": BUDGET,
        "tau": config["tau"],
        "screen_selected_target": chosen,
        "one_step_holdout": one_step,
        "full_exact_clone": compare(clean_runs, clone_runs, 10000),
        "label_blind_wrapper": {
            "metadata": blind_meta,
            "comparison": compare(clean_runs, blind_runs, 11000),
        },
        "oracle_aware_wrapper": {
            "metadata": aware_meta,
            "comparison": compare(clean_runs, aware_runs, 12000),
        },
        "natural_same_provider_pairs": pairs,
        "natural_pairs_within_wrapper_radius": [
            row for row in pairs if row["response_disagreement"] <= dose / responses.shape[1]
        ],
        "provenance": data["provenance"],
        "response_digest": hashlib.sha256(
            json.dumps(
                responses.tolist(), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "oracle_digest": hashlib.sha256(oracle.tobytes()).hexdigest(),
    }


def main() -> None:
    started = time.time()
    results = {name: audit_task(name, config) for name, config in TASKS.items()}
    result = {
        "gate": "STEP25_CURRENT_METHOD_EXTERNAL_GENERALITY",
        "created_local_date": "2026-08-06",
        "source": BASE,
        "paper": "https://arxiv.org/abs/2605.24981",
        "scope": (
            "equation-faithful independent instantiations on public HELM-Lite; "
            "not the authors' unreleased processed artifact"
        ),
        "full_seeds": FULL_SEEDS,
        "tasks": results,
        "elapsed_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "elapsed_seconds": result["elapsed_seconds"],
                "summary": {
                    name: {
                        "target": row["screen_selected_target"],
                        "one_step": row["one_step_holdout"],
                        "clone": row["full_exact_clone"],
                        "blind": row["label_blind_wrapper"],
                        "aware": row["oracle_aware_wrapper"],
                    }
                    for name, row in results.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
