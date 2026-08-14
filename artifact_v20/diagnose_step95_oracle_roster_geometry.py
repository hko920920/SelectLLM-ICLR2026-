from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import step93_common as selector
import step95_common as step95
from diagnose_step95_terminal_bottleneck import trace_active


ROOT = Path(__file__).resolve().parent
ARRAYS = ROOT / "STEP95_STAGEA_DEVELOPMENT_ARRAYS_2026-08-13.npz"
OUTPUT = ROOT / "STEP95_POSTHOC_ORACLE_ROSTER_GEOMETRY_2026-08-13.json"
SEEDS = tuple(range(954000, 954100))
VALID_ROOTS = (1, 3, 4, 5, 6, 7)  # Step-95 mappings with explicit semantic metadata or plausible realized accuracy.


def summarize(
    parent: int,
    roster: tuple[int, ...],
    clean_codes: np.ndarray,
    labels: np.ndarray,
    pools: np.ndarray,
) -> dict[str, Any]:
    parent_position = roster.index(parent)
    parent_codes = clean_codes[:, parent_position]
    parent_wrong = parent_codes != labels
    aliases = np.repeat(parent_codes[:, None], step95.N_ALIASES, axis=1)
    for alias in range(step95.N_ALIASES):
        aliases[parent_wrong, alias] = step95.N_CLASSES + alias
    refined_codes = np.column_stack([clean_codes, aliases])
    clean_parents = np.arange(len(roster), dtype=np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.full(step95.N_ALIASES, parent_position, dtype=np.int64)]
    )
    clean = trace_active(
        clean_codes, labels, clean_parents, clean_codes, pools, SEEDS, 20, 0.05
    )
    refined = trace_active(
        refined_codes, labels, refined_parents, clean_codes, pools, SEEDS, 20, 0.05
    )
    core_feedback = selector.exact_match_feedback(clean_codes, labels)
    root_quality = np.mean(core_feedback[pools], axis=1)
    rows: list[dict[str, Any]] = []
    for budget in (5, 10, 20):
        step = budget - 1
        delta = refined["regrets"][:, step] - clean["regrets"][:, step]
        changed = clean["roots"][:, step] != refined["roots"][:, step]
        rows.append(
            {
                "budget": budget,
                "terminal_pp": float(100.0 * np.mean(delta)),
                "root_change_rate": float(np.mean(changed)),
                "positive_fraction": float(np.mean(delta > 0)),
                "negative_fraction": float(np.mean(delta < 0)),
                "conditional_terminal_pp": float(100.0 * np.mean(delta[changed]))
                if np.any(changed)
                else 0.0,
            }
        )
    accuracies = np.mean(clean_codes == labels[:, None], axis=0)
    complement = {}
    for position, root in enumerate(roster):
        if root == parent:
            continue
        complement[step95.ROOT_SPECS[root].key] = float(
            np.mean((clean_codes[:, position] == labels)[parent_wrong])
        )
    return {
        "parent": step95.ROOT_SPECS[parent].key,
        "roster": [step95.ROOT_SPECS[root].key for root in roster],
        "global_accuracy": {
            step95.ROOT_SPECS[root].key: float(value)
            for root, value in zip(roster, accuracies)
        },
        "challenger_correct_given_parent_wrong": complement,
        "parent_error_rate": float(np.mean(parent_wrong)),
        "budget_results": rows,
    }


def main() -> None:
    arrays = np.load(ARRAYS, allow_pickle=False)
    indices = arrays["verify_indices"].astype(np.int64)
    labels = arrays["labels"][indices].astype(np.int64)
    predictions = arrays["root_predictions"][indices].astype(np.int64)
    accuracies = np.mean(predictions == labels[:, None], axis=0)
    pools = selector.sample_pools(len(labels), SEEDS, pool_size=step95.POOL_SIZE)
    configurations: list[tuple[int, tuple[int, ...]]] = []
    for parent in VALID_ROOTS:
        lower = [
            root
            for root in VALID_ROOTS
            if root != parent and accuracies[root] <= accuracies[parent] - 0.005
        ]
        lower.sort(key=lambda root: (-float(accuracies[root]), root))
        for challenger in lower:
            configurations.append((parent, (parent, challenger)))
        if len(lower) >= 3:
            configurations.append((parent, tuple([parent, *lower[:3]])))
        if len(lower) >= 5:
            configurations.append((parent, tuple([parent, *lower[:5]])))
    rows = []
    for index, (parent, roster) in enumerate(configurations, start=1):
        row = summarize(parent, roster, predictions[:, roster], labels, pools)
        rows.append(row)
        best = max(item["terminal_pp"] for item in row["budget_results"])
        print(f"{index}/{len(configurations)} {row['parent']} n={len(roster)} max={best:+.3f}pp", flush=True)
    ranked = sorted(
        rows,
        key=lambda row: max(item["terminal_pp"] for item in row["budget_results"]),
        reverse=True,
    )
    output = {
        "diagnostic_id": "STEP95_POSTHOC_ORACLE_ROSTER_GEOMETRY_V1",
        "scope": {
            "partition": "Step 95 selector_verify development only",
            "seeds": [SEEDS[0], SEEDS[-1]],
            "sealed_test_opened": False,
            "confirmatory_use": False,
            "oracle": "aliases abstain on every parent-wrong coordinate; this is a reference-aware upper-bound diagnostic, not a reference-free result",
        },
        "rows": rows,
        "top_10": ranked[:10],
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": OUTPUT.name, "top": ranked[0]}, indent=2))


if __name__ == "__main__":
    main()
