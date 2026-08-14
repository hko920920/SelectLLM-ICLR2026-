from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import step100_common as common
from step101_infer_candidate import CANDIDATES


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / f"STEP101_IMDB_DEVELOPMENT_PARENT_ROSTER_SELECTION_PROTOCOL_{common.DATE}.md"
DEV = ROOT / f"STEP100_IMDB_DEVELOPMENT_ROWS_{common.DATE}.json"
LEGACY = ROOT / "step100_stagea_work" / "root_predictions.npz"
CANDIDATE_DIR = ROOT / "step101_candidate_predictions"
OUTPUT = ROOT / f"STEP101_IMDB_DEVELOPMENT_PARENT_ROSTER_SELECTION_{common.DATE}.json"


def effect(
    predictions: np.ndarray, labels: np.ndarray, indices: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    codes = predictions[indices]
    references = labels[indices]
    parent_wrong = codes[:, 0] != references
    aliases = np.repeat(codes[:, 0, None], common.N_ALIASES, axis=1)
    for alias in range(common.N_ALIASES):
        aliases[parent_wrong, alias] = common.N_CLASSES + alias
    refined_codes = np.column_stack([codes, aliases])
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)]
    )
    pools = common.sample_pools(len(indices), seeds, common.POOL_SIZE)
    clean = common.run_active(codes, references, clean_parents, codes, pools, seeds, 10, 0.025)
    refined = common.run_active(
        refined_codes, references, refined_parents, codes, pools, seeds, 10, 0.025
    )
    fixed = common.run_fixed(
        refined_codes, references, refined_parents, codes, pools, clean.queries, seeds
    )
    terminal = refined.terminal - clean.terminal
    fixed_terminal = fixed.terminal - clean.terminal
    fixed_cumulative = fixed.cumulative - clean.cumulative
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    accuracies = np.mean(codes == references[:, None], axis=0)
    complementarity = [
        float(np.mean((codes[:, root] == references)[parent_wrong]))
        for root in range(1, common.ROSTER_SIZE)
    ]
    row: dict[str, Any] = {
        "rows": len(indices),
        "root_accuracies": accuracies.tolist(),
        "parent_minus_best_challenger": float(accuracies[0] - np.max(accuracies[1:])),
        "parent_error_prevalence": float(np.mean(parent_wrong)),
        "max_challenger_correct_given_parent_wrong": max(complementarity),
        "oracle_terminal_harm": float(np.mean(terminal)),
        "oracle_terminal_harm_pp": 100 * float(np.mean(terminal)),
        "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
        "clean_parent_terminal_rate": float(np.mean(clean_final == 0)),
        "refined_parent_terminal_rate": float(np.mean(refined_final == 0)),
        "parent_to_challenger_rate": float(np.mean((clean_final == 0) & (refined_final != 0))),
        "challenger_to_parent_rate": float(np.mean((clean_final != 0) & (refined_final == 0))),
        "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
        "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
        "fixed_root_history_exact": bool(np.array_equal(fixed.roots, clean.roots)),
    }
    row["gates"] = {
        "parent_gap_at_least_2_5pp": row["parent_minus_best_challenger"] >= 0.025,
        "parent_error_at_least_3pct": row["parent_error_prevalence"] >= 0.03,
        "challenger_complementarity_at_least_15pct": row["max_challenger_correct_given_parent_wrong"] >= 0.15,
        "oracle_terminal_at_least_1pp": row["oracle_terminal_harm"] >= 0.01,
        "path_change_at_least_half": row["path_change_rate"] >= 0.50,
        "directionality": row["parent_to_challenger_rate"] > row["challenger_to_parent_rate"],
        "fixed_exact_zero": row["max_abs_fixed_terminal_delta"] == 0
        and row["max_abs_fixed_cumulative_delta"] == 0
        and row["fixed_root_history_exact"],
    }
    row["eligible"] = all(row["gates"].values())
    return row


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    payload = json.loads(DEV.read_text(encoding="utf-8"))
    rows = payload["rows"]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    parts = {
        name: np.asarray([i for i, row in enumerate(rows) if row["partition"] == name], dtype=np.int64)
        for name in ("selector_search", "selector_verify")
    }
    legacy = np.load(LEGACY, allow_pickle=False)["root_predictions"].astype(np.int64)
    results: dict[str, Any] = {}
    order = {key: index for index, key in enumerate(CANDIDATES)}
    for key, spec in CANDIDATES.items():
        prediction_path = CANDIDATE_DIR / f"{key}.npz"
        audit_path = CANDIDATE_DIR / f"{key}.audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["repo_id"] != spec.repo_id or audit["revision"] != spec.revision:
            raise AssertionError({"candidate_binding_drift": key})
        if audit["prediction_file_sha256"] != common.sha256_path(prediction_path):
            raise AssertionError({"candidate_prediction_drift": key})
        parent = np.load(prediction_path, allow_pickle=False)["predictions"].astype(np.int64)
        roster = np.column_stack([parent, legacy[:, 2], legacy[:, 0], legacy[:, 1]])
        search = effect(roster, labels, parts["selector_search"], common.SEARCH_SEEDS)
        verify = effect(roster, labels, parts["selector_verify"], common.VERIFY_SEEDS)
        results[key] = {
            "candidate": key,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "prediction_file": audit["prediction_file"],
            "prediction_file_sha256": audit["prediction_file_sha256"],
            "roster_keys": [key, "textattack_albert", "lvwerra_distilbert_parent", "textattack_distilbert"],
            "selector_search": search,
            "selector_verify": verify,
            "eligible": bool(search["eligible"] and verify["eligible"]),
            "selection_primary": min(search["oracle_terminal_harm"], verify["oracle_terminal_harm"]),
            "selection_secondary": verify["oracle_terminal_harm"],
            "selection_gap": min(search["parent_minus_best_challenger"], verify["parent_minus_best_challenger"]),
        }
        print(json.dumps({
            "candidate": key,
            "eligible": results[key]["eligible"],
            "search_gap_pp": 100 * search["parent_minus_best_challenger"],
            "verify_gap_pp": 100 * verify["parent_minus_best_challenger"],
            "search_oracle_terminal_pp": search["oracle_terminal_harm_pp"],
            "verify_oracle_terminal_pp": verify["oracle_terminal_harm_pp"],
        }), flush=True)
    eligible = [row for row in results.values() if row["eligible"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["selection_primary"], row["selection_secondary"],
                row["selection_gap"], -order[row["candidate"]],
            ),
        )
        decision = "GO_STEP101_SELECTED_PARENT_TO_LEARNED_DEVELOPMENT"
        selected_parent = selected["candidate"]
    else:
        decision = "NO_GO_STEP101_PARENT_GEOMETRY_STOP"
        selected_parent = None
    output = {
        "selection_id": "STEP101_IMDB_DEVELOPMENT_PARENT_ROSTER_SELECTION_V1",
        "date": common.DATE,
        "decision": decision,
        "selected_parent": selected_parent,
        "protocol_sha256": common.sha256_path(PROTOCOL),
        "development_sha256": common.sha256_path(DEV),
        "legacy_prediction_sha256": common.sha256_path(LEGACY),
        "candidates": results,
        "sealed_outcome_opened": False,
        "code_sha256": {
            "step93_common.py": common.sha256_path(ROOT / "step93_common.py"),
            "step100_common.py": common.sha256_path(ROOT / "step100_common.py"),
            "step101_infer_candidate.py": common.sha256_path(ROOT / "step101_infer_candidate.py"),
            "step101_select_parent_roster.py": common.sha256_path(Path(__file__)),
        },
    }
    common.json_dump(OUTPUT, output)
    print(json.dumps({
        "decision": decision,
        "selected_parent": selected_parent,
        "output": OUTPUT.name,
        "sealed_outcome_opened": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
