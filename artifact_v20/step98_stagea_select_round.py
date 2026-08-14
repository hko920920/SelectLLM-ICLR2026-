from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import step98_common as common


ROOT = Path(__file__).resolve().parent
TRAIN_ROWS = ROOT / f"STEP98_ANLI_TRAIN_ONLY_ROWS_{common.DATE}.json"
STAGE0 = ROOT / f"STEP98_STAGE0_TRAIN_ONLY_MANIFEST_{common.DATE}.json"
WORK = ROOT / "step98_selection_work"
OUTPUT = ROOT / f"STEP98_TRAIN_ONLY_ROUND_SELECTION_{common.DATE}.json"


def geometry(
    round_key: str,
    partition: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    accuracies = np.mean(predictions == labels[:, None], axis=0)
    parent = predictions[:, 0]
    parent_wrong = parent != labels
    complementarity = [
        float(np.mean((predictions[:, root] == labels)[parent_wrong]))
        for root in range(1, common.ROSTER_SIZE)
    ]
    aliases = np.repeat(parent[:, None], common.N_ALIASES, axis=1)
    for alias in range(common.N_ALIASES):
        aliases[parent_wrong, alias] = common.N_CLASSES + alias
    refined_codes = np.column_stack([predictions, aliases])
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)]
    )
    pools = common.sample_pools(len(labels), seeds, common.POOL_SIZE)
    clean = common.run_active(
        predictions, labels, clean_parents, predictions, pools, seeds,
        common.BUDGET, common.TAU,
    )
    refined = common.run_active(
        refined_codes, labels, refined_parents, predictions, pools, seeds,
        common.BUDGET, common.TAU,
    )
    fixed = common.run_fixed(
        refined_codes, labels, refined_parents, predictions, pools,
        clean.queries, seeds,
    )
    terminal = refined.terminal - clean.terminal
    fixed_terminal = fixed.terminal - clean.terminal
    fixed_cumulative = fixed.cumulative - clean.cumulative
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    parent_to_challenger = (clean_final == 0) & (refined_final != 0)
    challenger_to_parent = (clean_final != 0) & (refined_final == 0)
    parent_gap = float(accuracies[0] - np.max(accuracies[1:]))
    row = {
        "round": round_key,
        "partition": partition,
        "rows": len(labels),
        "root_accuracies": accuracies.tolist(),
        "root_accuracy_by_key": {
            spec.key: float(value) for spec, value in zip(common.ROOT_SPECS, accuracies)
        },
        "parent_minus_best_challenger": parent_gap,
        "parent_error_prevalence": float(np.mean(parent_wrong)),
        "challenger_correct_given_parent_wrong": {
            common.ROOT_SPECS[index].key: complementarity[index - 1]
            for index in range(1, common.ROSTER_SIZE)
        },
        "max_challenger_correct_given_parent_wrong": max(complementarity),
        "oracle_terminal_harm": float(np.mean(terminal)),
        "oracle_terminal_harm_pp": float(100.0 * np.mean(terminal)),
        "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
        "final_root_change_rate": float(np.mean(clean_final != refined_final)),
        "clean_parent_terminal_rate": float(np.mean(clean_final == 0)),
        "refined_parent_terminal_rate": float(np.mean(refined_final == 0)),
        "parent_to_challenger_rate": float(np.mean(parent_to_challenger)),
        "challenger_to_parent_rate": float(np.mean(challenger_to_parent)),
        "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
        "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
        "fixed_root_history_exact": bool(np.array_equal(fixed.roots, clean.roots)),
    }
    row["gates"] = {
        "parent_gap_at_least_2_5pp": parent_gap >= 0.025,
        "parent_error_at_least_10pct": row["parent_error_prevalence"] >= 0.10,
        "challenger_complementarity_at_least_15pct": row["max_challenger_correct_given_parent_wrong"] >= 0.15,
        "oracle_terminal_harm_at_least_1pp": row["oracle_terminal_harm"] >= 0.01,
        "parent_to_challenger_exceeds_reverse": row["parent_to_challenger_rate"] > row["challenger_to_parent_rate"],
        "fixed_query_exact_zero": row["max_abs_fixed_terminal_delta"] == 0
        and row["max_abs_fixed_cumulative_delta"] == 0
        and row["fixed_root_history_exact"],
    }
    row["eligible"] = all(row["gates"].values())
    return row


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    payload = json.loads(TRAIN_ROWS.read_text(encoding="utf-8"))
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    if stage0["protocol_sha256"] != common.sha256_path(common.PROTOCOL):
        raise AssertionError("protocol binding drift")
    if stage0["train_rows_sha256"] != common.sha256_path(TRAIN_ROWS):
        raise AssertionError("train-row binding drift")
    WORK.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, dict[str, Any]] = {}
    for round_key in common.ROUNDS:
        if round_key not in payload["rounds"]:
            continue
        rows = [
            row for row in payload["rounds"][round_key]
            if row["partition"] in ("geometry_search", "geometry_verify")
        ]
        premises = [str(row["premise"]) for row in rows]
        hypotheses = [str(row["hypothesis"]) for row in rows]
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        cache = WORK / f"{round_key}_geometry_root_predictions.npz"
        audit_path = WORK / f"{round_key}_geometry_root_audits.json"
        if cache.is_file() and audit_path.is_file():
            predictions = np.load(cache, allow_pickle=False)["root_predictions"].astype(np.int64)
            audits = json.loads(audit_path.read_text(encoding="utf-8"))
            resumed = True
        elif cache.exists() or audit_path.exists():
            raise RuntimeError(f"incomplete geometry cache for {round_key}")
        else:
            columns: list[np.ndarray] = []
            audits: list[dict[str, Any]] = []
            for root_index, spec in enumerate(common.ROOT_SPECS):
                column, audit = common.infer_root_predictions(
                    common.as_learned_spec(spec), premises, hypotheses, batch_size=64
                )
                columns.append(column)
                audits.append(audit)
                print(f"{round_key}: root {root_index + 1}/{common.ROSTER_SIZE} {spec.key}", flush=True)
            predictions = np.stack(columns, axis=1)
            np.savez_compressed(cache, root_predictions=predictions.astype(np.int16))
            common.json_dump(audit_path, audits)
            resumed = False
        if predictions.shape != (len(rows), common.ROSTER_SIZE):
            raise AssertionError({"round": round_key, "shape": predictions.shape})
        partitions = {
            name: np.asarray(
                [index for index, row in enumerate(rows) if row["partition"] == name],
                dtype=np.int64,
            )
            for name in ("geometry_search", "geometry_verify")
        }
        search = geometry(
            round_key, "geometry_search",
            predictions[partitions["geometry_search"]], labels[partitions["geometry_search"]],
            common.GEOMETRY_SEARCH_SEEDS,
        )
        verify = geometry(
            round_key, "geometry_verify",
            predictions[partitions["geometry_verify"]], labels[partitions["geometry_verify"]],
            common.GEOMETRY_VERIFY_SEEDS,
        )
        all_rows[round_key] = {
            "round": round_key,
            "cache_resumed": resumed,
            "prediction_file": str(cache.relative_to(ROOT)).replace("\\", "/"),
            "prediction_sha256": common.sha256_path(cache),
            "audit_file": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
            "audit_sha256": common.sha256_path(audit_path),
            "root_audits": audits,
            "geometry_search": search,
            "geometry_verify": verify,
            "eligible": bool(search["eligible"] and verify["eligible"]),
            "selection_primary": min(search["oracle_terminal_harm"], verify["oracle_terminal_harm"]),
            "selection_secondary": verify["oracle_terminal_harm"],
            "selection_gap": min(search["parent_minus_best_challenger"], verify["parent_minus_best_challenger"]),
        }
        print(json.dumps({
            "round": round_key, "eligible": all_rows[round_key]["eligible"],
            "search_terminal_pp": search["oracle_terminal_harm_pp"],
            "verify_terminal_pp": verify["oracle_terminal_harm_pp"],
            "search_gap_pp": 100 * search["parent_minus_best_challenger"],
            "verify_gap_pp": 100 * verify["parent_minus_best_challenger"],
        }), flush=True)
    eligible = [row for row in all_rows.values() if row["eligible"]]
    if not eligible:
        selected_round = None
        decision = "NO_GO_STEP98_TRAIN_ONLY_GEOMETRY_STOP"
    else:
        order = {key: index for index, key in enumerate(common.ROUNDS)}
        selected = max(
            eligible,
            key=lambda row: (
                row["selection_primary"], row["selection_secondary"],
                row["selection_gap"], -order[row["round"]],
            ),
        )
        selected_round = selected["round"]
        decision = "GO_STEP98_SELECTED_ROUND_TO_LEARNED_DEVELOPMENT"
    output = {
        "selection_id": "STEP98_TRAIN_ONLY_ROUND_SELECTION_V1",
        "date": common.DATE,
        "decision": decision,
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "amendment_a_sha256": common.sha256_path(ROOT / f"STEP98_PREREGISTRATION_AMENDMENT_A_STAGE0_ELIGIBILITY_{common.DATE}.md"),
        "stage0_manifest_sha256": common.sha256_path(STAGE0),
        "train_rows_sha256": common.sha256_path(TRAIN_ROWS),
        "rounds": all_rows,
        "selected_round": selected_round,
        "dev_accessed": False,
        "code_sha256": {
            "step93_common.py": common.sha256_path(ROOT / "step93_common.py"),
            "step95_common.py": common.sha256_path(ROOT / "step95_common.py"),
            "step96_common.py": common.sha256_path(ROOT / "step96_common.py"),
            "step98_common.py": common.sha256_path(ROOT / "step98_common.py"),
            "step98_stagea_select_round.py": common.sha256_path(Path(__file__)),
        },
    }
    common.json_dump(OUTPUT, output)
    print(json.dumps({"decision": decision, "selected_round": selected_round, "output": OUTPUT.name}, indent=2))


if __name__ == "__main__":
    main()
