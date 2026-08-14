from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import step89_stagea_develop as common


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-12"
MANIFEST_PATH = ROOT / f"STEP90_HIGH_CONFIDENCE_ADAPTER_STAGE0_MANIFEST_{DATE}.json"
MODEL_DIR = ROOT / "step90_models"
TASKS: dict[str, dict[str, Any]] = {
    "mnli": {
        "model": "cross-encoder/nli-distilroberta-base",
        "revision": "b14d131f9d32668a5e6a982729b57ff6ed5dfcbd",
        "classes": 3,
    },
    "qqp": {
        "model": "cross-encoder/quora-distilroberta-base",
        "revision": "f62e7a4b20b97195c2868e53ec59126df5eac743",
        "classes": 2,
    },
}
PILOT_SEEDS = tuple(range(20))
VERIFY_SEEDS = tuple(range(300))
ROSTER_SIZES = (8, 12, 20)
BUDGETS = (20, 30, 50)
TAUS = (0.25, 1.0, 4.0)
TRIGGER_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)
POOL_SIZE = 400
TOP_TO_VERIFY = 12


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_logits(task: str, rows: list[dict[str, Any]]) -> np.ndarray:
    spec = TASKS[task]
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["model"], revision=spec["revision"], use_safetensors=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            batch = rows[start : start + 64]
            encoded = tokenizer(
                [str(row["text_a"]) for row in batch],
                [str(row["text_b"]) for row in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs.append(model(**encoded).logits.float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def map_predictions(logits: np.ndarray, labels: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if logits.shape[1] == 1:
        raw = (logits[:, 0] >= 0).astype(np.int64)
        candidates = [(0, 1), (1, 0)]
        mapping = max(candidates, key=lambda item: float(np.mean(np.asarray(item)[raw] == labels)))
        hard = np.asarray(mapping, dtype=np.int64)[raw]
        confidence = 1.0 / (1.0 + np.exp(-np.abs(logits[:, 0])))
        return hard.astype(np.int8), confidence, list(mapping)
    if logits.shape[1] != classes:
        raise AssertionError((logits.shape, classes))
    raw = np.argmax(logits, axis=1)
    mappings = list(itertools.permutations(range(classes)))
    mapping = max(mappings, key=lambda item: float(np.mean(np.asarray(item)[raw] == labels)))
    hard = np.asarray(mapping, dtype=np.int64)[raw]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= np.sum(probability, axis=1, keepdims=True)
    confidence = np.max(probability, axis=1)
    return hard.astype(np.int8), confidence, list(mapping)


def threshold_for_fraction(confidence: np.ndarray, fraction: float) -> float:
    return float(np.quantile(confidence, 1.0 - fraction, method="higher"))


def adapter_codes(hard: np.ndarray, confidence: np.ndarray, threshold: float, classes: int) -> np.ndarray:
    triggered = confidence >= threshold
    columns = []
    for alias in range(4):
        values = hard.astype(np.int64).copy()
        values[triggered] = classes + hard[triggered].astype(np.int64) * 4 + alias
        columns.append(values)
    return np.stack(columns, axis=1)


def evaluate(
    official: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    confidence: np.ndarray,
    classes: int,
    config: dict[str, Any],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    roster = common.top_roster(official, labels, int(config["roster_size"]))
    original = np.column_stack([official[:, roster], hard]).astype(np.int8)
    core_feedback = (original == labels[:, None]).astype(np.float64)
    clean_codes = original.astype(np.int64)
    clean_feedback = core_feedback.copy()
    clean_parents = np.arange(original.shape[1], dtype=np.int64)
    threshold = threshold_for_fraction(confidence, float(config["trigger_fraction"]))
    aliases = adapter_codes(hard, confidence, threshold, classes)
    refined_codes = np.column_stack([clean_codes, aliases])
    refined_feedback = np.column_stack(
        [clean_feedback, np.repeat(core_feedback[:, [-1]], 4, axis=1)]
    )
    parent_root = clean_codes.shape[1] - 1
    refined_parents = np.concatenate([clean_parents, np.full(4, parent_root, dtype=np.int64)])
    pools = common.sample_pools(len(labels), seeds)
    clean = common.run_active(
        clean_codes,
        clean_feedback,
        clean_parents,
        core_feedback,
        pools,
        seeds,
        int(config["budget"]),
        float(config["tau"]),
    )
    refined = common.run_active(
        refined_codes,
        refined_feedback,
        refined_parents,
        core_feedback,
        pools,
        seeds,
        int(config["budget"]),
        float(config["tau"]),
    )
    delta = refined.terminal - clean.terminal
    cumulative = refined.cumulative - clean.cumulative
    return {
        **config,
        "seeds": len(seeds),
        "threshold": threshold,
        "realized_trigger_fraction": float(np.mean(confidence >= threshold)),
        "roster_indices": roster.tolist(),
        "parent_calibration_accuracy": float(np.mean(hard == labels)),
        "best_calibration_accuracy": float(np.max(np.mean(core_feedback, axis=0))),
        "mean_terminal_delta": float(np.mean(delta)),
        "mean_cumulative_delta": float(np.mean(cumulative)),
        "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
        "final_root_change_rate": float(np.mean(clean.roots[:, -1] != refined.roots[:, -1])),
        "terminal_delta_values": delta.tolist() if len(seeds) >= 300 else None,
        "clean_query_sha256": hashlib.sha256(clean.queries.tobytes()).hexdigest(),
        "refined_query_sha256": hashlib.sha256(refined.queries.tobytes()).hexdigest(),
    }


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    config = json.dumps(
        {key: row[key] for key in ("roster_size", "budget", "tau", "trigger_fraction")},
        sort_keys=True,
    )
    return (
        float(row["mean_terminal_delta"]),
        float(row["path_change_rate"]),
        float(row["mean_cumulative_delta"]),
        config,
    )


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    MODEL_DIR.mkdir(exist_ok=True)
    frozen_tasks: dict[str, Any] = {}
    for task, spec in TASKS.items():
        print(f"[{task}] parent inference", flush=True)
        package_path = ROOT / f"STEP90_{task.upper()}_CALIBRATION_PACKAGE_{DATE}.json"
        rows = json.loads(package_path.read_text(encoding="utf-8"))["rows"]
        labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
        official = np.asarray([row["official_predictions"] for row in rows], dtype=np.int8)
        logits = infer_logits(task, rows)
        hard, confidence, mapping = map_predictions(logits, labels, int(spec["classes"]))
        output_path = ROOT / f"STEP90_{task.upper()}_CALIBRATION_PARENT_OUTPUTS_{DATE}.npz"
        np.savez_compressed(
            output_path,
            positions=np.asarray([row["position"] for row in rows], dtype=np.int64),
            logits=logits,
            hard_predictions=hard,
            confidence=confidence,
            labels=labels,
        )

        grid = list(itertools.product(ROSTER_SIZES, BUDGETS, TAUS, TRIGGER_FRACTIONS))
        pilot: list[dict[str, Any]] = []
        for index, (roster_size, budget, tau, fraction) in enumerate(grid, start=1):
            config = {
                "roster_size": roster_size,
                "pool_size": POOL_SIZE,
                "budget": budget,
                "tau": tau,
                "trigger_fraction": fraction,
            }
            pilot.append(evaluate(official, labels, hard, confidence, int(spec["classes"]), config, PILOT_SEEDS))
            if index % 25 == 0 or index == len(grid):
                print(f"[{task}] pilot {index}/{len(grid)}", flush=True)
        ranked = sorted(pilot, key=rank_key, reverse=True)
        verified: list[dict[str, Any]] = []
        for index, row in enumerate(ranked[:TOP_TO_VERIFY], start=1):
            config = {key: row[key] for key in ("roster_size", "pool_size", "budget", "tau", "trigger_fraction")}
            verified.append(evaluate(official, labels, hard, confidence, int(spec["classes"]), config, VERIFY_SEEDS))
            print(f"[{task}] verify {index}/{TOP_TO_VERIFY}", flush=True)
        selected = max(verified, key=rank_key)

        adapter_hashes: dict[str, str] = {}
        for alias in range(4):
            adapter_path = MODEL_DIR / f"step90_{task}_adapter_{alias}.safetensors"
            save_file(
                {
                    "confidence_threshold": torch.tensor([selected["threshold"]], dtype=torch.float64),
                    "style_id": torch.tensor([alias], dtype=torch.int64),
                    "class_count": torch.tensor([int(spec["classes"])], dtype=torch.int64),
                },
                str(adapter_path),
                metadata={
                    "base_model": str(spec["model"]),
                    "base_revision": str(spec["revision"]),
                    "hard_label_invariant": "true",
                    "runtime_inputs": "raw_text_pair_and_parent_confidence",
                },
            )
            adapter_hashes[adapter_path.name] = sha256_path(adapter_path)

        ledger = {
            "ledger_id": f"STEP90_{task.upper()}_STAGEA_DEVELOPMENT_V1",
            "stage0_manifest_sha256": sha256_path(MANIFEST_PATH),
            "calibration_package_sha256": sha256_path(package_path),
            "parent_output_sha256": sha256_path(output_path),
            "model": spec["model"],
            "model_revision": spec["revision"],
            "label_mapping_raw_to_glue": mapping,
            "search_space": {
                "pilot_seeds": [PILOT_SEEDS[0], PILOT_SEEDS[-1]],
                "verification_seeds": [VERIFY_SEEDS[0], VERIFY_SEEDS[-1]],
                "roster_sizes": list(ROSTER_SIZES),
                "pool_size": POOL_SIZE,
                "budgets": list(BUDGETS),
                "taus": list(TAUS),
                "trigger_fractions": list(TRIGGER_FRACTIONS),
                "pilot_configurations": len(pilot),
                "verified_configurations": len(verified),
            },
            "pilot_results": pilot,
            "verification_results": verified,
            "selection_rule": "maximum verified terminal delta, then path change, cumulative delta, serialized config",
            "selected": selected,
        }
        ledger_path = ROOT / f"STEP90_{task.upper()}_STAGEA_SEARCH_LEDGER_{DATE}.json"
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
        frozen_tasks[task] = {
            "model": spec["model"],
            "model_revision": spec["revision"],
            "classes": int(spec["classes"]),
            "label_mapping_raw_to_glue": mapping,
            "selected": {key: selected[key] for key in (
                "roster_size", "pool_size", "budget", "tau", "trigger_fraction", "threshold", "roster_indices"
            )},
            "adapter_sha256": adapter_hashes,
            "ledger_sha256": sha256_path(ledger_path),
            "calibration_parent_accuracy": selected["parent_calibration_accuracy"],
            "calibration_terminal_delta": selected["mean_terminal_delta"],
            "calibration_path_change_rate": selected["path_change_rate"],
        }

    config = {
        "config_id": "STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_CONFIRMATORY_V1",
        "stage0_manifest_sha256": sha256_path(MANIFEST_PATH),
        "tasks": frozen_tasks,
        "construction_access": {
            "calibration_labels_and_peer_rows_for_condition_selection": True,
            "runtime_parent_confidence_only": True,
            "holdout_labels_peer_rows_or_trajectories": False,
            "item_lookup_table": False,
        },
    }
    config_path = ROOT / f"STEP90_HIGH_CONFIDENCE_STAGEA_FROZEN_CONFIG_{DATE}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "config_sha256": sha256_path(config_path),
        "selected": {
            task: {
                "condition": row["selected"],
                "calibration_terminal_delta": row["calibration_terminal_delta"],
                "calibration_path_change_rate": row["calibration_path_change_rate"],
            }
            for task, row in frozen_tasks.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
