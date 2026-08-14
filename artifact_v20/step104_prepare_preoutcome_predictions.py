from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import step102_common as common


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / f"STEP104_IMDB_HELDOUT_LEARNED_SAME_S_PRIMARY_PROTOCOL_{common.DATE}.md"
STAGE0 = ROOT / f"STEP100_STAGE0_DATA_AND_SEAL_MANIFEST_{common.DATE}.json"
CONFIG = ROOT / f"STEP103_ROBUST_HELDOUT_FROZEN_CONFIG_{common.DATE}.json"
STEP103 = ROOT / f"STEP103_ROBUST_DUAL_DEVELOPMENT_LEDGER_{common.DATE}.json"
OUTPUT = ROOT / f"STEP104_PREOUTCOME_PREDICTIONS_{common.DATE}.npz"
LEDGER = ROOT / f"STEP104_PREOUTCOME_PREDICTION_LEDGER_{common.DATE}.json"


def main() -> None:
    if OUTPUT.exists() or LEDGER.exists():
        raise FileExistsError("Step 104 pre-outcome output already exists")
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    step103 = json.loads(STEP103.read_text(encoding="utf-8"))
    if step103["decision"] != "GO_STEP103_TO_HELDOUT_LOCK":
        raise AssertionError("development did not authorize held-out preparation")
    if config["thresholds"] != [
        0.9515677094459534, 0.9398181438446045,
        0.9488589763641357, 0.9394304752349854,
    ]:
        raise AssertionError("selected threshold drift")
    input_path = ROOT / stage0["input_file"]
    sealed_path = ROOT / stage0["sealed_file"]
    if common.sha256_path(input_path) != stage0["input_sha256"]:
        raise AssertionError("input binding drift")
    if common.sha256_path(sealed_path) != stage0["sealed_sha256"]:
        raise AssertionError("sealed byte binding drift")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = payload["rows"]
    if payload.get("labels_present") is not False:
        raise AssertionError("input label-free marker drift")
    if any("label" in row or "reference" in row for row in rows):
        raise AssertionError("held-out outcome field present in input")
    texts = [str(row["text"]) for row in rows]
    source_indices = np.asarray([int(row["source_index"]) for row in rows], dtype=np.int64)
    uid_sha256 = np.asarray([str(row["uid_sha256"]) for row in rows], dtype="U64")

    root_columns: list[np.ndarray] = []
    root_audits: list[dict[str, Any]] = []
    for index, spec in enumerate(common.ROOT_SPECS):
        prediction, audit = common.infer_root_predictions(spec, texts)
        root_columns.append(prediction)
        root_audits.append(audit)
        print(f"pre-outcome root {index + 1}/{common.ROSTER_SIZE}: {spec.key}", flush=True)
    root_predictions = np.stack(root_columns, axis=1).astype(np.int64)

    score_columns: list[np.ndarray] = []
    adapter_audits: list[dict[str, Any]] = []
    for alias, relative in enumerate(config["adapter_paths"]):
        path = ROOT / relative
        expected = config["adapter_sha256"][relative]
        if common.sha256_path(path) != expected:
            raise AssertionError({"adapter_hash_drift": relative})
        model, tokenizer = common.load_adapter_model(path)
        scores = common.infer_error_scores(model, tokenizer, texts)
        score_columns.append(scores)
        trainable_count = int(sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ))
        adapter_audits.append({
            "alias": alias,
            "path": relative,
            "sha256": expected,
            "trainable_parameter_count": trainable_count,
            "error_score_sha256": common.array_sha256(scores),
        })
        model.cpu()
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"pre-outcome adapter {alias + 1}/{common.N_ALIASES}", flush=True)
    error_scores = np.stack(score_columns, axis=1).astype(np.float64)
    aliases, triggers = common.make_alias_codes(
        root_predictions[:, 0], error_scores, config["thresholds"]
    )
    np.savez_compressed(
        OUTPUT,
        source_indices=source_indices,
        uid_sha256=uid_sha256,
        root_predictions=root_predictions.astype(np.int16),
        error_scores=error_scores,
        alias_predictions=aliases.astype(np.int16),
        triggers=triggers.astype(np.uint8),
        thresholds=np.asarray(config["thresholds"], dtype=np.float64),
    )
    ledger = {
        "ledger_id": "STEP104_PREOUTCOME_PREDICTIONS_V1",
        "date": common.DATE,
        "rows": len(rows),
        "protocol_sha256": common.sha256_path(PROTOCOL),
        "stage0_sha256": common.sha256_path(STAGE0),
        "step103_heldout_config_sha256": common.sha256_path(CONFIG),
        "step103_ledger_sha256": common.sha256_path(STEP103),
        "input_file": stage0["input_file"],
        "input_sha256": stage0["input_sha256"],
        "sealed_file": stage0["sealed_file"],
        "sealed_sha256_verified_without_loading_arrays": stage0["sealed_sha256"],
        "sealed_outcome_loaded": False,
        "root_audits": root_audits,
        "adapter_audits": adapter_audits,
        "thresholds": config["thresholds"],
        "prediction_file": OUTPUT.name,
        "prediction_sha256": common.sha256_path(OUTPUT),
        "array_sha256": {
            "source_indices": common.array_sha256(source_indices),
            "uid_sha256": common.array_sha256(uid_sha256),
            "root_predictions": common.array_sha256(root_predictions),
            "error_scores": common.array_sha256(error_scores),
            "alias_predictions": common.array_sha256(aliases),
            "triggers": common.array_sha256(triggers.astype(np.uint8)),
        },
        "code_sha256": {
            "step93_common.py": common.sha256_path(ROOT / "step93_common.py"),
            "step100_common.py": common.sha256_path(ROOT / "step100_common.py"),
            "step102_common.py": common.sha256_path(ROOT / "step102_common.py"),
            "step104_prepare_preoutcome_predictions.py": common.sha256_path(Path(__file__)),
        },
    }
    common.json_dump(LEDGER, ledger)
    print(json.dumps({
        "rows": len(rows),
        "prediction_file": OUTPUT.name,
        "prediction_sha256": ledger["prediction_sha256"],
        "trigger_counts_without_outcomes": np.sum(triggers, axis=0).tolist(),
        "sealed_outcome_loaded": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
