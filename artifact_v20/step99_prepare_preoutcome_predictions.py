from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import step98_common as common


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / f"STEP99_ANLI_DEVELOPMENT_INFORMED_SEALED_PRIMARY_PROTOCOL_{common.DATE}.md"
CONFIG = ROOT / f"STEP98_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
SEAL = ROOT / f"STEP98_SELECTED_DEV_SEAL_MANIFEST_{common.DATE}.json"
OUTPUT = ROOT / f"STEP99_PREOUTCOME_PREDICTIONS_{common.DATE}.npz"
LEDGER = ROOT / f"STEP99_PREOUTCOME_PREDICTION_LEDGER_{common.DATE}.json"


def main() -> None:
    if OUTPUT.exists() or LEDGER.exists():
        raise FileExistsError("Step 99 pre-outcome output already exists")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    input_path = ROOT / seal["input_file"]
    if common.sha256_path(input_path) != seal["input_sha256"]:
        raise AssertionError("label-free input binding drift")
    if common.sha256_path(ROOT / seal["sealed_file"]) != seal["sealed_sha256"]:
        raise AssertionError("sealed byte binding drift")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = payload["rows"]
    if payload.get("labels_present") is not False:
        raise AssertionError("label-free input marker drift")
    if any("label" in row or "reference" in row for row in rows):
        raise AssertionError("held-out outcome field present in prediction input")
    premises = [str(row["premise"]) for row in rows]
    hypotheses = [str(row["hypothesis"]) for row in rows]
    source_indices = np.asarray([int(row["source_index"]) for row in rows], dtype=np.int64)
    uid_sha256 = np.asarray([str(row["uid_sha256"]) for row in rows], dtype="U64")

    root_columns: list[np.ndarray] = []
    root_audits: list[dict[str, Any]] = []
    for index, spec in enumerate(common.ROOT_SPECS):
        prediction, audit = common.infer_root_predictions(
            common.as_learned_spec(spec), premises, hypotheses, batch_size=64
        )
        root_columns.append(prediction)
        root_audits.append(audit)
        print(f"pre-outcome root {index + 1}/{common.ROSTER_SIZE}: {spec.key}", flush=True)
    root_predictions = np.stack(root_columns, axis=1).astype(np.int64)

    score_columns: list[np.ndarray] = []
    adapter_audits: list[dict[str, Any]] = []
    for index, relative in enumerate(config["adapter_paths"]):
        path = ROOT / relative
        expected = config["adapter_sha256"][relative]
        if common.sha256_path(path) != expected:
            raise AssertionError({"adapter_hash_drift": relative})
        model, tokenizer = common.load_adapter_model(
            common.as_learned_spec(common.ROOT_SPECS[0]), path
        )
        scores = common.infer_error_scores(model, tokenizer, premises, hypotheses)
        score_columns.append(scores)
        adapter_audits.append({
            "alias": index,
            "path": relative,
            "sha256": expected,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "error_score_sha256": common.array_sha256(scores),
        })
        model.cpu()
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"pre-outcome adapter {index + 1}/{common.N_ALIASES}", flush=True)
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
        "ledger_id": "STEP99_PREOUTCOME_PREDICTIONS_V1",
        "date": common.DATE,
        "rows": len(rows),
        "protocol_sha256": common.sha256_path(PROTOCOL),
        "step98_config_sha256": common.sha256_path(CONFIG),
        "input_file": seal["input_file"],
        "input_sha256": seal["input_sha256"],
        "sealed_file": seal["sealed_file"],
        "sealed_sha256_verified_without_loading_arrays": seal["sealed_sha256"],
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
            "step95_common.py": common.sha256_path(ROOT / "step95_common.py"),
            "step96_common.py": common.sha256_path(ROOT / "step96_common.py"),
            "step98_common.py": common.sha256_path(ROOT / "step98_common.py"),
            "step99_prepare_preoutcome_predictions.py": common.sha256_path(Path(__file__)),
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
