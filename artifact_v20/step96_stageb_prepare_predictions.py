from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import step96_common as common


ROOT = Path(__file__).resolve().parent
LOCK = ROOT / f"STEP96_CONFIRMATORY_EXECUTION_LOCK_{common.DATE}.json"
CONFIG = ROOT / f"STEP96_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
TEST_INPUT = ROOT / "step96_test_inputs" / "snli_test_inputs.json"
OUTPUT = ROOT / f"STEP96_CONFIRMATORY_PREOUTCOME_PREDICTIONS_{common.DATE}.npz"
LEDGER = ROOT / f"STEP96_CONFIRMATORY_PREOUTCOME_LEDGER_{common.DATE}.json"


def main() -> None:
    for output in (OUTPUT, LEDGER):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite {output}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if lock["sealed_outcome_opened_at_lock"]:
        raise AssertionError("invalid execution lock")
    if lock["code_sha256"][Path(__file__).name] != common.sha256_path(Path(__file__)):
        raise AssertionError("preoutcome code drift")
    if lock["test_input"]["sha256"] != common.sha256_path(TEST_INPUT):
        raise AssertionError("test input drift")
    payload = json.loads(TEST_INPUT.read_text(encoding="utf-8"))
    premises = [str(row["premise"]) for row in payload["rows"]]
    hypotheses = [str(row["hypothesis"]) for row in payload["rows"]]
    source_indices = np.asarray([row["source_index"] for row in payload["rows"]], dtype=np.int64)
    root_columns: list[np.ndarray] = []
    root_audits = []
    for index, spec in enumerate(common.ROOT_SPECS):
        predictions, audit = common.infer_root_predictions(spec, premises, hypotheses)
        root_columns.append(predictions)
        root_audits.append(audit)
        print(f"test preoutcome root {index + 1}/{common.ROSTER_SIZE}: {spec.key}", flush=True)
    root_predictions = np.stack(root_columns, axis=1)
    parent = common.ROOT_SPECS[0]
    score_columns: list[np.ndarray] = []
    adapter_audits = []
    for alias, relative in enumerate(config["adapter_paths"]):
        path = ROOT / relative
        expected = lock["adapter_sha256"][relative]
        if common.sha256_path(path) != expected:
            raise AssertionError("adapter drift")
        model, tokenizer = common.load_adapter_model(parent, path)
        scores = common.infer_error_scores(model, tokenizer, premises, hypotheses)
        score_columns.append(scores)
        adapter_audits.append(
            {"alias": alias, "path": relative, "sha256": expected, "score_sha256": common.array_sha256(scores)}
        )
        model.cpu()
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"test preoutcome adapter {alias + 1}/{common.N_ALIASES}", flush=True)
    error_scores = np.stack(score_columns, axis=1)
    np.savez_compressed(
        OUTPUT,
        source_indices=source_indices,
        root_predictions=root_predictions.astype(np.int16),
        error_scores=error_scores.astype(np.float64),
    )
    ledger = {
        "ledger_id": "STEP96_CONFIRMATORY_PREOUTCOME_PREDICTIONS_V1",
        "date": common.DATE,
        "lock_sha256": common.sha256_path(LOCK),
        "test_input_sha256": common.sha256_path(TEST_INPUT),
        "rows": len(source_indices),
        "root_audits": root_audits,
        "adapter_audits": adapter_audits,
        "output_file": OUTPUT.name,
        "output_sha256": common.sha256_path(OUTPUT),
        "contains_reference_labels": False,
        "sealed_outcome_opened": False,
        "code_sha256": common.sha256_path(Path(__file__)),
    }
    common.json_dump(LEDGER, ledger)
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
