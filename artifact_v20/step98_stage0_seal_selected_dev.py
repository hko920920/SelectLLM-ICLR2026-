from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

import step98_common as common


ROOT = Path(__file__).resolve().parent
SELECTION = ROOT / f"STEP98_TRAIN_ONLY_ROUND_SELECTION_{common.DATE}.json"
INPUT_DIR = ROOT / "step98_test_inputs"
SEALED_DIR = ROOT / "external_data" / "step98_sealed"
MANIFEST = ROOT / f"STEP98_SELECTED_DEV_SEAL_MANIFEST_{common.DATE}.json"


def uid_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    if MANIFEST.exists() or INPUT_DIR.exists() or SEALED_DIR.exists():
        raise FileExistsError("Step 98 selected-dev seal output already exists")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection["decision"] != "GO_STEP98_SELECTED_ROUND_TO_LEARNED_DEVELOPMENT":
        raise AssertionError("round selection did not authorize one selected dev")
    selected_round = str(selection["selected_round"])
    if selected_round not in common.ROUNDS:
        raise AssertionError(selected_round)
    dataset = load_dataset(
        common.DATASET_ID,
        common.DATASET_CONFIG,
        split=f"dev_{selected_round}",
        revision=common.DATASET_REVISION,
    )
    inputs = []
    source_indices = []
    labels = []
    uid_hashes = []
    for index, raw in enumerate(dataset):
        label = int(raw["label"])
        if label not in (0, 1, 2):
            raise AssertionError({"source_index": index, "label_out_of_range": True})
        uid = str(raw["uid"])
        inputs.append({
            "source_index": int(index),
            "uid_sha256": uid_digest(uid),
            "premise": str(raw["premise"]),
            "hypothesis": str(raw["hypothesis"]),
        })
        source_indices.append(index)
        labels.append(label)
        uid_hashes.append(uid_digest(uid))
    INPUT_DIR.mkdir(parents=True)
    SEALED_DIR.mkdir(parents=True)
    input_path = INPUT_DIR / f"anli_{selected_round}_dev_inputs.json"
    sealed_path = SEALED_DIR / f"anli_{selected_round}_dev_sealed_outcomes.npz"
    common.json_dump(input_path, {
        "dataset_id": common.DATASET_ID,
        "revision": common.DATASET_REVISION,
        "split": f"dev_{selected_round}",
        "selected_by_train_only_rule": True,
        "labels_present": False,
        "rows": inputs,
    })
    np.savez_compressed(
        sealed_path,
        source_indices=np.asarray(source_indices, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int16),
        uid_sha256=np.asarray(uid_hashes, dtype="U64"),
    )
    manifest = {
        "manifest_id": "STEP98_SELECTED_DEV_SEAL_V1",
        "date": common.DATE,
        "selected_round": selected_round,
        "selected_split": f"dev_{selected_round}",
        "selection_sha256": common.sha256_path(SELECTION),
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "rows": len(inputs),
        "input_file": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": common.sha256_path(input_path),
        "input_contains_label_field": False,
        "sealed_file": str(sealed_path.relative_to(ROOT)).replace("\\", "/"),
        "sealed_sha256": common.sha256_path(sealed_path),
        "sealed_outcome_opened_for_analysis": False,
        "console_exposed_labels_or_label_counts": False,
        "code_sha256": {
            "step98_common.py": common.sha256_path(ROOT / "step98_common.py"),
            "step98_stage0_seal_selected_dev.py": common.sha256_path(Path(__file__)),
        },
    }
    common.json_dump(MANIFEST, manifest)
    print(json.dumps({
        "selected_round": selected_round,
        "rows": len(inputs),
        "input_file": manifest["input_file"],
        "input_sha256": manifest["input_sha256"],
        "sealed_file": manifest["sealed_file"],
        "sealed_sha256": manifest["sealed_sha256"],
        "sealed_outcome_opened_for_analysis": False,
    }, indent=2))


if __name__ == "__main__":
    main()
