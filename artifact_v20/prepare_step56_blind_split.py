"""Prepare separated calibration and blind-holdout packages for Step 56."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_step54_matrix_free_realism_audit as step54
from step56_reference_free_common import (
    TASKS,
    digest_json,
    file_sha256,
    raw_helm_manifest,
    split_indices,
)


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "STEP56_CALIBRATION_TO_HOLDOUT_PREREGISTRATION_2026-08-09.json"
STAGEA_LOCK = ROOT / "STEP56_CALIBRATION_STAGEA_EXECUTION_LOCK_2026-08-09.json"
CALIBRATION_OUT = ROOT / "STEP56_LABELED_CALIBRATION_PACKAGE_2026-08-09.json"
BLIND_OUT = ROOT / "STEP56_BLIND_HOLDOUT_ATTACK_INPUT_2026-08-09.json"
SPLIT_OUT = ROOT / "STEP56_SPLIT_AUDIT_2026-08-09.json"
EXPECTED_PREREG_SHA256 = "909626b48342171b82f91d6f61a51b2a1f2d3db0df1a2a75e4611d12e3842896"

INSTANCE_FILES = {
    "medqa": ROOT
    / "external_data/helm_lite_medqa_v1_0_0_raw/openai_gpt-4-0613__instances.json",
    "gsm8k": ROOT
    / "external_data/helm_lite_step25_raw/gsm8k__openai_gpt-4-0613__instances.json",
    "openbookqa": ROOT
    / "external_data/helm_lite_step25_raw/openbookqa__openai_gpt-4-0613__instances.json",
}


def select(values: Any, indices: list[int]) -> list[Any]:
    return [values[index] for index in indices]


def main() -> None:
    assert file_sha256(PREREG) == EXPECTED_PREREG_SHA256
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    lock = json.loads(STAGEA_LOCK.read_text(encoding="utf-8"))
    assert lock["manifest_id"] == "STEP56_CALIBRATION_STAGEA_EXECUTION_LOCK_V1"
    assert lock["outcomes_seen_before_lock"] is False
    assert lock["preregistration_sha256"] == EXPECTED_PREREG_SHA256
    assert lock["preparation_runner_sha256"] == file_sha256(Path(__file__).resolve())
    records, raw_digest, raw_bytes = raw_helm_manifest(ROOT)
    assert len(records) == prereg["locked_sources"]["raw_helm_file_count"]
    assert raw_digest == prereg["locked_sources"]["raw_helm_file_manifest_sha256"]
    assert raw_bytes == prereg["locked_sources"]["raw_helm_total_bytes"]

    step54.step46.configure_legacy_paths()
    all_data = step54.step46.phase_a.load_all_data()
    calibration_payload: dict[str, Any] = {
        "manifest_id": "STEP56_LABELED_CALIBRATION_PACKAGE_V1",
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "raw_helm_file_manifest_sha256": raw_digest,
        "tasks": {},
    }
    blind_payload: dict[str, Any] = {
        "manifest_id": "STEP56_BLIND_HOLDOUT_ATTACK_INPUT_V1",
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "explicitly_absent": [
            "correct",
            "oracle",
            "peer_responses",
            "registry_responses",
            "pool_seed",
            "selector_state",
        ],
        "tasks": {},
    }
    split_payload: dict[str, Any] = {
        "manifest_id": "STEP56_SPLIT_AUDIT_V1",
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "raw_helm_file_manifest_sha256": raw_digest,
        "tasks": {},
    }

    for task in TASKS:
        data = all_data[task]
        spec = prereg["frozen_tasks"][task]
        target = int(spec["target_index"])
        instances = json.loads(INSTANCE_FILES[task].read_text(encoding="utf-8"))
        assert [str(item["id"]) for item in instances] == [
            str(value) for value in data["ids"]
        ]
        prompts = [str(item["input"]["text"]) for item in instances]
        options = (
            [[str(value) for value in row] for row in data["public_options"]]
            if data["public_options"] is not None
            else [[] for _ in data["ids"]]
        )
        calibration_index, holdout_index = split_indices(task, list(data["ids"]))
        assert set(calibration_index).isdisjoint(holdout_index)
        assert sorted(calibration_index + holdout_index) == list(range(len(data["ids"])))
        assert len(calibration_index) == prereg["split"]["calibration_examples"][task]
        assert len(holdout_index) == prereg["split"]["holdout_examples"][task]

        calibration_payload["tasks"][task] = {
            "target_index": target,
            "target": spec["target"],
            "ids": select(list(data["ids"]), calibration_index),
            "prompts": select(prompts, calibration_index),
            "public_options": select(options, calibration_index),
            "parent_responses": select(
                [str(value) for value in data["responses"][target]], calibration_index
            ),
            "parent_oracle": select(
                [float(value) for value in data["oracle"][target]], calibration_index
            ),
            "correct": select([str(value) for value in data["correct"]], calibration_index),
            "registry_responses": [
                select([str(value) for value in row], calibration_index)
                for row in data["responses"]
            ],
            "registry_oracle": [
                select([float(value) for value in row], calibration_index)
                for row in data["oracle"]
            ],
            "source_indices_digest": digest_json(calibration_index),
        }
        blind_payload["tasks"][task] = {
            "target_index": target,
            "target": spec["target"],
            "ids": select(list(data["ids"]), holdout_index),
            "prompts": select(prompts, holdout_index),
            "public_options": select(options, holdout_index),
            "parent_responses": select(
                [str(value) for value in data["responses"][target]], holdout_index
            ),
            "source_indices_digest": digest_json(holdout_index),
        }
        split_payload["tasks"][task] = {
            "examples": len(data["ids"]),
            "calibration_count": len(calibration_index),
            "holdout_count": len(holdout_index),
            "calibration_index_digest": digest_json(calibration_index),
            "holdout_index_digest": digest_json(holdout_index),
            "calibration_id_digest": digest_json(
                select([str(value) for value in data["ids"]], calibration_index)
            ),
            "holdout_id_digest": digest_json(
                select([str(value) for value in data["ids"]], holdout_index)
            ),
            "id_intersection_count": 0,
        }

    CALIBRATION_OUT.write_text(
        json.dumps(calibration_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    BLIND_OUT.write_text(
        json.dumps(blind_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    SPLIT_OUT.write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "calibration_sha256": file_sha256(CALIBRATION_OUT),
                "blind_holdout_sha256": file_sha256(BLIND_OUT),
                "split_audit_sha256": file_sha256(SPLIT_OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
