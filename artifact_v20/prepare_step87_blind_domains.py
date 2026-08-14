"""Download the frozen Step 87 domains and create separated calibration/holdout packages."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from step87_terminal_common import (
    PREREG,
    ROOT,
    STAGE0_LOCK,
    TASKS,
    digest_json,
    file_sha256,
    load_preregistration,
    safe_name,
    split_indices,
)


RAW_ROOT = ROOT / "external_data" / "step87_raw"
BLIND_ROOT = ROOT / "external_data" / "step87_blind_parent_inputs"
SEALED_ROOT = ROOT / "external_data" / "step87_sealed"
CALIBRATION_OUT = ROOT / "STEP87_LABELED_CALIBRATION_PACKAGE_2026-08-12.json"
BLIND_MANIFEST_OUT = ROOT / "STEP87_BLIND_PARENT_INPUT_MANIFEST_2026-08-12.json"
SEALED_OUT = SEALED_ROOT / "STEP87_SEALED_HOLDOUT_OUTCOMES_2026-08-12.json"
RAW_MANIFEST_OUT = ROOT / "STEP87_RAW_SOURCE_MANIFEST_2026-08-12.json"
SPLIT_AUDIT_OUT = ROOT / "STEP87_SPLIT_AUDIT_2026-08-12.json"
RUNNER = Path(__file__).resolve()


def verify_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = load_preregistration()
    lock = json.loads(STAGE0_LOCK.read_text(encoding="utf-8"))
    assert lock["manifest_id"] == "STEP87_STAGE0_EXECUTION_LOCK_V1"
    assert lock["new_domain_contents_seen_before_lock"] is False
    assert lock["preregistration_sha256"] == file_sha256(PREREG)
    assert lock["common_module_sha256"] == file_sha256(
        ROOT / "step87_terminal_common.py"
    )
    assert lock["preparation_runner_sha256"] == file_sha256(RUNNER)
    return prereg, lock


def fetch_json(base: str, relative: str, path: Path) -> tuple[Any, dict[str, Any]]:
    url = base + urllib.parse.quote(relative, safe="/:,@=-._")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = path.read_bytes()
        source = "cache"
    else:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        path.write_bytes(payload)
        source = "download"
    return json.loads(payload), {
        "url": url,
        "relative_cache_path": path.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(path),
        "bytes": len(payload),
        "source": source,
    }


def load_scenario(
    *, base: str, task: str, scenario_index: int, scenario: str, models: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenario_dir = RAW_ROOT / task / f"scenario_{scenario_index:02d}"

    def load_model(model: str):
        relative = f"{scenario},model={model}/display_predictions.json"
        path = scenario_dir / f"{safe_name(model)}__display_predictions.json"
        rows, provenance = fetch_json(base, relative, path)
        return model, rows, provenance

    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(load_model, model) for model in models]
        for future in as_completed(futures):
            model, rows, record = future.result()
            rows_by_model[model] = rows
            provenance.append(
                {"kind": "display_predictions", "model": model, **record}
            )

    instance_model = "openai_gpt-4-0613"
    relative = f"{scenario},model={instance_model}/instances.json"
    path = scenario_dir / f"{safe_name(instance_model)}__instances.json"
    instance_rows, record = fetch_json(base, relative, path)
    provenance.append({"kind": "instances", "model": instance_model, **record})

    ids = [str(row["instance_id"]) for row in rows_by_model[models[0]]]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{scenario}: duplicate prediction IDs")
    keyed = {
        model: {str(row["instance_id"]): row for row in rows_by_model[model]}
        for model in models
    }
    if any(set(keyed[model]) != set(ids) for model in models):
        raise AssertionError(f"{scenario}: candidate ID mismatch")
    instances = {str(row["id"]): row for row in instance_rows}
    if set(instances) != set(ids):
        raise AssertionError(f"{scenario}: instance/prediction ID mismatch")

    prompts: list[str] = []
    options: list[list[str]] = []
    correct: list[str] = []
    for instance_id in ids:
        item = instances[instance_id]
        prompts.append(str(item["input"]["text"]).strip())
        refs = item["references"]
        values = [str(ref["output"]["text"]).strip() for ref in refs]
        tagged = [
            str(ref["output"]["text"]).strip()
            for ref in refs
            if "correct" in ref.get("tags", [])
        ]
        if len(tagged) != 1 or len(values) < 2 or len(set(values)) != len(values):
            raise AssertionError(f"{scenario}:{instance_id}: malformed options")
        options.append(values)
        correct.append(tagged[0])

    responses: list[list[str]] = []
    for model in models:
        values: list[str] = []
        for instance_id in ids:
            row = keyed[model][instance_id]
            mapped = row.get("mapped_output")
            if mapped is None:
                mapped = "INVALID:" + str(row.get("predicted_text", ""))
            values.append(str(mapped).strip())
        responses.append(values)
    response_array = np.asarray(responses, dtype=object)
    correct_array = np.asarray(correct, dtype=object)
    oracle = (response_array == correct_array[None, :]).astype(np.float64)
    composite_ids = [
        json.dumps([scenario, value], ensure_ascii=False, separators=(",", ":"))
        for value in ids
    ]
    return {
        "ids": composite_ids,
        "prompts": prompts,
        "public_options": options,
        "correct": correct,
        "responses": response_array.tolist(),
        "oracle": oracle.tolist(),
        "scenario": scenario,
    }, provenance


def concatenate(parts: list[dict[str, Any]], model_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ids": [],
        "prompts": [],
        "public_options": [],
        "correct": [],
        "responses": [[] for _ in range(model_count)],
        "oracle": [[] for _ in range(model_count)],
    }
    for part in parts:
        for key in ("ids", "prompts", "public_options", "correct"):
            result[key].extend(part[key])
        for model in range(model_count):
            result["responses"][model].extend(part["responses"][model])
            result["oracle"][model].extend(part["oracle"][model])
    if len(result["ids"]) != len(set(result["ids"])):
        raise AssertionError("aggregate IDs are not unique")
    return result


def select(values: list[Any], indices: list[int]) -> list[Any]:
    return [values[index] for index in indices]


def main() -> None:
    prereg, lock = verify_lock()
    base = prereg["public_source"]["base_url"]
    models = list(prereg["public_source"]["candidate_models"])
    calibration: dict[str, Any] = {
        "manifest_id": "STEP87_LABELED_CALIBRATION_PACKAGE_V1",
        "preregistration_sha256": file_sha256(PREREG),
        "models": models,
        "tasks": {},
    }
    sealed: dict[str, Any] = {
        "manifest_id": "STEP87_SEALED_HOLDOUT_OUTCOMES_V1",
        "stage_a_must_not_read": True,
        "preregistration_sha256": file_sha256(PREREG),
        "models": models,
        "tasks": {},
    }
    blind_manifest: dict[str, Any] = {
        "manifest_id": "STEP87_BLIND_PARENT_INPUT_MANIFEST_V1",
        "preregistration_sha256": file_sha256(PREREG),
        "tasks": {},
        "forbidden_fields_absent": True,
    }
    split_audit: dict[str, Any] = {
        "manifest_id": "STEP87_SPLIT_AUDIT_V1",
        "preregistration_sha256": file_sha256(PREREG),
        "tasks": {},
    }
    provenance: list[dict[str, Any]] = []

    for task in TASKS:
        print(f"[{task}] downloading frozen subjects", flush=True)
        parts = []
        for scenario_index, scenario in enumerate(
            prereg["frozen_domain_blocks"][task]
        ):
            part, rows = load_scenario(
                base=base,
                task=task,
                scenario_index=scenario_index,
                scenario=scenario,
                models=models,
            )
            parts.append(part)
            provenance.extend(
                {"task": task, "scenario": scenario, **row} for row in rows
            )
        data = concatenate(parts, len(models))
        calibration_indices, holdout_indices = split_indices(task, data["ids"])
        if min(len(calibration_indices), len(holdout_indices)) < 100:
            raise AssertionError(f"{task}: split below 100 examples")
        if set(calibration_indices) & set(holdout_indices):
            raise AssertionError(f"{task}: split overlap")

        calibration["tasks"][task] = {
            "ids": select(data["ids"], calibration_indices),
            "prompts": select(data["prompts"], calibration_indices),
            "public_options": select(data["public_options"], calibration_indices),
            "correct": select(data["correct"], calibration_indices),
            "registry_responses": [
                select(row, calibration_indices) for row in data["responses"]
            ],
            "registry_oracle": [
                select(row, calibration_indices) for row in data["oracle"]
            ],
        }
        sealed["tasks"][task] = {
            "ids": select(data["ids"], holdout_indices),
            "correct": select(data["correct"], holdout_indices),
            "registry_responses": [
                select(row, holdout_indices) for row in data["responses"]
            ],
            "registry_oracle": [
                select(row, holdout_indices) for row in data["oracle"]
            ],
        }
        task_blind: dict[str, Any] = {"models": {}}
        for model_index, model in enumerate(models):
            payload = {
                "manifest_id": "STEP87_SINGLE_PARENT_BLIND_INPUT_V1",
                "task": task,
                "parent_index": model_index,
                "parent": model,
                "ids": select(data["ids"], holdout_indices),
                "prompts": select(data["prompts"], holdout_indices),
                "public_options": select(data["public_options"], holdout_indices),
                "parent_outputs": select(
                    data["responses"][model_index], holdout_indices
                ),
            }
            rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            lowered = rendered.casefold()
            for forbidden in (
                '"correct"',
                '"oracle"',
                '"peer"',
                '"registry_responses"',
                '"pool"',
                '"selector_state"',
            ):
                if forbidden in lowered:
                    raise AssertionError(f"forbidden field {forbidden}")
            path = BLIND_ROOT / task / f"{safe_name(model)}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
            task_blind["models"][model] = {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        blind_manifest["tasks"][task] = task_blind
        split_audit["tasks"][task] = {
            "total": len(data["ids"]),
            "calibration": len(calibration_indices),
            "holdout": len(holdout_indices),
            "id_overlap": 0,
            "calibration_id_digest": digest_json(
                select(data["ids"], calibration_indices)
            ),
            "holdout_id_digest": digest_json(select(data["ids"], holdout_indices)),
            "subjects": prereg["frozen_domain_blocks"][task],
        }
        print(
            f"[{task}] total={len(data['ids'])} cal={len(calibration_indices)} "
            f"holdout={len(holdout_indices)}",
            flush=True,
        )

    CALIBRATION_OUT.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    SEALED_ROOT.mkdir(parents=True, exist_ok=True)
    SEALED_OUT.write_text(
        json.dumps(sealed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    BLIND_MANIFEST_OUT.write_text(
        json.dumps(blind_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    SPLIT_AUDIT_OUT.write_text(
        json.dumps(split_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    raw_manifest = {
        "manifest_id": "STEP87_RAW_SOURCE_MANIFEST_V1",
        "preregistration_sha256": file_sha256(PREREG),
        "stage0_lock_sha256": file_sha256(STAGE0_LOCK),
        "records": sorted(
            provenance,
            key=lambda row: (row["task"], row["scenario"], row["kind"], row["model"]),
        ),
        "counts": {
            "records": len(provenance),
            "downloaded": sum(row["source"] == "download" for row in provenance),
            "cached": sum(row["source"] == "cache" for row in provenance),
        },
        "output_sha256": {
            "calibration": file_sha256(CALIBRATION_OUT),
            "blind_manifest": file_sha256(BLIND_MANIFEST_OUT),
            "sealed_outcomes": file_sha256(SEALED_OUT),
            "split_audit": file_sha256(SPLIT_AUDIT_OUT),
        },
    }
    RAW_MANIFEST_OUT.write_text(
        json.dumps(raw_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("STEP87_PREPARATION_COMPLETE_WITH_SEALED_OUTCOMES", flush=True)


if __name__ == "__main__":
    main()
