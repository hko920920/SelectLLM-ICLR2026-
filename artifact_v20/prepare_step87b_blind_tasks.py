"""Prepare separated calibration, blind-parent, and sealed Step 87B packages."""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from step87b_openended_common import (
    METRIC_KEYS,
    PREREG,
    ROOT,
    STAGE0_LOCK,
    TASKS,
    digest_json,
    file_sha256,
    load_preregistration,
    safe_name,
    scenario_run_prefix,
    split_indices,
    validate_tokens_against_references,
)


RUNNER = Path(__file__).resolve()
RAW_ROOT = ROOT / "external_data" / "step87b_raw"
BLIND_ROOT = ROOT / "external_data" / "step87b_blind_parent_inputs"
SEALED_ROOT = ROOT / "external_data" / "step87b_sealed"
CALIBRATION_OUT = ROOT / "STEP87B_LABELED_CALIBRATION_PACKAGE_2026-08-12.json"
BLIND_MANIFEST_OUT = ROOT / "STEP87B_BLIND_PARENT_INPUT_MANIFEST_2026-08-12.json"
SEALED_OUT = SEALED_ROOT / "STEP87B_SEALED_HOLDOUT_OUTCOMES_2026-08-12.json"
RAW_MANIFEST_OUT = ROOT / "STEP87B_RAW_SOURCE_MANIFEST_2026-08-12.json"
SPLIT_AUDIT_OUT = ROOT / "STEP87B_SPLIT_AND_SCHEMA_AUDIT_2026-08-12.json"


def verify_lock() -> dict[str, Any]:
    prereg = load_preregistration()
    lock = json.loads(STAGE0_LOCK.read_text(encoding="utf-8"))
    if lock.get("manifest_id") != "STEP87B_STAGE0_EXECUTION_LOCK_V1":
        raise AssertionError("unexpected Step 87B Stage-0 lock")
    if lock.get("new_task_contents_seen_before_lock") is not False:
        raise AssertionError("Stage-0 lock does not certify content blindness")
    expected = {
        "preregistration": file_sha256(PREREG),
        "common_module": file_sha256(ROOT / "step87b_openended_common.py"),
        "preparation_runner": file_sha256(RUNNER),
    }
    if lock.get("sha256") != expected:
        raise AssertionError({"locked": lock.get("sha256"), "observed": expected})
    return prereg


def fetch_json(base: str, relative: str, path: Path) -> tuple[Any, dict[str, Any]]:
    url = base + urllib.parse.quote(relative, safe="/:,@=-._")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = path.read_bytes()
        source = "cache"
    else:
        with urllib.request.urlopen(url, timeout=180) as response:
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


def canonical_response(row: dict[str, Any]) -> str:
    mapped = row.get("mapped_output")
    value = row.get("predicted_text", "") if mapped is None else mapped
    rendered = str(value).strip()
    return rendered if rendered else "EMPTY:"


def load_task(
    *, base: str, task: str, scenario: str, models: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_dir = RAW_ROOT / task

    def load_model(model: str):
        prefix = scenario_run_prefix(scenario, model)
        relative = f"{prefix}/display_predictions.json"
        path = task_dir / f"{safe_name(model)}__display_predictions.json"
        rows, provenance = fetch_json(base, relative, path)
        return model, rows, provenance

    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(load_model, model) for model in models]
        for future in as_completed(futures):
            model, rows, record = future.result()
            rows_by_model[model] = rows
            provenance.append({"kind": "display_predictions", "model": model, **record})

    instance_model = "openai_gpt-4-0613"
    instance_prefix = scenario_run_prefix(scenario, instance_model)
    relative = f"{instance_prefix}/instances.json"
    path = task_dir / f"{safe_name(instance_model)}__instances.json"
    instances_list, record = fetch_json(base, relative, path)
    provenance.append({"kind": "instances", "model": instance_model, **record})

    ids = [str(row["instance_id"]) for row in rows_by_model[models[0]]]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{task}: duplicate prediction IDs")
    keyed = {
        model: {str(row["instance_id"]): row for row in rows_by_model[model]}
        for model in models
    }
    if any(set(keyed[model]) != set(ids) for model in models):
        raise AssertionError(f"{task}: candidate ID mismatch")
    instances = {str(row["id"]): row for row in instances_list}
    if set(instances) != set(ids):
        raise AssertionError(f"{task}: instance/prediction ID mismatch")

    present_everywhere = []
    for metric in METRIC_KEYS:
        if all(
            metric in keyed[model][instance_id].get("stats", {})
            for model in models
            for instance_id in ids
        ):
            present_everywhere.append(metric)
    if not present_everywhere:
        observed_keys = sorted(
            {
                key
                for model in models
                for instance_id in ids
                for key in keyed[model][instance_id].get("stats", {})
            }
        )
        raise AssertionError(
            f"{task}: no frozen utility key is complete; observed={observed_keys}"
        )
    metric = present_everywhere[0]

    prompts: list[str] = []
    references: list[list[str]] = []
    for instance_id in ids:
        item = instances[instance_id]
        prompts.append(str(item["input"]["text"]).strip())
        refs = [
            str(reference["output"]["text"]).strip()
            for reference in item.get("references", [])
        ]
        if not refs:
            raise AssertionError(f"{task}:{instance_id}: no references")
        references.append(refs)
    validate_tokens_against_references(task, references)

    responses: list[list[str]] = []
    utilities: list[list[float]] = []
    for model in models:
        model_responses: list[str] = []
        model_utilities: list[float] = []
        for instance_id in ids:
            row = keyed[model][instance_id]
            model_responses.append(canonical_response(row))
            value = float(row["stats"][metric])
            if not math.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-12:
                raise AssertionError(f"{task}:{model}:{instance_id}: utility {value}")
            model_utilities.append(min(1.0, max(0.0, value)))
        responses.append(model_responses)
        utilities.append(model_utilities)
    return {
        "ids": ids,
        "prompts": prompts,
        "references": references,
        "registry_responses": responses,
        "registry_utilities": utilities,
        "metric": metric,
        "complete_metric_keys": present_everywhere,
    }, provenance


def select(values: list[Any], indices: list[int]) -> list[Any]:
    return [values[index] for index in indices]


def main() -> None:
    prereg = verify_lock()
    base = prereg["public_source"]["base_url"]
    models = list(prereg["public_source"]["candidate_models"])
    calibration: dict[str, Any] = {
        "manifest_id": "STEP87B_LABELED_CALIBRATION_PACKAGE_V1",
        "preregistration_sha256": file_sha256(PREREG),
        "models": models,
        "tasks": {},
    }
    sealed: dict[str, Any] = {
        "manifest_id": "STEP87B_SEALED_HOLDOUT_OUTCOMES_V1",
        "stage_a_must_not_read": True,
        "preregistration_sha256": file_sha256(PREREG),
        "models": models,
        "tasks": {},
    }
    blind_manifest: dict[str, Any] = {
        "manifest_id": "STEP87B_BLIND_PARENT_INPUT_MANIFEST_V1",
        "preregistration_sha256": file_sha256(PREREG),
        "forbidden_fields_absent": True,
        "tasks": {},
    }
    split_audit: dict[str, Any] = {
        "manifest_id": "STEP87B_SPLIT_AND_SCHEMA_AUDIT_V1",
        "preregistration_sha256": file_sha256(PREREG),
        "tasks": {},
    }
    provenance: list[dict[str, Any]] = []

    for task in TASKS:
        scenario = prereg["frozen_scenarios"][task]
        print(f"[{task}] downloading frozen scenario {scenario}", flush=True)
        data, records = load_task(base=base, task=task, scenario=scenario, models=models)
        provenance.extend({"task": task, "scenario": scenario, **row} for row in records)
        calibration_indices, holdout_indices = split_indices(task, data["ids"])
        if min(len(calibration_indices), len(holdout_indices)) < 100:
            raise AssertionError(f"{task}: split below 100 examples")
        if set(calibration_indices) & set(holdout_indices):
            raise AssertionError(f"{task}: split overlap")

        calibration["tasks"][task] = {
            "scenario": scenario,
            "metric": data["metric"],
            "ids": select(data["ids"], calibration_indices),
            "prompts": select(data["prompts"], calibration_indices),
            "references": select(data["references"], calibration_indices),
            "registry_responses": [
                select(row, calibration_indices) for row in data["registry_responses"]
            ],
            "registry_utilities": [
                select(row, calibration_indices) for row in data["registry_utilities"]
            ],
        }
        sealed["tasks"][task] = {
            "scenario": scenario,
            "metric": data["metric"],
            "ids": select(data["ids"], holdout_indices),
            "references": select(data["references"], holdout_indices),
            "registry_responses": [
                select(row, holdout_indices) for row in data["registry_responses"]
            ],
            "registry_utilities": [
                select(row, holdout_indices) for row in data["registry_utilities"]
            ],
        }

        task_blind: dict[str, Any] = {"models": {}}
        for model_index, model in enumerate(models):
            payload = {
                "manifest_id": "STEP87B_SINGLE_PARENT_BLIND_INPUT_V1",
                "task": task,
                "scenario": scenario,
                "parent_index": model_index,
                "parent": model,
                "ids": select(data["ids"], holdout_indices),
                "prompts": select(data["prompts"], holdout_indices),
                "parent_outputs": select(
                    data["registry_responses"][model_index], holdout_indices
                ),
            }
            rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            lowered = rendered.casefold()
            for forbidden in (
                '"reference"',
                '"utility"',
                '"correct"',
                '"oracle"',
                '"peer"',
                '"registry_responses"',
                '"pool"',
                '"selector_state"',
            ):
                if forbidden in lowered:
                    raise AssertionError(f"{task}: forbidden blind field {forbidden}")
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
            "scenario": scenario,
            "metric": data["metric"],
            "complete_metric_keys_in_frozen_order": data["complete_metric_keys"],
            "total": len(data["ids"]),
            "calibration": len(calibration_indices),
            "holdout": len(holdout_indices),
            "id_overlap": 0,
            "calibration_id_digest": digest_json(select(data["ids"], calibration_indices)),
            "holdout_id_digest": digest_json(select(data["ids"], holdout_indices)),
            "all_31_models": len(data["registry_responses"]) == 31,
            "utility_range": [
                float(np.min(np.asarray(data["registry_utilities"], dtype=np.float64))),
                float(np.max(np.asarray(data["registry_utilities"], dtype=np.float64))),
            ],
            "opaque_tokens_absent_from_references": True,
        }
        print(
            f"[{task}] metric={data['metric']} total={len(data['ids'])} "
            f"cal={len(calibration_indices)} holdout={len(holdout_indices)}",
            flush=True,
        )

    CALIBRATION_OUT.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
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
        json.dumps(split_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    raw_manifest = {
        "manifest_id": "STEP87B_RAW_SOURCE_MANIFEST_V1",
        "preregistration_sha256": file_sha256(PREREG),
        "stage0_lock_sha256": file_sha256(STAGE0_LOCK),
        "records": sorted(
            provenance,
            key=lambda row: (row["task"], row["kind"], row["model"]),
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
        json.dumps(raw_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("STEP87B_PREPARATION_COMPLETE_WITH_SEALED_OUTCOMES", flush=True)


if __name__ == "__main__":
    main()

