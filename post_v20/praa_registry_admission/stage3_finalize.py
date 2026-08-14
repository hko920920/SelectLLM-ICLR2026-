#!/usr/bin/env python3
"""Close the two-task PRAA Stage-3 family from task receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

TASKS = ("emotion", "language_identification")
TASK_SCHEMA = "praa.stage3.error_head_and_threshold.task.v1"


class FinalizeError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalizeError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def find_receipt(root: Path, task: str) -> Path:
    candidates = [
        path
        for path in root.rglob("PRAA_STAGE3_RECEIPT.json")
        if path.parent.name == task
    ]
    if len(candidates) != 1:
        raise FinalizeError(
            f"expected one Stage-3 receipt for {task}, observed {[str(p) for p in candidates]}"
        )
    return candidates[0]


def inspect_task(
    receipt_path: Path,
    task: str,
    stage1_lock_sha256: str,
    stage2_ledger_sha256: str,
) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    if receipt.get("schema") != TASK_SCHEMA or receipt.get("task") != task:
        raise FinalizeError(f"task receipt identity drift for {task}")
    if receipt.get("stage1_lock_sha256") != stage1_lock_sha256:
        raise FinalizeError(f"Stage-1 binding drift for {task}")
    if receipt.get("stage2_ledger_sha256") != stage2_ledger_sha256:
        raise FinalizeError(f"Stage-2 binding drift for {task}")
    boundary = receipt.get("information_boundary") or {}
    forbidden = [
        key
        for key in (
            "target_threshold_labels_opened",
            "primary_inputs_opened",
            "primary_labels_opened",
            "deployment_inputs_opened",
            "deployment_labels_opened",
            "selector_executed",
        )
        if bool(boundary.get(key))
    ]
    if forbidden:
        raise FinalizeError(f"information-boundary violation for {task}: {forbidden}")
    gates = receipt.get("gates") or {}
    failed_from_gates = sorted(name for name, value in gates.items() if not bool(value))
    recorded_failed = sorted(receipt.get("failed_gates") or [])
    if failed_from_gates != recorded_failed:
        raise FinalizeError(
            f"failed-gate ledger drift for {task}: {failed_from_gates} != {recorded_failed}"
        )
    expected_decision = (
        "PASS_PRAA_STAGE3_TASK" if not failed_from_gates else "STOP_PRAA_STAGE3_TASK"
    )
    if receipt.get("decision") != expected_decision:
        raise FinalizeError(f"task decision drift for {task}")
    checkpoints = [head["checkpoint"]["sha256"] for head in receipt.get("heads") or []]
    if len(checkpoints) != 4 or len(set(checkpoints)) != 4:
        raise FinalizeError(f"head-checkpoint cardinality/hash failure for {task}")
    return {
        "task": task,
        "decision": expected_decision,
        "selected_parent": receipt.get("selected_parent"),
        "receipt_file_sha256": sha256_file(receipt_path),
        "receipt_canonical_sha256": receipt.get("receipt_sha256"),
        "failed_gates": failed_from_gates,
        "thresholds": [float(head["threshold"]) for head in receipt["heads"]],
        "target_threshold_trigger_counts": [
            int(head["target_threshold_trigger_count"]) for head in receipt["heads"]
        ],
        "safety_trigger_counts": [
            int(head["safety_trigger_count"]) for head in receipt["heads"]
        ],
        "head_checkpoint_sha256": checkpoints,
        "safety_quality": receipt["safety_quality"],
        "gates": gates,
        "arrays_sha256": receipt["arrays_sha256"],
    }


def markdown(closure: dict[str, Any]) -> str:
    lines = [
        "# PRAA Stage-3 family closure",
        "",
        f"Decision: `{closure['decision']}`",
        "",
        f"Closure SHA-256: `{closure['closure_sha256']}`",
        "",
        "No primary/deployment input, label, prediction, selector path, or effect was opened in Stage 3.",
        "",
    ]
    for task, record in closure["tasks"].items():
        lines.extend(
            [
                f"## {task}",
                "",
                f"- decision: `{record['decision']}`",
                f"- selected parent: `{record['selected_parent']}`",
                f"- thresholds: `{record['thresholds']}`",
                f"- target-threshold triggers: `{record['target_threshold_trigger_counts']}`",
                f"- safety triggers: `{record['safety_trigger_counts']}`",
                f"- safety parent accuracy: `{record['safety_quality']['parent_accuracy']}`",
                f"- safety derived accuracies: `{record['safety_quality']['alias_accuracy']}`",
                f"- failed gates: `{record['failed_gates']}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument("--stage1-lock", type=Path, required=True)
    parser.add_argument("--stage2-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage1_hash = sha256_file(args.stage1_lock)
    stage2_hash = sha256_file(args.stage2_ledger)
    task_records = {
        task: inspect_task(
            find_receipt(args.receipts_root, task),
            task,
            stage1_hash,
            stage2_hash,
        )
        for task in TASKS
    }
    pass_family = all(
        record["decision"] == "PASS_PRAA_STAGE3_TASK"
        for record in task_records.values()
    )
    closure: dict[str, Any] = {
        "schema": "praa.stage3.family_closure.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage1_lock_sha256": stage1_hash,
        "stage2_ledger_sha256": stage2_hash,
        "tasks": task_records,
        "information_boundary": {
            "target_threshold_labels_opened": False,
            "primary_inputs_opened": False,
            "primary_labels_opened": False,
            "deployment_inputs_opened": False,
            "deployment_labels_opened": False,
            "selector_executed": False,
        },
        "decision": (
            "PASS_PRAA_STAGE3_ERROR_HEAD_AND_THRESHOLD"
            if pass_family
            else "STOP_PRAA_STAGE3_ERROR_HEAD_AND_THRESHOLD"
        ),
    }
    closure["closure_sha256"] = hashlib.sha256(
        canonical(closure).encode("utf-8")
    ).hexdigest()
    json_path = args.output_dir / "PRAA_STAGE3_FAMILY_CLOSURE.json"
    md_path = args.output_dir / "PRAA_STAGE3_FAMILY_CLOSURE.md"
    json_path.write_text(json.dumps(closure, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(closure), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    if not pass_family:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
