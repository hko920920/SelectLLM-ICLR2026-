#!/usr/bin/env python3
"""Generic PRAA Stage-3 task builder routed by the frozen parent family.

The scientific recipe is fixed in the Stage-3 protocol. This wrapper chooses
only the already-declared runtime-family implementation required by the parent
selected under Stage 2; it does not inspect outcome data or selector effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import stage2_clean_safety_endpoint as stage2
import stage3_build_transformer_task as builder
import stage3_lid_native_features as native_features
from stage3_learned_common import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise builder.Stage3Error(f"expected object at {path}")
    return value


def postprocess_native_receipt(output_root: Path, task: str) -> None:
    output_dir = output_root / task
    path = output_dir / "PRAA_STAGE3_RECEIPT.json"
    if not path.is_file():
        return
    receipt = load_json(path)
    code = receipt.setdefault("code_sha256", {})
    code["generic_router"] = sha256_file(Path(__file__).resolve())
    code["native_feature_extractor"] = sha256_file(
        ROOT / "stage3_lid_native_features.py"
    )
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    markdown = output_dir / "PRAA_STAGE3_RECEIPT.md"
    markdown.write_text(builder.markdown_receipt(receipt), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, choices=["emotion", "language_identification"]
    )
    parser.add_argument("--stage1-public-root", type=Path, required=True)
    parser.add_argument("--stage2-ledger", type=Path, required=True)
    parser.add_argument("--stage2-predictions-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    ledger = load_json(args.stage2_ledger)
    parent = str(ledger["tasks"][args.task]["selected_parent"])
    native = parent not in stage2.TRANSFORMER_ROOTS
    if native:
        if args.task != "language_identification":
            raise builder.Stage3Error(
                f"non-Transformer parent is not authorized for {args.task}: {parent}"
            )

        def routed_extract(
            task: str,
            runtime_spec: dict[str, Any],
            texts_by_partition: dict[str, list[str]],
        ):
            if task != "language_identification":
                raise builder.Stage3Error(task)
            return native_features.extract_partitions(
                parent, runtime_spec, texts_by_partition
            )

        # The legacy builder's check means only "a frozen feature
        # implementation exists". Extend that set for this invocation after
        # binding the exact selected parent; no scientific setting changes.
        stage2.TRANSFORMER_ROOTS.add(parent)
        builder.extract_partitions = routed_extract

    try:
        builder.build_task(
            args.task,
            args.stage1_public_root,
            args.stage2_ledger,
            args.stage2_predictions_root,
            args.output_root,
        )
    except builder.ScientificStop:
        if native:
            postprocess_native_receipt(args.output_root, args.task)
        raise
    if native:
        postprocess_native_receipt(args.output_root, args.task)


if __name__ == "__main__":
    main()
