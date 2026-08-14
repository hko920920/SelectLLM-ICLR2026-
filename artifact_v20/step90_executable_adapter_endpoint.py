"""Run one Step 90 calibration adapter on a raw text pair.

Example:
  python step90_executable_adapter_endpoint.py --task mnli --adapter 0 \
    --text-a "A person is outdoors." --text-b "Someone is outside."

The endpoint reads no dataset item identifier, label, peer response, or selector
state.  It fetches the hash-pinned parent named in the frozen configuration and
loads one local safetensors adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "STEP90_HIGH_CONFIDENCE_STAGEA_FROZEN_CONFIG_2026-08-12.json"
MODEL_HASHES = {
    "mnli": "9df3eb5d37118f952f4ba4fb46fde6889e3a9ccedeee0bad09b0110fc64c5c29",
    "qqp": "73fd14ad7d08f3ef30eb25841c8f4ba89e91230f48159279233b37015ccb33fb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("mnli", "qqp"))
    parser.add_argument("--adapter", required=True, type=int, choices=range(4))
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))["tasks"][args.task]
    adapter_path = ROOT / "step90_models" / f"step90_{args.task}_adapter_{args.adapter}.safetensors"
    if sha256(adapter_path) != config["adapter_sha256"][adapter_path.name]:
        raise AssertionError("adapter hash mismatch")
    adapter = load_file(str(adapter_path))
    threshold = float(adapter["confidence_threshold"].item())
    style = int(adapter["style_id"].item())
    if style != args.adapter:
        raise AssertionError("adapter style mismatch")

    model_path = Path(
        hf_hub_download(
            config["model"],
            filename="model.safetensors",
            revision=config["model_revision"],
        )
    )
    if sha256(model_path) != MODEL_HASHES[args.task]:
        raise AssertionError("parent model hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"], revision=config["model_revision"], use_safetensors=True
    ).eval()
    encoded = tokenizer(
        args.text_a,
        args.text_b,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**encoded).logits[0].double().numpy()
    mapping = np.asarray(config["label_mapping_raw_to_glue"], dtype=np.int64)
    if len(logits) == 1:
        raw_label = int(logits[0] >= 0.0)
        confidence = float(1.0 / (1.0 + np.exp(-abs(float(logits[0])))))
    else:
        raw_label = int(np.argmax(logits))
        shifted = logits - float(np.max(logits))
        probability = np.exp(shifted)
        probability /= float(np.sum(probability))
        confidence = float(np.max(probability))
    label = int(mapping[raw_label])
    response = f"LABEL_{label}"
    triggered = confidence >= threshold
    if triggered:
        response += f"::STYLE_{style}"
    print(json.dumps({
        "task": args.task,
        "adapter": args.adapter,
        "response": response,
        "parsed_hard_label": label,
        "parent_confidence": confidence,
        "triggered": triggered,
        "reads_reference": False,
        "reads_item_id": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
