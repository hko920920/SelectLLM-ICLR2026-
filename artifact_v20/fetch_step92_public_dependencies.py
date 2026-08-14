"""Fetch and verify the public encoder needed for Step 92 endpoint replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer


MODEL_ID = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
EXPECTED_WEIGHT_SHA256 = "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    weight_path = Path(
        hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION)
    )
    observed = sha256(weight_path)
    if observed != EXPECTED_WEIGHT_SHA256:
        raise AssertionError(
            {"public_encoder_weight_hash": observed, "expected": EXPECTED_WEIGHT_SHA256}
        )

    # Populate all tokenizer/config files subsequently requested with
    # local_files_only=True by the independent validator.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION)
    print(
        json.dumps(
            {
                "verdict": "PASS_STEP92_PUBLIC_DEPENDENCY_FETCH",
                "model_id": MODEL_ID,
                "revision": REVISION,
                "weight_sha256": observed,
                "tokenizer_class": tokenizer.__class__.__name__,
                "config_class": config.__class__.__name__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
