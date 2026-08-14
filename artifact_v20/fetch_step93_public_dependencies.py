from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "STEP93_STAGE0_MANIFEST_2026-08-13.json"
BASE_MODEL = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
BASE_SHA256 = "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
DATASET = "mteb/banking77"
DATASET_REVISION = "18072d2685ea682290f7b8924d94c62acc19c0b2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base = Path(
        hf_hub_download(BASE_MODEL, "model.safetensors", revision=BASE_REVISION)
    )
    if sha256(base) != BASE_SHA256:
        raise AssertionError("public base encoder hash mismatch")
    observed: dict[str, str] = {}
    for split in ("train", "test"):
        path = Path(
            hf_hub_download(
                repo_id=DATASET,
                filename=f"data/{split}-00000-of-00001.parquet",
                repo_type="dataset",
                revision=DATASET_REVISION,
            )
        )
        observed[split] = sha256(path)
    if observed != manifest["dataset"]["raw_parquet_sha256"]:
        raise AssertionError(
            {"dataset_raw_hashes": observed, "expected": manifest["dataset"]["raw_parquet_sha256"]}
        )
    print(
        json.dumps(
            {
                "verdict": "PASS_STEP93_PUBLIC_DEPENDENCY_FETCH",
                "base_model_sha256": BASE_SHA256,
                "dataset_raw_parquet_sha256": observed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
