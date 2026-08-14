from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import step100_common as common


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / f"STEP101_IMDB_DEVELOPMENT_PARENT_ROSTER_SELECTION_PROTOCOL_{common.DATE}.md"
DEV = ROOT / f"STEP100_IMDB_DEVELOPMENT_ROWS_{common.DATE}.json"
OUTPUT_DIR = ROOT / "step101_candidate_predictions"

CANDIDATES = {
    "aychang_roberta": common.RootSpec(
        "aychang_roberta", "aychang/roberta-base-imdb",
        "cb6bcadd0540b61c9623bd6295d51ac445ceb135",
    ),
    "wrmurray_roberta": common.RootSpec(
        "wrmurray_roberta", "wrmurray/roberta-base-finetuned-imdb",
        "7aa8ca3fae56a1860d8b4c6bf727b91370821ad5",
    ),
    "dfurman_deberta": common.RootSpec(
        "dfurman_deberta", "dfurman/deberta-v3-base-imdb",
        "cbfcf7b54b2fb47d13b75f3c3a517f95808a2285",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=tuple(CANDIDATES))
    args = parser.parse_args()
    spec = CANDIDATES[args.candidate]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{spec.key}.npz"
    audit_path = OUTPUT_DIR / f"{spec.key}.audit.json"
    if output.exists() or audit_path.exists():
        raise FileExistsError(spec.key)
    payload = json.loads(DEV.read_text(encoding="utf-8"))
    texts = [str(row["text"]) for row in payload["rows"]]
    prediction, audit = common.infer_root_predictions(spec, texts)
    np.savez_compressed(output, predictions=prediction.astype(np.int16))
    record = {
        "candidate": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "protocol_sha256": common.sha256_path(PROTOCOL),
        "development_sha256": common.sha256_path(DEV),
        "prediction_file": str(output.relative_to(ROOT)).replace("\\", "/"),
        "prediction_file_sha256": common.sha256_path(output),
        "prediction_array_sha256": common.array_sha256(prediction),
        "model_audit": audit,
        "sealed_outcome_opened": False,
        "code_sha256": {
            "step100_common.py": common.sha256_path(ROOT / "step100_common.py"),
            "step101_infer_candidate.py": common.sha256_path(Path(__file__)),
        },
    }
    common.json_dump(audit_path, record)
    print(json.dumps({
        "candidate": spec.key,
        "prediction_sha256": record["prediction_array_sha256"],
        "rows": len(prediction),
        "sealed_outcome_opened": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
