from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import step93_common as selector
import step95_common as adapter
import step96_common as learned


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-13"
PROTOCOL = ROOT / f"STEP98_ANLI_FRESH_PRIMARY_PREREGISTRATION_{DATE}.md"
DATASET_ID = "facebook/anli"
DATASET_CONFIG = "plain_text"
DATASET_REVISION = "8e4813d81f46d313dac7892e1c28076917cfcdf9"
SALT = "step98-anli-fresh-primary-v1"
ROUNDS = ("r1", "r2", "r3")
N_CLASSES = 3
N_ALIASES = 4
ROSTER_SIZE = 4
POOL_SIZE = 500
BUDGET = 10
TAU = 0.025
TRAIN_QUOTAS_PER_LABEL = {
    "geometry_search": 250,
    "geometry_verify": 250,
    "adapter_train": 1800,
    "threshold_calibration": 400,
    "threshold_safety": 500,
    "selector_search": 600,
    "selector_verify": 1000,
}
GEOMETRY_SEARCH_SEEDS = tuple(range(980000, 980300))
GEOMETRY_VERIFY_SEEDS = tuple(range(980300, 980600))
SELECTOR_SEARCH_SEEDS = tuple(range(982000, 982800))
SELECTOR_VERIFY_SEEDS = tuple(range(983000, 984500))
TEST_SEEDS = tuple(range(985000, 988000))


@dataclass(frozen=True)
class RootSpec:
    key: str
    repo_id: str
    revision: str
    raw_to_dataset: tuple[int, int, int]
    parent: bool = False
    slow_tokenizer: bool = False


ROOT_SPECS: tuple[RootSpec, ...] = (
    RootSpec(
        "crossencoder_deberta_small",
        "cross-encoder/nli-deberta-v3-small",
        "fa2804872c3b4bd748f38c0185cc85775361e735",
        (2, 0, 1),
        parent=True,
        slow_tokenizer=True,
    ),
    RootSpec(
        "moritz_minilm",
        "MoritzLaurer/MiniLM-L6-mnli",
        "6e0917f1a395b7a6c0f054a56b91c45d8e3af92f",
        (0, 1, 2),
    ),
    RootSpec(
        "mfac_bert_mini",
        "M-FAC/bert-mini-finetuned-mnli",
        "780061727f47254ff763de653920bb8b7e2fd5f2",
        (0, 1, 2),
    ),
    RootSpec(
        "mfac_bert_tiny",
        "M-FAC/bert-tiny-finetuned-mnli",
        "618f766f89b50853abc1bea92fd38e1973818f0b",
        (0, 1, 2),
    ),
)


sha256_path = selector.sha256_path
array_sha256 = selector.array_sha256
stable_u64 = selector.stable_u64
sample_pools = selector.sample_pools
run_active = selector.run_active
run_fixed = selector.run_fixed
effect_summary = selector.effect_summary
exact_match_feedback = selector.exact_match_feedback
threshold_from_caps = adapter.threshold_from_caps
make_alias_codes = adapter.make_alias_codes
infer_error_scores = adapter.infer_error_scores
save_adapter = adapter.save_adapter
load_adapter_model = learned.load_adapter_model
train_error_adapter = learned.train_error_adapter
tokenizer_for = learned.tokenizer_for
infer_root_predictions = learned.infer_root_predictions


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def row_digest(round_key: str, row: dict[str, Any]) -> str:
    payload = "\x1f".join(
        (
            DATASET_REVISION,
            round_key,
            str(row["uid"]),
            str(row["premise"]),
            str(row["hypothesis"]),
            str(int(row["label"])),
            SALT,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def as_learned_spec(spec: RootSpec) -> learned.RootSpec:
    return learned.RootSpec(
        spec.key, spec.repo_id, spec.revision, spec.raw_to_dataset,
        parent=spec.parent, slow_tokenizer=spec.slow_tokenizer,
    )
