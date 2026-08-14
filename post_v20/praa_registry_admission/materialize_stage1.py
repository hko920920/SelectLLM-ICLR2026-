#!/usr/bin/env python3
"""Materialize and seal PRAA Stage-1 artifacts without model inference.

The executable information boundary is intentionally narrow:

* exact public dataset rows and labels may be read to create deterministic,
  separately stored input and label packages;
* exact endpoint files may be downloaded and hashed;
* model tensors are never loaded and no endpoint or selector is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import requests
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

SCRIPT_DIR = Path(__file__).resolve().parent
LOCK_PATH = SCRIPT_DIR / "PRAA_STAGE0_METADATA_LOCK_2026-08-14.json"
SCHEMA = "praa.stage1.artifact_and_data_seal.v1"
PARTITION_DOMAIN = "PRAA-stage1-partition-v1"
USER_AGENT = "SelectLLM-PRAA-stage1-seal/1.0"

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LID_LABELS = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja",
    "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh",
]

# Native NLLB / GlotLID-style labels that deterministically map into the
# frozen 20-code target space. Wrappers in later stages must also map all
# unlisted native labels to a private always-incorrect code.
LID_THREE_LETTER_SCRIPT = {
    "ar": ["arb_Arab", "ara_Arab"],
    "bg": ["bul_Cyrl"],
    "de": ["deu_Latn", "ger_Latn"],
    "el": ["ell_Grek", "gre_Grek"],
    "en": ["eng_Latn"],
    "es": ["spa_Latn"],
    "fr": ["fra_Latn", "fre_Latn"],
    "hi": ["hin_Deva"],
    "it": ["ita_Latn"],
    "ja": ["jpn_Jpan"],
    "nl": ["nld_Latn", "dut_Latn"],
    "pl": ["pol_Latn"],
    "pt": ["por_Latn"],
    "ru": ["rus_Cyrl"],
    "sw": ["swh_Latn", "swa_Latn"],
    "th": ["tha_Thai"],
    "tr": ["tur_Latn"],
    "ur": ["urd_Arab"],
    "vi": ["vie_Latn"],
    "zh": ["zho_Hans", "zho_Hant", "cmn_Hans", "cmn_Hant", "chi_Hans", "chi_Hant"],
}

WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".ftz",
    ".pt",
    ".pth",
    ".onnx",
    ".h5",
)
CONFIG_BASENAMES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "preprocessor_config.json",
}
WEIGHT_EXCLUDES = (
    "optimizer",
    "scheduler",
    "training_args",
    "rng_state",
    "trainer_state",
    "checkpoint-",
)
GENERIC_LABEL_RE = re.compile(r"^(?:label[_ -]?)?(\d+)$", re.IGNORECASE)


class Stage1Error(RuntimeError):
    """A deterministic Stage-1 materialization or validation failure."""


@dataclass(frozen=True)
class PartitionSpec:
    source_split: str
    name: str
    start: int
    stop: int


PARTITIONS: dict[str, list[PartitionSpec]] = {
    "emotion": [
        PartitionSpec("train", "error_head_train", 0, 12_000),
        PartitionSpec("train", "safety", 12_000, 16_000),
        PartitionSpec("validation", "target_threshold", 0, 2_000),
        PartitionSpec("test", "primary_outcome", 0, 1_500),
        PartitionSpec("test", "deployment", 1_500, 2_000),
    ],
    "language_identification": [
        PartitionSpec("train", "error_head_train", 0, 50_000),
        PartitionSpec("train", "safety", 50_000, 60_000),
        PartitionSpec("train", "development_reserve", 60_000, 70_000),
        PartitionSpec("validation", "target_threshold", 0, 10_000),
        PartitionSpec("test", "primary_outcome", 0, 8_000),
        PartitionSpec("test", "deployment", 8_000, 10_000),
    ],
}

EXPECTED_SPLITS = {
    "emotion": {"train": 16_000, "validation": 2_000, "test": 2_000},
    "language_identification": {"train": 70_000, "validation": 10_000, "test": 10_000},
}


class HttpClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def get_json(self, url: str, attempts: int = 4) -> Any:
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.get(url, timeout=60)
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f"transient HTTP {response.status_code} for {url}",
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last = exc
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
        raise Stage1Error(f"failed to retrieve {url}: {last}")

    def download(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.session.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage1Error(f"expected JSON object at {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_object_sha256(value: Any) -> str:
    return sha256_text(canonical_json(value))


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        value = str(value)
    return unicodedata.normalize("NFC", value)


def normalize_label_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def uid_for(
    task_key: str,
    revision: str,
    source_split: str,
    source_index: int,
    raw_input: Mapping[str, str],
) -> str:
    preimage = "|".join(
        [
            task_key,
            revision,
            source_split,
            str(source_index),
            canonical_json(raw_input),
        ]
    )
    return sha256_text(preimage)


def partition_key(uid: str) -> str:
    return sha256_text(f"{PARTITION_DOMAIN}|{uid}")


def digest_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def inventory_tree(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def select_model_files(file_names: Sequence[str]) -> tuple[list[str], list[str]]:
    weight_files: list[str] = []
    support_files: list[str] = []
    for name in sorted(set(file_names)):
        lower = name.lower()
        base = Path(name).name.lower()
        if any(token in lower for token in WEIGHT_EXCLUDES):
            continue
        if lower.endswith(WEIGHT_SUFFIXES):
            weight_files.append(name)
            continue
        if base in CONFIG_BASENAMES or base.endswith(".model"):
            support_files.append(name)
            continue
        if base.endswith(".index.json") and (
            "safetensors" in base or "pytorch_model" in base
        ):
            support_files.append(name)
    return weight_files, support_files


def parse_config(path_by_name: Mapping[str, Path]) -> dict[str, Any]:
    config_path = path_by_name.get("config.json")
    if config_path is None:
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def emotion_label_map(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("id2label") or {}
    if not isinstance(raw, dict):
        raw = {}
    id2label = {str(key): str(value) for key, value in raw.items()}
    num_labels = config.get("num_labels")
    if num_labels is None and id2label:
        num_labels = len(id2label)
    try:
        count = int(num_labels)
    except (TypeError, ValueError):
        count = -1
    if count != len(EMOTION_LABELS):
        raise Stage1Error(f"emotion endpoint num_labels={num_labels!r}, expected 6")

    if not id2label:
        raise Stage1Error("emotion endpoint config lacks id2label")
    observed = {normalize_label_name(value) for value in id2label.values()}
    expected = {normalize_label_name(value) for value in EMOTION_LABELS}
    if observed == expected:
        native_to_target = {
            str(native): EMOTION_LABELS.index(str(label).strip().lower())
            for native, label in id2label.items()
        }
        return {
            "mode": "semantic_id2label",
            "id2label": id2label,
            "native_index_to_target_index": native_to_target,
        }

    generic_pairs: list[tuple[int, str]] = []
    for native, label in id2label.items():
        match = GENERIC_LABEL_RE.fullmatch(str(label).strip())
        if match is None:
            raise Stage1Error(
                f"emotion endpoint labels are neither semantic nor generic: {id2label}"
            )
        generic_pairs.append((int(native), match.group(1)))
    native_indices = sorted(index for index, _ in generic_pairs)
    generic_indices = sorted(int(value) for _, value in generic_pairs)
    if native_indices != list(range(6)) or generic_indices != list(range(6)):
        raise Stage1Error(f"generic emotion label indices are not 0..5: {id2label}")
    return {
        "mode": "generic_indices_bound_to_dataset_order",
        "dataset_order": EMOTION_LABELS,
        "id2label": id2label,
        "native_index_to_target_index": {str(index): index for index in range(6)},
    }


def lid_label_map(repo_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    if repo_id == "papluca/xlm-roberta-base-language-detection":
        raw = config.get("id2label") or {}
        if not isinstance(raw, dict) or len(raw) != len(LID_LABELS):
            raise Stage1Error(
                f"XLM-R LID config must expose 20 id2label entries, observed {raw!r}"
            )
        id2label = {str(key): str(value).strip().lower() for key, value in raw.items()}
        observed = set(id2label.values())
        if observed != set(LID_LABELS):
            raise Stage1Error(
                f"XLM-R LID labels differ from frozen target set: {sorted(observed)}"
            )
        return {
            "mode": "semantic_id2label",
            "id2label": id2label,
            "native_index_to_target_index": {
                native: LID_LABELS.index(label) for native, label in id2label.items()
            },
            "out_of_target_policy": "private_always_incorrect_code",
        }
    if repo_id == "facebook/fasttext-language-identification":
        return {
            "mode": "nllb_three_letter_script_to_iso639_1",
            "target_to_accepted_native_suffixes": LID_THREE_LETTER_SCRIPT,
            "accepted_prefixes": ["__label__"],
            "out_of_target_policy": "private_always_incorrect_code",
        }
    if repo_id == "cis-lmu/glotlid":
        return {
            "mode": "glotlid_three_letter_script_to_iso639_1",
            "target_to_accepted_native_suffixes": LID_THREE_LETTER_SCRIPT,
            "accepted_prefixes": ["__label__"],
            "out_of_target_policy": "private_always_incorrect_code",
        }
    raise Stage1Error(f"no frozen LID mapping for {repo_id}")


def materialize_hf_endpoint(
    api: HfApi,
    repo_id: str,
    revision: str,
    task_key: str,
) -> dict[str, Any]:
    info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=True)
    if str(info.sha) != revision:
        raise Stage1Error(
            f"revision mismatch for {repo_id}: expected {revision}, observed {info.sha}"
        )
    file_names = [sibling.rfilename for sibling in info.siblings]
    weight_files, support_files = select_model_files(file_names)
    if not weight_files:
        raise Stage1Error(f"no model weight artifact selected for {repo_id}@{revision}")
    selected = sorted(set(weight_files + support_files))

    with tempfile.TemporaryDirectory(prefix="praa-stage1-model-") as cache:
        paths: dict[str, Path] = {}
        records: list[dict[str, Any]] = []
        for filename in selected:
            materialized = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=cache,
                )
            )
            if not materialized.is_file() or materialized.stat().st_size <= 0:
                raise Stage1Error(f"empty artifact {repo_id}@{revision}:{filename}")
            paths[Path(filename).name] = materialized
            records.append(
                {
                    "path": filename,
                    "bytes": materialized.stat().st_size,
                    "sha256": sha256_file(materialized),
                    "role": "weight" if filename in weight_files else "support",
                }
            )
        config = parse_config(paths)
        if task_key == "emotion":
            mapping = emotion_label_map(config)
        else:
            mapping = lid_label_map(repo_id, config)
        return {
            "kind": "huggingface_model",
            "repo_id": repo_id,
            "revision": revision,
            "resolved_revision": str(info.sha),
            "library_name": getattr(info, "library_name", None),
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "files": records,
            "total_materialized_bytes": sum(item["bytes"] for item in records),
            "label_mapping": mapping,
            "model_inference_executed": False,
        }


def materialize_pypi_langid(client: HttpClient, output_dir: Path) -> dict[str, Any]:
    package = "langid"
    version = "1.1.6"
    metadata = client.get_json(
        f"https://pypi.org/pypi/{quote(package)}/{quote(version)}/json"
    )
    urls = metadata.get("urls") or []
    if not urls:
        raise Stage1Error("PyPI langid==1.1.6 exposes no distributions")
    distribution_records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(urls, key=lambda row: str(row.get("filename"))):
        filename = str(item.get("filename") or "")
        url = str(item.get("url") or "")
        expected = str((item.get("digests") or {}).get("sha256") or "")
        if not filename or not url or not expected:
            raise Stage1Error(f"incomplete PyPI distribution metadata: {item!r}")
        destination = output_dir / filename
        client.download(url, destination)
        observed = sha256_file(destination)
        if observed != expected:
            raise Stage1Error(
                f"PyPI SHA-256 mismatch for {filename}: {expected} != {observed}"
            )
        distribution_records.append(
            {
                "filename": filename,
                "bytes": destination.stat().st_size,
                "sha256": observed,
                "packagetype": item.get("packagetype"),
                "python_version": item.get("python_version"),
            }
        )
    return {
        "kind": "pypi_package",
        "package": package,
        "version": version,
        "files": distribution_records,
        "total_materialized_bytes": sum(item["bytes"] for item in distribution_records),
        "label_mapping": {
            "mode": "native_iso639_1",
            "target_codes": LID_LABELS,
            "out_of_target_policy": "private_always_incorrect_code",
        },
        "model_inference_executed": False,
    }


def snapshot_dataset(repo_id: str, revision: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="praa-stage1-dataset-snapshot-") as tmp:
        local = Path(tmp) / "snapshot"
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local,
        )
        records = inventory_tree(local)
        if not records:
            raise Stage1Error(f"empty dataset snapshot for {repo_id}@{revision}")
        return {
            "repo_id": repo_id,
            "revision": revision,
            "files": records,
            "total_materialized_bytes": sum(item["bytes"] for item in records),
        }


def load_exact_dataset(
    task_key: str,
    repo_id: str,
    revision: str,
    cache_dir: Path,
) -> tuple[DatasetDict, str | None]:
    configurations: list[str | None]
    if task_key == "emotion":
        configurations = ["split", None]
    else:
        configurations = [None]
    failures: list[str] = []
    for configuration in configurations:
        try:
            if configuration is None:
                dataset = load_dataset(
                    repo_id,
                    revision=revision,
                    cache_dir=str(cache_dir),
                )
            else:
                dataset = load_dataset(
                    repo_id,
                    configuration,
                    revision=revision,
                    cache_dir=str(cache_dir),
                )
            if not isinstance(dataset, DatasetDict):
                raise Stage1Error(f"{repo_id} did not return a DatasetDict")
            return dataset, configuration
        except Exception as exc:
            failures.append(f"configuration={configuration!r}: {type(exc).__name__}: {exc}")
    raise Stage1Error(
        f"unable to load {repo_id}@{revision}; attempts: {' | '.join(failures)}"
    )


def validate_dataset(
    task_key: str,
    dataset: DatasetDict,
) -> dict[str, Any]:
    expected_splits = EXPECTED_SPLITS[task_key]
    observed = {name: len(dataset[name]) for name in expected_splits if name in dataset}
    if observed != expected_splits:
        raise Stage1Error(
            f"{task_key} split cardinality mismatch: expected {expected_splits}, observed {observed}"
        )

    if task_key == "emotion":
        for split in expected_splits:
            columns = set(dataset[split].column_names)
            if not {"text", "label"}.issubset(columns):
                raise Stage1Error(f"emotion {split} fields mismatch: {sorted(columns)}")
        feature = dataset["train"].features["label"]
        names = list(getattr(feature, "names", []) or [])
        if [str(item).lower() for item in names] != EMOTION_LABELS:
            raise Stage1Error(
                f"emotion ClassLabel order mismatch: expected {EMOTION_LABELS}, observed {names}"
            )
        observed_labels = set()
        for split in expected_splits:
            observed_labels.update(int(value) for value in dataset[split]["label"])
        if observed_labels != set(range(6)):
            raise Stage1Error(f"emotion observed label indices mismatch: {observed_labels}")
        return {
            "input_fields": ["text"],
            "label_field": "label",
            "label_names": EMOTION_LABELS,
            "split_cardinalities": observed,
        }

    for split in expected_splits:
        columns = set(dataset[split].column_names)
        if not {"text", "labels"}.issubset(columns):
            raise Stage1Error(
                f"language-identification {split} fields mismatch: {sorted(columns)}"
            )
    observed_codes = set()
    for split in expected_splits:
        observed_codes.update(str(value).strip().lower() for value in dataset[split]["labels"])
    if observed_codes != set(LID_LABELS):
        raise Stage1Error(
            f"language-identification label set mismatch: {sorted(observed_codes)}"
        )
    return {
        "input_fields": ["text"],
        "label_field": "labels",
        "label_names": LID_LABELS,
        "split_cardinalities": observed,
    }


def raw_input_for(task_key: str, row: Mapping[str, Any]) -> dict[str, str]:
    if "text" not in row:
        raise Stage1Error(f"{task_key} row lacks text")
    return {"text": normalize_text(row["text"])}


def label_index_for(task_key: str, row: Mapping[str, Any]) -> int:
    if task_key == "emotion":
        label = int(row["label"])
        if not 0 <= label < len(EMOTION_LABELS):
            raise Stage1Error(f"emotion label out of range: {label}")
        return label
    code = str(row["labels"]).strip().lower()
    try:
        return LID_LABELS.index(code)
    except ValueError as exc:
        raise Stage1Error(f"LID label outside frozen target set: {code!r}") from exc


def build_split_records(
    task_key: str,
    revision: str,
    source_split: str,
    dataset: Dataset,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(len(dataset)):
        row = dataset[index]
        raw_input = raw_input_for(task_key, row)
        uid = uid_for(task_key, revision, source_split, index, raw_input)
        if uid in seen:
            raise Stage1Error(f"duplicate UID in {task_key}/{source_split}: {uid}")
        seen.add(uid)
        records.append(
            {
                "uid": uid,
                "partition_key": partition_key(uid),
                "source_split": source_split,
                "source_index": index,
                "input": raw_input,
                "label": label_index_for(task_key, row),
            }
        )
    return sorted(records, key=lambda item: (item["partition_key"], item["uid"]))


def write_partition(
    task_key: str,
    spec: PartitionSpec,
    ordered_records: Sequence[Mapping[str, Any]],
    public_root: Path,
    sealed_root: Path,
) -> dict[str, Any]:
    selected = list(ordered_records[spec.start : spec.stop])
    expected = spec.stop - spec.start
    if len(selected) != expected:
        raise Stage1Error(
            f"partition {task_key}/{spec.name} expected {expected}, observed {len(selected)}"
        )
    input_path = public_root / "inputs" / task_key / f"{spec.name}.jsonl"
    label_path = sealed_root / "labels" / task_key / f"{spec.name}.labels.jsonl"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    uids: list[str] = []
    label_bindings: list[str] = []
    with input_path.open("w", encoding="utf-8", newline="\n") as input_handle, label_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as label_handle:
        for item in selected:
            uid = str(item["uid"])
            uids.append(uid)
            input_payload = {
                "uid": uid,
                "partition": spec.name,
                "source_split": spec.source_split,
                "source_index": int(item["source_index"]),
                "input": item["input"],
            }
            label_payload = {"uid": uid, "label": int(item["label"])}
            input_handle.write(canonical_json(input_payload) + "\n")
            label_handle.write(canonical_json(label_payload) + "\n")
            label_bindings.append(f"{uid}\t{int(item['label'])}")

    uid_digest = digest_lines(uids)
    label_binding_digest = digest_lines(label_bindings)
    return {
        "task": task_key,
        "partition": spec.name,
        "source_split": spec.source_split,
        "source_hash_slice": [spec.start, spec.stop],
        "rows": expected,
        "uid_sha256": uid_digest,
        "label_binding_sha256": label_binding_digest,
        "input_file": input_path.relative_to(public_root).as_posix(),
        "input_bytes": input_path.stat().st_size,
        "input_sha256": sha256_file(input_path),
        "sealed_label_file": label_path.relative_to(sealed_root).as_posix(),
        "sealed_label_bytes": label_path.stat().st_size,
        "sealed_label_sha256": sha256_file(label_path),
        "label_values_logged": False,
    }


def materialize_task_data(
    task_key: str,
    task_lock: Mapping[str, Any],
    public_root: Path,
    sealed_root: Path,
) -> dict[str, Any]:
    dataset_lock = task_lock["dataset"]
    repo_id = str(dataset_lock["repo_id"])
    revision = str(dataset_lock["revision"])
    snapshot = snapshot_dataset(repo_id, revision)

    with tempfile.TemporaryDirectory(prefix="praa-stage1-datasets-cache-") as cache:
        dataset, configuration = load_exact_dataset(
            task_key, repo_id, revision, Path(cache)
        )
        validation = validate_dataset(task_key, dataset)
        ordered_by_split = {
            split: build_split_records(task_key, revision, split, dataset[split])
            for split in EXPECTED_SPLITS[task_key]
        }
        partition_records = [
            write_partition(
                task_key,
                spec,
                ordered_by_split[spec.source_split],
                public_root,
                sealed_root,
            )
            for spec in PARTITIONS[task_key]
        ]

    # Prove that each source split is partitioned without overlap and with the
    # exact authorized coverage. No label is involved in this accounting.
    coverage: dict[str, list[tuple[int, int, str]]] = {}
    for spec in PARTITIONS[task_key]:
        coverage.setdefault(spec.source_split, []).append((spec.start, spec.stop, spec.name))
    for split, expected_rows in EXPECTED_SPLITS[task_key].items():
        slices = sorted(coverage.get(split, []))
        cursor = 0
        for start, stop, name in slices:
            if start != cursor or stop <= start:
                raise Stage1Error(
                    f"non-contiguous or invalid partition coverage for {task_key}/{split}: "
                    f"cursor={cursor}, slice={(start, stop, name)}"
                )
            cursor = stop
        if cursor != expected_rows:
            raise Stage1Error(
                f"incomplete partition coverage for {task_key}/{split}: {cursor}/{expected_rows}"
            )

    return {
        "dataset_snapshot": snapshot,
        "load_configuration": configuration,
        "schema_validation": validation,
        "partitions": partition_records,
        "dataset_rows_accessed": True,
        "dataset_labels_accessed_only_for_separate_seal": True,
        "model_outputs_accessed": False,
        "selector_executed": False,
    }


def verify_cross_package_bindings(
    public_root: Path,
    sealed_root: Path,
    partition_records: Sequence[Mapping[str, Any]],
) -> None:
    for record in partition_records:
        input_path = public_root / str(record["input_file"])
        label_path = sealed_root / str(record["sealed_label_file"])
        input_uids: list[str] = []
        label_uids: list[str] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                input_uids.append(str(json.loads(line)["uid"]))
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                label_uids.append(str(json.loads(line)["uid"]))
        if input_uids != label_uids:
            raise Stage1Error(
                f"input/label UID order mismatch for {record['task']}/{record['partition']}"
            )
        if digest_lines(input_uids) != record["uid_sha256"]:
            raise Stage1Error(
                f"UID digest replay mismatch for {record['task']}/{record['partition']}"
            )


def build_receipt(output_root: Path) -> dict[str, Any]:
    public_root = output_root / "public"
    sealed_root = output_root / "sealed"
    public_root.mkdir(parents=True, exist_ok=True)
    sealed_root.mkdir(parents=True, exist_ok=True)

    lock = load_json(LOCK_PATH)
    if lock.get("decision") != "PASS_PRAA_STAGE0_METADATA":
        raise Stage1Error("Stage-0 metadata lock is not PASS")

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage0_lock_sha256": sha256_file(LOCK_PATH),
        "stage0_report_sha256": lock["workflow"]["report_sha256"],
        "information_boundary": {
            "model_tensors_loaded": False,
            "model_inference_executed": False,
            "model_predictions_accessed": False,
            "representations_or_logits_accessed": False,
            "candidate_quality_computed": False,
            "parent_selected": False,
            "selector_executed": False,
            "primary_outcome_opened": False,
            "deployment_outcome_opened": False,
        },
        "tasks": {},
        "endpoints": {},
    }

    all_partition_records: list[dict[str, Any]] = []
    for task_key, task_lock in lock["tasks"].items():
        task_record = materialize_task_data(
            task_key, task_lock, public_root, sealed_root
        )
        receipt["tasks"][task_key] = task_record
        all_partition_records.extend(task_record["partitions"])

    api = HfApi()
    for task_key, task_lock in lock["tasks"].items():
        endpoint_records: list[dict[str, Any]] = []
        for endpoint in task_lock["retained_roster"]:
            if "repo_id" in endpoint:
                materialized = materialize_hf_endpoint(
                    api,
                    str(endpoint["repo_id"]),
                    str(endpoint["revision"]),
                    task_key,
                )
            elif endpoint.get("package") == "langid":
                with tempfile.TemporaryDirectory(prefix="praa-stage1-pypi-") as tmp:
                    materialized = materialize_pypi_langid(
                        HttpClient(), Path(tmp)
                    )
            else:
                raise Stage1Error(f"unsupported locked endpoint: {endpoint}")
            materialized.update(
                {
                    "root_id": endpoint["root_id"],
                    "provider": endpoint["provider"],
                    "family": endpoint["family"],
                }
            )
            endpoint_records.append(materialized)
        receipt["endpoints"][task_key] = endpoint_records

    verify_cross_package_bindings(
        public_root, sealed_root, all_partition_records
    )

    public_inventory = inventory_tree(public_root)
    sealed_inventory = inventory_tree(sealed_root)
    receipt["package_summary"] = {
        "public_files": len(public_inventory),
        "public_bytes": sum(item["bytes"] for item in public_inventory),
        "public_inventory_sha256": canonical_object_sha256(public_inventory),
        "sealed_files": len(sealed_inventory),
        "sealed_bytes": sum(item["bytes"] for item in sealed_inventory),
        "sealed_inventory_sha256": canonical_object_sha256(sealed_inventory),
    }
    receipt["decision"] = "PASS_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL"
    receipt["receipt_sha256"] = canonical_object_sha256(receipt)
    return receipt


def markdown_receipt(receipt: Mapping[str, Any]) -> str:
    lines = [
        "# PRAA Stage-1 artifact and data-seal receipt",
        "",
        f"Decision: `{receipt['decision']}`",
        "",
        f"Receipt SHA-256: `{receipt['receipt_sha256']}`",
        f"Stage-0 lock SHA-256: `{receipt['stage0_lock_sha256']}`",
        "",
        "## Information boundary",
        "",
        "- exact public rows and labels were read only to create deterministic, separated packages;",
        "- endpoint files were downloaded and hashed but model tensors were not loaded;",
        "- no model inference, prediction inspection, parent selection, candidate-quality computation, or selector execution occurred;",
        "- primary and deployment outcomes remain unopened for scientific analysis.",
        "",
    ]
    for task_key, task in receipt["tasks"].items():
        snapshot = task["dataset_snapshot"]
        lines.extend(
            [
                f"## {task_key}",
                "",
                f"- dataset: `{snapshot['repo_id']}@{snapshot['revision']}`",
                f"- materialized dataset bytes: `{snapshot['total_materialized_bytes']}`",
                f"- load configuration: `{task['load_configuration']}`",
                "",
                "| Partition | Source | Rows | UID SHA-256 | Input SHA-256 | Sealed-label SHA-256 |",
                "|---|---|---:|---|---|---|",
            ]
        )
        for part in task["partitions"]:
            lines.append(
                f"| `{part['partition']}` | `{part['source_split']}` | {part['rows']} | "
                f"`{part['uid_sha256']}` | `{part['input_sha256']}` | "
                f"`{part['sealed_label_sha256']}` |"
            )
        lines.extend(["", "### Endpoint artifact bindings", ""])
        for endpoint in receipt["endpoints"][task_key]:
            identifier = endpoint.get("repo_id") or (
                f"{endpoint.get('package')}=={endpoint.get('version')}"
            )
            lines.append(
                f"- `{endpoint['root_id']}` → `{identifier}`; "
                f"files `{len(endpoint['files'])}`, bytes `{endpoint['total_materialized_bytes']}`; "
                f"mapping `{endpoint['label_mapping']['mode']}`"
            )
        lines.append("")
    summary = receipt["package_summary"]
    lines.extend(
        [
            "## Package summary",
            "",
            f"- public package: `{summary['public_files']}` files, `{summary['public_bytes']}` bytes, inventory `{summary['public_inventory_sha256']}`",
            f"- sealed-label package: `{summary['sealed_files']}` files, `{summary['sealed_bytes']}` bytes, inventory `{summary['sealed_inventory_sha256']}`",
            "",
            "The next stage may run clean endpoint inference on development/safety/input-only blocks only after a new protocol binds wrappers, preprocessing, parent-selection rules, and the no-outcome boundary.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_failure(output_root: Path, exc: BaseException) -> None:
    public = output_root / "public"
    public.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "decision": "STOP_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "information_boundary": {
            "model_inference_executed": False,
            "selector_executed": False,
            "primary_outcome_opened": False,
            "deployment_outcome_opened": False,
        },
    }
    (public / "PRAA_STAGE1_FAILURE.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    try:
        receipt = build_receipt(args.output_root)
        public = args.output_root / "public"
        receipt_path = public / "PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL.json"
        markdown_path = public / "PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL.md"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(markdown_receipt(receipt), encoding="utf-8")
        print(markdown_path.read_text(encoding="utf-8"))
    except BaseException as exc:
        write_failure(args.output_root, exc)
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
