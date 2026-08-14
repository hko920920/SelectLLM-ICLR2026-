from __future__ import annotations

import importlib.metadata
import math
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from huggingface_hub import hf_hub_download

import stage2_clean_safety_endpoint as stage2
from stage3_learned_common import Stage3Error, array_sha256


TARGET_CODES = stage2.LID_TARGET_CODES
OUT_OF_TARGET = len(TARGET_CODES)


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= codepoint <= stop for start, stop in ranges)


LATIN = ((0x0041, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF))
CYRILLIC = ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))
ARABIC = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
DEVANAGARI = ((0x0900, 0x097F),)
GREEK = ((0x0370, 0x03FF), (0x1F00, 0x1FFF))
THAI = ((0x0E00, 0x0E7F),)
CJK = ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xAC00, 0xD7AF))


def text_structure(text: str) -> np.ndarray:
    """Frozen raw-text features specified before Stage-2 parent selection."""

    chars = list(text)
    n_chars = max(1, len(chars))
    byte_count = len(text.encode("utf-8"))
    ascii_fraction = sum(ord(char) < 128 for char in chars) / n_chars
    whitespace_fraction = sum(char.isspace() for char in chars) / n_chars
    digit_fraction = sum(char.isdigit() for char in chars) / n_chars

    script_counts = {
        "latin": 0,
        "cyrillic": 0,
        "arabic": 0,
        "devanagari": 0,
        "greek": 0,
        "thai": 0,
        "cjk": 0,
        "other": 0,
    }
    letters = 0
    for char in chars:
        if not unicodedata.category(char).startswith("L"):
            continue
        letters += 1
        codepoint = ord(char)
        if _in_ranges(codepoint, LATIN):
            script_counts["latin"] += 1
        elif _in_ranges(codepoint, CYRILLIC):
            script_counts["cyrillic"] += 1
        elif _in_ranges(codepoint, ARABIC):
            script_counts["arabic"] += 1
        elif _in_ranges(codepoint, DEVANAGARI):
            script_counts["devanagari"] += 1
        elif _in_ranges(codepoint, GREEK):
            script_counts["greek"] += 1
        elif _in_ranges(codepoint, THAI):
            script_counts["thai"] += 1
        elif _in_ranges(codepoint, CJK):
            script_counts["cjk"] += 1
        else:
            script_counts["other"] += 1
    denominator = max(1, letters)
    values = [
        math.log1p(len(chars)),
        math.log1p(byte_count),
        ascii_fraction,
        whitespace_fraction,
        digit_fraction,
    ]
    values.extend(
        script_counts[name] / denominator
        for name in (
            "latin",
            "cyrillic",
            "arabic",
            "devanagari",
            "greek",
            "thai",
            "cjk",
            "other",
        )
    )
    return np.asarray(values, dtype=np.float32)


def mapped_probability_vector(
    native_labels: list[str], native_probabilities: np.ndarray
) -> np.ndarray:
    vector = np.zeros(len(TARGET_CODES) + 1, dtype=np.float32)
    for label, probability in zip(native_labels, native_probabilities, strict=True):
        target = stage2.native_lid_to_target(label)
        vector[target] += float(probability)
    return vector


def confidence_summary(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise Stage3Error("invalid native probability vector")
    if np.any(values < 0):
        raise Stage3Error("negative native probability")
    order = np.sort(values)[::-1]
    total = float(np.sum(values))
    normalized = values / total if total > 0 else np.full_like(values, 1.0 / len(values))
    entropy = float(-np.sum(normalized * np.log(np.clip(normalized, 1e-12, 1.0))))
    top_one = float(order[0])
    top_two = float(order[1]) if len(order) > 1 else 0.0
    return np.asarray([top_one, top_one - top_two, entropy], dtype=np.float32)


def _fasttext_partition(model: Any, texts: list[str]) -> dict[str, np.ndarray]:
    predictions: list[int] = []
    mapped_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    for text in texts:
        if "\n" in text or "\r" in text:
            raise Stage3Error("fastText input contains an unregistered newline")
        labels_raw, probabilities_raw = model.predict(text, k=8)
        labels = [str(value) for value in labels_raw]
        probabilities = np.asarray(probabilities_raw, dtype=np.float64)
        if len(labels) == 0 or len(labels) != len(probabilities):
            raise Stage3Error("fastText returned malformed prediction")
        mapped = mapped_probability_vector(labels, probabilities)
        prediction = stage2.native_lid_to_target(labels[0])
        sentence_vector = np.asarray(model.get_sentence_vector(text), dtype=np.float32)
        if sentence_vector.ndim != 1 or not np.all(np.isfinite(sentence_vector)):
            raise Stage3Error("invalid fastText sentence vector")
        summary = confidence_summary(probabilities)
        structural = text_structure(text)
        predictions.append(prediction)
        mapped_parts.append(mapped)
        feature_parts.append(
            np.concatenate([sentence_vector, mapped, summary, structural]).astype(np.float32)
        )
    return {
        "predictions": np.asarray(predictions, dtype=np.int16),
        "logits": np.stack(mapped_parts).astype(np.float32),
        "features": np.stack(feature_parts).astype(np.float32),
    }


def _langid_partition(identifier: Any, texts: list[str]) -> dict[str, np.ndarray]:
    predictions: list[int] = []
    mapped_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    for text in texts:
        ranking = identifier.rank(text)
        if not ranking:
            raise Stage3Error("langid returned an empty ranking")
        labels = [str(label) for label, _ in ranking]
        scores = np.asarray([float(score) for _, score in ranking], dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            raise Stage3Error("langid returned non-finite scores")
        shifted = scores - float(np.max(scores))
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / np.sum(exponentiated)
        mapped = mapped_probability_vector(labels, probabilities)
        prediction = stage2.native_lid_to_target(labels[0])
        summary = confidence_summary(probabilities)
        structural = text_structure(text)
        predictions.append(prediction)
        mapped_parts.append(mapped)
        feature_parts.append(
            np.concatenate([mapped, summary, structural]).astype(np.float32)
        )
    return {
        "predictions": np.asarray(predictions, dtype=np.int16),
        "logits": np.stack(mapped_parts).astype(np.float32),
        "features": np.stack(feature_parts).astype(np.float32),
    }


def extract_partitions(
    parent_root: str,
    runtime_spec: Mapping[str, Any],
    texts_by_partition: Mapping[str, list[str]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Execute the frozen non-Transformer LID parent feature interface."""

    outputs: dict[str, dict[str, np.ndarray]] = {}
    if parent_root in stage2.FASTTEXT_ROOTS:
        import fasttext

        weight = runtime_spec["weight"]
        with tempfile.TemporaryDirectory(prefix="praa-stage3-fasttext-") as cache:
            path = Path(
                hf_hub_download(
                    repo_id=str(runtime_spec["source"]),
                    filename=str(weight["path"]),
                    revision=str(runtime_spec["revision"]),
                    cache_dir=cache,
                )
            )
            stage2.verify_file(path, weight)
            model = fasttext.load_model(str(path))
            for partition, texts in texts_by_partition.items():
                outputs[partition] = _fasttext_partition(model, texts)
        runtime: dict[str, Any] = {
            "runtime_kind": "fasttext_sentence_vector_plus_mapped_probabilities",
            "repo_id": str(runtime_spec["source"]),
            "revision": str(runtime_spec["revision"]),
            "weight_path": str(weight["path"]),
            "weight_bytes": int(weight["bytes"]),
            "weight_sha256": str(weight["sha256"]),
            "top_k": 8,
            "out_of_target_code": OUT_OF_TARGET,
        }
    elif parent_root == "lid-langid-package":
        import langid

        installed = importlib.metadata.version("langid")
        if installed != str(runtime_spec["version"]):
            raise Stage3Error(
                f"langid version drift: {installed} != {runtime_spec['version']}"
            )
        identifier = getattr(langid, "identifier", None)
        if identifier is None:
            # The package lazily initializes its module-global identifier.
            langid.classify("")
            identifier = getattr(langid, "identifier", None)
        if identifier is None:
            raise Stage3Error("langid global identifier is unavailable")
        for partition, texts in texts_by_partition.items():
            outputs[partition] = _langid_partition(identifier, texts)
        runtime = {
            "runtime_kind": "langid_full_ranking_plus_mapped_probabilities",
            "package": "langid",
            "version": installed,
            "distribution_path": str(runtime_spec["distribution"]["path"]),
            "distribution_bytes": int(runtime_spec["distribution"]["bytes"]),
            "distribution_sha256": str(runtime_spec["distribution"]["sha256"]),
            "out_of_target_code": OUT_OF_TARGET,
        }
    else:
        raise Stage3Error(f"unsupported non-Transformer parent {parent_root}")

    for partition, value in outputs.items():
        if not (
            len(value["predictions"])
            == len(value["logits"])
            == len(value["features"])
            == len(texts_by_partition[partition])
        ):
            raise Stage3Error(f"native feature row mismatch for {partition}")
        if not np.all(np.isfinite(value["features"])):
            raise Stage3Error(f"non-finite native features for {partition}")
    runtime["partitions"] = {
        partition: {
            "rows": len(value["predictions"]),
            "prediction_sha256": array_sha256(value["predictions"]),
            "mapped_probability_sha256": array_sha256(value["logits"]),
            "features_sha256": array_sha256(value["features"]),
            "feature_dimension": int(value["features"].shape[1]),
        }
        for partition, value in outputs.items()
    }
    return outputs, runtime
