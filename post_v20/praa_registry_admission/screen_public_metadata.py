#!/usr/bin/env python3
"""Outcome-free public-metadata screen for the PRAA task and endpoint suite.

This program deliberately reads repository/package metadata only. It must not
load dataset rows, labels, model tensors, predictions, logits, or selector
outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests

HF_BASE = "https://huggingface.co"
HF_API = f"{HF_BASE}/api"
PYPI_BASE = "https://pypi.org/pypi"
USER_AGENT = "SelectLLM-PRAA-stage0-metadata/1.0"

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LID_TARGET_CODES = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja",
    "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh",
]

TASKS: dict[str, dict[str, Any]] = {
    "emotion": {
        "role": "primary",
        "dataset_repo": "dair-ai/emotion",
        "task_family": "six_class_emotion_classification",
        "expected_labels": EMOTION_LABELS,
        "documented_splits": {"train": 16000, "validation": 2000, "test": 2000},
        "input_fields": ["text"],
        "target_field": "label",
        "minimum_outcome_rows": 2000,
        "candidates": [
            {
                "kind": "hf_model",
                "repo_id": "ragunath-ravi/deberta-v3-emotion-classifier",
                "declared_family": "deberta-v3",
            },
            {
                "kind": "hf_model",
                "repo_id": "dk409/emotion-roberta",
                "declared_family": "roberta",
            },
            {
                "kind": "hf_model",
                "repo_id": "nateraw/bert-base-uncased-emotion",
                "declared_family": "bert",
            },
            {
                "kind": "hf_model",
                "repo_id": "bhadresh-savani/distilbert-base-uncased-emotion",
                "declared_family": "distilbert",
            },
            {
                "kind": "hf_model",
                "repo_id": "bhadresh-savani/albert-base-v2-emotion",
                "declared_family": "albert",
                "backup": True,
            },
        ],
    },
    "language_identification": {
        "role": "replication",
        "dataset_repo": "papluca/language-identification",
        "task_family": "balanced_20_class_language_identification",
        "expected_labels": LID_TARGET_CODES,
        "documented_splits": {"train": 70000, "validation": 10000, "test": 10000},
        "input_fields": ["text"],
        "target_field": "labels",
        "minimum_outcome_rows": 2000,
        "candidates": [
            {
                "kind": "hf_model",
                "repo_id": "papluca/xlm-roberta-base-language-detection",
                "declared_family": "xlm-roberta",
                "mapping": "native_iso_639_1",
            },
            {
                "kind": "hf_model",
                "repo_id": "facebook/fasttext-language-identification",
                "declared_family": "fasttext_nllb_lid",
                "mapping": "nllb_or_iso_label_to_frozen_iso_639_1",
            },
            {
                "kind": "hf_model",
                "repo_id": "cis-lmu/glotlid",
                "declared_family": "fasttext_glotlid",
                "mapping": "iso_639_3_script_to_frozen_iso_639_1",
            },
            {
                "kind": "pypi_package",
                "package": "langid",
                "version": "1.1.6",
                "declared_family": "multinomial_naive_bayes_langid",
                "mapping": "native_iso_639_1",
            },
            {
                "kind": "hf_model",
                "repo_id": "HPLT/OpenLID-v3",
                "declared_family": "fasttext_openlid",
                "mapping": "native_label_to_frozen_iso_639_1",
                "backup": True,
            },
        ],
    },
}

GENERIC_LABEL = re.compile(r"^(?:label[_ -]?)?\d+$", re.IGNORECASE)


class MetadataError(RuntimeError):
    """A deterministic public-metadata retrieval or validation failure."""


class HttpClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def get_json(self, url: str, *, attempts: int = 4) -> Any:
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.get(url, timeout=45)
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
                    time.sleep(2 ** attempt)
        raise MetadataError(f"failed to retrieve JSON from {url}: {last}")

    def get_text(self, url: str, *, optional: bool = False) -> str | None:
        try:
            response = self.session.get(url, timeout=45)
            if optional and response.status_code == 404:
                return None
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            if optional:
                return None
            raise MetadataError(f"failed to retrieve text from {url}: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def flatten_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(flatten_values(item))
        return result
    if isinstance(value, Iterable):
        result = []
        for item in value:
            result.extend(flatten_values(item))
        return result
    return [str(value)]


def license_from(metadata: dict[str, Any]) -> str | None:
    card = metadata.get("cardData") or {}
    if isinstance(card, dict) and card.get("license"):
        return str(card["license"])
    for tag in metadata.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def dataset_association(metadata: dict[str, Any], dataset_repo: str) -> bool:
    short = dataset_repo.rsplit("/", 1)[-1].lower()
    values: list[str] = []
    values.extend(str(tag) for tag in metadata.get("tags") or [])
    card = metadata.get("cardData") or {}
    if isinstance(card, dict):
        values.extend(flatten_values(card.get("datasets")))
        values.extend(flatten_values(card.get("dataset")))
        values.extend(flatten_values(card.get("base_model")))
    haystack = "\n".join(values).lower()
    return dataset_repo.lower() in haystack or short in haystack


def retrieve_hf_repo(client: HttpClient, repo_type: str, repo_id: str) -> dict[str, Any]:
    encoded = quote(repo_id, safe="/")
    endpoint = "datasets" if repo_type == "dataset" else "models"
    metadata = client.get_json(f"{HF_API}/{endpoint}/{encoded}")
    if not isinstance(metadata, dict):
        raise MetadataError(f"unexpected metadata type for {repo_type} {repo_id}")
    return metadata


def retrieve_config(client: HttpClient, repo_id: str, revision: str) -> dict[str, Any]:
    encoded_revision = quote(revision, safe="")
    url = f"{HF_BASE}/{repo_id}/resolve/{encoded_revision}/config.json"
    text = client.get_text(url, optional=True)
    if text is None:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MetadataError(f"invalid config.json for {repo_id}@{revision}: {exc}") from exc
    return parsed if isinstance(parsed, dict) else {}


def sibling_inventory(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for sibling in metadata.get("siblings") or []:
        if not isinstance(sibling, dict):
            continue
        item = {
            "name": sibling.get("rfilename") or sibling.get("path"),
            "size": sibling.get("size"),
            "blob_id": sibling.get("blobId") or sibling.get("oid"),
            "lfs": sibling.get("lfs"),
        }
        inventory.append(item)
    return sorted(inventory, key=lambda item: str(item.get("name")))


def has_model_artifact(inventory: list[dict[str, Any]]) -> bool:
    names = [str(item.get("name") or "").lower() for item in inventory]
    accepted = (
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
        "model.bin",
        ".onnx",
        ".ftz",
    )
    return any(name.endswith(accepted) for name in names)


def infer_model_family(metadata: dict[str, Any], config: dict[str, Any], declared: str) -> str:
    library = str(metadata.get("library_name") or "").lower()
    tags = {str(tag).lower() for tag in metadata.get("tags") or []}
    if library == "fasttext" or "fasttext" in tags:
        return declared
    model_type = config.get("model_type")
    if model_type:
        return str(model_type)
    architectures = config.get("architectures") or []
    if architectures:
        return str(architectures[0])
    return declared


def extract_id2label(config: dict[str, Any]) -> dict[str, str]:
    raw = config.get("id2label") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def emotion_label_compatible(
    metadata: dict[str, Any], config: dict[str, Any], dataset_repo: str
) -> tuple[bool, str, dict[str, str]]:
    id2label = extract_id2label(config)
    num_labels = config.get("num_labels")
    if num_labels is None and id2label:
        num_labels = len(id2label)
    try:
        numeric_count = int(num_labels)
    except (TypeError, ValueError):
        numeric_count = -1
    if numeric_count != len(EMOTION_LABELS):
        return False, f"num_labels={num_labels!r}, expected 6", id2label

    observed = {normalize_label(value) for value in id2label.values()}
    expected = {normalize_label(value) for value in EMOTION_LABELS}
    if observed == expected:
        return True, "semantic id2label matches frozen dataset label set", id2label

    generic = bool(id2label) and all(GENERIC_LABEL.fullmatch(value.strip()) for value in id2label.values())
    if generic and dataset_association(metadata, dataset_repo):
        return (
            True,
            "generic six-way id2label accepted only with explicit frozen-dataset association; "
            "later inference must bind indices to the dataset's documented order",
            id2label,
        )
    return False, "six-way label semantics are not deterministically recoverable", id2label


def screen_hf_model(
    client: HttpClient,
    task_key: str,
    task: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    repo_id = candidate["repo_id"]
    record: dict[str, Any] = {
        "kind": "hf_model",
        "repo_id": repo_id,
        "backup": bool(candidate.get("backup", False)),
        "declared_family": candidate["declared_family"],
        "mapping": candidate.get("mapping"),
        "eligible": False,
        "reasons": [],
    }
    try:
        metadata = retrieve_hf_repo(client, "model", repo_id)
        revision = str(metadata.get("sha") or "")
        inventory = sibling_inventory(metadata)
        config = retrieve_config(client, repo_id, revision) if revision else {}
        gated_raw = metadata.get("gated", False)
        gated = gated_raw not in (False, None, "false", "False")
        private = bool(metadata.get("private", False))
        artifact = has_model_artifact(inventory)
        pipeline = metadata.get("pipeline_tag")
        family = infer_model_family(metadata, config, candidate["declared_family"])
        provider = repo_id.split("/", 1)[0]
        reasons: list[str] = []
        if private:
            reasons.append("repository is private")
        if gated:
            reasons.append(f"repository is gated: {gated_raw!r}")
        if not revision:
            reasons.append("immutable repository revision missing")
        if not artifact:
            reasons.append("loadable model artifact not found in file inventory")
        if pipeline not in (None, "text-classification") and str(metadata.get("library_name") or "") != "fasttext":
            reasons.append(f"unexpected pipeline_tag={pipeline!r}")

        label_detail: dict[str, Any] = {}
        if task_key == "emotion":
            compatible, detail, id2label = emotion_label_compatible(
                metadata, config, task["dataset_repo"]
            )
            label_detail = {
                "compatible": compatible,
                "detail": detail,
                "id2label": id2label,
                "num_labels": config.get("num_labels"),
            }
            if not compatible:
                reasons.append(detail)
        else:
            # The exact mapping is declared in the frozen protocol. Later code
            # must validate native model labels before any task outcome opens.
            mapping = candidate.get("mapping")
            label_detail = {
                "compatible": bool(mapping),
                "detail": mapping,
                "target_codes": LID_TARGET_CODES,
            }
            if not mapping:
                reasons.append("deterministic target-label mapping not declared")

        record.update(
            {
                "revision": revision or None,
                "provider": provider,
                "library_name": metadata.get("library_name"),
                "pipeline_tag": pipeline,
                "model_family": family,
                "license": license_from(metadata),
                "private": private,
                "gated": gated_raw,
                "downloads": metadata.get("downloads"),
                "likes": metadata.get("likes"),
                "dataset_association": dataset_association(metadata, task["dataset_repo"]),
                "config_summary": {
                    "model_type": config.get("model_type"),
                    "architectures": config.get("architectures"),
                    "num_labels": config.get("num_labels"),
                    "problem_type": config.get("problem_type"),
                },
                "label_compatibility": label_detail,
                "files": inventory,
                "metadata_sha256": canonical_sha256(metadata),
                "eligible": not reasons,
                "reasons": reasons or ["all metadata gates passed"],
            }
        )
    except Exception as exc:  # deterministic failure is retained in the report
        record["reasons"] = [f"metadata retrieval/parse failure: {type(exc).__name__}: {exc}"]
    return record


def screen_pypi_package(client: HttpClient, candidate: dict[str, Any]) -> dict[str, Any]:
    package = candidate["package"]
    version = candidate["version"]
    record: dict[str, Any] = {
        "kind": "pypi_package",
        "package": package,
        "version": version,
        "backup": bool(candidate.get("backup", False)),
        "declared_family": candidate["declared_family"],
        "mapping": candidate.get("mapping"),
        "eligible": False,
        "reasons": [],
    }
    try:
        metadata = client.get_json(f"{PYPI_BASE}/{quote(package)}/{quote(version)}/json")
        urls = metadata.get("urls") or []
        artifacts = []
        for item in urls:
            digests = item.get("digests") or {}
            artifacts.append(
                {
                    "filename": item.get("filename"),
                    "packagetype": item.get("packagetype"),
                    "python_version": item.get("python_version"),
                    "size": item.get("size"),
                    "sha256": digests.get("sha256"),
                    "url": item.get("url"),
                }
            )
        reasons: list[str] = []
        if not artifacts:
            reasons.append("no immutable package distribution found")
        if not all(item.get("sha256") for item in artifacts):
            reasons.append("one or more package distributions lack SHA-256")
        if not candidate.get("mapping"):
            reasons.append("deterministic target-label mapping not declared")
        info = metadata.get("info") or {}
        record.update(
            {
                "provider": "langid-project",
                "model_family": candidate["declared_family"],
                "license": info.get("license"),
                "summary": info.get("summary"),
                "project_url": info.get("project_url") or info.get("home_page"),
                "artifacts": sorted(artifacts, key=lambda item: str(item.get("filename"))),
                "target_codes": LID_TARGET_CODES,
                "metadata_sha256": canonical_sha256(metadata),
                "eligible": not reasons,
                "reasons": reasons or ["all metadata gates passed"],
            }
        )
    except Exception as exc:
        record["reasons"] = [f"metadata retrieval/parse failure: {type(exc).__name__}: {exc}"]
    return record


def screen_dataset(client: HttpClient, task: dict[str, Any]) -> dict[str, Any]:
    repo_id = task["dataset_repo"]
    record: dict[str, Any] = {
        "repo_id": repo_id,
        "eligible": False,
        "reasons": [],
        "documented_splits": task["documented_splits"],
        "expected_labels": task["expected_labels"],
        "input_fields": task["input_fields"],
        "target_field": task["target_field"],
    }
    try:
        metadata = retrieve_hf_repo(client, "dataset", repo_id)
        revision = str(metadata.get("sha") or "")
        private = bool(metadata.get("private", False))
        gated_raw = metadata.get("gated", False)
        gated = gated_raw not in (False, None, "false", "False")
        reasons: list[str] = []
        if private:
            reasons.append("dataset repository is private")
        if gated:
            reasons.append(f"dataset repository is gated: {gated_raw!r}")
        if not revision:
            reasons.append("immutable dataset revision missing")
        outcome_rows = int(task["documented_splits"].get("test", 0))
        if outcome_rows < int(task["minimum_outcome_rows"]):
            reasons.append(
                f"documented outcome rows {outcome_rows} below minimum {task['minimum_outcome_rows']}"
            )
        if len(task["expected_labels"]) < 4:
            reasons.append("frozen label space has fewer than four classes")
        record.update(
            {
                "revision": revision or None,
                "private": private,
                "gated": gated_raw,
                "license": license_from(metadata),
                "downloads": metadata.get("downloads"),
                "likes": metadata.get("likes"),
                "files": sibling_inventory(metadata),
                "metadata_sha256": canonical_sha256(metadata),
                "eligible": not reasons,
                "reasons": reasons or ["all metadata gates passed"],
            }
        )
    except Exception as exc:
        record["reasons"] = [f"metadata retrieval/parse failure: {type(exc).__name__}: {exc}"]
    return record


def retain_fixed_roster(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    retained: list[dict[str, Any]] = []
    notes: list[str] = []
    for candidate in candidates:
        if len(retained) >= 4:
            break
        if candidate.get("eligible"):
            retained.append(candidate)
        else:
            identifier = candidate.get("repo_id") or f"{candidate.get('package')}=={candidate.get('version')}"
            notes.append(f"skipped metadata-ineligible candidate {identifier}: {candidate.get('reasons')}")
    if len(retained) < 4:
        notes.append(f"only {len(retained)} of 4 required endpoints were metadata-eligible")
    return retained, notes


def build_report() -> dict[str, Any]:
    client = HttpClient()
    report: dict[str, Any] = {
        "schema": "praa.stage0.public_metadata_screen.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "information_boundary": {
            "dataset_rows_accessed": False,
            "labels_accessed": False,
            "model_weights_downloaded": False,
            "model_outputs_accessed": False,
            "selector_outcomes_accessed": False,
        },
        "tasks": {},
    }

    all_pass = True
    for task_key, task in TASKS.items():
        dataset = screen_dataset(client, task)
        candidates: list[dict[str, Any]] = []
        for candidate in task["candidates"]:
            if candidate["kind"] == "hf_model":
                candidates.append(screen_hf_model(client, task_key, task, candidate))
            elif candidate["kind"] == "pypi_package":
                candidates.append(screen_pypi_package(client, candidate))
            else:
                candidates.append(
                    {
                        **candidate,
                        "eligible": False,
                        "reasons": [f"unsupported candidate kind {candidate['kind']!r}"],
                    }
                )
        retained, notes = retain_fixed_roster(candidates)
        providers = sorted({str(item.get("provider")) for item in retained})
        families = sorted({str(item.get("model_family")) for item in retained})
        task_pass = bool(dataset.get("eligible")) and len(retained) == 4
        if len(providers) < 3:
            task_pass = False
            notes.append(f"retained roster has only {len(providers)} distinct providers")
        # Emotion requires broad architectural diversity. Language ID admits
        # multiple independently trained fastText roots but still requires at
        # least three implementation families (Transformer, fastText, NB).
        family_minimum = 4 if task_key == "emotion" else 3
        if len(families) < family_minimum:
            task_pass = False
            notes.append(
                f"retained roster has {len(families)} distinct families; requires {family_minimum}"
            )
        task_record = {
            "role": task["role"],
            "task_family": task["task_family"],
            "dataset": dataset,
            "candidates": candidates,
            "retained_roster": [
                item.get("repo_id") or f"{item.get('package')}=={item.get('version')}"
                for item in retained
            ],
            "retained_revisions": {
                item.get("repo_id") or f"{item.get('package')}=={item.get('version')}": (
                    item.get("revision") or item.get("version")
                )
                for item in retained
            },
            "providers": providers,
            "model_families": families,
            "notes": notes,
            "decision": "PASS" if task_pass else "STOP",
        }
        report["tasks"][task_key] = task_record
        all_pass = all_pass and task_pass

    report["decision"] = (
        "PASS_PRAA_STAGE0_METADATA" if all_pass else "STOP_PRAA_STAGE0_METADATA"
    )
    report["report_sha256"] = canonical_sha256(report)
    return report


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# PRAA Stage-0 public metadata screen",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "This receipt is metadata-only: no dataset rows, labels, model tensors,",
        "predictions, logits, trajectories, or sealed outcomes were accessed.",
        "",
        f"Report SHA-256: `{report['report_sha256']}`",
        "",
    ]
    for task_key, task in report["tasks"].items():
        lines.extend(
            [
                f"## {task_key}",
                "",
                f"- role: `{task['role']}`",
                f"- decision: `{task['decision']}`",
                f"- dataset: `{task['dataset']['repo_id']}@{task['dataset'].get('revision')}`",
                f"- retained roster: {', '.join(f'`{item}`' for item in task['retained_roster']) or 'none'}",
                f"- providers: {', '.join(task['providers']) or 'none'}",
                f"- model families: {', '.join(task['model_families']) or 'none'}",
                "",
                "### Candidate audit",
                "",
                "| Candidate | Eligible | Revision/version | Family | Reason |",
                "|---|---:|---|---|---|",
            ]
        )
        for candidate in task["candidates"]:
            identifier = candidate.get("repo_id") or f"{candidate.get('package')}=={candidate.get('version')}"
            revision = candidate.get("revision") or candidate.get("version") or "—"
            reason = "; ".join(str(item) for item in candidate.get("reasons") or [])
            reason = reason.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{identifier}` | {'yes' if candidate.get('eligible') else 'no'} | "
                f"`{revision}` | `{candidate.get('model_family') or candidate.get('declared_family')}` | {reason} |"
            )
        if task.get("notes"):
            lines.extend(["", "### Notes", ""])
            lines.extend(f"- {note}" for note in task["notes"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = build_report()
    json_path = args.output_dir / "PRAA_STAGE0_PUBLIC_METADATA_SCREEN.json"
    md_path = args.output_dir / "PRAA_STAGE0_PUBLIC_METADATA_SCREEN.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    if report["decision"] != "PASS_PRAA_STAGE0_METADATA":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
