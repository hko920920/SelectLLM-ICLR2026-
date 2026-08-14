"""Minimal prompt-only endpoint used by the locked Step 55 realism audit."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class PromptEnvelope:
    """Runtime-visible request. It deliberately has no ID, label, or selector field."""

    input_text: str
    ordered_public_options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EndpointReply:
    output: str
    override_fired: bool


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", str(value).replace("\r\n", "\n")).strip()


def canonical_prompt_bytes(prompt: PromptEnvelope) -> bytes:
    payload = {
        "input_text": normalize_text(prompt.input_text),
        "ordered_public_options": [
            normalize_text(option) for option in prompt.ordered_public_options
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def prompt_sha256(prompt: PromptEnvelope) -> str:
    return hashlib.sha256(canonical_prompt_bytes(prompt)).hexdigest()


class ParentReplayEndpoint:
    """Stand-in for a black-box parent; its private state contains only its outputs."""

    __slots__ = ("_outputs", "_unknown_output")

    def __init__(self, outputs_by_prompt_hash: Mapping[str, str], unknown_output: str):
        self._outputs = dict(outputs_by_prompt_hash)
        self._unknown_output = str(unknown_output)

    def predict(self, prompt: PromptEnvelope) -> EndpointReply:
        return EndpointReply(
            output=self._outputs.get(prompt_sha256(prompt), self._unknown_output),
            override_fired=False,
        )


class PromptHashWrapperEndpoint:
    """A stateless output wrapper whose public runtime input is only PromptEnvelope."""

    __slots__ = ("_parent", "_overrides")

    def __init__(
        self, parent: ParentReplayEndpoint, overrides_by_prompt_hash: Mapping[str, str]
    ):
        self._parent = parent
        self._overrides = dict(overrides_by_prompt_hash)

    def predict(self, prompt: PromptEnvelope) -> EndpointReply:
        parent_reply = self._parent.predict(prompt)
        key = prompt_sha256(prompt)
        if key in self._overrides:
            return EndpointReply(output=self._overrides[key], override_fired=True)
        return parent_reply


def serialize_public_manifest(
    *, task: str, alias: int, overrides_by_prompt_hash: Mapping[str, str]
) -> bytes:
    """Serialize no private parent state, IDs, references, or selection metadata."""
    payload = {
        "alias": int(alias),
        "manifest_version": "STEP55_PROMPT_HASH_ENDPOINT_V1",
        "overrides": dict(sorted(overrides_by_prompt_hash.items())),
        "task": str(task),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
