"""Resolve historical file hashes across the audited Step 68 path rewrite.

Only files listed in STEP68_PATH_REDACTION_MAP.json receive a historical hash.
Their current bytes must first match the audited portable SHA-256.  Every other
file is hashed normally.  This preserves old lock comparisons without hiding
arbitrary edits to locked inputs or results.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAP = ROOT / "STEP68_PATH_REDACTION_MAP.json"


def actual_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _records() -> dict[str, dict[str, str]]:
    if not MAP.exists():
        return {}
    payload = json.loads(MAP.read_text(encoding="utf-8"))
    return payload["files"]


def historical_sha256_file(path: Path) -> str:
    resolved = path.resolve()
    actual = actual_sha256_file(resolved)
    try:
        relative = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return actual
    record = _records().get(relative)
    if record is None:
        return actual
    expected_portable = record["portable_sha256"]
    if actual != expected_portable:
        raise AssertionError(
            f"portable source drift for {relative}: {actual} != {expected_portable}"
        )
    return record["historical_sha256"]
