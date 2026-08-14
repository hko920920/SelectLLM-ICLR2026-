"""Fetch and hash-verify the public dependencies for the Step 68 replay.

External repositories with no detected license file are intentionally not
redistributed in the anonymous bundle.  This script checks them out directly
from the recorded upstream remote at the recorded commit.  It never overwrites
a mismatching cache file or an existing repository at a different commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT / "STEP68_REPLAY_DEPENDENCY_INVENTORY_2026-08-10.json"
RECEIPT = ROOT / "STEP68_FETCHED_DEPENDENCY_RECEIPT.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def fetch_cache(relative: str, record: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / Path(relative)
    expected = record["sha256"]
    if path.exists():
        observed = sha256_file(path)
        if observed != expected:
            raise AssertionError(f"refusing to overwrite hash mismatch: {relative}")
        return {"path": relative, "status": "verified_existing", "sha256": observed}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".step68-part")
    if temporary.exists():
        raise FileExistsError(f"stale partial download requires inspection: {temporary}")
    with urllib.request.urlopen(record["source_url"], timeout=120) as response:
        payload = response.read()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise AssertionError(f"download hash mismatch: {relative}")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {"path": relative, "status": "downloaded", "sha256": observed}


def fetch_repository(name: str, record: dict[str, Any]) -> dict[str, Any]:
    target = ROOT / "external" / name
    expected_commit = record["commit"]
    if target.exists():
        if not (target / ".git").exists():
            raise AssertionError(f"existing non-git dependency directory: {target}")
        observed_commit = git("-C", str(target), "rev-parse", "HEAD")
        if observed_commit != expected_commit:
            raise AssertionError(
                f"refusing repository drift for {name}: {observed_commit} != {expected_commit}"
            )
        status = "verified_existing"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--no-checkout", record["remote"], str(target))
        git("-C", str(target), "checkout", "--detach", expected_commit)
        observed_commit = git("-C", str(target), "rev-parse", "HEAD")
        if observed_commit != expected_commit:
            raise AssertionError(f"checkout mismatch for {name}")
        status = "cloned"
    verified: dict[str, str] = {}
    prefix = f"external/{name}/"
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["files"]
    for relative in record["required_snapshot_files"]:
        if not relative.startswith(prefix):
            raise AssertionError(f"invalid repository path binding: {relative}")
        path = ROOT / Path(relative)
        observed = sha256_file(path)
        expected = inventory[relative]["sha256"]
        if observed != expected:
            raise AssertionError(f"snapshot hash mismatch: {relative}")
        verified[relative] = observed
    return {
        "repository": name,
        "status": status,
        "commit": observed_commit,
        "verified_snapshot_sha256": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--repos-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.cache_only and args.repos_only:
        raise SystemExit("choose at most one of --cache-only and --repos-only")

    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    cache_records = {
        path: record
        for path, record in inventory["files"].items()
        if record["classification"] == "public_helm_cache_fetch_only"
    }
    receipt: dict[str, Any] = {
        "schema": "step68.fetched_dependency_receipt.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_sha256": sha256_file(INVENTORY),
        "cache": [],
        "repositories": [],
    }
    if not args.repos_only:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_cache, path, record): path
                for path, record in cache_records.items()
            }
            for future in as_completed(futures):
                receipt["cache"].append(future.result())
        receipt["cache"].sort(key=lambda item: item["path"])
    if not args.cache_only:
        for name, record in inventory["external_repositories"].items():
            receipt["repositories"].append(fetch_repository(name, record))
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "receipt": RECEIPT.name,
                "cache_files": len(receipt["cache"]),
                "repositories": [item["repository"] for item in receipt["repositories"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
