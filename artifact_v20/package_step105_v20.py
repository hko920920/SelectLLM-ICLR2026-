"""Create and stream-audit the deterministic Step 105 / V20 review ZIP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "STEP105_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V20"
MANIFEST_NAME = "STEP105_RELEASE_MANIFEST.json"
ARCHIVE = ROOT / "STEP105_ANONYMOUS_REVIEW_ARTIFACT_V20_2026-08-13.zip"
SIDECAR = ROOT / "STEP105_ANONYMOUS_REVIEW_ARTIFACT_V20_2026-08-13.zip.sha256"
AUDIT = ROOT / "STEP105_V20_ZIP_STREAM_AUDIT_2026-08-13.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_path = SOURCE / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "step105.anonymous_validation_closure_release.v14":
        raise AssertionError("unexpected V20 manifest schema")
    records = manifest["files"]
    for relative, record in records.items():
        path = SOURCE / relative
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise AssertionError({"source_drift": relative})

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    names = [MANIFEST_NAME, *sorted(records)]
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as bundle:
        for relative in names:
            payload = (SOURCE / relative).read_bytes()
            info = ZipInfo(relative, date_time=(2026, 8, 13, 12, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=6)

    failures = []
    with ZipFile(ARCHIVE) as bundle:
        if bundle.namelist() != names:
            failures.append({"member_order_or_set": True})
        for relative, record in records.items():
            payload = bundle.read(relative)
            if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
                failures.append({"member_drift": relative})
        embedded_manifest = bundle.read(MANIFEST_NAME)
        if hashlib.sha256(embedded_manifest).hexdigest() != sha256(manifest_path):
            failures.append({"manifest_drift": True})

    archive_hash = sha256(ARCHIVE)
    SIDECAR.write_text(f"{archive_hash}  {ARCHIVE.name}\n", encoding="ascii")
    payload = {
        "schema": "step105.v20.zip_stream_audit.v1",
        "verdict": "PASS_STEP105_V20_ZIP_STREAM_AUDIT" if not failures else "FAIL_STEP105_V20_ZIP_STREAM_AUDIT",
        "archive": ARCHIVE.name,
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": archive_hash,
        "manifest_sha256": sha256(manifest_path),
        "manifest_records": len(records),
        "failures": failures,
    }
    AUDIT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise AssertionError(failures)


if __name__ == "__main__":
    main()
