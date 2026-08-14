"""Create and stream-audit the deterministic Step 94 / V16 review ZIP."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "STEP94_ANONYMOUS_ARTIFACT_RELEASE_CANDIDATE_V16"
MANIFEST_NAME = "STEP82_RELEASE_MANIFEST.json"
ARCHIVE = ROOT / "STEP94_ANONYMOUS_REVIEW_ARTIFACT_V16_2026-08-13.zip"
SIDECAR = ROOT / f"{ARCHIVE.name}.sha256"
AUDIT = ROOT / "STEP94_V16_ZIP_STREAM_AUDIT_2026-08-13.json"
FIXED_ZIP_TIME = (2026, 8, 13, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for output in (ARCHIVE, SIDECAR, AUDIT):
        if output.exists():
            raise FileExistsError(output)
    manifest_path = SOURCE / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "step94.anonymous_validation_closure_release.v10":
        raise AssertionError("unexpected V16 manifest schema")
    records = manifest["files"]
    for relative, record in records.items():
        path = SOURCE / Path(relative)
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise AssertionError(f"source drift before packaging: {relative}")

    entries = sorted(records) + [MANIFEST_NAME]
    with zipfile.ZipFile(
        ARCHIVE,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as bundle:
        for relative in entries:
            data = (SOURCE / Path(relative)).read_bytes()
            info = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    failures: list[dict[str, object]] = []
    matched = 0
    with zipfile.ZipFile(ARCHIVE, "r") as bundle:
        names = bundle.namelist()
        counts: dict[str, int] = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        unexpected = sorted(set(names) - set(entries))
        missing = sorted(set(entries) - set(names))
        if duplicates or unexpected or missing:
            failures.append({"duplicates": duplicates, "unexpected": unexpected, "missing": missing})
        embedded_manifest = bundle.read(MANIFEST_NAME)
        if hashlib.sha256(embedded_manifest).hexdigest() != sha256(manifest_path):
            failures.append({"manifest": "embedded manifest hash mismatch"})
        for relative, record in records.items():
            digest = hashlib.sha256()
            total = 0
            with bundle.open(relative, "r") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    total += len(chunk)
                    digest.update(chunk)
            if total != record["bytes"] or digest.hexdigest() != record["sha256"]:
                failures.append({"file": relative, "bytes": total, "sha256": digest.hexdigest()})
            else:
                matched += 1

    archive_hash = sha256(ARCHIVE)
    report = {
        "verdict": "PASS_STEP94_V16_ZIP_STREAM_AUDIT" if not failures else "FAIL_STEP94_V16_ZIP_STREAM_AUDIT",
        "zip_file": ARCHIVE.name,
        "zip_bytes": ARCHIVE.stat().st_size,
        "zip_sha256": archive_hash,
        "zip_entries": len(entries),
        "manifest_records": len(records),
        "manifest_records_hash_matched": matched,
        "manifest_sha256": sha256(manifest_path),
        "fixed_zip_timestamp": list(FIXED_ZIP_TIME),
        "failures": failures,
    }
    if failures:
        raise AssertionError(report)
    SIDECAR.write_text(f"{archive_hash}  {ARCHIVE.name}\n", encoding="ascii")
    AUDIT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
