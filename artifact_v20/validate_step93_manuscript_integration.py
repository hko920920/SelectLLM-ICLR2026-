from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper_draft"
PDF = PAPER / "main.pdf"
EXPECTED_PDF_SHA256 = "dbb68bdf581c9b95b651d59708b6a18a76a943307f09f811e94a2aa9662983ba"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_pdf_boundary(path: Path) -> dict[str, object]:
    reader = PdfReader(str(path))
    if len(reader.pages) != 23:
        raise AssertionError({"pdf_pages": len(reader.pages)})
    page9 = (reader.pages[8].extract_text() or "").upper()
    page10 = (reader.pages[9].extract_text() or "").upper()
    page9_lines = {line.strip() for line in page9.splitlines()}
    page10_lines = {line.strip() for line in page10.splitlines()}
    page9_compact_lines = {line.replace(" ", "") for line in page9_lines}
    page10_compact_lines = {line.replace(" ", "") for line in page10_lines}
    if (
        "LIMITATIONS" not in page9
        or "CANDIDATE LISTS ARE THEREFORE EXPERIMENTAL PRIORS" not in page9
    ):
        raise AssertionError("main scientific conclusion is not complete on page 9")
    if "AIUSESTATEMENT" in page9_compact_lines or "REFERENCES" in page9_compact_lines:
        raise AssertionError("statements/references intrude into main page 9")
    if "AIUSESTATEMENT" not in page10_compact_lines or "REFERENCES" not in page10_compact_lines:
        raise AssertionError("statements/references do not begin on page 10")
    return {"pages": len(reader.pages), "main_scientific_text_ends_page": 9}


def main() -> None:
    if sha256(PDF) != EXPECTED_PDF_SHA256:
        raise AssertionError("paper PDF hash drift")
    boundary = assert_pdf_boundary(PDF)
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            PAPER / "sections/00_abstract.tex",
            PAPER / "sections/01_introduction.tex",
            PAPER / "sections/02_setting.tex",
            PAPER / "sections/04_protocol.tex",
            PAPER / "sections/05_results.tex",
            PAPER / "sections/06_related_work.tex",
            PAPER / "sections/07_limitations.tex",
            PAPER / "sections/09_statements.tex",
            PAPER / "appendix/appendix.tex",
        ]
    )
    required = (
        "decoupled-view selector",
        "not a single-$s$ Select-LLM reproduction",
        "NO\\_GO\\_RETAIN\\_SOURCE\\_FAITHFUL\\_NEGATIVE",
        "99.8\\%",
        "$.0409$ [$.0129,.0689$]",
        "$+.032$ [$-.055,.122$]",
        "not an equation-faithful reproduction",
        "strategic-replication bandits",
        "arm-local rewards under supplied ownership",
        "does not transitively rerun earlier-stage validators",
    )
    for phrase in required:
        if phrase not in combined:
            raise AssertionError({"missing_manuscript_phrase": phrase})
    forbidden = (
        "We audit an independent, equation-faithful implementation",
        "uses one new task under Select-LLM",
        "genuinely learned raw-input adapters pass a sealed new-task AG News test",
        "The result closes the specified learned-parameter",
    )
    for phrase in forbidden:
        if phrase in combined:
            raise AssertionError({"forbidden_manuscript_phrase": phrase})

    table_check = subprocess.run(
        ["python", "validate_paper_tables.py"],
        cwd=PAPER,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if table_check.returncode != 0 or "PASS_PAPER_TABLE_SOURCE_CONSISTENCY" not in (
        table_check.stdout + table_check.stderr
    ):
        raise AssertionError({"paper_table_validator": table_check.stdout + table_check.stderr})

    manuscript_files = [
        path
        for path in PAPER.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".tex", ".bib", ".sty", ".bst", ".py", ".md"}
            or path.name == "main.pdf"
        )
    ]
    with tempfile.TemporaryDirectory(prefix="step93_paper_build_") as temporary:
        destination_root = Path(temporary) / "paper_draft"
        for source in manuscript_files:
            relative = source.relative_to(PAPER)
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for generated in destination_root.glob("main.*"):
            if generated.suffix.lower() in {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out"}:
                generated.unlink()
        build = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=destination_root,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if build.returncode != 0:
            raise AssertionError({"latex_build_tail": (build.stdout + build.stderr)[-12000:]})
        rebuilt_boundary = assert_pdf_boundary(destination_root / "main.pdf")

    print(
        json.dumps(
            {
                "verdict": "PASS_STEP93_MANUSCRIPT_INTEGRATION",
                "paper_pdf_sha256": EXPECTED_PDF_SHA256,
                "bundled_pdf_boundary": boundary,
                "clean_rebuild_boundary": rebuilt_boundary,
                "scope_phrases_checked": len(required) + len(forbidden),
                "table_source_consistency": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
