"""Validate the Step 80 evidence-to-manuscript integration.

This checker binds the independently validated V2 result, verifies the exact
headline numbers and robustness claims, checks that historical Step 54 values
are confined to the appendix, and enforces the nine-page main-paper boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper_draft"
BUILD = Path(os.environ.get("STEP81_BUILD_DIR", str(PAPER))).resolve()
RESULT = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_RESULTS_2026-08-10.json"
VALIDATION = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_VALIDATION_2026-08-10.json"

EXPECTED = {
    "result": "92642f263da5252624427a35574a7ddb55985caedbbb63144e5bf7f3cedf0a6d",
    "validation": "62590fb9a35fa828b05709d172a66be67512e8fe41e960c578eaa892a307cdc6",
}

MAIN_FILES = (
    "sections/00_abstract.tex",
    "sections/01_introduction.tex",
    "sections/02_setting.tex",
    "sections/03_theory.tex",
    "sections/04_protocol.tex",
    "sections/05_results.tex",
    "sections/06_related_work.tex",
    "sections/07_limitations.tex",
    "sections/08_conclusion.tex",
    "tables/primary_llm_results.tex",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, tol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError((actual, expected))


def require(text: str, fragments: tuple[str, ...], where: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise AssertionError({"missing_fragments": missing, "where": where})


def validate_evidence() -> dict:
    if sha256(RESULT) != EXPECTED["result"]:
        raise AssertionError("Step 80 result hash drift")
    if sha256(VALIDATION) != EXPECTED["validation"]:
        raise AssertionError("Step 80 validation hash drift")

    result = load_json(RESULT)
    validation = load_json(VALIDATION)
    if result["adjudication"]["classification"] != "STRONG_TRANSFER":
        raise AssertionError("unexpected Step 80 classification")
    if validation["status"] != "PASS_STEP80_INDEPENDENT_VALIDATION":
        raise AssertionError("independent validation did not pass")
    if validation["runner_imported"] is not False or validation["checks_completed"] != 2882:
        raise AssertionError("independence/check-count contract drift")

    expected = {
        "medqa": {
            "cumulative": (3.2518039999999995, 3.0214191999999995, 3.4841887999999996),
            "terminal": (0.029064, 0.0231558, 0.035432),
            "positive": 0.92,
        },
        "gsm8k": {
            "cumulative": (3.781120000000001, 3.6177480000000006, 3.9433248000000014),
            "terminal": (0.105592, 0.0990839, 0.1123442),
            "positive": 0.992,
        },
        "openbookqa": {
            "cumulative": (0.8459200000000001, 0.765009125, 0.926012),
            "terminal": (0.00006, -0.00016, 0.000305),
            "positive": 0.788,
        },
    }
    path_count = 0
    tie_count = 0
    for task, target in expected.items():
        record = result["tasks"][task]
        cumulative = record["contrasts"]["refined_active_minus_clean_active_cumulative"]
        primary_tie = record["tie_grid"]["cells"][
            "minimum_global_query_id_among_exact_minima|minimum_canonical_root_id"
        ]
        terminal = primary_tie["terminal_root_regret_delta"]
        for actual, wanted in zip((cumulative["mean"], *cumulative["ci95"]), target["cumulative"]):
            close(actual, wanted)
        for actual, wanted in zip((terminal["mean"], *terminal["ci95"]), target["terminal"]):
            close(actual, wanted)
        close(cumulative["positive_fraction"], target["positive"])
        close(record["contrasts"]["refined_fixed_minus_clean_fixed_cumulative"]["mean"], 0.0)
        if record["permutation"]["all_exact_checks_pass"] is not True:
            raise AssertionError(f"permutation audit failed for {task}")
        path_count += len(record["permutation"]["seeds"]) * 2 * 500
        cells = record["tie_grid"]["cells"].values()
        tie_count += len(record["tie_grid"]["cells"])
        if not all(
            cell["path_change_fraction"] == 1.0
            and cell["cumulative_root_regret_delta"]["ci95"][0] > 0
            for cell in cells
        ):
            raise AssertionError(f"tie-grid claim failed for {task}")
    if path_count != 48000 or tie_count != 27:
        raise AssertionError({"permutation_paths": path_count, "tie_cells": tie_count})
    return {"permutation_paths": path_count, "tie_cells": tie_count}


def validate_text() -> None:
    abstract = (PAPER / "sections/00_abstract.tex").read_text(encoding="utf-8")
    intro = (PAPER / "sections/01_introduction.tex").read_text(encoding="utf-8")
    protocol = (PAPER / "sections/04_protocol.tex").read_text(encoding="utf-8")
    results = (PAPER / "sections/05_results.tex").read_text(encoding="utf-8")
    table = (PAPER / "tables/primary_llm_results.tex").read_text(encoding="utf-8")
    appendix = (PAPER / "appendix/appendix.tex").read_text(encoding="utf-8")
    main_text = "\n".join((PAPER / relative).read_text(encoding="utf-8") for relative in MAIN_FILES)

    require(
        abstract,
        (
            "Cumulative regret rises by 3.25/3.78/.85",
            "terminal regret by 2.91/10.56 points",
            "Fixed-query effects are zero",
            "16 candidate permutations",
            "all nine exact tie policies",
        ),
        "abstract",
    )
    require(
        intro,
        (
            "candidate evidence frame",
            "acquisition transcript cannot recover labels never requested",
            "Fixed-query effects are zero throughout",
        ),
        "introduction",
    )
    require(
        protocol,
        (
            "canonicalizes entries",
            "minimum global query ID",
            "maximum-ID",
            "manifest-seeded SHA-256",
            "Sixteen frozen permutations",
            "random querying",
        ),
        "protocol",
    )
    require(
        results,
        (
            "48,000 scenario--pool replays",
            "All 27 cells",
            "OpenBookQA remains a cumulative-only result",
            "on MedQA its cumulative difference is inconclusive",
        ),
        "results",
    )
    require(
        table,
        (
            "$+3.2518\\ [3.0214,3.4842]$",
            "$+3.7811\\ [3.6177,3.9433]$",
            "$+0.8459\\ [0.7650,0.9260]$",
            "$+0.006\\ [-0.016,0.031]$",
        ),
        "primary table",
    )
    require(
        appendix,
        (
            "Under the historical backend",
            "Main-text values use this V2 backend; Step 54 values above remain the historical record",
            "PASS\\_STEP80\\_INDEPENDENT\\_VALIDATION",
        ),
        "appendix",
    )
    for stale in ("3.2657", "3.7851", ".9552"):
        if stale in main_text:
            raise AssertionError(f"historical Step 54 value leaked into main paper: {stale}")


def validate_build() -> dict:
    table = subprocess.run(
        [sys.executable, "validate_paper_tables.py"],
        cwd=PAPER,
        capture_output=True,
        text=True,
        check=False,
    )
    if table.returncode != 0 or "PASS_PAPER_TABLE_SOURCE_CONSISTENCY" not in table.stdout:
        raise AssertionError({"paper_table_validator": table.stdout + table.stderr})

    log = (BUILD / "main.log").read_text(encoding="utf-8", errors="replace")
    forbidden = (
        "Overfull \\hbox",
        "Undefined control sequence",
        "There were undefined references",
        "Citation `",
    )
    hits = [token for token in forbidden if token in log]
    if hits:
        raise AssertionError({"latex_log_failures": hits})
    page_matches = re.findall(r"\((\d+) pages,\s*\d+ bytes\)", log)
    if not page_matches:
        raise AssertionError("could not recover PDF page count from LaTeX log")

    aux = (BUILD / "main.aux").read_text(encoding="utf-8", errors="replace")
    related_pattern = rf"\\contentsline \{{section\}}.*{re.escape('Related Work')}\}}\{{9\}}"
    if not re.search(related_pattern, aux):
        raise AssertionError("Related Work is not closed on main-paper page 9")
    combined = re.search(
        rf"\\contentsline \{{section\}}.*{re.escape('Limitations, Implications, and Conclusion')}\}}\{{9\}}",
        aux,
    )
    legacy = all(
        re.search(
            rf"\\contentsline \{{section\}}.*{re.escape(label)}\}}\{{9\}}",
            aux,
        )
        for label in ("Limitations and Implications", "Conclusion")
    )
    if not (combined or legacy):
        raise AssertionError("limitations/conclusion are not closed on main-paper page 9")

    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    if "\\author{Anonymous Authors}" not in main_tex or "\\iclrfinalcopy" not in main_tex:
        raise AssertionError("anonymous submission configuration missing")
    if re.search(r"(?i)[A-Z]:\\Users\\[^\\\"'\s]+", "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in PAPER.rglob("*") if path.is_file() and path.suffix.lower() in {".tex", ".bib", ".md", ".py"}
    )):
        raise AssertionError("identity or workstation path found in manuscript sources")
    return {"pdf_pages_total": int(page_matches[-1]), "main_text_last_page": 9}


def main() -> None:
    evidence = validate_evidence()
    validate_text()
    build = validate_build()
    print(json.dumps({
        "verdict": "PASS_STEP81_MANUSCRIPT_INTEGRATION",
        "result_sha256": EXPECTED["result"],
        "validation_sha256": EXPECTED["validation"],
        **evidence,
        **build,
        "historical_values_confined_to_appendix": True,
        "anonymous_source_hits": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
