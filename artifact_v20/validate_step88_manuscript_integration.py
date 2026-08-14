"""Validate Step 88 evidence, manuscript claims, and the nine-page boundary."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper_draft"
FILES = {
    "preregistration": ROOT / "STEP88_LLM_SELECTOR_PREREGISTRATION_2026-08-12.md",
    "runner": ROOT / "run_step88_llm_selector_audit.py",
    "results": ROOT / "STEP88_LLM_SELECTOR_RESULTS_2026-08-12.json",
    "independent_validator": ROOT / "validate_step88_llm_selector_audit.py",
    "receipt": ROOT / "STEP88_INDEPENDENT_VALIDATION_RECEIPT_2026-08-12.json",
}
HASHES = {
    "preregistration": "b630370652c88cb4150a5630d04cdf6fd93fde90cf67cb4e64ca694bb03ee4df",
    "runner": "751f4f250627a8854c460c028ee79fc218dba18df3fdd611a2cd04f9b41f8ccb",
    "results": "24c3fcdab2016bf6004844c242603bfd346fe0222911124f304fd81e4cea7020",
    "independent_validator": "4eee854c0a958877072efd1447ce9bc5bc7c071ab5d88875fa1b9d645c345a09",
    "receipt": "9e3f062cbc2e76f552738a5e0fb4e3f62b043f3b8eb6d51c146ac4a41eb993fd",
}
EXPECTED = {
    "alpacaeval": {
        "terminal": 0.0032125,
        "ci": (0.00193625, 0.00452875),
        "q": 0.00002999970000299997,
        "cumulative": -0.24478625,
        "set_change": 0.995,
        "root_change": 0.232,
        "pass": True,
    },
    "arena-hard": {
        "terminal": -0.003695,
        "ci": (-0.0053075, -0.00212625),
        "q": 1.0,
        "cumulative": -0.37898,
        "set_change": 0.991,
        "root_change": 0.067,
        "pass": False,
    },
    "bingo": {
        "terminal": 0.0049775,
        "ci": (0.00373875, 0.0062275),
        "q": 0.00002999970000299997,
        "cumulative": 0.11693625,
        "set_change": 0.982,
        "root_change": 0.324,
        "pass": True,
    },
    "flickr30k": {
        "terminal": 0.00182625,
        "ci": (-0.000025, 0.00367625),
        "q": 0.052979470205297946,
        "cumulative": -0.029095,
        "set_change": 1.0,
        "root_change": 0.667,
        "pass": False,
    },
    "medi_qa": {
        "terminal": 0.0,
        "ci": (0.0, 0.0),
        "q": 1.0,
        "cumulative": 0.02666666666666634,
        "set_change": 1.0,
        "root_change": 0.0,
        "pass": False,
    },
    "mt-bench": {
        "terminal": 0.0,
        "ci": (0.0, 0.0),
        "q": 1.0,
        "cumulative": 0.71875,
        "set_change": 1.0,
        "root_change": 0.0,
        "pass": False,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError((left, right))


def require(text: str, fragments: tuple[str, ...], source: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise AssertionError({"source": source, "missing": missing})


def verify_evidence() -> dict[str, object]:
    observed = {name: sha256(path) for name, path in FILES.items()}
    if observed != HASHES:
        raise AssertionError({"expected": HASHES, "observed": observed})

    result = json.loads(FILES["results"].read_text(encoding="utf-8"))
    receipt = json.loads(FILES["receipt"].read_text(encoding="utf-8"))
    if result["decision"] != "GO_STRONG_CROSS_SELECTOR":
        raise AssertionError(result["decision"])
    if result["passing_datasets"] != ["alpacaeval", "bingo"]:
        raise AssertionError(result["passing_datasets"])
    if set(result["summaries"]) != set(EXPECTED):
        raise AssertionError("six-dataset retention drift")
    if receipt["status"] != "PASS_STEP88_INDEPENDENT_VALIDATION":
        raise AssertionError(receipt["status"])
    if receipt["decision_reconstructed"] != result["decision"]:
        raise AssertionError("independent decision mismatch")
    if receipt["passing_datasets_reconstructed"] != result["passing_datasets"]:
        raise AssertionError("independent passing-set mismatch")
    if receipt["checks"] != 85934 or receipt["literal_official_path_replays"] != 300:
        raise AssertionError("independent-validation coverage drift")
    receipt_hash_keys = {
        "preregistration": "preregistration_sha256",
        "runner": "runner_sha256",
        "results": "results_sha256",
        "independent_validator": "validator_sha256",
    }
    for name, receipt_key in receipt_hash_keys.items():
        if receipt["hashes"][receipt_key] != HASHES[name]:
            raise AssertionError(f"receipt binding mismatch: {name}")

    for task, expected in EXPECTED.items():
        source = result["datasets"][task]
        summary = result["summaries"][task]
        if source["strong_utility_max_abs_difference"] != 0:
            raise AssertionError(f"utility equivalence failed: {task}")
        close(summary["mean_terminal_delta"], expected["terminal"])
        close(summary["terminal_bootstrap_95_ci"][0], expected["ci"][0])
        close(summary["terminal_bootstrap_95_ci"][1], expected["ci"][1])
        close(summary["terminal_signflip_q_bh"], expected["q"])
        close(summary["mean_cumulative_delta"], expected["cumulative"])
        close(summary["ordered_path_change_rate"], 1.0)
        close(summary["query_set_change_rate"], expected["set_change"])
        close(summary["terminal_root_change_rate"], expected["root_change"])
        close(summary["max_abs_fixed_terminal_delta"], 0.0)
        close(summary["max_abs_fixed_cumulative_delta"], 0.0)
        if summary["passes_terminal_harm_gate"] is not expected["pass"]:
            raise AssertionError(f"gate drift: {task}")
    return {
        "decision": result["decision"],
        "passing_datasets": result["passing_datasets"],
        "paired_trajectories": 6000,
        "independent_checks": receipt["checks"],
        "literal_official_path_replays": receipt["literal_official_path_replays"],
    }


def verify_text() -> None:
    abstract = (PAPER / "sections/00_abstract.tex").read_text(encoding="utf-8")
    intro = (PAPER / "sections/01_introduction.tex").read_text(encoding="utf-8")
    protocol = (PAPER / "sections/04_protocol.tex").read_text(encoding="utf-8")
    results = (PAPER / "sections/05_results.tex").read_text(encoding="utf-8")
    limitations = (PAPER / "sections/07_limitations.tex").read_text(encoding="utf-8")
    statements = (PAPER / "sections/09_statements.tex").read_text(encoding="utf-8")
    appendix = (PAPER / "appendix/appendix.tex").read_text(encoding="utf-8")
    table = (PAPER / "tables/step88_llm_selector_results.tex").read_text(encoding="utf-8")
    bibliography = (PAPER / "references.bib").read_text(encoding="utf-8")

    require(
        abstract,
        (
            "second official pairwise-judge selector",
            "AlpacaEval and Bingo ($2/6$)",
            "fixed-query effects are again zero",
            "all outcomes are retained",
        ),
        "abstract",
    )
    require(intro, ("official pairwise-judge selector passes on $2/6$ datasets",), "introduction")
    require(
        protocol,
        (
            "Official cross-selector audits",
            "all six official LLM Selector datasets",
            "score-matrix decision-view intervention",
            "not a raw-response endpoint",
        ),
        "protocol",
    )
    require(
        results,
        (
            "AlpacaEval and Bingo pass",
            "$.321$ [$.194,.453$]",
            "$.498$ [$.374,.623$]",
            "Arena-Hard improves",
            "Flickr30k misses correction",
            "MEDIQA/MT-Bench are terminal-null",
            "pairwise weak-view-distinct, not executed raw responses",
        ),
        "results",
    )
    require(
        limitations,
        (
            "uses weak-judge score views rather than response endpoints",
            "two of six pass",
            "over three selectors",
            "Candidate lists are experimental priors",
        ),
        "limitations/conclusion",
    )
    require(
        statements,
        ("6,000 paired arrays", "85,934", "300 paths from the literal official entropy equation"),
        "reproducibility statement",
    )
    require(
        appendix,
        (
            "Preregistered LLM Selector Audit",
            "GO\\_STRONG\\_CROSS\\_SELECTOR",
            "AlpacaEval and Bingo pass",
            "PASS\\_STEP88\\_INDEPENDENT\\_VALIDATION",
            "all six upstream datasets",
        ),
        "appendix",
    )
    for token in (
        "AlpacaEval$^\\dagger$",
        "$.321$ [$ .194,.453$]",
        "Arena-Hard",
        "$-.370$ [$-.531,-.213$]",
        "Bingo$^\\dagger$",
        "$.498$ [$ .374,.623$]",
        "Flickr30k",
        "$.05298$",
        "MEDIQA",
        "MT-Bench",
    ):
        if token not in table:
            raise AssertionError({"source": "step88 table", "missing": token})
    require(bibliography, ("@article{durmazkeser2025llmselector",), "bibliography")


def run_checked(command: list[str], cwd: Path, marker: str | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or (marker is not None and marker not in output):
        raise AssertionError(output[-12000:])
    return output


def verify_build() -> dict[str, int]:
    run_checked(
        [sys.executable, "validate_paper_tables.py"],
        PAPER,
        "PASS_PAPER_TABLE_SOURCE_CONSISTENCY",
    )
    with tempfile.TemporaryDirectory(prefix="step88_paper_build_", dir=ROOT) as temporary:
        build_dir = Path(temporary)
        run_checked(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-outdir={build_dir}",
                "main.tex",
            ],
            PAPER,
        )
        log = (build_dir / "main.log").read_text(encoding="utf-8", errors="replace")
        forbidden = (
            "Overfull \\hbox",
            "Undefined control sequence",
            "There were undefined references",
            "Citation `",
        )
        found = [token for token in forbidden if token in log]
        if found:
            raise AssertionError({"latex_warning_gate": found})
        aux = (build_dir / "main.aux").read_text(encoding="utf-8", errors="replace")
        for label in ("Related Work", "Limitations, Implications, and Conclusion"):
            if not re.search(rf"\\contentsline \{{section\}}.*{re.escape(label)}\}}\{{9\}}", aux):
                raise AssertionError(f"{label} is outside page 9")

        pdf = build_dir / "main.pdf"
        info = run_checked(["pdfinfo", str(pdf)], ROOT)
        match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
        if not match:
            raise AssertionError("PDF page count missing")
        pages = int(match.group(1))
        page9 = run_checked(["pdftotext", "-f", "9", "-l", "9", str(pdf), "-"], ROOT)
        page10 = run_checked(["pdftotext", "-f", "10", "-l", "10", str(pdf), "-"], ROOT)
        require(
            page9,
            (
                "Deployment boundary and conclusion.",
                "Candidate lists are experimental priors",
            ),
            "rendered page 9",
        )
        require(page10, ("AI U SE S TATEMENT",), "rendered page 10")
        return {"pdf_pages_total": pages, "main_text_last_page": 9}


def main() -> None:
    evidence = verify_evidence()
    verify_text()
    build = verify_build()
    print(
        json.dumps(
            {
                "verdict": "PASS_STEP88_MANUSCRIPT_INTEGRATION",
                **evidence,
                **build,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
