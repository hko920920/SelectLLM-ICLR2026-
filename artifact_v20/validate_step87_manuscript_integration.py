"""Validate the Step 87 evidence-to-manuscript integration and page boundary."""

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
    "preregistration": ROOT / "STEP87B_REFERENCE_FREE_OPEN_ENDED_PREREGISTRATION_2026-08-12.json",
    "stage_b_lock": ROOT / "STEP87B_STAGEB_EXECUTION_LOCK_2026-08-12.json",
    "results": ROOT / "STEP87B_REFERENCE_FREE_OPEN_ENDED_RESULTS_2026-08-12.json",
    "raw": ROOT / "STEP87B_REFERENCE_FREE_OPEN_ENDED_RAW_2026-08-12.npz",
    "validation": ROOT / "STEP87B_INDEPENDENT_VALIDATION_2026-08-12.json",
}
HASHES = {
    "preregistration": "a95c6f0b2a11177577cd03574d019475bc7d8c169b71f02ab8de778a99ed8093",
    "stage_b_lock": "0fa650a7c60429711f5a0b293802f23f96213ee3cf030b368250193e7011ce9c",
    "results": "fc953c876ef67e3b5a6b915434667e4051d6ece47973574f8d82ee3e917cf556",
    "raw": "26422130d6b7d053c4b78fbc292e3deceb05db5eb72113c2b6e9175f2cb0ddad",
    "validation": "1ef6c0f7df2c11f871ead0f7162c480d600a3730d0f4d455f9e88500841cd9db",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(text: str, fragments: tuple[str, ...], source: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise AssertionError({"source": source, "missing": missing})


def verify_evidence() -> dict[str, object]:
    observed = {name: sha256(path) for name, path in FILES.items()}
    if observed != HASHES:
        raise AssertionError({"expected": HASHES, "observed": observed})
    result = json.loads(FILES["results"].read_text(encoding="utf-8"))
    validation = json.loads(FILES["validation"].read_text(encoding="utf-8"))
    expected_pass = ["narrativeqa", "wmt_fr_en", "wmt_ru_en"]
    if result["decision"] != {
        "global_separation_and_completeness": True,
        "passing_scenarios": expected_pass,
        "passing_count": 3,
        "status": "PROMOTE_REFERENCE_FREE_TERMINAL_BRIDGE",
    }:
        raise AssertionError(result["decision"])
    if validation["status"] != "PASS_STEP87B_INDEPENDENT_RECONSTRUCTION":
        raise AssertionError(validation["status"])
    if validation["decision_match"] is not True or not all(
        validation["stage_b_lock_checks"].values()
    ):
        raise AssertionError("independent decision or lock validation failed")
    if set(result["tasks"]) != {
        "narrativeqa",
        "naturalqa_closed",
        "naturalqa_open",
        "wmt_cs_en",
        "wmt_de_en",
        "wmt_fr_en",
        "wmt_hi_en",
        "wmt_ru_en",
    }:
        raise AssertionError("eight-task freeze drift")
    for task, row in result["tasks"].items():
        if row["gate"]["exact_fixed_query_mediation"] is not True:
            raise AssertionError(f"nonzero fixed-query effect: {task}")
        if not math.isclose(
            row["statistics"]["fixed_terminal_root_delta"]["mean"],
            0.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise AssertionError(f"fixed terminal drift: {task}")
        if row["quality"]["minimum_gap"] < -0.010000000001:
            raise AssertionError(f"quality-bound failure: {task}")
        if row["gate"]["task_pass"]:
            if row["statistics"]["active_terminal_root_delta"]["ci95"][0] <= 0:
                raise AssertionError(f"active CI failure: {task}")
            if row["statistics"]["terminal_root_did"]["ci95"][0] <= 0:
                raise AssertionError(f"DID CI failure: {task}")
            if max(row["multiplicity_adjustment"].values()) >= 0.05:
                raise AssertionError(f"BH failure: {task}")
    signs = {
        task: math.copysign(
            1.0, row["statistics"]["active_terminal_root_delta"]["mean"]
        )
        for task, row in result["tasks"].items()
    }
    if not all(signs[task] < 0 for task in ("naturalqa_closed", "naturalqa_open", "wmt_de_en")):
        raise AssertionError("negative findings were not retained")
    return {"passing_scenarios": expected_pass, "all_fixed_query_effects_zero": True}


def verify_text() -> None:
    abstract = (PAPER / "sections/00_abstract.tex").read_text(encoding="utf-8")
    intro = (PAPER / "sections/01_introduction.tex").read_text(encoding="utf-8")
    theory = (PAPER / "sections/03_theory.tex").read_text(encoding="utf-8")
    protocol = (PAPER / "sections/04_protocol.tex").read_text(encoding="utf-8")
    results = (PAPER / "sections/05_results.tex").read_text(encoding="utf-8")
    related = (PAPER / "sections/06_related_work.tex").read_text(encoding="utf-8")
    limitations = (PAPER / "sections/07_limitations.tex").read_text(encoding="utf-8")
    appendix = (PAPER / "appendix/appendix.tex").read_text(encoding="utf-8")
    table = (PAPER / "tables/step87b_reference_free_results.tex").read_text(
        encoding="utf-8"
    )
    require(abstract, ("two of five WMT14 pairs", "$3/8$", "five null or improve"), "abstract")
    require(intro, ("three of eight reference-free scenarios", "Fixed-query effects are zero throughout"), "introduction")
    require(theory, ("Selector-agnostic root factorization", "uncertainty, diversity, and mutual-information"), "theory")
    require(protocol, ("eight untouched", "1,000 paired holdout pools", "Every scenario and null is retained"), "protocol")
    require(results, ("three of eight untouched scenarios", "3.021 points [2.766,3.282]", "WMT14 fr--en and ru--en", "NaturalQA closed/open and WMT de--en improve"), "results")
    require(related, ("closest sequential precedent is UNREAL", "duplicate mass affecting an active query is not our novelty"), "related work")
    require(limitations, ("five of eight do not pass", "rather than an independent checkpoint"), "limitations")
    require(appendix, ("Outcome-blind task freeze", "PROMOTE\\_REFERENCE\\_FREE\\_TERMINAL\\_BRIDGE", "PASS\\_STEP87B\\_INDEPENDENT\\_RECONSTRUCTION"), "appendix")
    require(table, ("All rows were frozen and retained", "$+3.021$ [$2.766,3.282$]", "$+.139$ [$+.102,.177$]", "$+.250$ [$+.154,.350$]"), "table")


def verify_build() -> dict[str, int]:
    completed = subprocess.run(
        [sys.executable, "validate_paper_tables.py"],
        cwd=PAPER,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or "PASS_PAPER_TABLE_SOURCE_CONSISTENCY" not in completed.stdout:
        raise AssertionError(completed.stdout + completed.stderr)
    with tempfile.TemporaryDirectory(prefix="step87_paper_build_", dir=ROOT) as temporary:
        build_dir = Path(temporary)
        latex = subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-outdir={build_dir}",
                "main.tex",
            ],
            cwd=PAPER,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if latex.returncode != 0:
            raise AssertionError(latex.stdout[-6000:] + latex.stderr[-6000:])
        log = (build_dir / "main.log").read_text(encoding="utf-8", errors="replace")
        forbidden = (
            "Overfull \\hbox",
            "Undefined control sequence",
            "There were undefined references",
            "Citation `",
        )
        if any(token in log for token in forbidden):
            raise AssertionError("LaTeX warning gate failed")
        aux = (build_dir / "main.aux").read_text(encoding="utf-8", errors="replace")
        for label in ("Related Work", "Limitations, Implications, and Conclusion"):
            if not re.search(rf"\\contentsline \{{section\}}.*{re.escape(label)}\}}\{{9\}}", aux):
                raise AssertionError(f"{label} is outside page 9")
        page_matches = re.findall(r"\((\d+) pages,\s*\d+ bytes\)", log)
        if not page_matches:
            raise AssertionError("PDF page count missing from LaTeX log")
        return {"pdf_pages_total": int(page_matches[-1]), "main_text_last_page": 9}


def main() -> None:
    evidence = verify_evidence()
    verify_text()
    build = verify_build()
    print(
        json.dumps(
            {
                "verdict": "PASS_STEP87_MANUSCRIPT_INTEGRATION",
                **evidence,
                **build,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
