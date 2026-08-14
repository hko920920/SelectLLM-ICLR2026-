"""Validate the Step 90 evidence-to-manuscript closure and page boundary.

This validator deliberately reuses only the historical Step 88 evidence/build
checks.  It does not import either Step 89/90 experimental runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import validate_step88_manuscript_integration as step88


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper_draft"
FILES = {
    "stage0_protocol": ROOT / "STEP90_HIGH_CONFIDENCE_ADAPTER_STAGE0_PROTOCOL_2026-08-12.md",
    "stage0_manifest": ROOT / "STEP90_HIGH_CONFIDENCE_ADAPTER_STAGE0_MANIFEST_2026-08-12.json",
    "frozen_config": ROOT / "STEP90_HIGH_CONFIDENCE_STAGEA_FROZEN_CONFIG_2026-08-12.json",
    "preregistration": ROOT / "STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_PREREGISTRATION_2026-08-12.md",
    "execution_lock": ROOT / "STEP90_HIGH_CONFIDENCE_CONFIRMATORY_EXECUTION_LOCK_2026-08-12.json",
    "runner": ROOT / "step90_stageb_confirmatory.py",
    "endpoint": ROOT / "step90_executable_adapter_endpoint.py",
    "results": ROOT / "STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_RESULTS_2026-08-12.json",
    "raw": ROOT / "STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_RAW_2026-08-12.npz",
    "independent_validator": ROOT / "validate_step90_independent.py",
    "receipt": ROOT / "STEP90_HIGH_CONFIDENCE_INDEPENDENT_VALIDATION_2026-08-12.json",
    "retained_qnli": ROOT / "STEP89_QNLI_EXECUTABLE_ADAPTER_RESULTS_2026-08-12.json",
}
HASHES = {
    "stage0_protocol": "1b459457a222b070953e52079a83a1ceda0bb5d6a78fc0c8c07e23e36491354f",
    "stage0_manifest": "4aa672d6657d1324636c74a29510c8a06a5ea804c6f5db0fd51d3272c9df87cb",
    "frozen_config": "3dfda0e5deb6db51bba510fcf074360f4a8764a95271c16f427bbec1b585ecb4",
    "preregistration": "70147e0b5b1c70024e95492f23d31dbe9b739866468a407237bc0cb661b05c55",
    "execution_lock": "b73730c970f6e8d78a59978fe1e379168ebed498ca4bbc1123c39b8bf3ff42b3",
    "runner": "a50f7c192a5fab220a576019e3927f5463c8b6afa39d9fb7fb35037c4f1c8783",
    "endpoint": "bc8918f06c8fd2a9bca8ecf24ca543c108e6454c0500fe3b14d18d64cf2e7dfe",
    "results": "f699eee78c4fd84139933e469d1c1c2bfbefb9ffe4a6cc3f278a2996f55d5654",
    "raw": "5ef206b57d2204c128e621f07c83f8a621d1e94ab6856cbddc05d8e3bbd44d6d",
    "independent_validator": "ee3aa0b1377733c25fcf3d6c356faf8d0a6696eee368127b83ebdceadc5aa918",
    "receipt": "0226d76e493905bfed74b62647729f4ed3e83d37bc127c57d81d20a41bf4aa79",
    "retained_qnli": "efb16dbc16830d2398f98592747018fb7f1da952a4022e76ea0f06270b892ec0",
}
ADAPTER_HASHES = {
    "step90_mnli_adapter_0.safetensors": "750f2e2d98d92cd440f4d11044f296ee9d057b6cd8345eb923a6776ded8b18a2",
    "step90_mnli_adapter_1.safetensors": "c54105ec135b1d57257abb68543abbdf8855ded8a0123806546c105074926dc1",
    "step90_mnli_adapter_2.safetensors": "56c7b637492d65bd097119eacf8ed39041e01196f27c889edfba4aef50012dc9",
    "step90_mnli_adapter_3.safetensors": "0d12dc05eabf0f6d6cd88b98903e413508b7d2e31f93161dfa64dc7acb9935dd",
    "step90_qqp_adapter_0.safetensors": "7e4cc6f70dd45490d49b004071ea2dd4c403d829d92531c47053dadb5a44d136",
    "step90_qqp_adapter_1.safetensors": "92a7be6d812222757c7faa75191f35265bc594d6973cc54aa02dbcb10e53886b",
    "step90_qqp_adapter_2.safetensors": "f50624038c24592115612cf39cb5a9f67c79f68c69c71eacef64459d872baaa7",
    "step90_qqp_adapter_3.safetensors": "d7029f9701641f8833c5f28d9d6f26ebf3c2d6dc3d3f734f511acf328a919b4c",
}
EXPECTED = {
    "mnli": {
        "terminal": 0.007017500000000001,
        "terminal_ci": (0.005857500000000002, 0.0082275),
        "adaptive_ci": (0.005850000000000001, 0.008215000000000002),
        "holm": 1.999980000199998e-05,
        "path": 0.998,
        "root": 0.328,
        "cumulative": 0.22081000000000003,
    },
    "qqp": {
        "terminal": 0.012695,
        "terminal_ci": (0.012009937499999996, 0.013399999999999999),
        "adaptive_ci": (0.012015000000000003, 0.01339),
        "holm": 1.999980000199998e-05,
        "path": 1.0,
        "root": 0.823,
        "cumulative": 0.42470750000000007,
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


def verify_step90_evidence() -> dict[str, object]:
    observed = {name: sha256(path) for name, path in FILES.items()}
    if observed != HASHES:
        raise AssertionError({"expected": HASHES, "observed": observed})
    for name, expected in ADAPTER_HASHES.items():
        observed_hash = sha256(ROOT / "step90_models" / name)
        if observed_hash != expected:
            raise AssertionError({"adapter": name, "expected": expected, "observed": observed_hash})

    result = json.loads(FILES["results"].read_text(encoding="utf-8"))
    receipt = json.loads(FILES["receipt"].read_text(encoding="utf-8"))
    qnli = json.loads(FILES["retained_qnli"].read_text(encoding="utf-8"))
    if result["decision"] != "GO_STRONG_EXECUTABLE_ADAPTER_TRANSFER":
        raise AssertionError(result["decision"])
    if result["passing_tasks"] != ["mnli", "qqp"]:
        raise AssertionError(result["passing_tasks"])
    if result["config_sha256"] != HASHES["frozen_config"]:
        raise AssertionError("frozen-config binding drift")
    if result["execution_lock_sha256"] != HASHES["execution_lock"]:
        raise AssertionError("execution-lock binding drift")
    if result["raw_sha256"] != HASHES["raw"]:
        raise AssertionError("raw-result binding drift")

    for task, expected in EXPECTED.items():
        summary = result["tasks"][task]
        close(summary["terminal_regret_delta"]["mean"], expected["terminal"])
        for value, target in zip(summary["terminal_regret_delta"]["bootstrap_95"], expected["terminal_ci"]):
            close(value, target)
        for value, target in zip(summary["active_minus_fixed_terminal"]["bootstrap_95"], expected["adaptive_ci"]):
            close(value, target)
        close(summary["holm_adjusted_p"], expected["holm"])
        close(summary["path_change_rate"], expected["path"])
        close(summary["final_root_change_rate"], expected["root"])
        close(summary["cumulative_regret_delta"]["mean"], expected["cumulative"])
        if summary["adapter_parent_accuracy_gaps"] != [0.0, 0.0, 0.0, 0.0]:
            raise AssertionError(f"hard-utility drift: {task}")
        if summary["max_abs_fixed_terminal_delta"] != 0.0 or summary["max_abs_fixed_cumulative_delta"] != 0.0:
            raise AssertionError(f"fixed-query drift: {task}")
        if not summary["passed"] or not all(summary["gate"].values()):
            raise AssertionError(f"gate drift: {task}")

    if qnli["decision"] != "NO_CONFIRMATORY_EXECUTABLE_ADAPTER_BRIDGE":
        raise AssertionError("retained QNLI decision drift")
    close(qnli["terminal_regret_delta"]["mean"], -0.0006799999999999991)
    if qnli["path_change_rate"] != 0.969 or qnli["max_abs_fixed_terminal_delta"] != 0.0:
        raise AssertionError("retained QNLI summary drift")

    if receipt["verdict"] != "PASS_STEP90_INDEPENDENT_VALIDATION":
        raise AssertionError(receipt["verdict"])
    if receipt["validator_imports_runner_code"] is not False:
        raise AssertionError("validator independence drift")
    if receipt["decision"] != result["decision"] or receipt["passing_tasks"] != result["passing_tasks"]:
        raise AssertionError("independent decision reconstruction drift")
    if receipt["checks"] != 12077 or receipt["literal_trajectory_replays"] != 4000:
        raise AssertionError("active validation coverage drift")
    if receipt["fixed_query_replays"] != 2000:
        raise AssertionError("fixed-query validation coverage drift")
    if receipt["raw_sha256"] != HASHES["raw"] or receipt["result_sha256"] != HASHES["results"]:
        raise AssertionError("receipt hash binding drift")
    return {
        "decision": result["decision"],
        "passing_tasks": result["passing_tasks"],
        "paired_active_trajectories": 4000,
        "fixed_query_replays": 2000,
        "independent_checks": 12077,
        "retained_negative": "qnli",
    }


def verify_text() -> None:
    sources = {
        "title": (PAPER / "main.tex").read_text(encoding="utf-8"),
        "abstract": (PAPER / "sections/00_abstract.tex").read_text(encoding="utf-8"),
        "introduction": (PAPER / "sections/01_introduction.tex").read_text(encoding="utf-8"),
        "protocol": (PAPER / "sections/04_protocol.tex").read_text(encoding="utf-8"),
        "results": (PAPER / "sections/05_results.tex").read_text(encoding="utf-8"),
        "related": (PAPER / "sections/06_related_work.tex").read_text(encoding="utf-8"),
        "limitations": (PAPER / "sections/07_limitations.tex").read_text(encoding="utf-8"),
        "statements": (PAPER / "sections/09_statements.tex").read_text(encoding="utf-8"),
        "appendix": (PAPER / "appendix/appendix.tex").read_text(encoding="utf-8"),
        "table": (PAPER / "tables/step90_executable_adapter_results.tex").read_text(encoding="utf-8"),
        "bibliography": (PAPER / "references.bib").read_text(encoding="utf-8"),
    }
    require(sources["title"], ("Evidence-Frame Dependence in Active Model Selection",), "title")
    require(
        sources["abstract"],
        (
            "raw-input calibration adapters",
            "sealed MNLI/QQP",
            "a prior QNLI audit improves and is retained",
            "not universal instability or production compromise",
        ),
        "abstract",
    )
    require(
        sources["introduction"],
        (
            "Sequential clone sensitivity is prior",
            "neither the first sequential clone effect nor the first multiplicity attack",
            "raw-input answer-preserving adapters pass sealed MNLI/QQP",
        ),
        "introduction",
    )
    require(
        sources["protocol"],
        (
            "Executable answer-preserving adapter transfer",
            "all 135 roster/budget/temperature/threshold conditions",
            "requiring at least $.5$ terminal and active-minus-fixed harm",
            "prior preregistered QNLI failure remains binding",
        ),
        "protocol",
    )
    require(
        sources["results"],
        (
            "Raw-input calibration adapters transfer terminal harm to both sealed tasks",
            "$.702$ [$.586,.823$]",
            "$1.270$ [$1.201,1.340$]",
            "not a full checkpoint or registry-admission test",
        ),
        "results",
    )
    require(
        sources["related"],
        (
            "Closest sequentially",
            "perturbed, separately ranked clones",
            "We do not claim the first sequential clone effect",
            "shared reference-label unit",
            "same-root exact-utility refinement",
        ),
        "related work",
    )
    require(
        sources["limitations"],
        (
            "raw-input response adapters sharing pinned parents",
            "not full checkpoints or registry admission",
            "QNLI improves",
        ),
        "limitations",
    )
    require(
        sources["statements"],
        ("Steps~89--90", "4,000 active plus 2,000 fixed-query", "12,077 checks"),
        "reproducibility statement",
    )
    require(
        sources["appendix"],
        (
            "Executable Calibration-Adapter Transfer",
            "PASS\\_STEP90\\_INDEPENDENT\\_VALIDATION",
            "closes a raw-input executable-adapter bridge, not hidden-label or live-production realism",
        ),
        "appendix",
    )
    require(
        sources["table"],
        (
            "QNLI & all confidence views & $-.068$ [$-.149,.014$]",
            "MNLI & top 30\\% confidence & $+.702$ [$+.586,.823$]",
            "QQP & top 20\\% confidence & $+1.270$ [$+1.201,+1.340$]",
        ),
        "Step 90 table",
    )
    require(sources["bibliography"], ("@inproceedings{lanctot2026activeevaluation",), "bibliography")


def main() -> None:
    historical = step88.verify_evidence()
    current = verify_step90_evidence()
    verify_text()
    build = step88.verify_build()
    print(json.dumps({
        "verdict": "PASS_STEP90_MANUSCRIPT_INTEGRATION",
        "historical_step88": historical,
        "step90": current,
        **build,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
