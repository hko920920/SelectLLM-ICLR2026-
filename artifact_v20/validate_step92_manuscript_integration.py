"""Validate the Step 92 learned-adapter evidence-to-manuscript closure."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from safetensors.torch import load_file

import validate_step88_manuscript_integration as build_validation
import validate_step90_manuscript_integration as historical_step90


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper_draft"
FILES = {
    "protocol": ROOT / "STEP92_LEARNED_CONTEXTUAL_ADAPTER_PREREGISTRATION_2026-08-12.md",
    "stage0_manifest": ROOT / "STEP92_STAGE0_MANIFEST_2026-08-12.json",
    "frozen_config": ROOT / "STEP92_STAGEA_FROZEN_CONFIG_2026-08-12.json",
    "stagea_ledger": ROOT / "STEP92_STAGEA_COMPLETE_SEARCH_LEDGER_2026-08-12.json",
    "calibration_outputs": ROOT / "STEP92_STAGEA_CALIBRATION_OUTPUTS_2026-08-12.npz",
    "execution_lock": ROOT / "STEP92_CONFIRMATORY_EXECUTION_LOCK_2026-08-12.json",
    "results": ROOT / "STEP92_LEARNED_ADAPTER_CONFIRMATORY_RESULTS_2026-08-12.json",
    "raw": ROOT / "STEP92_LEARNED_ADAPTER_CONFIRMATORY_RAW_2026-08-12.npz",
    "independent_validator": ROOT / "validate_step92_independent.py",
    "independent_receipt": ROOT / "STEP92_INDEPENDENT_VALIDATION_2026-08-12.json",
    "numeric_audit": ROOT / "STEP92_NUMERIC_TIE_ROBUSTNESS_2026-08-12.json",
    "numeric_raw": ROOT / "STEP92_NUMERIC_TIE_ROBUSTNESS_RAW_2026-08-12.npz",
    "numeric_runner": ROOT / "audit_step92_numeric_tie_robustness.py",
}
EXPECTED_HASHES = {
    "protocol": "47c63712082f5b119b7495195f6134bdfa9dae1947931aae93d6cc13cca7451c",
    "stage0_manifest": "9a6bc9d60a1e339fc94eddf62c0fea16ae611ec64515f0e95695bb8bd8208147",
    "frozen_config": "2c2d51aa81c4a99d942c0a129fed99785149315adfc20092ec62b869e442e9f1",
    "stagea_ledger": "562fbc63ab223c2cff5cbebcb013a1eb9745398d45b98e50f213b33f56f0d525",
    "calibration_outputs": "813a9f6e4fcf1af1afb6f81c18c73b7eab4ff9d21acdb2b8d3b79aa612c59a32",
    "execution_lock": "c219349a46e4a326f4d3cf8853ac69093014d0bdfe73ae44f0359f4259d426cc",
    "results": "8e20db3d7bbeb6fd4dff550ba8a24e2e992d88eb7bcaeb3d1233428deca011e4",
    "raw": "fdcaf9b09c648dcf13ae21b4a28e5358afe6df85a5f97b497bc866f3f07816ff",
    "independent_validator": "9fc7c1e90fe9c003167de87c8dcd4f08f99daf73f5af8338aded0c2dd7c88bdf",
    "independent_receipt": "8ac808aaa8b68686b1d4190c92a90035207c5d3de3bbb7d1d099f0bcc167015b",
    "numeric_audit": "e78c017dd34cc425022dcf3aaf4e9c503b0a428bd6b840b836692478649ff32a",
    "numeric_raw": "ea98945cfae3216ca9a9efd905c91e23035e2afb61652ca30f63ef754b47ce2f",
    "numeric_runner": "dbf2a9c3ce9e23fb7bde151fbadd0958882603235bc2e9eae3932cc0ef24605a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError((left, right, tolerance))


def require(text: str, fragments: tuple[str, ...], source: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise AssertionError({"source": source, "missing": missing})


def verify_evidence() -> dict[str, object]:
    observed = {name: sha256(path) for name, path in FILES.items()}
    if observed != EXPECTED_HASHES:
        raise AssertionError({"expected": EXPECTED_HASHES, "observed": observed})

    manifest = json.loads(FILES["stage0_manifest"].read_text(encoding="utf-8"))
    config = json.loads(FILES["frozen_config"].read_text(encoding="utf-8"))
    ledger = json.loads(FILES["stagea_ledger"].read_text(encoding="utf-8"))
    lock = json.loads(FILES["execution_lock"].read_text(encoding="utf-8"))
    result = json.loads(FILES["results"].read_text(encoding="utf-8"))
    receipt = json.loads(FILES["independent_receipt"].read_text(encoding="utf-8"))
    numeric = json.loads(FILES["numeric_audit"].read_text(encoding="utf-8"))

    if manifest["protocol_sha256"] != EXPECTED_HASHES["protocol"]:
        raise AssertionError("protocol/manifest binding drift")
    if config["stage0_manifest_sha256"] != EXPECTED_HASHES["stage0_manifest"]:
        raise AssertionError("manifest/config binding drift")
    if config["stagea_ledger_sha256"] != EXPECTED_HASHES["stagea_ledger"]:
        raise AssertionError("ledger/config binding drift")
    if config["calibration_outputs_sha256"] != EXPECTED_HASHES["calibration_outputs"]:
        raise AssertionError("calibration/config binding drift")
    if lock["confirmatory_outputs_absent_at_lock"] is not True or lock["test_outcome_access_before_lock"] is not False:
        raise AssertionError("outcome separation drift")
    for relative, expected in lock["locked_sha256"].items():
        if sha256(ROOT / Path(relative)) != expected:
            raise AssertionError(f"locked file drift: {relative}")

    search = ledger["search_space"]
    if search["pilot_configurations"] != 300 or search["verified_configurations"] != 12:
        raise AssertionError("complete Stage-A grid drift")
    if ledger["holdout_access"] is not False:
        raise AssertionError("Stage-A holdout-access drift")
    if ledger["selected"]["parent_root"] != 8:
        raise AssertionError("selected parent drift")

    if result["decision"] != "GO_SCORE_CHANGING_LEARNED_ADAPTER_BRIDGE":
        raise AssertionError(result["decision"])
    if result["config_sha256"] != EXPECTED_HASHES["frozen_config"]:
        raise AssertionError("result/config binding drift")
    if result["execution_lock_sha256"] != EXPECTED_HASHES["execution_lock"]:
        raise AssertionError("result/lock binding drift")
    if result["raw_sha256"] != EXPECTED_HASHES["raw"]:
        raise AssertionError("result/raw binding drift")
    task = result["task"]
    if task["n_holdout"] != 7600 or task["paired_runs"] != 1000:
        raise AssertionError("holdout/run count drift")
    if task["adapter_parent_accuracy_gaps"] != [0.0, 0.0, 0.0, 0.0]:
        raise AssertionError("hard-utility drift")
    close(task["terminal_regret_delta"]["mean"], 0.005467500000000004)
    close(task["terminal_regret_delta"]["bootstrap_95"][0], 0.005022500000000003)
    close(task["terminal_regret_delta"]["bootstrap_95"][1], 0.0059250000000000014)
    close(task["active_minus_fixed_terminal"]["mean"], 0.005467500000000004)
    close(task["path_change_rate"], 1.0)
    close(task["query_set_change_rate"], 0.993)
    close(task["final_root_change_rate"], 0.552)
    if task["max_abs_fixed_terminal_delta"] != 0.0 or task["max_abs_fixed_cumulative_delta"] != 0.0:
        raise AssertionError("fixed-query drift")
    if not all(task["gates"].values()):
        raise AssertionError("confirmatory gate drift")

    adapter_hashes: list[str] = []
    for name, expected in config["learned_adapter_sha256"].items():
        path = ROOT / "step92_models" / name
        if sha256(path) != expected:
            raise AssertionError(f"adapter hash drift: {name}")
        tensors = load_file(str(path))
        count = sum(tensor.numel() for tensor in tensors.values())
        if count != 50817:
            raise AssertionError({"adapter": name, "parameter_count": count})
        adapter_hashes.append(expected)
    if len(set(adapter_hashes)) != 4:
        raise AssertionError("learned adapter hashes are not distinct")

    if receipt["verdict"] != "PASS_STEP92_INDEPENDENT_VALIDATION":
        raise AssertionError(receipt["verdict"])
    if receipt["checks"] != 4170 or receipt["active_trajectory_replays"] != 2000:
        raise AssertionError("active validation coverage drift")
    if receipt["fixed_query_replays"] != 2000 or receipt["endpoint_replay"]["sample_size"] != 256:
        raise AssertionError("fixed/endpoint validation coverage drift")
    if receipt["result_sha256"] != EXPECTED_HASHES["results"] or receipt["raw_sha256"] != EXPECTED_HASHES["raw"]:
        raise AssertionError("receipt binding drift")

    if numeric["decision"] != "PASS_NUMERIC_TIE_ROBUSTNESS" or not numeric["all_policies_pass_original_harm_gate"]:
        raise AssertionError("numeric robustness drift")
    if numeric["primary_raw_sha256"] != EXPECTED_HASHES["raw"] or numeric["raw_sha256"] != EXPECTED_HASHES["numeric_raw"]:
        raise AssertionError("numeric audit binding drift")
    means = [row["mean_terminal_delta_pp"] for row in numeric["policies"].values()]
    close(min(means), 0.5457500000000004)
    close(max(means), 0.5480000000000003)
    return {
        "decision": result["decision"],
        "task": "ag_news",
        "learned_parameters_per_alias": 50817,
        "paired_active_trajectories": 2000,
        "fixed_query_replays": 2000,
        "independent_checks": 4170,
        "endpoint_replay_samples": 256,
        "numeric_policies": len(numeric["policies"]),
    }


def verify_text() -> None:
    sources = {
        "abstract": (PAPER / "sections/00_abstract.tex").read_text(encoding="utf-8"),
        "introduction": (PAPER / "sections/01_introduction.tex").read_text(encoding="utf-8"),
        "protocol": (PAPER / "sections/04_protocol.tex").read_text(encoding="utf-8"),
        "results": (PAPER / "sections/05_results.tex").read_text(encoding="utf-8"),
        "limitations": (PAPER / "sections/07_limitations.tex").read_text(encoding="utf-8"),
        "statements": (PAPER / "sections/09_statements.tex").read_text(encoding="utf-8"),
        "appendix": (PAPER / "appendix/appendix.tex").read_text(encoding="utf-8"),
        "table": (PAPER / "tables/step92_learned_adapter_results.tex").read_text(encoding="utf-8"),
        "readme": (PAPER / "README.md").read_text(encoding="utf-8"),
    }
    require(
        sources["abstract"],
        (
            "previously unused AG News",
            "50,817-parameter contextual adapters",
            "$.547$ [$.502,.593$]",
            "not universal instability or production compromise",
        ),
        "abstract",
    )
    require(
        sources["introduction"],
        (
            "genuinely learned raw-input adapters pass a sealed new-task AG News test",
            "configuration-only MNLI/QQP successes",
        ),
        "introduction",
    )
    require(
        sources["protocol"],
        (
            "Learned raw-input adapter transfer",
            "all 300 parent/temperature/bin/trigger conditions",
            "four 50,817-parameter contextual temperature adapters",
            "Earlier configuration-only QNLI/MNLI/QQP wrappers",
        ),
        "protocol",
    )
    require(
        sources["results"],
        (
            "A learned raw-input adapter crosses the new-task bridge",
            "24.1--26.3\\%",
            "Terminal regret rises by $.547$ [$.502,.593$]",
            "from $.546$ to $.548$ points",
        ),
        "results",
    )
    require(
        sources["limitations"],
        (
            "genuinely learned contextual calibration adapters",
            "one task under an already audited selector",
            "not independently fine-tuned full checkpoints or registry admission",
            "configuration-only",
        ),
        "limitations",
    )
    require(
        sources["statements"],
        (
            "Step~92's learned-adapter protocol",
            "2,000 active plus 2,000 fixed-query trajectories in 4,170 checks",
            "256 raw-input endpoint replays",
        ),
        "reproducibility statement",
    )
    require(
        sources["appendix"],
        (
            "Preregistered Learned-Adapter Bridge",
            "30,400 adapter--item hard answers",
            "GO\\_SCORE\\_CHANGING\\_LEARNED\\_ADAPTER\\_BRIDGE",
            "PASS\\_STEP92\\_INDEPENDENT\\_VALIDATION",
            "This robustness audit does not replace the preregistered primary result",
        ),
        "appendix",
    )
    require(
        sources["table"],
        (
            "AG News & 50,817",
            "100.0/55.2\\%",
            "$+.547$ [$+.502,+.593$]",
        ),
        "Step 92 table",
    )
    require(sources["readme"], ("Step 92 review-ready manuscript",), "README")


def main() -> None:
    historical = historical_step90.verify_step90_evidence()
    current = verify_evidence()
    verify_text()
    build = build_validation.verify_build()
    page10 = build_validation.run_checked(
        ["pdftotext", "-f", "10", "-l", "10", str(PAPER / "main.pdf"), "-"],
        ROOT,
    )
    spill_tokens = (
        "Deployment boundary and conclusion.",
        "Candidate lists are experimental priors",
        "similarity cannot authenticate ownership",
        "evidence-frame and multiplicity sensitivity",
    )
    spilled = [token for token in spill_tokens if token in page10]
    if spilled:
        raise AssertionError({"page10_scientific_text_spill": spilled})
    print(
        json.dumps(
            {
                "verdict": "PASS_STEP92_MANUSCRIPT_INTEGRATION",
                "historical_step90": historical,
                "step92": current,
                "page10_scientific_text_spill_tokens": 0,
                **build,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
