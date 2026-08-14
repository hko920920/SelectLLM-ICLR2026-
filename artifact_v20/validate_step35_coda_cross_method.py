from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "STEP35_CODA_CROSS_METHOD_RESULTS_2026-08-08.json"
EXPECTED_RESULT_SHA256 = "58C23357EE93C11D715F25BB77D397511F66E61473B8977EC650D511302AEFA7"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    assert sha256(RESULT) == EXPECTED_RESULT_SHA256
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    assert set(report["tasks"]) == {
        "glue_cola",
        "glue_mrpc",
        "glue_rte",
        "glue_sst2",
        "glue_wnli",
    }

    task_deltas = []
    path_count = 0
    harm_count = 0
    for task, row in report["tasks"].items():
        assert row["validity"]["all_pass"]
        seeds = row["seeds_executed"]
        assert seeds == [run["seed"] for run in row["runs"]["clean"]]
        for condition in ("clean", "exact4", "near4"):
            assert seeds == [run["seed"] for run in row["runs"][condition]]
            for run in row["runs"][condition]:
                assert len(run["queries"]) == 30
                assert len(set(run["queries"])) == 30
                assert len(run["trajectory"]) == 31
                cumulative = 0.0
                for step, state in enumerate(run["trajectory"]):
                    assert state["step"] == step
                    if step:
                        cumulative += state["regret"]
                    assert close(cumulative, state["cumulative_regret"])

        summary = row["comparisons"]["near4"]
        paired = summary["paired"]
        deltas = [x["checkpoints"]["30"]["cumulative_regret_delta"] for x in paired]
        divergence = [not x["identical_query_sequence"] for x in paired]
        mean_delta = float(np.mean(deltas))
        rate = float(np.mean(divergence))
        assert close(mean_delta, summary["mean_cumulative_regret_delta"])
        assert close(rate, summary["query_divergence_rate"])
        assert summary["task_counts_as_path_changed"] == (rate >= 0.5)
        assert summary["task_counts_as_harmful"] == (mean_delta > 1e-12)
        task_deltas.append(mean_delta)
        path_count += int(rate >= 0.5)
        harm_count += int(mean_delta > 1e-12)

        # This construction is soft-response-distinct but hard-equivalent; CODA's
        # realized paths match the exact4 control under the locked settings.
        exact = row["comparisons"]["exact4"]
        assert close(
            exact["mean_cumulative_regret_delta"],
            summary["mean_cumulative_regret_delta"],
        )

    median_delta = float(np.median(task_deltas))
    decision = "GO_STRONG" if path_count >= 4 and harm_count >= 3 and median_delta > 1e-12 else (
        "GO_LIMITED" if path_count >= 2 and harm_count >= 1 else "NO_CROSS_METHOD"
    )
    stored = report["decision"]
    assert stored["decision"] == decision == "GO_STRONG"
    assert stored["tasks_with_query_path_divergence"] == path_count == 5
    assert stored["tasks_with_positive_cumulative_regret_delta"] == harm_count == 3
    assert close(stored["median_task_level_cumulative_regret_delta"], median_delta)
    assert report["tasks"]["glue_wnli"]["role"] == "previously unused task confirmation"
    assert stored["unused_task_glue_wnli"]["path_changed"]
    assert not stored["unused_task_glue_wnli"]["harmful"]
    print("PASS_STEP35_RESULT_CONSISTENCY")
    print(json.dumps(stored, indent=2))


if __name__ == "__main__":
    main()
