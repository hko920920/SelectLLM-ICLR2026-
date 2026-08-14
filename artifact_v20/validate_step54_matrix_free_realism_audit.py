"""Independent validation for the locked Step 54 realism kill test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import run_step54_matrix_free_realism_audit as step54


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "STEP54_MATRIX_FREE_REALISM_RESULTS_2026-08-09.json"
EXPECTED_RESULT_SHA256 = "4ab5428c368991f8c9696edda63660ac76b1d516bb7c78d52eeb553ad56a41dd"
CONDITION_INDEX = {
    "legacy_matrix_t2_5pct": 0,
    "matrix_free_public_reference_5pct": 1,
    "matrix_free_sequential_reference_5pct": 2,
    "matrix_free_public_reference_1pct": 3,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_hash(row: np.ndarray) -> str:
    payload = json.dumps(
        row.tolist(), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def isolated_public_rows(
    task: str,
    *,
    target_response: np.ndarray,
    target_oracle: np.ndarray,
    correct: np.ndarray,
    ids: list[Any],
    public_options: list[list[Any]] | None,
    distance_queries: int,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """Reproduce aliases without accepting any peer response row as input."""
    eligible = [
        query
        for query in range(len(ids))
        if step54.usable_wrong_coordinate(
            task,
            query,
            float(target_oracle[query]),
            target_response,
            correct,
            public_options,
        )
    ]
    rows: list[np.ndarray] = []
    supports: list[list[int]] = []
    for alias in range(step54.ALIASES):
        ordered = sorted(
            eligible,
            key=lambda query: step54.hash_rank(
                step54.MANIFEST_ID,
                "public-support",
                task,
                alias,
                ids[query],
            ),
        )
        changed = [int(query) for query in ordered[:distance_queries]]
        row = target_response.copy()
        for rank, query in enumerate(changed):
            row[query] = step54.wrong_answer(
                task,
                query=query,
                rank=rank,
                alias=alias,
                example_id=ids[query],
                parent_response=target_response[query],
                correct=correct[query],
                public_options=(
                    public_options[query] if public_options is not None else None
                ),
            )
        rows.append(row)
        supports.append(changed)
    return rows, supports


def isolated_sequential_rows(
    task: str,
    *,
    target_response: np.ndarray,
    target_oracle: np.ndarray,
    correct: np.ndarray,
    ids: list[Any],
    public_options: list[list[Any]] | None,
    distance_queries: int,
) -> tuple[list[np.ndarray], list[list[int]], list[int]]:
    """Use a logged label stream; uninspected references never reach generation."""
    order = sorted(
        range(len(ids)),
        key=lambda query: step54.hash_rank(
            step54.MANIFEST_ID, "sequential-support", task, ids[query]
        ),
    )
    inspected: list[int] = []
    revealed: dict[int, Any] = {}
    support: list[int] = []
    for query in order:
        inspected.append(int(query))
        revealed[query] = correct[query]
        local_correct = np.asarray(
            [revealed.get(index, "") for index in range(len(ids))], dtype=object
        )
        if step54.usable_wrong_coordinate(
            task,
            query,
            float(target_oracle[query]),
            target_response,
            local_correct,
            public_options,
        ):
            support.append(int(query))
            if len(support) == distance_queries:
                break
    rows: list[np.ndarray] = []
    for alias in range(step54.ALIASES):
        row = target_response.copy()
        for rank, query in enumerate(support):
            row[query] = step54.wrong_answer(
                task,
                query=query,
                rank=rank,
                alias=alias,
                example_id=ids[query],
                parent_response=target_response[query],
                correct=revealed[query],
                public_options=(
                    public_options[query] if public_options is not None else None
                ),
            )
        rows.append(row)
    return rows, [list(support) for _ in range(step54.ALIASES)], inspected


def validate_effect(effect: dict[str, Any], seed: int) -> None:
    phase_a = step54.step46.phase_a
    values = np.asarray(
        effect["paired_delta_cumulative_deployed_regret"], dtype=np.float64
    )
    finals = np.asarray(effect["paired_delta_final_deployed_regret"], dtype=np.float64)
    assert len(values) == 500 and len(finals) == 500
    assert np.isclose(values.mean(), effect["mean_delta_cumulative_deployed_regret"])
    assert np.isclose(
        np.median(values), effect["median_delta_cumulative_deployed_regret"]
    )
    assert np.allclose(
        phase_a.bootstrap_ci(values, seed),
        effect["delta_cumulative_deployed_regret_95ci"],
    )
    assert np.isclose(
        phase_a.sign_flip_pvalue(values, seed + 1),
        effect["delta_cumulative_deployed_regret_signflip_p"],
    )
    assert np.isclose(finals.mean(), effect["mean_delta_final_deployed_regret"])
    assert np.allclose(
        phase_a.bootstrap_ci(finals, seed + 2),
        effect["delta_final_deployed_regret_95ci"],
    )


def main() -> None:
    assert file_sha256(RESULT) == EXPECTED_RESULT_SHA256
    prereg, _, observed_inputs = step54.verify_locks()
    step54.step46.configure_legacy_paths()
    data = step54.step46.phase_a.load_all_data()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["complete"] is True
    assert result["input_sha256"] == observed_inputs
    assert result["fresh_seed_block"] == [91000, 91499]

    for task_index, task in enumerate(step54.TASKS):
        task_result = result["tasks"][task]
        task_data = data[task]
        spec = prereg["frozen_tasks"][task]
        target = int(spec["target_index"])
        target_response = np.asarray(task_data["responses"][target], dtype=object)
        target_oracle = np.asarray(task_data["oracle"][target], dtype=np.float64)
        correct = np.asarray(task_data["correct"], dtype=object)
        ids = list(task_data["ids"])
        public_options = task_data["public_options"]

        for distance_label, distance in (
            ("matrix_free_public_reference_5pct", int(spec["primary_distance_queries"])),
            ("matrix_free_public_reference_1pct", int(spec["secondary_distance_queries"])),
        ):
            rows, supports = isolated_public_rows(
                task,
                target_response=target_response,
                target_oracle=target_oracle,
                correct=correct,
                ids=ids,
                public_options=public_options,
                distance_queries=distance,
            )
            metadata = task_result["construction"][distance_label]
            assert [row_hash(row) for row in rows] == metadata["alias_response_hashes"]
            assert [step54.digest_jsonable(values) for values in supports] == metadata[
                "changed_set_digests"
            ]
            assert all(int(np.sum(row != target_response)) == distance for row in rows)

        seq_rows, seq_supports, inspected = isolated_sequential_rows(
            task,
            target_response=target_response,
            target_oracle=target_oracle,
            correct=correct,
            ids=ids,
            public_options=public_options,
            distance_queries=int(spec["primary_distance_queries"]),
        )
        seq_metadata = task_result["construction"][
            "matrix_free_sequential_reference_5pct"
        ]
        assert [row_hash(row) for row in seq_rows] == seq_metadata[
            "alias_response_hashes"
        ]
        assert [step54.digest_jsonable(values) for values in seq_supports] == seq_metadata[
            "changed_set_digests"
        ]
        assert len(inspected) == seq_metadata["reference_labels_inspected"]
        assert step54.digest_jsonable(inspected) == seq_metadata["inspected_index_digest"]

        pools = step54.step46.phase_a.pools_from_seeds(
            list(range(91000, 91500)), int(spec["examples"]), int(spec["pool_size"])
        )
        assert step54.digest_array(pools) == task_result["pool_matrix_digest"]
        fixed = step54.fixed_random_queries(pools, int(spec["budget"]), task_index)
        assert step54.digest_array(fixed) == task_result["fixed_query_control"][
            "query_matrix_digest_clean"
        ]
        assert task_result["fixed_query_control"]["query_matrices_byte_identical"]

        for condition, condition_index in CONDITION_INDEX.items():
            validate_effect(
                task_result["effects"][condition],
                954000 + task_index * 1000 + condition_index * 10,
            )

    assert step54.adjudicate(result["tasks"]) == result["gates"]
    assert result["gates"]["decision"] == "STRONG_GO"
    print("PASS_STEP54_MATRIX_FREE_REALISM_AUDIT")


if __name__ == "__main__":
    main()
