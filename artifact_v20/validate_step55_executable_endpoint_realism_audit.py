"""Independent validation of the locked Step 55 executable-endpoint audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
import unicodedata
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

import run_step54_matrix_free_realism_audit as step54
import validate_step54_matrix_free_realism_audit as validate54
from step55_prompt_hash_endpoint import (
    ParentReplayEndpoint,
    PromptEnvelope,
    PromptHashWrapperEndpoint,
    prompt_sha256,
    serialize_public_manifest,
)


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "STEP55_EXECUTABLE_ENDPOINT_REALISM_RESULTS_2026-08-09.json"
PREREG = ROOT / "STEP55_EXECUTABLE_ENDPOINT_REALISM_PREREGISTRATION_2026-08-09.json"
LOCK = ROOT / "STEP55_EXECUTABLE_ENDPOINT_REALISM_EXECUTION_LOCK_2026-08-09.json"
RUNNER = ROOT / "run_step55_executable_endpoint_realism_audit.py"
ENDPOINT_MODULE = ROOT / "step55_prompt_hash_endpoint.py"
STEP54_RESULT = ROOT / "STEP54_MATRIX_FREE_REALISM_RESULTS_2026-08-09.json"
STEP54_PREREG = ROOT / "STEP54_MATRIX_FREE_REALISM_PREREGISTRATION_2026-08-09.json"

EXPECTED_RESULT_SHA256 = "a6bcf13fc15aef392906a3ce2a782d6a2c746f514e53b8e9ab35afb85f2bed02"
EXPECTED_PREREG_SHA256 = "5058ede34407249bd818cce47110957b3124a4bc761bb2bc9f41df58ccccdedd"
EXPECTED_LOCK_SHA256 = "b3ff41a6845c70095188c20840b97d9bec456a49eb0b41d510744e14d582062c"
EXPECTED_RUNNER_SHA256 = "d29a3123f138398458b32d63ceb59733bbd9bcdb08b11a58a3178b059556f294"
EXPECTED_ENDPOINT_SHA256 = "16105271ea7c9c9c5e80e225d8b12cb9a4f5e49227c6adf7c2fb99de795d7684"

TASKS = ("medqa", "gsm8k", "openbookqa")
INSTANCE_FILES = {
    "medqa": ROOT
    / "external_data/helm_lite_medqa_v1_0_0_raw/openai_gpt-4-0613__instances.json",
    "gsm8k": ROOT
    / "external_data/helm_lite_step25_raw/gsm8k__openai_gpt-4-0613__instances.json",
    "openbookqa": ROOT
    / "external_data/helm_lite_step25_raw/openbookqa__openai_gpt-4-0613__instances.json",
}
RUN_FIELDS = (
    "queried",
    "selected_local_path",
    "final_local",
    "final_parent",
    "deploy_regret_path",
    "expanded_regret_path",
    "root_regret_path",
    "cumulative_deploy_regret",
    "final_deploy_regret",
    "cumulative_expanded_regret",
    "final_expanded_regret",
    "cumulative_root_regret",
    "final_root_regret",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def row_hash(row: np.ndarray) -> str:
    payload = json.dumps(
        row.tolist(), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def independent_prompt_hash(prompt: PromptEnvelope) -> str:
    def norm(value: str) -> str:
        return unicodedata.normalize("NFC", str(value).replace("\r\n", "\n")).strip()

    payload = {
        "input_text": norm(prompt.input_text),
        "ordered_public_options": [norm(value) for value in prompt.ordered_public_options],
    }
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def repo_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def validate_locks(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    assert file_sha256(RESULT) == EXPECTED_RESULT_SHA256
    assert file_sha256(PREREG) == EXPECTED_PREREG_SHA256
    assert file_sha256(LOCK) == EXPECTED_LOCK_SHA256
    assert file_sha256(RUNNER) == EXPECTED_RUNNER_SHA256
    assert file_sha256(ENDPOINT_MODULE) == EXPECTED_ENDPOINT_SHA256
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["outcomes_seen_before_lock"] is False
    assert lock["preregistration_sha256"] == EXPECTED_PREREG_SHA256
    assert lock["runner_sha256"] == EXPECTED_RUNNER_SHA256
    assert result["preregistration_sha256"] == EXPECTED_PREREG_SHA256
    assert result["execution_lock_sha256"] == EXPECTED_LOCK_SHA256
    assert result["runner_sha256"] == EXPECTED_RUNNER_SHA256
    assert result["endpoint_module_sha256"] == EXPECTED_ENDPOINT_SHA256

    for name, digest in result["input_sha256"].items():
        filename = prereg["locked_inputs"][name]["file"]
        assert file_sha256(ROOT / filename) == digest
        assert digest == prereg["locked_inputs"][name]["sha256"]
        assert digest == lock["input_sha256"][name]
    for name, digest in lock["code_dependency_sha256"].items():
        mapping = {
            "endpoint_module": ENDPOINT_MODULE,
            "step54_runner": ROOT / "run_step54_matrix_free_realism_audit.py",
            "step54_validator": ROOT / "validate_step54_matrix_free_realism_audit.py",
            "step29_runner": ROOT / "run_step29_locked_phase_a.py",
            "step46_runner": ROOT / "run_step46_clone_robust_soft_weighting.py",
            "medqa_loader": ROOT / "run_step24_real_response_current_method_gate.py",
            "external_task_loader": ROOT
            / "run_step25_current_method_external_generality_gate.py",
            "model_selector_interface": ROOT
            / "external/model-selector/src/models/interface_inferable.py",
            "model_selector_npy_adapter": ROOT
            / "external/model-selector/src/models/npy_data_model.py",
            "coda_dataset_interface": ROOT / "external/coda/coda/datasets.py",
            "older_llm_selector_main": ROOT / "external/llm-selector/main.py",
        }
        assert file_sha256(mapping[name]) == digest
        assert result["code_dependency_sha256"][name] == digest
    return prereg, lock


def build_envelopes(
    task: str, instances: list[dict[str, Any]], data: dict[str, Any]
) -> list[PromptEnvelope]:
    assert [str(item["id"]) for item in instances] == [str(value) for value in data["ids"]]
    envelopes: list[PromptEnvelope] = []
    for query, item in enumerate(instances):
        if task in {"medqa", "openbookqa"}:
            options = tuple(str(ref["output"]["text"]) for ref in item["references"])
            assert options == tuple(str(value) for value in data["public_options"][query])
        else:
            options = ()
        envelopes.append(
            PromptEnvelope(
                input_text=str(item["input"]["text"]),
                ordered_public_options=options,
            )
        )
    assert [field.name for field in fields(PromptEnvelope)] == [
        "input_text",
        "ordered_public_options",
    ]
    module_hashes = [prompt_sha256(prompt) for prompt in envelopes]
    own_hashes = [independent_prompt_hash(prompt) for prompt in envelopes]
    assert module_hashes == own_hashes
    assert len(module_hashes) == len(set(module_hashes))
    return envelopes


def validate_task(
    task: str,
    task_index: int,
    data: dict[str, Any],
    step54_result: dict[str, Any],
    step54_prereg: dict[str, Any],
    recorded: dict[str, Any],
) -> None:
    phase_a = step54.step46.phase_a
    spec = step54_prereg["frozen_tasks"][task]
    target = int(spec["target_index"])
    distance = int(spec["primary_distance_queries"])
    clean = phase_a.base_scenario(data)
    target_response = np.asarray(clean["responses"][target], dtype=object)
    target_oracle = np.asarray(clean["oracle"][target], dtype=np.float64)
    instances = json.loads(INSTANCE_FILES[task].read_text(encoding="utf-8"))
    envelopes = build_envelopes(task, instances, data)
    prompt_hashes = [prompt_sha256(prompt) for prompt in envelopes]
    assert compact_digest(sorted(prompt_hashes)) == recorded["prompt_integrity"][
        "prompt_hash_set_digest"
    ]

    rows, supports = validate54.isolated_public_rows(
        task,
        target_response=target_response,
        target_oracle=target_oracle,
        correct=np.asarray(data["correct"], dtype=object),
        ids=list(data["ids"]),
        public_options=data["public_options"],
        distance_queries=distance,
    )
    locked_rows = step54_result["tasks"][task]["construction"][
        "matrix_free_public_reference_5pct"
    ]["alias_response_hashes"]
    assert [row_hash(row) for row in rows] == locked_rows
    assert recorded["endpoint_row_hashes"] == locked_rows

    parent = ParentReplayEndpoint(
        {
            prompt_hashes[query]: str(target_response[query])
            for query in range(len(envelopes))
        },
        unknown_output="STEP55_UNSEEN_PROMPT",
    )
    replayed_rows: list[np.ndarray] = []
    for alias, support in enumerate(supports):
        overrides = {
            prompt_hashes[query]: str(rows[alias][query]) for query in support
        }
        endpoint = PromptHashWrapperEndpoint(parent, overrides)
        replies = [endpoint.predict(prompt) for prompt in envelopes]
        replayed = np.asarray([reply.output for reply in replies], dtype=object)
        assert np.array_equal(replayed, rows[alias])
        assert sum(reply.override_fired for reply in replies) == distance
        manifest = serialize_public_manifest(
            task=task, alias=alias, overrides_by_prompt_hash=overrides
        )
        payload = json.loads(manifest.decode("utf-8"))
        assert set(payload) == {"alias", "manifest_version", "overrides", "task"}
        assert len(manifest) <= 32768
        endpoint_record = recorded["endpoints"][alias]
        assert len(manifest) == endpoint_record["manifest_bytes"]
        assert hashlib.sha256(manifest).hexdigest() == endpoint_record["manifest_sha256"]
        assert row_hash(replayed) == endpoint_record["response_hash"]
        for query in support:
            prompt = envelopes[query]
            mutated = PromptEnvelope(
                input_text=prompt.input_text + "?",
                ordered_public_options=prompt.ordered_public_options,
            )
            assert endpoint.predict(mutated).override_fired is False
        replayed_rows.append(replayed)

    endpoint_scenario = step54.assemble_refined_scenario(
        clean,
        target,
        replayed_rows,
        [target_oracle.copy() for _ in replayed_rows],
        "independent_step55_validation",
    )
    direct_scenario, _ = step54.build_public_reference_variant(
        task, data, clean, target=target, distance_queries=distance
    )
    assert np.array_equal(endpoint_scenario["responses"], direct_scenario["responses"])
    assert np.array_equal(endpoint_scenario["oracle"], direct_scenario["oracle"])
    assert np.array_equal(endpoint_scenario["codes"], direct_scenario["codes"])

    pools = phase_a.pools_from_seeds(
        list(range(91000, 91500)), int(spec["examples"]), int(spec["pool_size"])
    )
    clean_runs = phase_a.run_scenario(
        clean, data["oracle"], pools, int(spec["budget"]), float(spec["temperature"])
    )
    endpoint_runs = phase_a.run_scenario(
        endpoint_scenario,
        data["oracle"],
        pools,
        int(spec["budget"]),
        float(spec["temperature"]),
    )
    direct_runs = phase_a.run_scenario(
        direct_scenario,
        data["oracle"],
        pools,
        int(spec["budget"]),
        float(spec["temperature"]),
    )
    for field in RUN_FIELDS:
        assert np.array_equal(endpoint_runs[field], direct_runs[field])
        digest = hashlib.sha256(
            np.ascontiguousarray(np.asarray(endpoint_runs[field])).tobytes()
        ).hexdigest()
        assert digest == recorded["endpoint_run_digests"][field]
    comparison = phase_a.compare_runs(
        clean_runs,
        endpoint_runs,
        954000 + task_index * 1000 + 10,
        include_arrays=True,
    )
    locked_effect = step54_result["tasks"][task]["effects"][
        "matrix_free_public_reference_5pct"
    ]
    assert comparison == locked_effect
    assert comparison == recorded["comparison"]


def validate_interfaces(prereg: dict[str, Any], result: dict[str, Any]) -> None:
    evidence = prereg["official_interface_evidence_frozen_before_endpoint_execution"]
    commits = {
        "model_selector": repo_commit(ROOT / "external/model-selector"),
        "coda": repo_commit(ROOT / "external/coda"),
        "older_llm_selector_artifact": repo_commit(ROOT / "external/llm-selector"),
    }
    for name, commit in commits.items():
        assert commit == evidence[name]["commit"]
    interface = (ROOT / "external/model-selector/src/models/interface_inferable.py").read_text(encoding="utf-8")
    adapter = (ROOT / "external/model-selector/src/models/npy_data_model.py").read_text(encoding="utf-8")
    coda = (ROOT / "external/coda/coda/datasets.py").read_text(encoding="utf-8")
    older = (ROOT / "external/llm-selector/main.py").read_text(encoding="utf-8")
    assert "def make_prediction(self, x)" in interface
    assert "return self.predictions[x]" in adapter
    assert "tensor of shape (H,N,C)" in coda
    assert "self.preds = torch.load" in coda
    assert "Shape of judge_scores is (num_models, num_queries)" in older
    assert result["official_interface_audit"][
        "independent_official_implementation_contract_count"
    ] >= 2
    assert result["official_interface_audit"]["production_untrusted_registration_primary_evidence_found"] is False
    assert result["official_interface_audit"]["scope_classification"] == (
        "RESEARCH_OUTPUT_INTERFACE_FEASIBLE_PRODUCTION_REGISTRY_ADMISSION_UNPROVEN"
    )


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    prereg, _ = validate_locks(result)
    assert result["manifest_id"] == "STEP55_EXECUTABLE_ENDPOINT_REALISM_RESULTS_V1"
    assert result["complete"] is True
    step54.step46.configure_legacy_paths()
    all_data = step54.step46.phase_a.load_all_data()
    step54_result = json.loads(STEP54_RESULT.read_text(encoding="utf-8"))
    step54_prereg = json.loads(STEP54_PREREG.read_text(encoding="utf-8"))
    for task_index, task in enumerate(TASKS):
        validate_task(
            task,
            task_index,
            all_data[task],
            step54_result,
            step54_prereg,
            result["tasks"][task],
        )
    validate_interfaces(prereg, result)
    assert all(result["gates"][key]["pass"] for key in (
        "G0_prompt_corpus_integrity",
        "G1_exact_endpoint_replay",
        "G2_runtime_capability_and_size",
        "G3_research_interface_support",
        "G4_scope_honesty",
    ))
    assert result["gates"]["G4_scope_honesty"]["production_registry_admission_proven"] is False
    assert result["gates"]["decision"] == "STRONG_INTERFACE_PASS_WITH_SCOPE_LIMIT"
    print("PASS_STEP55_EXECUTABLE_ENDPOINT_REALISM_AUDIT")


if __name__ == "__main__":
    main()
