"""Step 55: locked executable prompt-only endpoint realism audit."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import time
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
PREREG = ROOT / "STEP55_EXECUTABLE_ENDPOINT_REALISM_PREREGISTRATION_2026-08-09.json"
EXECUTION_LOCK = ROOT / "STEP55_EXECUTABLE_ENDPOINT_REALISM_EXECUTION_LOCK_2026-08-09.json"
OUT = ROOT / "STEP55_EXECUTABLE_ENDPOINT_REALISM_RESULTS_2026-08-09.json"
RUNNER = ROOT / "run_step55_executable_endpoint_realism_audit.py"
ENDPOINT_MODULE = ROOT / "step55_prompt_hash_endpoint.py"
EXPECTED_PREREG_SHA256 = "5058ede34407249bd818cce47110957b3124a4bc761bb2bc9f41df58ccccdedd"

INPUTS = {
    "step54_results": ROOT / "STEP54_MATRIX_FREE_REALISM_RESULTS_2026-08-09.json",
    "step54_preregistration": ROOT
    / "STEP54_MATRIX_FREE_REALISM_PREREGISTRATION_2026-08-09.json",
    "medqa_instances": ROOT
    / "external_data/helm_lite_medqa_v1_0_0_raw/openai_gpt-4-0613__instances.json",
    "gsm8k_instances": ROOT
    / "external_data/helm_lite_step25_raw/gsm8k__openai_gpt-4-0613__instances.json",
    "openbookqa_instances": ROOT
    / "external_data/helm_lite_step25_raw/openbookqa__openai_gpt-4-0613__instances.json",
}
EXPECTED_INPUT_SHA256 = {
    "step54_results": "4ab5428c368991f8c9696edda63660ac76b1d516bb7c78d52eeb553ad56a41dd",
    "step54_preregistration": "cf45cdd3e6b1425db2f2c17d4e4283bde95d58356a3fd1ade09c2ef6813e8ed9",
    "medqa_instances": "3b80d9db281936cf24054b8272161a1a583be3b7883ffa19cdab4653e0bd62b2",
    "gsm8k_instances": "050e1d1f860f83ae0e00f1f9e4328593034e39170fdef3fb86f7dd3ee771fa76",
    "openbookqa_instances": "fa6dd7e8c7ac9c335fc6af5bdc3efe7552845d47ed091b4473a95629553656eb",
}
CODE_DEPENDENCIES = {
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
TASKS = ("medqa", "gsm8k", "openbookqa")
INSTANCE_KEYS = {
    "medqa": "medqa_instances",
    "gsm8k": "gsm8k_instances",
    "openbookqa": "openbookqa_instances",
}
ALIASES = 4


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_locks() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    if file_sha256(PREREG) != EXPECTED_PREREG_SHA256:
        raise AssertionError("Step 55 preregistration hash mismatch")
    prereg = load_json(PREREG)
    if prereg["manifest_id"] != "STEP55_EXECUTABLE_ENDPOINT_REALISM_V1":
        raise AssertionError("unexpected Step 55 preregistration")
    observed_inputs = {name: file_sha256(path) for name, path in INPUTS.items()}
    if observed_inputs != EXPECTED_INPUT_SHA256:
        raise AssertionError(
            {"expected_inputs": EXPECTED_INPUT_SHA256, "observed": observed_inputs}
        )
    for name, record in prereg["locked_inputs"].items():
        if observed_inputs[name] != record["sha256"]:
            raise AssertionError(f"preregistered input mismatch: {name}")
    lock = load_json(EXECUTION_LOCK)
    if lock["manifest_id"] != "STEP55_EXECUTABLE_ENDPOINT_REALISM_EXECUTION_LOCK_V1":
        raise AssertionError("unexpected Step 55 execution lock")
    if lock["preregistration_sha256"] != EXPECTED_PREREG_SHA256:
        raise AssertionError("execution lock does not bind preregistration")
    if lock["runner_sha256"] != file_sha256(RUNNER):
        raise AssertionError("execution lock does not bind runner")
    if lock["input_sha256"] != observed_inputs:
        raise AssertionError("execution lock input mismatch")
    observed_code = {
        name: file_sha256(path) for name, path in CODE_DEPENDENCIES.items()
    }
    if lock["code_dependency_sha256"] != observed_code:
        raise AssertionError(
            {"locked_code": lock["code_dependency_sha256"], "observed": observed_code}
        )
    return prereg, lock, observed_inputs


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def build_prompt_envelopes(
    task: str, instances: list[dict[str, Any]], data: dict[str, Any]
) -> tuple[list[PromptEnvelope], dict[str, Any]]:
    ids = list(data["ids"])
    instance_ids = [str(instance["id"]) for instance in instances]
    id_order_exact = instance_ids == [str(value) for value in ids]
    if not id_order_exact:
        raise AssertionError(f"{task}: instance ID order mismatch")
    envelopes: list[PromptEnvelope] = []
    options_match = True
    correctness_tags_absent = True
    for query, instance in enumerate(instances):
        input_text = str(instance["input"]["text"])
        if task in {"medqa", "openbookqa"}:
            options = tuple(
                str(reference["output"]["text"])
                for reference in instance["references"]
            )
            expected_options = tuple(str(value) for value in data["public_options"][query])
            options_match = options_match and options == expected_options
        else:
            options = ()
        envelope = PromptEnvelope(input_text=input_text, ordered_public_options=options)
        correctness_tags_absent = correctness_tags_absent and set(
            field.name for field in fields(envelope)
        ) == {"input_text", "ordered_public_options"}
        envelopes.append(envelope)
    hashes = [prompt_sha256(envelope) for envelope in envelopes]
    unique = len(set(hashes)) == len(hashes)
    if not (options_match and correctness_tags_absent and unique):
        raise AssertionError(
            {
                "task": task,
                "options_match": options_match,
                "runtime_fields_valid": correctness_tags_absent,
                "unique_prompt_hashes": unique,
            }
        )
    return envelopes, {
        "instance_id_order_exact_before_discard": id_order_exact,
        "runtime_prompt_fields": [field.name for field in fields(PromptEnvelope)],
        "forbidden_runtime_fields_absent": correctness_tags_absent,
        "public_options_exact": options_match,
        "prompt_hashes_unique": unique,
        "prompt_count": len(hashes),
        "prompt_hash_set_digest": digest_json(sorted(hashes)),
    }


def row_hash(row: np.ndarray) -> str:
    return hashlib.sha256(
        json.dumps(
            row.tolist(), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def build_and_run_endpoints(
    task: str,
    task_index: int,
    data: dict[str, Any],
    instances: list[dict[str, Any]],
    step54_result: dict[str, Any],
    step54_prereg: dict[str, Any],
) -> dict[str, Any]:
    phase_a = step54.step46.phase_a
    spec = step54_prereg["frozen_tasks"][task]
    target = int(spec["target_index"])
    distance = int(spec["primary_distance_queries"])
    clean = phase_a.base_scenario(data)
    target_response = np.asarray(clean["responses"][target], dtype=object)
    target_oracle = np.asarray(clean["oracle"][target], dtype=np.float64)
    envelopes, prompt_integrity = build_prompt_envelopes(task, instances, data)

    isolated_rows, supports = validate54.isolated_public_rows(
        task,
        target_response=target_response,
        target_oracle=target_oracle,
        correct=np.asarray(data["correct"], dtype=object),
        ids=list(data["ids"]),
        public_options=data["public_options"],
        distance_queries=distance,
    )
    locked_metadata = step54_result["tasks"][task]["construction"][
        "matrix_free_public_reference_5pct"
    ]
    if [row_hash(row) for row in isolated_rows] != locked_metadata[
        "alias_response_hashes"
    ]:
        raise AssertionError(f"{task}: isolated rows drift from Step 54")

    parent_map = {
        prompt_sha256(envelope): str(target_response[query])
        for query, envelope in enumerate(envelopes)
    }
    parent = ParentReplayEndpoint(parent_map, unknown_output="STEP55_UNSEEN_PROMPT")
    endpoint_rows: list[np.ndarray] = []
    endpoint_metadata: list[dict[str, Any]] = []
    for alias in range(ALIASES):
        overrides = {
            prompt_sha256(envelopes[query]): str(isolated_rows[alias][query])
            for query in supports[alias]
        }
        endpoint = PromptHashWrapperEndpoint(parent, overrides)
        outputs: list[str] = []
        trigger_count = 0
        started = time.perf_counter_ns()
        for envelope in envelopes:
            reply = endpoint.predict(envelope)
            outputs.append(reply.output)
            trigger_count += int(reply.override_fired)
        elapsed_ns = time.perf_counter_ns() - started
        row = np.asarray(outputs, dtype=object)
        manifest = serialize_public_manifest(
            task=task, alias=alias, overrides_by_prompt_hash=overrides
        )
        manifest_payload = json.loads(manifest.decode("utf-8"))
        allowed_manifest_keys = {
            "alias",
            "manifest_version",
            "overrides",
            "task",
        }
        manifest_keys_valid = set(manifest_payload) == allowed_manifest_keys
        override_schema_valid = all(
            len(key) == 64
            and all(character in "0123456789abcdef" for character in key)
            and isinstance(value, str)
            for key, value in manifest_payload["overrides"].items()
        )
        mutation_trigger_count = 0
        for query in supports[alias]:
            original = envelopes[query]
            mutated = PromptEnvelope(
                input_text=original.input_text + "?",
                ordered_public_options=original.ordered_public_options,
            )
            mutation_trigger_count += int(endpoint.predict(mutated).override_fired)
        exact_row = bool(np.array_equal(row, isolated_rows[alias]))
        if not (
            exact_row
            and trigger_count == distance
            and manifest_keys_valid
            and override_schema_valid
            and len(manifest) <= 32768
            and mutation_trigger_count == 0
        ):
            raise AssertionError(
                {
                    "task": task,
                    "alias": alias,
                    "exact_row": exact_row,
                    "trigger_count": trigger_count,
                    "manifest_keys_valid": manifest_keys_valid,
                    "override_schema_valid": override_schema_valid,
                    "manifest_bytes": len(manifest),
                    "mutation_trigger_count": mutation_trigger_count,
                }
            )
        endpoint_rows.append(row)
        endpoint_metadata.append(
            {
                "alias": alias,
                "response_hash": row_hash(row),
                "exact_step54_row": exact_row,
                "override_entries": len(overrides),
                "runtime_override_count": trigger_count,
                "manifest_bytes": len(manifest),
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "manifest_keys_valid": manifest_keys_valid,
                "override_schema_valid": override_schema_valid,
                "mutated_supported_prompts_tested": len(supports[alias]),
                "mutated_prompt_override_count": mutation_trigger_count,
                "runtime_calls": len(envelopes),
                "runtime_total_ns": elapsed_ns,
                "runtime_mean_microseconds_per_call": elapsed_ns
                / len(envelopes)
                / 1000.0,
            }
        )

    endpoint_scenario = step54.assemble_refined_scenario(
        clean,
        target,
        endpoint_rows,
        [target_oracle.copy() for _ in range(ALIASES)],
        "step55_prompt_only_endpoint_replay",
    )
    direct_scenario, _ = step54.build_public_reference_variant(
        task, data, clean, target=target, distance_queries=distance
    )
    scenario_response_exact = bool(
        np.array_equal(endpoint_scenario["responses"], direct_scenario["responses"])
    )
    scenario_oracle_exact = bool(
        np.array_equal(endpoint_scenario["oracle"], direct_scenario["oracle"])
    )
    scenario_codes_exact = bool(
        np.array_equal(endpoint_scenario["codes"], direct_scenario["codes"])
    )
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
    run_fields = (
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
    run_field_exact = {
        field: bool(np.array_equal(endpoint_runs[field], direct_runs[field]))
        for field in run_fields
    }
    comparison = phase_a.compare_runs(
        clean_runs,
        endpoint_runs,
        954000 + task_index * 1000 + 10,
        include_arrays=True,
    )
    locked_effect = step54_result["tasks"][task]["effects"][
        "matrix_free_public_reference_5pct"
    ]
    comparison_exact = comparison == locked_effect
    if not (
        scenario_response_exact
        and scenario_oracle_exact
        and scenario_codes_exact
        and all(run_field_exact.values())
        and comparison_exact
    ):
        raise AssertionError(
            {
                "task": task,
                "scenario_response_exact": scenario_response_exact,
                "scenario_oracle_exact": scenario_oracle_exact,
                "scenario_codes_exact": scenario_codes_exact,
                "run_field_exact": run_field_exact,
                "comparison_exact": comparison_exact,
            }
        )

    return {
        "task": task,
        "target_index": target,
        "target": spec["target"],
        "distance_queries": distance,
        "prompt_integrity": prompt_integrity,
        "endpoint_runtime_contract": {
            "prompt_envelope_fields": [field.name for field in fields(PromptEnvelope)],
            "predict_signature": str(inspect.signature(PromptHashWrapperEndpoint.predict)),
            "instance_id_runtime_access": False,
            "reference_runtime_access": False,
            "peer_response_build_or_runtime_access": False,
            "pool_or_selector_runtime_access": False,
        },
        "endpoints": endpoint_metadata,
        "endpoint_row_hashes": [row_hash(row) for row in endpoint_rows],
        "locked_step54_row_hashes": locked_metadata["alias_response_hashes"],
        "scenario_equivalence": {
            "responses_exact": scenario_response_exact,
            "oracle_exact": scenario_oracle_exact,
            "codes_exact": scenario_codes_exact,
        },
        "run_field_exact": run_field_exact,
        "endpoint_run_digests": {
            field: digest_array(np.asarray(endpoint_runs[field])) for field in run_fields
        },
        "locked_step54_comparison_exact": comparison_exact,
        "comparison": comparison,
    }


def audit_interfaces(prereg: dict[str, Any]) -> dict[str, Any]:
    evidence = prereg["official_interface_evidence_frozen_before_endpoint_execution"]
    commits = {
        "model_selector": git_commit(ROOT / "external/model-selector"),
        "coda": git_commit(ROOT / "external/coda"),
        "older_llm_selector_artifact": git_commit(ROOT / "external/llm-selector"),
    }
    commits_exact = {
        name: commits[name] == evidence[name]["commit"] for name in commits
    }
    interface_text = CODE_DEPENDENCIES["model_selector_interface"].read_text(
        encoding="utf-8"
    )
    adapter_text = CODE_DEPENDENCIES["model_selector_npy_adapter"].read_text(
        encoding="utf-8"
    )
    coda_text = CODE_DEPENDENCIES["coda_dataset_interface"].read_text(
        encoding="utf-8"
    )
    older_text = CODE_DEPENDENCIES["older_llm_selector_main"].read_text(
        encoding="utf-8"
    )
    model_selector_contract = (
        "only thing a model needs to provide" in interface_text
        and "def make_prediction(self, x)" in interface_text
        and "return self.predictions[x]" in adapter_text
    )
    coda_contract = (
        "tensor of shape (H,N,C)" in coda_text
        and "self.preds = torch.load" in coda_text
    )
    older_llm_contract = (
        "Shape of judge_scores is (num_models, num_queries)" in older_text
        and 'judge_data = json.load(open(f"{data_path}/judge.json"' in older_text
    )
    forbidden_tokens = ("lineage", "attestation", "provenance")
    no_lineage_fields = {
        "model_selector_core": not any(
            token in (interface_text + adapter_text).casefold()
            for token in forbidden_tokens
        ),
        "coda_dataset_core": not any(
            token in coda_text.casefold() for token in forbidden_tokens
        ),
        "older_llm_selector_core": not any(
            token in older_text.casefold() for token in forbidden_tokens
        ),
    }
    independent_contract_count = int(model_selector_contract) + int(coda_contract)
    pass_gate = bool(
        all(commits_exact.values())
        and independent_contract_count >= 2
        and all(no_lineage_fields.values())
        and evidence["select_llm_2026"]["primary_source"]
        == "https://arxiv.org/abs/2605.24981"
    )
    return {
        "local_repository_commits": commits,
        "local_commits_match_preregistered_remote_heads": commits_exact,
        "model_selector_make_prediction_or_array_contract": model_selector_contract,
        "coda_prediction_tensor_contract": coda_contract,
        "older_llm_selector_output_table_contract": older_llm_contract,
        "independent_official_implementation_contract_count": independent_contract_count,
        "core_input_files_have_no_lineage_attestation_or_provenance_field": no_lineage_fields,
        "select_llm_primary_source_output_only_black_box_evidence_locked": True,
        "production_untrusted_registration_primary_evidence_found": False,
        "scope_classification": "RESEARCH_OUTPUT_INTERFACE_FEASIBLE_PRODUCTION_REGISTRY_ADMISSION_UNPROVEN",
        "pass": pass_gate,
    }


def adjudicate(
    task_results: dict[str, dict[str, Any]], interface_audit: dict[str, Any]
) -> dict[str, Any]:
    g0_by_task = {
        task: all(
            result["prompt_integrity"][key]
            for key in (
                "instance_id_order_exact_before_discard",
                "forbidden_runtime_fields_absent",
                "public_options_exact",
                "prompt_hashes_unique",
            )
        )
        for task, result in task_results.items()
    }
    g1_by_task = {
        task: bool(
            result["endpoint_row_hashes"] == result["locked_step54_row_hashes"]
            and all(result["scenario_equivalence"].values())
            and all(result["run_field_exact"].values())
            and result["locked_step54_comparison_exact"]
        )
        for task, result in task_results.items()
    }
    g2_by_task = {
        task: bool(
            result["endpoint_runtime_contract"]["instance_id_runtime_access"] is False
            and result["endpoint_runtime_contract"]["reference_runtime_access"] is False
            and result["endpoint_runtime_contract"][
                "peer_response_build_or_runtime_access"
            ]
            is False
            and result["endpoint_runtime_contract"]["pool_or_selector_runtime_access"]
            is False
            and all(
                endpoint["manifest_bytes"] <= 32768
                and endpoint["runtime_override_count"]
                == result["distance_queries"]
                and endpoint["mutated_prompt_override_count"] == 0
                and endpoint["manifest_keys_valid"]
                and endpoint["override_schema_valid"]
                for endpoint in result["endpoints"]
            )
        )
        for task, result in task_results.items()
    }
    g0 = all(g0_by_task.values())
    g1 = all(g1_by_task.values())
    g2 = all(g2_by_task.values())
    g3 = bool(interface_audit["pass"])
    if g0 and g1 and g2 and g3:
        decision = "STRONG_INTERFACE_PASS_WITH_SCOPE_LIMIT"
    elif g0 and g1 and g2:
        decision = "NARROWED_INTERFACE_PASS"
    else:
        decision = "NO_GO_ENDPOINT_REALISM"
    return {
        "G0_prompt_corpus_integrity": {"pass": g0, "by_task": g0_by_task},
        "G1_exact_endpoint_replay": {"pass": g1, "by_task": g1_by_task},
        "G2_runtime_capability_and_size": {"pass": g2, "by_task": g2_by_task},
        "G3_research_interface_support": {"pass": g3},
        "G4_scope_honesty": {
            "pass": True,
            "production_registry_admission_proven": False,
            "required_scope_language": interface_audit["scope_classification"],
        },
        "decision": decision,
    }


def main() -> None:
    started = time.time()
    prereg, lock, observed_inputs = verify_locks()
    step54.step46.configure_legacy_paths()
    data = step54.step46.phase_a.load_all_data()
    step54_result = load_json(INPUTS["step54_results"])
    step54_prereg = load_json(INPUTS["step54_preregistration"])
    task_results: dict[str, dict[str, Any]] = {}
    for task_index, task in enumerate(TASKS):
        instances = json.loads(INPUTS[INSTANCE_KEYS[task]].read_text(encoding="utf-8"))
        task_results[task] = build_and_run_endpoints(
            task,
            task_index,
            data[task],
            instances,
            step54_result,
            step54_prereg,
        )
        print(
            task,
            "exact=",
            task_results[task]["locked_step54_comparison_exact"],
            "manifest_bytes=",
            [endpoint["manifest_bytes"] for endpoint in task_results[task]["endpoints"]],
            flush=True,
        )
    interface_audit = audit_interfaces(prereg)
    gates = adjudicate(task_results, interface_audit)
    payload = {
        "manifest_id": "STEP55_EXECUTABLE_ENDPOINT_REALISM_RESULTS_V1",
        "complete": True,
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "execution_lock_sha256": file_sha256(EXECUTION_LOCK),
        "runner_sha256": file_sha256(RUNNER),
        "endpoint_module_sha256": file_sha256(ENDPOINT_MODULE),
        "input_sha256": observed_inputs,
        "code_dependency_sha256": lock["code_dependency_sha256"],
        "tasks": task_results,
        "official_interface_audit": interface_audit,
        "gates": gates,
        "elapsed_seconds": time.time() - started,
    }
    OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(gates, indent=2), flush=True)
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
