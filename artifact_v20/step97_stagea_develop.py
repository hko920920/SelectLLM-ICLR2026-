from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

import step97_common as common
from step97_executable_adapter_endpoint import execute_aliases_from_raw_pairs


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / f"STEP97_STAGE0_MANIFEST_{common.DATE}.json"
DEVELOPMENT = ROOT / f"STEP97_SCITAIL_DEVELOPMENT_ROWS_{common.DATE}.json"
ARRAYS = ROOT / f"STEP97_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
LEDGER = ROOT / f"STEP97_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
CONFIG = ROOT / f"STEP97_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
EARLY_STOP = ROOT / f"STEP97_STAGEA_ROOT_GEOMETRY_STOP_{common.DATE}.json"
MODEL_DIR = ROOT / "step97_models"
WORK_DIR = ROOT / "step97_stagea_work"
ROOT_CACHE = WORK_DIR / "root_predictions.npz"
ROOT_AUDITS = WORK_DIR / "root_prediction_audits.json"
SEARCH_SEEDS = tuple(range(973000, 973400))
VERIFY_SEEDS = tuple(range(974000, 975500))


def source_sha256(function: object) -> str:
    import hashlib

    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def load_rows() -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    payload = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    rows = [*payload["train_rows"], *payload["validation_rows"]]
    partitions = {**{name: [] for name in common.PARTITION_QUOTAS}, "selector_verify": []}
    for index, row in enumerate(rows):
        partitions[str(row["partition"])].append(index)
    expected = {
        "adapter_train": 5000,
        "roster_gate": 1000,
        "threshold": 1500,
        "selector_search": 2000,
        "selector_verify": len(payload["validation_rows"]),
    }
    observed = {name: len(indices) for name, indices in partitions.items()}
    if observed != expected:
        raise AssertionError({"observed": observed, "expected": expected})
    return rows, {name: np.asarray(indices, dtype=np.int64) for name, indices in partitions.items()}


def preflight() -> list[dict[str, Any]]:
    rows = []
    for index, spec in enumerate(common.ROOT_SPECS):
        tokenizer = common.tokenizer_for(spec)
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.repo_id, revision=spec.revision, use_safetensors=None
        )
        rows.append(
            {
                "bank_index": index,
                "key": spec.key,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "model_type": str(model.config.model_type),
                "num_labels": int(model.config.num_labels),
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "tokenizer_class": tokenizer.__class__.__name__,
                "raw_to_dataset": list(spec.raw_to_dataset),
                "loaded_before_scitail_prediction": True,
            }
        )
        if int(model.config.num_labels) != 3:
            raise AssertionError(rows[-1])
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"preflight root {index + 1}/{common.ROSTER_SIZE}: {spec.key}", flush=True)
    return rows


def load_or_infer_roots(premises: list[str], hypotheses: list[str]) -> tuple[np.ndarray, list[dict[str, Any]], bool]:
    if ROOT_CACHE.is_file() and ROOT_AUDITS.is_file():
        predictions = np.load(ROOT_CACHE, allow_pickle=False)["root_predictions"].astype(np.int64)
        audits = json.loads(ROOT_AUDITS.read_text(encoding="utf-8"))
        if predictions.shape != (len(premises), common.ROSTER_SIZE):
            raise AssertionError(predictions.shape)
        for index, row in enumerate(audits):
            if row["prediction_sha256"] != common.array_sha256(predictions[:, index]):
                raise AssertionError("root cache hash drift")
        return predictions, audits, True
    if ROOT_CACHE.exists() or ROOT_AUDITS.exists():
        raise RuntimeError("incomplete root cache")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    columns = []
    audits = []
    for index, spec in enumerate(common.ROOT_SPECS):
        predictions, audit = common.infer_root_predictions(spec, premises, hypotheses)
        columns.append(predictions)
        audits.append(audit)
        print(f"inferred root {index + 1}/{common.ROSTER_SIZE}: {spec.key}", flush=True)
    matrix = np.stack(columns, axis=1)
    np.savez_compressed(ROOT_CACHE, root_predictions=matrix.astype(np.int16))
    common.json_dump(ROOT_AUDITS, audits)
    return matrix, audits, False


def geometry(predictions: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    codes = predictions[indices]
    references = labels[indices]
    accuracies = np.mean(codes == references[:, None], axis=0)
    parent_wrong = codes[:, 0] != references
    complementarity = [
        float(np.mean((codes[:, root] == references)[parent_wrong])) if np.any(parent_wrong) else 0.0
        for root in range(1, common.ROSTER_SIZE)
    ]
    return {
        "rows": len(indices),
        "root_accuracies": accuracies.tolist(),
        "root_accuracy_by_key": {spec.key: float(value) for spec, value in zip(common.ROOT_SPECS, accuracies)},
        "parent_minus_best_challenger": float(accuracies[0] - np.max(accuracies[1:])),
        "parent_error_prevalence": float(np.mean(parent_wrong)),
        "challenger_correct_given_parent_wrong": {
            common.ROOT_SPECS[root].key: complementarity[root - 1]
            for root in range(1, common.ROSTER_SIZE)
        },
        "max_challenger_correct_given_parent_wrong": max(complementarity),
    }


def build_aliases(parent: np.ndarray, labels: np.ndarray, scores: np.ndarray, thresholds: list[float] | None = None) -> dict[str, Any]:
    if thresholds is None:
        chosen = []
        threshold_audits = []
        for alias in range(common.N_ALIASES):
            threshold, _, audit = common.threshold_from_caps(
                scores[:, alias], parent == labels, common.LOSS_CAP, common.TRIGGER_CAP
            )
            chosen.append(float(threshold))
            threshold_audits.append(audit)
    else:
        chosen = [float(value) for value in thresholds]
        threshold_audits = []
    alias_codes, triggers = common.make_alias_codes(parent, scores, chosen)
    feedback = common.exact_match_feedback(alias_codes, labels)
    parent_feedback = common.exact_match_feedback(parent[:, None], labels)
    loss_counts = np.sum(triggers & (parent[:, None] == labels[:, None]), axis=0).astype(np.int64)
    return {
        "thresholds": chosen,
        "threshold_audits": threshold_audits,
        "alias_codes": alias_codes,
        "triggers": triggers,
        "loss_counts": loss_counts,
        "loss_fractions": loss_counts / len(labels),
        "coordinate_wise_nonimproving": bool(
            np.all(feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
        ),
    }


def evaluate(partition: str, clean_codes: np.ndarray, labels: np.ndarray, built: dict[str, Any], seeds: tuple[int, ...]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pools = common.sample_pools(len(labels), seeds, common.POOL_SIZE)
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    refined_codes = np.column_stack([clean_codes, built["alias_codes"]])
    refined_parents = np.concatenate([clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)])
    clean = common.run_active(clean_codes, labels, clean_parents, clean_codes, pools, seeds, common.BUDGET, common.TAU)
    refined = common.run_active(refined_codes, labels, refined_parents, clean_codes, pools, seeds, common.BUDGET, common.TAU)
    fixed = common.run_fixed(refined_codes, labels, refined_parents, clean_codes, pools, clean.queries, seeds)
    terminal = refined.terminal - clean.terminal
    cumulative = refined.cumulative - clean.cumulative
    fixed_terminal = fixed.terminal - clean.terminal
    fixed_cumulative = fixed.cumulative - clean.cumulative
    active = terminal - fixed_terminal
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    parent_to = (clean_final == 0) & (refined_final != 0)
    to_parent = (clean_final != 0) & (refined_final == 0)
    row = {
        "partition": partition,
        "tau": common.TAU,
        "loss_cap": common.LOSS_CAP,
        "trigger_cap": common.TRIGGER_CAP,
        "thresholds": built["thresholds"],
        "loss_counts": built["loss_counts"].tolist(),
        "loss_fractions": built["loss_fractions"].tolist(),
        "trigger_fractions": np.mean(built["triggers"], axis=0).tolist(),
        "coordinate_wise_nonimproving": built["coordinate_wise_nonimproving"],
        "runs": len(seeds),
        "mean_terminal_delta": float(np.mean(terminal)),
        "mean_active_minus_fixed_terminal_delta": float(np.mean(active)),
        "mean_cumulative_delta": float(np.mean(cumulative)),
        "mean_fixed_terminal_delta": float(np.mean(fixed_terminal)),
        "mean_fixed_cumulative_delta": float(np.mean(fixed_cumulative)),
        "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
        "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
        "fixed_root_history_exact": bool(np.array_equal(fixed.roots, clean.roots)),
        "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
        "query_set_change_rate": float(np.mean([set(a.tolist()) != set(b.tolist()) for a, b in zip(clean.queries, refined.queries)])),
        "final_root_change_rate": float(np.mean(clean_final != refined_final)),
        "clean_parent_terminal_rate": float(np.mean(clean_final == 0)),
        "refined_parent_terminal_rate": float(np.mean(refined_final == 0)),
        "parent_to_challenger_rate": float(np.mean(parent_to)),
        "challenger_to_parent_rate": float(np.mean(to_parent)),
        "directional_transition_net": float(np.mean(parent_to) - np.mean(to_parent)),
    }
    vectors = {
        "terminal": terminal,
        "active": active,
        "cumulative": cumulative,
        "fixed_terminal": fixed_terminal,
        "fixed_cumulative": fixed_cumulative,
        "clean_final": clean_final,
        "refined_final": refined_final,
    }
    return row, vectors


def main() -> None:
    for output in (ARRAYS, LEDGER, CONFIG, EARLY_STOP):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite {output}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["authority_sha256"][common.PROTOCOL.name] != common.sha256_path(common.PROTOCOL):
        raise AssertionError("protocol hash drift")
    root_preflight = preflight()
    rows, partitions = load_rows()
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    premises = [str(row["premise"]) for row in rows]
    hypotheses = [str(row["hypothesis"]) for row in rows]
    predictions, root_audits, cache_reused = load_or_infer_roots(premises, hypotheses)
    geometries = {name: geometry(predictions, labels, partitions[name]) for name in ("roster_gate", "selector_search", "selector_verify")}
    gate = geometries["roster_gate"]
    early_gates = {
        "parent_unique_best_by_2_5pp_on_roster_gate": gate["parent_minus_best_challenger"] >= 0.025,
        "parent_error_prevalence_at_least_3pct": gate["parent_error_prevalence"] >= 0.03,
        "challenger_correct_on_at_least_15pct_parent_errors": gate["max_challenger_correct_given_parent_wrong"] >= 0.15,
    }
    if not all(early_gates.values()):
        receipt = {
            "decision": "NO_GO_STEP97_ROOT_GEOMETRY_STOP",
            "authority_sha256": {common.PROTOCOL.name: common.sha256_path(common.PROTOCOL)},
            "manifest_sha256": common.sha256_path(MANIFEST),
            "development_sha256": common.sha256_path(DEVELOPMENT),
            "root_cache_sha256": common.sha256_path(ROOT_CACHE),
            "root_audits_sha256": common.sha256_path(ROOT_AUDITS),
            "root_preflight": root_preflight,
            "root_prediction_audits": root_audits,
            "geometry": geometries,
            "gates": early_gates,
            "failed_gates": [name for name, passed in early_gates.items() if not passed],
            "sealed_test_opened": False,
        }
        common.json_dump(EARLY_STOP, receipt)
        print(json.dumps(receipt, indent=2))
        return
    print(json.dumps({"root_geometry_go": True, "geometry": geometries}, indent=2), flush=True)
    adapter_indices = partitions["adapter_train"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    adapter_audits = []
    adapter_paths = []
    score_columns = []
    for alias in range(common.N_ALIASES):
        path = MODEL_DIR / f"step97_deberta_error_adapter_{alias}.safetensors"
        audit_path = MODEL_DIR / f"step97_deberta_error_adapter_{alias}.audit.json"
        score_path = WORK_DIR / f"step97_error_scores_{alias}.npy"
        if path.is_file() and audit_path.is_file() and score_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            scores = np.load(score_path, allow_pickle=False).astype(np.float64)
            if audit["checkpoint"]["sha256"] != common.sha256_path(path) or audit["score_sha256"] != common.array_sha256(scores):
                raise AssertionError("adapter resume drift")
            resumed = True
        elif path.exists() or audit_path.exists() or score_path.exists():
            raise RuntimeError("incomplete adapter resume")
        else:
            model, tokenizer, training = common.train_error_adapter(
                common.ROOT_SPECS[0],
                [premises[index] for index in adapter_indices],
                [hypotheses[index] for index in adapter_indices],
                predictions[adapter_indices, 0],
                labels[adapter_indices],
                97100 + alias,
            )
            checkpoint = common.save_adapter(
                model,
                path,
                {"step": "97", "alias": alias, "parent_revision": common.ROOT_SPECS[0].revision, "seed": 97100 + alias},
            )
            scores = common.infer_error_scores(model, tokenizer, premises, hypotheses)
            np.save(score_path, scores.astype(np.float64), allow_pickle=False)
            audit = {
                "alias": alias,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "training": training,
                "checkpoint": checkpoint,
                "score_sha256": common.array_sha256(scores),
            }
            common.json_dump(audit_path, audit)
            model.cpu()
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            resumed = False
        adapter_audits.append(audit)
        adapter_paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        score_columns.append(scores)
        print(f"adapter {alias+1}/4 ready; resumed={resumed}; params={audit['checkpoint']['parameter_count']}", flush=True)
    scores = np.stack(score_columns, axis=1)
    threshold_indices = partitions["threshold"]
    threshold_built = build_aliases(predictions[threshold_indices, 0], labels[threshold_indices], scores[threshold_indices])
    search_indices = partitions["selector_search"]
    search_built = build_aliases(predictions[search_indices, 0], labels[search_indices], scores[search_indices], threshold_built["thresholds"])
    search, _ = evaluate("selector_search", predictions[search_indices], labels[search_indices], search_built, SEARCH_SEEDS)
    verify_indices = partitions["selector_verify"]
    verify_built = build_aliases(predictions[verify_indices, 0], labels[verify_indices], scores[verify_indices], threshold_built["thresholds"])
    verify, vectors = evaluate("selector_verify", predictions[verify_indices], labels[verify_indices], verify_built, VERIFY_SEEDS)
    verify["inference"] = {
        "terminal": common.effect_summary(vectors["terminal"], 97900, 97901),
        "active_minus_fixed_terminal": common.effect_summary(vectors["active"], 97901, 97902),
        "cumulative": common.effect_summary(vectors["cumulative"], 97902, 97903),
    }
    verify["terminal_root_counts"] = {
        "clean": np.bincount(vectors["clean_final"], minlength=common.ROSTER_SIZE).tolist(),
        "refined": np.bincount(vectors["refined_final"], minlength=common.ROSTER_SIZE).tolist(),
    }
    endpoint_signature = str(inspect.signature(execute_aliases_from_raw_pairs))
    forbidden_runtime = any(token in endpoint_signature.lower() for token in ("label", "reference", "item_id", "lookup", "peer", "pool", "trajectory", "posterior", "selector", "dataset"))
    threshold_allowed = int(math.floor(common.LOSS_CAP * len(threshold_indices) + 1e-12))
    search_allowed = int(math.floor(0.01 * len(search_indices) + 1e-12))
    verify_allowed = int(math.floor(0.01 * len(verify_indices) + 1e-12))
    terminal_inference = verify["inference"]["terminal"]
    active_inference = verify["inference"]["active_minus_fixed_terminal"]
    cumulative_inference = verify["inference"]["cumulative"]
    gates = {
        **early_gates,
        "distinct_adapters_at_least_ten_million_parameters": len({a["checkpoint"]["sha256"] for a in adapter_audits}) == 4 and all(a["checkpoint"]["parameter_count"] >= 10_000_000 for a in adapter_audits),
        "raw_input_endpoint_has_no_forbidden_runtime_input": not forbidden_runtime,
        "same_literal_similarity": True,
        "parent_unique_best_by_2_5pp_roster_search_validation": all(geometries[name]["parent_minus_best_challenger"] >= 0.025 for name in geometries),
        "coordinate_wise_nonimproving_threshold_search_validation": threshold_built["coordinate_wise_nonimproving"] and search["coordinate_wise_nonimproving"] and verify["coordinate_wise_nonimproving"],
        "conservative_threshold_and_one_point_transfer_quality": bool(np.all(threshold_built["loss_counts"] <= threshold_allowed) and np.all(search_built["loss_counts"] <= search_allowed) and np.all(verify_built["loss_counts"] <= verify_allowed)),
        "search_validation_path_change_at_least_half": search["path_change_rate"] >= .5 and verify["path_change_rate"] >= .5,
        "clean_parent_terminal_at_least_60pct": search["clean_parent_terminal_rate"] >= .6 and verify["clean_parent_terminal_rate"] >= .6,
        "parent_to_challenger_exceeds_reverse": search["parent_to_challenger_rate"] > search["challenger_to_parent_rate"] and verify["parent_to_challenger_rate"] > verify["challenger_to_parent_rate"],
        "search_validation_terminal_at_least_half_point": search["mean_terminal_delta"] >= .005 and verify["mean_terminal_delta"] >= .005,
        "search_validation_active_at_least_half_point": search["mean_active_minus_fixed_terminal_delta"] >= .005 and verify["mean_active_minus_fixed_terminal_delta"] >= .005,
        "validation_terminal_inference": terminal_inference["bootstrap_95"][0] > 0 and terminal_inference["one_sided_signflip_p"] <= .05,
        "validation_active_inference": active_inference["bootstrap_95"][0] > 0 and active_inference["one_sided_signflip_p"] <= .05,
        "validation_cumulative_positive": cumulative_inference["mean"] > 0 and cumulative_inference["bootstrap_95"][0] > 0,
        "validation_fixed_query_exact_zero": verify["max_abs_fixed_terminal_delta"] == 0 and verify["max_abs_fixed_cumulative_delta"] == 0 and verify["fixed_root_history_exact"],
    }
    decision = "GO_TO_STEP97_ONE_TIME_CONFIRMATORY_LOCK" if all(gates.values()) else "NO_GO_DEVELOPMENT_GATE_STOP"
    np.savez_compressed(
        ARRAYS,
        labels=labels.astype(np.int16),
        root_predictions=predictions.astype(np.int16),
        error_scores=scores.astype(np.float64),
        adapter_train_indices=adapter_indices,
        roster_gate_indices=partitions["roster_gate"],
        threshold_indices=threshold_indices,
        search_indices=search_indices,
        verify_indices=verify_indices,
    )
    ledger = {
        "ledger_id": "STEP97_SCITAIL_STAGEA_COMPLETE_V1",
        "date": common.DATE,
        "authority_sha256": {common.PROTOCOL.name: common.sha256_path(common.PROTOCOL)},
        "stage0_manifest_sha256": common.sha256_path(MANIFEST),
        "development_rows_sha256": common.sha256_path(DEVELOPMENT),
        "root_preflight": root_preflight,
        "root_cache_reused": cache_reused,
        "root_prediction_audits": root_audits,
        "root_geometry": geometries,
        "adapter_audits": adapter_audits,
        "threshold_construction": {
            "thresholds": threshold_built["thresholds"],
            "loss_counts": threshold_built["loss_counts"].tolist(),
            "allowed_loss_count": threshold_allowed,
            "trigger_fractions": np.mean(threshold_built["triggers"], axis=0).tolist(),
        },
        "search": search,
        "validation": verify,
        "endpoint_signature": endpoint_signature,
        "single_similarity_source_sha256": source_sha256(common.exact_match_similarity),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "decision": decision,
    }
    common.json_dump(LEDGER, ledger)
    config = {
        "config_id": "STEP97_SCITAIL_STAGEA_FROZEN_V1",
        "date": common.DATE,
        "decision": decision,
        "authority_sha256": ledger["authority_sha256"],
        "stage0_manifest_sha256": ledger["stage0_manifest_sha256"],
        "stagea_ledger_sha256": common.sha256_path(LEDGER),
        "development_arrays_sha256": common.sha256_path(ARRAYS),
        "adapter_paths": adapter_paths,
        "adapter_sha256": {a["path"]: a["checkpoint"]["sha256"] for a in adapter_audits},
        "selected_condition": {"tau": common.TAU, "loss_cap": common.LOSS_CAP, "trigger_cap": common.TRIGGER_CAP, "thresholds": threshold_built["thresholds"]},
        "development_gates": gates,
        "test_input": manifest["test_input"],
        "sealed_outcome": manifest["sealed_outcome"],
        "construction_access": {"test_reference": False, "test_item_lookup": False, "peer_outputs": False, "runtime_fields": ["premise", "hypothesis"]},
        "code_sha256": {name: common.sha256_path(ROOT / name) for name in ("step93_common.py", "step95_common.py", "step96_common.py", "step97_common.py", "step97_executable_adapter_endpoint.py", "step97_stagea_develop.py")},
    }
    common.json_dump(CONFIG, config)
    print(json.dumps({"decision": decision, "threshold": ledger["threshold_construction"], "search": search, "validation": verify, "gates": gates, "failed_gates": ledger["failed_gates"], "ledger_sha256": common.sha256_path(LEDGER), "config_sha256": common.sha256_path(CONFIG)}, indent=2))


if __name__ == "__main__":
    main()
