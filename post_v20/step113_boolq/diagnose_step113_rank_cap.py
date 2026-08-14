from __future__ import annotations

import json

import numpy as np

import step113_common as common


OUTPUT = common.ROOT / f"STEP113_POSTOUTCOME_RANK_CAP_DIAGNOSTIC_{common.DATE}.json"


def main() -> None:
    if OUTPUT.exists():
        raise AssertionError("rank-cap diagnostic already exists")
    ledger = json.loads(common.PRIMARY_LEDGER.read_text(encoding="utf-8"))
    if ledger["decision"] != "NO_GO_STEP113_PRIMARY_CONFIRMATION":
        raise AssertionError(ledger["decision"])
    with np.load(common.PREOUTCOME_PREDICTIONS, allow_pickle=False) as package:
        roots = package["roots"].astype(np.int64)
        scores = package["scores"].astype(np.float64)
        uids = package["uids"]
    with np.load(
        common.ROOT / "external_data/step113_sealed/boolq_validation_sealed_outcomes.npz",
        allow_pickle=False,
    ) as package:
        labels = package["labels"].astype(np.int64)
    cap = int(np.floor(common.QUALITY_LOSS_MAX * len(labels)))
    aliases = np.repeat(roots[:, [0]], common.N_ALIASES, axis=1)
    triggers = np.zeros((len(labels), common.N_ALIASES), dtype=bool)
    for alias in range(common.N_ALIASES):
        order = np.lexsort((uids, -scores[:, alias]))
        chosen = order[:cap]
        triggers[chosen, alias] = True
        aliases[chosen, alias] = common.N_CLASSES + alias
    quality = common.alias_quality_audit(roots[:, 0], aliases, labels)
    raw, arrays = common.compute_raw_effects(roots, aliases, labels)
    summary = common.add_inference(raw, arrays)
    distinct = len(
        {common.array_sha256(aliases[:, alias]) for alias in range(common.N_ALIASES)}
    ) == common.N_ALIASES
    gates = common.primary_gates(summary, quality, distinct)
    result = {
        "schema": "step113.postoutcome_rank_cap_diagnostic.v1",
        "date": common.DATE,
        "status": "POSTOUTCOME_DIAGNOSTIC_NOT_CONFIRMATORY_NOT_PROMOTABLE",
        "motivation": "isolate fixed-threshold distribution shift from mechanism failure",
        "source_primary_decision": ledger["decision"],
        "source_preoutcome_predictions_sha256": common.sha256_path(
            common.PREOUTCOME_PREDICTIONS
        ),
        "rank_rule": {
            "per_alias_top_rows": cap,
            "fraction": cap / len(labels),
            "score_only": True,
            "label_used_to_construct_alias": False,
            "rule_defined_after_outcome": True,
        },
        "trigger_counts": np.sum(triggers, axis=0).astype(int).tolist(),
        "quality": quality,
        "summary": summary,
        "gates_if_the_rule_had_been_prelocked": gates,
        "interpretation_constraint": "may design a future sealed test; cannot rescue Step 113",
    }
    common.json_dump(OUTPUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
