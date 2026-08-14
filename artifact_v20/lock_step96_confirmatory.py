from __future__ import annotations

import hashlib
import json
from pathlib import Path

import step96_common as common


ROOT = Path(__file__).resolve().parent
PROTOCOL = common.PROTOCOL
AMENDMENT_A = ROOT / f"STEP96_PREREGISTRATION_AMENDMENT_A_INTEGER_GATE_{common.DATE}.md"
AMENDMENT_B = ROOT / f"STEP96_PREREGISTRATION_AMENDMENT_B_CONFIRMATORY_SEEDS_{common.DATE}.md"
MANIFEST = ROOT / f"STEP96_STAGE0_MANIFEST_{common.DATE}.json"
DEVELOPMENT = ROOT / f"STEP96_SNLI_DEVELOPMENT_ROWS_{common.DATE}.json"
ARRAYS = ROOT / f"STEP96_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
LEDGER = ROOT / f"STEP96_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
CONFIG = ROOT / f"STEP96_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
REPAIR = ROOT / f"STEP96_INTEGER_QUALITY_GATE_REPAIR_RECEIPT_{common.DATE}.json"
LOCK = ROOT / f"STEP96_CONFIRMATORY_EXECUTION_LOCK_{common.DATE}.json"
TEST_INPUT = ROOT / "step96_test_inputs" / "snli_test_inputs.json"
SEALED = ROOT / "external_data" / "step96_sealed" / "snli_test_sealed_outcomes.npz"


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    if LOCK.exists():
        raise RuntimeError(f"refusing to overwrite {LOCK}")
    required = (
        PROTOCOL,
        AMENDMENT_A,
        AMENDMENT_B,
        MANIFEST,
        DEVELOPMENT,
        ARRAYS,
        LEDGER,
        CONFIG,
        REPAIR,
        TEST_INPUT,
        SEALED,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    repair = json.loads(REPAIR.read_text(encoding="utf-8"))
    if ledger["decision"] != "GO_TO_STEP96_ONE_TIME_CONFIRMATORY_LOCK":
        raise AssertionError(ledger["decision"])
    if ledger["failed_gates"] or not all(ledger["gates"].values()):
        raise AssertionError("development gates are not all true")
    if repair["decision"] != ledger["decision"] or not repair["selected_condition_unchanged"]:
        raise AssertionError("repair receipt mismatch")
    if manifest["test_input"]["sha256"] != common.sha256_path(TEST_INPUT):
        raise AssertionError("test input drift")
    if manifest["sealed_outcome"]["sha256"] != common.sha256_path(SEALED):
        raise AssertionError("sealed outcome drift")
    absent_outputs = [
        ROOT / f"STEP96_CONFIRMATORY_PREOUTCOME_PREDICTIONS_{common.DATE}.npz",
        ROOT / f"STEP96_CONFIRMATORY_PREOUTCOME_LEDGER_{common.DATE}.json",
        ROOT / f"STEP96_CONFIRMATORY_RAW_{common.DATE}.npz",
        ROOT / f"STEP96_CONFIRMATORY_RESULTS_{common.DATE}.json",
        ROOT / f"STEP96_CONFIRMATORY_INDEPENDENT_VALIDATION_{common.DATE}.json",
    ]
    if any(path.exists() for path in absent_outputs):
        raise RuntimeError("Step 96 confirmatory output already exists")
    code_names = (
        "step93_common.py",
        "step95_common.py",
        "step96_common.py",
        "step96_executable_adapter_endpoint.py",
        "step96_stagea_develop.py",
        "repair_step96_integer_quality_gate.py",
        "lock_step96_confirmatory.py",
        "step96_stageb_prepare_predictions.py",
        "step96_stageb_confirmatory.py",
    )
    for name in code_names:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(ROOT / name)
    adapter_hashes = {}
    for relative, expected in config["adapter_sha256"].items():
        path = ROOT / relative
        observed = common.sha256_path(path)
        if observed != expected:
            raise AssertionError({"adapter": relative, "expected": expected, "observed": observed})
        adapter_hashes[relative] = observed
    lock = {
        "lock_id": "STEP96_SNLI_CONFIRMATORY_LOCK_V1",
        "date": common.DATE,
        "development_decision": ledger["decision"],
        "authority_sha256": {
            path.name: common.sha256_path(path)
            for path in (PROTOCOL, AMENDMENT_A, AMENDMENT_B, MANIFEST, DEVELOPMENT, ARRAYS, LEDGER, CONFIG, REPAIR)
        },
        "code_sha256": {name: common.sha256_path(ROOT / name) for name in code_names},
        "adapter_sha256": adapter_hashes,
        "public_roots": manifest["root_bank"],
        "selected_condition": config["selected_condition"],
        "selected_condition_canonical_sha256": canonical_hash(config["selected_condition"]),
        "test_input": manifest["test_input"],
        "sealed_outcome": manifest["sealed_outcome"],
        "test_trajectory_seeds": {"start": 966000, "stop_exclusive": 969000, "runs": 3000},
        "inference_seeds": {
            "terminal": {"bootstrap": 96910, "signflip": 96911},
            "active_minus_fixed_terminal": {"bootstrap": 96911, "signflip": 96912},
            "cumulative": {"bootstrap": 96912, "signflip": 96913},
        },
        "bootstrap_repetitions": 10000,
        "signflip_repetitions": 100000,
        "quality_gate": "integer_loss_count <= floor(0.01*n + 1e-12)",
        "confirmatory_success_label": "GO_SOURCE_FAITHFUL_DIRECTIONAL_TERMINAL_BRIDGE",
        "confirmatory_failure_label": "NO_GO_RETAIN_STEP96_NEGATIVE",
        "sealed_outcome_opened_at_lock": False,
        "confirmatory_outputs_absent_at_lock": [str(path.relative_to(ROOT)).replace("\\", "/") for path in absent_outputs],
    }
    common.json_dump(LOCK, lock)
    print(json.dumps({"lock": LOCK.name, "sha256": common.sha256_path(LOCK), "selected": lock["selected_condition"]}, indent=2))


if __name__ == "__main__":
    main()
