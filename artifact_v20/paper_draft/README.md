# ICLR 2027 working manuscript

Working title: **Candidate Lists Are Priors: Evidence-Frame Dependence in Active Model Selection**

This directory is the Step 105 / V20 review-ready manuscript. It uses an unmodified copy of the official ICLR 2027 style. The natural exact-alias anchor, canonical V2 quality-equivalent public-reference confirmation, calibration-trained holdout-reference-free audit, official CODA and LLM Selector audits, defense frontiers, every retained learned negative, the disjoint 13,000-row IMDB held-out same-similarity primary, and the prospective Yelp one-shot negative are linked to machine-readable evidence. Stage-specific runner-independent validators reconstruct the promoted and retained results; the current closure reconstructs both final learned studies' trajectories, inference, and gates, with optional raw-input re-inference against pinned public base weights.

The scientific main text ends on page 9; references begin on page 10. The PDF has 28 pages.

## Build

From this directory:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Anonymous mode is active because `\iclrfinalcopy` remains commented. Do not edit `iclr2027_conference.sty`, `iclr2027_conference.bst`, `natbib.sty`, or `fancyhdr.sty`.

## Evidence sources

- Formal statements: `../STEP37_PAPER_READY_THEORY_2026-08-08.md`
- Final claim boundaries: `../STEP36_FINAL_CONTRIBUTION_LOCK_2026-08-08.md`
- Primary peer-response-free LLM confirmation (historical backend): `../STEP54_MATRIX_FREE_REALISM_RESULTS_2026-08-09.json`
- Canonical V2 exact-condition transfer: `../STEP80_MAIN_T2A_V2_TRANSFER_RESULTS_2026-08-10.json`, raw arrays, and `../STEP80_MAIN_T2A_V2_TRANSFER_VALIDATION_2026-08-10.json`
- Matrix-based target/robustness audit: `../STEP30_PHASE_B1_ALL_TARGET_RESULTS_2026-08-06.json` and `../STEP31_PHASE_B2_RESULTS_2026-08-06.json`
- Natural exact-alias audit: `../STEP50_NATURAL_ALIAS_RESULTS_2026-08-09.json`
- Natural operational decomposition: `../STEP52_OPERATIONAL_SIGNIFICANCE_RESULTS_2026-08-09.json`
- Step 52 human-readable audit: `../STEP52_OPERATIONAL_SIGNIFICANCE_AUDIT_2026-08-09.md`
- Step 52 machine-readable audit: `../STEP52_OPERATIONAL_SIGNIFICANCE_AUDIT_2026-08-09.json`
- CODA results: `../STEP35_CODA_CROSS_METHOD_RESULTS_2026-08-08.json`
- Clone-robust soft-weight results: `../STEP46_CLONE_ROBUST_SOFT_WEIGHTING_RESULTS_2026-08-08.json`
- Step 50 replay validator: `../validate_step50_natural_alias_audit.py`
- Step 52 operational validator: `../validate_step52_operational_significance.py`
- Step 54 preregistration/report: `../STEP54_MATRIX_FREE_REALISM_PREREGISTRATION_2026-08-09.json` and `../STEP54_MATRIX_FREE_REALISM_KILL_TEST_REPORT_2026-08-09.md`
- Step 54 independent validator: `../validate_step54_matrix_free_realism_audit.py`
- Step 80 preregistration/execution locks and runner-independent validator: `../STEP80_MAIN_T2A_V2_TRANSFER_PREREGISTRATION_LOCK_2026-08-10.json`, `../STEP80_MAIN_T2A_V2_TRANSFER_EXECUTION_LOCK_2026-08-10.json`, and `../validate_step80_main_t2a_v2_transfer.py`
- Step 55 executable-endpoint results/report: `../STEP55_EXECUTABLE_ENDPOINT_REALISM_RESULTS_2026-08-09.json` and `../STEP55_EXECUTABLE_ENDPOINT_REALISM_AUDIT_2026-08-09.md`
- Step 55 endpoint implementation/validator: `../step55_prompt_hash_endpoint.py` and `../validate_step55_executable_endpoint_realism_audit.py`
- Step 56 calibration-to-holdout results/report: `../STEP56_CALIBRATION_TO_HOLDOUT_RESULTS_2026-08-09.json` and `../STEP56_CALIBRATION_TO_HOLDOUT_AUDIT_2026-08-09.md`
- Step 56 frozen blind attack/validator: `../STEP56_FROZEN_BLIND_HOLDOUT_ATTACK_2026-08-09.json` and `../validate_step56_calibration_to_holdout.py`
- Step 57 reviewer hierarchy protocol/result/report: `../STEP57_REVIEWER_HIERARCHY_AUDIT_PROTOCOL_2026-08-09.json`, `../STEP57_REVIEWER_HIERARCHY_AUDIT_2026-08-09.json`, and `../STEP57_REVIEWER_HIERARCHY_AUDIT_2026-08-09.md`
- Step 61 prospective negative control: `../STEP61B_TRUE_EFFECT_RESULTS_2026-08-10.json` and `../validate_step61b_results.py`
- Step 87 prospective reference-free transfer: `../STEP87B_REFERENCE_FREE_OPEN_ENDED_RESULTS_2026-08-12.json` and `../validate_step87b_sealed_holdout.py`.
- Step 88 official LLM Selector audit: `../STEP88_LLM_SELECTOR_RESULTS_2026-08-12.json` and `../validate_step88_llm_selector_audit.py`.
- Steps 89--90 configuration-wrapper audit: `../STEP90_HIGH_CONFIDENCE_EXECUTABLE_ADAPTER_RESULTS_2026-08-12.json`, retained `../STEP89_QNLI_EXECUTABLE_ADAPTER_RESULTS_2026-08-12.json`, endpoint `../step90_executable_adapter_endpoint.py`, and `../validate_step90_independent.py`.
- Step 92 learned-adapter audit: `../STEP92_LEARNED_ADAPTER_CONFIRMATORY_RESULTS_2026-08-12.json`, `../STEP92_INDEPENDENT_VALIDATION_2026-08-12.json`, and `../STEP92_NUMERIC_TIE_ROBUSTNESS_2026-08-12.json`.
- Step 93 source-faithful learned audit: `../STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RESULTS_2026-08-13.json` and `../STEP93_INDEPENDENT_VALIDATION_AND_ADJUDICATION_2026-08-13.json`.
- Step 94 fresh-task learned single-similarity replication: `../STEP94_CONFIRMATORY_RESULTS_2026-08-13.json`, `../STEP94_CONFIRMATORY_RAW_2026-08-13.npz`, and `../STEP94_INDEPENDENT_VALIDATION_2026-08-13.json`.
- Step 95 MultiNLI full-adapter development stop: `../STEP95_STAGEA_COMPLETE_LEDGER_2026-08-13.json`, `../STEP95_STAGEA_INDEPENDENT_VALIDATION_2026-08-13.json`, and `../STEP95_DEVELOPMENT_STOP_RECEIPT_2026-08-13.json`; no confirmatory lock or test result exists.
- Step 96 SNLI directional terminal audit: `../STEP96_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json`, `../STEP96_CONFIRMATORY_RESULTS_2026-08-13.json`, `../STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_SENSITIVITY_2026-08-13.json`, and `../STEP96_CONFIRMATORY_INDEPENDENT_VALIDATION_2026-08-13.json`. The primary decision remains no-go; the conservative top-eight sensitivity is secondary.
- Step 97 SciTail geometry stop: `../STEP97_STAGEA_ROOT_GEOMETRY_STOP_2026-08-13.json` and `../STEP97_ROOT_GEOMETRY_STOP_INDEPENDENT_VALIDATION_2026-08-13.json`; no adapters or confirmatory outputs exist and test remains unopened.
- Steps 98--99 ANLI trail: `../STEP98_STAGEA_COMPLETE_LEDGER_2026-08-13.json`, `../STEP98_STAGEA_INDEPENDENT_VALIDATION_2026-08-13.json`, `../STEP99_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json`, and `../STEP99_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json`; both no-go decisions are retained.
- Steps 100--103 IMDB development: `../STEP100_STAGE0_DATA_AND_SEAL_MANIFEST_2026-08-13.json`, `../STEP102_STAGEA_COMPLETE_LEDGER_2026-08-13.json`, `../STEP103_ROBUST_DUAL_DEVELOPMENT_LEDGER_2026-08-13.json`, and `../STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION_2026-08-13.json`. The robust Step 103 cell was frozen before held-out outcomes opened.
- Step 104 IMDB held-out primary: `../STEP104_PREOUTCOME_LOCK_2026-08-13.json`, `../STEP104_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json`, `../STEP104_PRIMARY_CONFIRMATORY_ARRAYS_2026-08-13.npz`, and `../STEP104_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json`. Its literal decision is `GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY`.
- Step 105 prospective Yelp one-shot: `../STEP105_PREDATA_LOCK_2026-08-13.json`, `../STEP105_PREOUTCOME_LOCK_2026-08-13.json`, `../STEP105_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json`, `../STEP105_PRIMARY_CONFIRMATORY_ARRAYS_2026-08-13.npz`, `../STEP105_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json`, and `../STEP105_ARTIFACT_RECONSTRUCTION_2026-08-13.json`. Its literal decision is `NO_GO_RETAIN_STEP105_PROSPECTIVE_NEGATIVE`; the positive effect misses only the frozen +.5-point terminal and active-minus-fixed materiality gates, and no replacement outcome is authorized.
- Current evidence-to-manuscript validator: `validate_paper_tables.py`
- Clean-room dependency inventory: `../STEP68_REPLAY_DEPENDENCY_INVENTORY_2026-08-10.json`
- The current anonymous release manifest and upload are generated by the Step 105 / V20 closure.

Numerical entries in the result tables are locked measurements. After changing source or result prose, rebuild and run `python validate_paper_tables.py`; the current anonymous artifact must then be rebuilt. Earlier integration validators target historical manuscript snapshots and are not the current page-boundary authority.
