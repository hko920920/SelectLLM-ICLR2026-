"""Check that manuscript tables agree with the locked machine-readable results."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def close(actual: float, expected: float, tol: float = 1e-9) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol), (
        actual,
        expected,
    )


def primary_llm_checks() -> None:
    historical_path = ROOT / "STEP54_MATRIX_FREE_REALISM_RESULTS_2026-08-09.json"
    assert hashlib.sha256(historical_path.read_bytes()).hexdigest() == (
        "4ab5428c368991f8c9696edda63660ac76b1d516bb7c78d52eeb553ad56a41dd"
    )
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    assert historical["gates"]["decision"] == "STRONG_GO"

    path = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_RESULTS_2026-08-10.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "92642f263da5252624427a35574a7ddb55985caedbbb63144e5bf7f3cedf0a6d"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["complete"] is True
    assert result["adjudication"]["classification"] == "STRONG_TRANSFER"
    validation_path = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_VALIDATION_2026-08-10.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["status"] == "PASS_STEP80_INDEPENDENT_VALIDATION"
    assert validation["validated_classification"] == "STRONG_TRANSFER"
    expected = {
        "medqa": (3.2518039999999995, (3.0214191999999995, 3.4841887999999996)),
        "gsm8k": (3.781120000000001, (3.6177480000000006, 3.9433248000000014)),
        "openbookqa": (0.8459200000000001, (0.765009125, 0.926012)),
    }
    for task, (mean, ci) in expected.items():
        task_result = result["tasks"][task]
        selected = task_result["contrasts"]["refined_active_minus_clean_active_cumulative"]
        close(selected["mean"], mean)
        close(selected["ci95"][0], ci[0])
        close(selected["ci95"][1], ci[1])
        metadata = task_result["constructor_metadata"]
        assert metadata["per_example_quality_equal"] is True
        assert metadata["aliases"] == 4
        assert metadata["competitor_response_rows_read_by_constructor"] == 0
        close(task_result["contrasts"]["refined_fixed_minus_clean_fixed_cumulative"]["mean"], 0.0)
        assert task_result["permutation"]["all_exact_checks_pass"] is True
        assert all(
            cell["path_change_fraction"] == 1.0
            and cell["cumulative_root_regret_delta"]["ci95"][0] > 0
            for cell in task_result["tie_grid"]["cells"].values()
        )


def natural_alias_checks() -> None:
    path = ROOT / "STEP50_NATURAL_ALIAS_RESULTS_2026-08-09.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "14b77bd3a4193ea3e60ad2f300524fc37393ab250adfa59af016d7758b0f8ea4"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["decision"]["classification"] == "LIMITED_NATURAL_OUTCOME_ANCHOR"
    assert result["natural_pair"]["cross_task_exact_pair_intersection"] == [[9, 10]]
    expected = {
        "medqa": (1.0, 40.744, 0.8437326138452613, -0.500212, (-0.6151536, -0.3883186), 0.094),
        "gsm8k": (1.0, 21.752, 0.9110875329603983, 0.059328, (0.027491, 0.092144), 0.002),
        "openbookqa": (1.0, 16.01, 0.9395068569949974, -0.027555, (-0.063410375, 0.008281), 0.034),
    }
    all_cells = []
    for task, (path_fraction, positions, jaccard, delta, ci, final_change) in expected.items():
        task_result = result["tasks"][task]
        primary = task_result["cells"][task_result["primary_cell_id"]]["comparison"]
        close(primary["path_change_fraction"], path_fraction)
        close(primary["mean_changed_query_positions"], positions)
        close(primary["mean_query_set_jaccard"], jaccard)
        close(primary["mean_delta_cumulative_root_regret"], delta)
        close(primary["delta_cumulative_root_regret_95ci"][0], ci[0])
        close(primary["delta_cumulative_root_regret_95ci"][1], ci[1])
        close(primary["final_selected_root_change_fraction"], final_change)
        assert task_result["random_query_control"]["pass"] is True
        assert len(task_result["cells"]) == 7
        all_cells.extend(cell["comparison"] for cell in task_result["cells"].values())
    assert len(all_cells) == 21
    assert sum(
        cell["path_change_fraction"] >= 0.1
        and cell["mean_changed_query_positions"] >= 1.0
        for cell in all_cells
    ) == 21
    significant = [cell for cell in all_cells if cell["all_21_cell_family_bh_q"] < 0.05]
    assert len(significant) == 14
    assert sum(cell["mean_delta_cumulative_root_regret"] > 0 for cell in significant) == 5
    assert sum(cell["mean_delta_cumulative_root_regret"] < 0 for cell in significant) == 9


def operational_significance_checks() -> None:
    path = ROOT / "STEP52_OPERATIONAL_SIGNIFICANCE_RESULTS_2026-08-09.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "bb70cefde9c7fa14ec842ee1dcfbeb4acd00835b69e2a7c45c05edecd9e372fa"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["decision"]["classification"] == (
        "TASK_LOCAL_LABEL_SET_SUBSTITUTION_WITH_GLOBAL_PATH_INSTABILITY"
    )
    expected = {
        "medqa": (4.358, 0.08716, 0.94, 1.486, 10.592),
        "gsm8k": (1.42, 0.04733333333333334, 0.874, 3.698, 2.902),
        "openbookqa": (0.96, 0.032, 0.66, 4.69, 2.012),
    }
    for task, (replaced, fraction, set_change, first, root_positions) in expected.items():
        task_result = result["tasks"][task]
        active = task_result["active"]
        close(active["label_set_substitution"]["base_only_queries"]["mean"], replaced)
        close(active["label_set_substitution"]["replacement_fraction"]["mean"], fraction)
        close(active["label_set_substitution"]["set_change_fraction"], set_change)
        close(active["temporal_divergence"]["first_divergence_step"]["mean"], first)
        close(
            active["interim_decision_displacement"]["changed_root_positions"]["mean"],
            root_positions,
        )
        assert task_result["random_query_control"]["pass"] is True


def reference_free_terminal_checks() -> None:
    result_path = ROOT / "STEP87B_REFERENCE_FREE_OPEN_ENDED_RESULTS_2026-08-12.json"
    raw_path = ROOT / "STEP87B_REFERENCE_FREE_OPEN_ENDED_RAW_2026-08-12.npz"
    validation_path = ROOT / "STEP87B_INDEPENDENT_VALIDATION_2026-08-12.json"
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == (
        "fc953c876ef67e3b5a6b915434667e4051d6ece47973574f8d82ee3e917cf556"
    )
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == (
        "26422130d6b7d053c4b78fbc292e3deceb05db5eb72113c2b6e9175f2cb0ddad"
    )
    assert hashlib.sha256(validation_path.read_bytes()).hexdigest() == (
        "1ef6c0f7df2c11f871ead0f7162c480d600a3730d0f4d455f9e88500841cd9db"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["status"] == "PASS_STEP87B_INDEPENDENT_RECONSTRUCTION"
    assert validation["decision_match"] is True
    assert result["decision"] == {
        "global_separation_and_completeness": True,
        "passing_scenarios": ["narrativeqa", "wmt_fr_en", "wmt_ru_en"],
        "passing_count": 3,
        "status": "PROMOTE_REFERENCE_FREE_TERMINAL_BRIDGE",
    }
    expected = {
        "narrativeqa": (-0.008510638297872353, 0.0302099779391832, 0.027656147969947913, 0.0328173486216773, 1.0, 0.456, True),
        "naturalqa_closed": (0.0, -0.01213217564512961, -0.014433300874742318, -0.009854417559627963, 0.976, 0.555, False),
        "naturalqa_open": (-0.008424242424242334, -0.0031711261137608465, -0.005517712175646173, -0.0007482408475942443, 0.996, 0.356, False),
        "wmt_cs_en": (-0.0010900852345343964, 0.0008048632675094674, -0.00033073728586330335, 0.0019169857205501054, 0.991, 0.395, False),
        "wmt_de_en": (-0.002752776452546901, -0.0008707439157358243, -0.0016897654145721418, -0.000015307852006992438, 0.942, 0.359, False),
        "wmt_fr_en": (-0.0006363636363636016, 0.0013939037818202324, 0.0010247991551159297, 0.001772162686427034, 0.751, 0.106, True),
        "wmt_hi_en": (0.0, 0.000014110036795892317, -0.0003320927612480443, 0.00034595276837290053, 1.0, 0.154, False),
        "wmt_ru_en": (0.0, 0.002503002327949595, 0.0015380779172164648, 0.003502166340648863, 0.974, 0.461, True),
    }
    for task, target in expected.items():
        row = result["tasks"][task]
        statistic = row["statistics"]["active_terminal_root_delta"]
        for actual, wanted in zip(
            (row["quality"]["minimum_gap"], statistic["mean"], *statistic["ci95"], row["path_effect"]["ordered_path_change_fraction"], row["path_effect"]["terminal_selected_root_change_fraction"]),
            target[:-1],
        ):
            close(actual, wanted, tol=1e-12)
        assert row["gate"]["task_pass"] is target[-1]
        assert row["gate"]["exact_fixed_query_mediation"] is True
        close(row["statistics"]["fixed_terminal_root_delta"]["mean"], 0.0)


def robustness_and_cover_checks() -> None:
    result = load("STEP31_PHASE_B2_RESULTS_2026-08-06.json")
    assert result["complete"] is True
    expected_counts = {
        "medqa": (240, 240, 0, 1000),
        "gsm8k": (180, 180, 0, 1000),
        "openbookqa": (240, 171, 69, 500),
    }
    primary = {
        "medqa": (3.0, 50),
        "gsm8k": (5.0, 30),
        "openbookqa": (5.0, 30),
    }
    for task, (total, positive, negative, examples) in expected_counts.items():
        assert result["tasks"][task]["examples"] == examples
        cells = result["tasks"][task]["grid_cells"]
        assert len(cells) == total
        values = [cell["mean_delta_cumulative_deployed_regret"] for cell in cells]
        assert sum(value > 0 for value in values) == positive
        assert sum(value < 0 for value in values) == negative

        tau, budget = primary[task]
        temperature_slice = [
            cell
            for cell in cells
            if cell["nominal_distance"] == 0.05
            and cell["aliases"] == 4
            and cell["budget"] == budget
        ]
        budget_slice = [
            cell
            for cell in cells
            if cell["nominal_distance"] == 0.05
            and cell["aliases"] == 4
            and cell["temperature"] == tau
        ]
        assert len(temperature_slice) == 5
        assert len(budget_slice) == 3
        assert all(cell["mean_delta_cumulative_deployed_regret"] > 0 for cell in temperature_slice)
        assert all(cell["mean_delta_cumulative_deployed_regret"] > 0 for cell in budget_slice)

    cover_expected = {
        "medqa": (3.0, 50, 4.852616, (4.6032704, 5.1022779), 0.968, 0.568588),
        "openbookqa": (5.0, 30, 1.00555, (0.94241825, 1.0692205), 0.86, 1.029555),
    }
    for task, (tau, budget, mean, ci, harmed, clean_cost) in cover_expected.items():
        task_result = result["tasks"][task]
        attack_cell = next(
            cell
            for cell in task_result["grid_cells"]
            if cell["nominal_distance"] == 0.1
            and cell["aliases"] == 4
            and cell["temperature"] == tau
            and cell["budget"] == budget
        )
        close(attack_cell["mean_delta_cumulative_deployed_regret"], mean)
        close(attack_cell["delta_cumulative_deployed_regret_95ci"][0], ci[0])
        close(attack_cell["delta_cumulative_deployed_regret_95ci"][1], ci[1])
        close(attack_cell["harmful_fraction"], harmed)

        cover = next(
            row for row in task_result["matched_cover"] if row["nominal_distance"] == 0.1
        )
        defended_cell = next(
            cell
            for cell in cover["cells"]
            if cell["temperature"] == tau and cell["budget"] == budget
        )
        close(defended_cell["residual_attack"]["mean_delta_cumulative_deployed_regret"], 0.0)
        close(defended_cell["clean_cover_cost"]["mean_delta_cumulative_deployed_regret"], clean_cost)

    gsm_cover = next(
        row
        for row in result["tasks"]["gsm8k"]["matched_cover"]
        if row["nominal_distance"] == 0.1
    )
    assert gsm_cover["status"] == "INELIGIBLE_TARGET_AT_DISTANCE"


def coda_checks() -> None:
    result = load("STEP35_CODA_CROSS_METHOD_RESULTS_2026-08-08.json")
    expected = {
        "glue_cola": (0.05196544528007507, -0.0019175410270690918, 1),
        "glue_mrpc": (0.05392146110534668, 0.0, 5),
        "glue_rte": (0.8411549925804138, 0.0, 2),
        "glue_sst2": (-0.271789014339447, 0.0022935867309570312, 4),
        "glue_wnli": (-0.01408451795578003, 0.0, 1),
    }
    for task, (cumulative, terminal, first_divergence) in expected.items():
        comparison = result["tasks"][task]["comparisons"]["near4"]
        assert comparison["task_counts_as_path_changed"] is True
        close(comparison["mean_cumulative_regret_delta"], cumulative)
        close(comparison["mean_final_regret_delta"], terminal)
        assert all(
            pair["first_query_divergence"] == first_divergence
            for pair in comparison["paired"]
        )


def llm_selector_step88_checks() -> None:
    result_path = ROOT / "STEP88_LLM_SELECTOR_RESULTS_2026-08-12.json"
    receipt_path = ROOT / "STEP88_INDEPENDENT_VALIDATION_RECEIPT_2026-08-12.json"
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == (
        "24c3fcdab2016bf6004844c242603bfd346fe0222911124f304fd81e4cea7020"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result["decision"] == "GO_STRONG_CROSS_SELECTOR"
    assert result["passing_datasets"] == ["alpacaeval", "bingo"]
    assert receipt["status"] == "PASS_STEP88_INDEPENDENT_VALIDATION"
    assert receipt["decision_reconstructed"] == result["decision"]
    expected = {
        "alpacaeval": (0.0032125, (0.00193625, 0.00452875), 0.00002999970000299997, -0.24478625, True),
        "arena-hard": (-0.003695, (-0.0053075, -0.00212625), 1.0, -0.37898, False),
        "bingo": (0.0049775, (0.00373875, 0.0062275), 0.00002999970000299997, 0.11693625, True),
        "flickr30k": (0.00182625, (-0.000025, 0.00367625), 0.052979470205297946, -0.029095, False),
        "medi_qa": (0.0, (0.0, 0.0), 1.0, 0.02666666666666634, False),
        "mt-bench": (0.0, (0.0, 0.0), 1.0, 0.71875, False),
    }
    for task, (mean, ci, qvalue, cumulative, passes) in expected.items():
        source = result["datasets"][task]
        summary = result["summaries"][task]
        assert source["strong_utility_max_abs_difference"] == 0
        close(summary["mean_terminal_delta"], mean, tol=1e-12)
        close(summary["terminal_bootstrap_95_ci"][0], ci[0], tol=1e-12)
        close(summary["terminal_bootstrap_95_ci"][1], ci[1], tol=1e-12)
        close(summary["terminal_signflip_q_bh"], qvalue, tol=1e-12)
        close(summary["mean_cumulative_delta"], cumulative, tol=1e-12)
        close(summary["ordered_path_change_rate"], 1.0)
        close(summary["max_abs_fixed_terminal_delta"], 0.0)
        close(summary["max_abs_fixed_cumulative_delta"], 0.0)
        assert summary["passes_terminal_harm_gate"] is passes


def soft_weight_checks() -> None:
    path = ROOT / "STEP46_CLONE_ROBUST_SOFT_WEIGHTING_RESULTS_2026-08-08.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "15b693f06f708e4583aac52ab21dd85cdc8326c87683ff88938c128ab7b7b136"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["decision"]["classification"] == "TRADEOFF_OR_FAILURE"

    clean_cells = []
    attack_cells = []
    for task_result in result["tasks"].values():
        assert len(task_result["configs"]) == 21
        for config in task_result["configs"]:
            clean_cells.append(config)
            for attack in config["attacks"].values():
                attack_cells.append(attack)

    assert len(clean_cells) == 63
    assert len(attack_cells) == 105
    assert sum(config["cheap_clean"] for config in clean_cells) == 0
    assert sum(attack["attack_suppressed"] for attack in attack_cells) == 0
    assert sum(
        attack["absolute_residual_to_undefended_ratio"] < 1.0
        for attack in attack_cells
    ) == 74
    assert sum(
        attack["absolute_residual_to_undefended_ratio"] > 1.0
        for attack in attack_cells
    ) == 31
    assert sum(
        attack["absolute_residual_to_undefended_ratio"] <= 0.1
        for attack in attack_cells
    ) == 4
    assert sum(
        attack["absolute_residual_to_undefended_ratio"] <= 0.5
        for attack in attack_cells
    ) == 24
    assert sum(
        attack["absolute_residual_to_undefended_ratio"] <= 0.25
        for attack in attack_cells
    ) == 11
    assert all(attack["residual"]["path_change_fraction"] == 1.0 for attack in attack_cells)

    clean_positive = sum(
        config["clean_cost"]["delta_cumulative_deployed_regret_95ci"][0] > 0
        for config in clean_cells
    )
    clean_negative = sum(
        config["clean_cost"]["delta_cumulative_deployed_regret_95ci"][1] < 0
        for config in clean_cells
    )
    assert (clean_positive, clean_negative) == (42, 21)

    terminal_positive = sum(
        config["clean_cost"]["delta_final_deployed_regret_95ci"][0] > 0
        for config in clean_cells
    )
    terminal_negative = sum(
        config["clean_cost"]["delta_final_deployed_regret_95ci"][1] < 0
        for config in clean_cells
    )
    terminal_zero = len(clean_cells) - terminal_positive - terminal_negative
    assert (terminal_positive, terminal_negative, terminal_zero) == (10, 8, 45)
    close(
        max(
            abs(config["clean_cost"]["mean_delta_final_deployed_regret"])
            for config in clean_cells
        ),
        0.00203,
    )

    config_ids = [config["config_id"] for config in result["tasks"]["gsm8k"]["configs"]]
    fixed_summaries = {}
    for config_id in config_ids:
        ratios = []
        clean_means = {}
        for task, task_result in result["tasks"].items():
            config = next(row for row in task_result["configs"] if row["config_id"] == config_id)
            ratios.extend(
                attack["absolute_residual_to_undefended_ratio"]
                for attack in config["attacks"].values()
            )
            clean_means[task] = config["clean_cost"][
                "mean_delta_cumulative_deployed_regret"
            ]
        fixed_summaries[config_id] = {
            "mean": sum(ratios) / len(ratios),
            "maximum": max(ratios),
            "mitigated": sum(value < 1.0 for value in ratios),
            "all_nonworsening": all(value <= 1.0 for value in ratios),
            "clean": clean_means,
        }
    assert sum(row["mean"] < 1.0 for row in fixed_summaries.values()) == 20
    assert sum(row["all_nonworsening"] for row in fixed_summaries.values()) == 0
    for row in fixed_summaries.values():
        assert row["clean"]["gsm8k"] < 0
        assert row["clean"]["medqa"] > 0
        assert row["clean"]["openbookqa"] > 0
    best_id = min(fixed_summaries, key=lambda key: fixed_summaries[key]["mean"])
    assert best_id == "mcca|alpha=0.132"
    close(fixed_summaries[best_id]["mean"], 0.5837317476310953)
    close(fixed_summaries[best_id]["maximum"], 1.1576190044880605)
    assert fixed_summaries[best_id]["mitigated"] == 4

    expected = {
        ("medqa", "0.05"): ("mcca", 0.1, 4.16456, 1.165688, 0.274952),
        ("medqa", "0.1"): ("mcca", 0.213, 4.852616, 3.034436, 0.637516),
        ("gsm8k", "0.05"): ("mccp", 0.01, 3.047196, 3.233452, -0.180468),
        ("openbookqa", "0.05"): ("mcca", 0.184, 1.45997, 0.02053, 2.010445),
        ("openbookqa", "0.1"): (
            "class_uniform",
            0.15,
            1.00555,
            -0.00023,
            1.107515,
        ),
    }
    for (task, distance), (rule, alpha, undefended, residual, clean) in expected.items():
        task_result = result["tasks"][task]
        config = next(
            item
            for item in task_result["configs"]
            if item["rule"] == rule and item["alpha"] == alpha
        )
        attack = config["attacks"][distance]
        close(
            task_result["undefended"][distance]["comparison"][
                "mean_delta_cumulative_deployed_regret"
            ],
            undefended,
        )
        close(attack["residual"]["mean_delta_cumulative_deployed_regret"], residual)
        close(config["clean_cost"]["mean_delta_cumulative_deployed_regret"], clean)
        close(attack["residual"]["path_change_fraction"], 1.0)


def rendered_source_checks() -> None:
    primary = (HERE / "tables" / "primary_llm_results.tex").read_text(encoding="utf-8")
    cover = (HERE / "tables" / "defense_frontier.tex").read_text(encoding="utf-8")
    coda = (HERE / "tables" / "coda_results.tex").read_text(encoding="utf-8")
    soft = (HERE / "tables" / "soft_weight_frontier.tex").read_text(encoding="utf-8")
    soft_all = (HERE / "tables" / "soft_weight_all_configs.tex").read_text(encoding="utf-8")
    reference_free = (HERE / "tables" / "step87b_reference_free_results.tex").read_text(encoding="utf-8")
    second_selector = (HERE / "tables" / "step88_llm_selector_results.tex").read_text(encoding="utf-8")
    learned_adapter = (HERE / "tables" / "step92_learned_adapter_results.tex").read_text(encoding="utf-8")
    source_faithful = (HERE / "tables" / "step93_source_faithful_results.tex").read_text(encoding="utf-8")
    step96_terminal = (HERE / "tables" / "step96_terminal_bridge_results.tex").read_text(encoding="utf-8")
    step104_imdb = (HERE / "tables" / "step104_imdb_heldout_primary.tex").read_text(encoding="utf-8")
    step105_yelp = (HERE / "tables" / "step105_yelp_prospective_one_shot.tex").read_text(encoding="utf-8")
    for token in (
        "Natural exact alias",
        "40.744/50",
        "21.752/30",
        "16.010/30",
        "-0.5002",
        "+0.0593",
        "-0.0276",
        "Unique repl.",
        "4.358",
        "8.72",
        "1.420",
        "4.73",
        "0.960",
        "3.20",
        "+3.2518",
        "+3.7811",
        "+0.8459",
    ):
        assert token in primary
    for token in ("+4.853", "+1.006", "Cover residual", "$0$"):
        assert token in cover
    for token in ("+0.051965", "+0.053921", "+0.841155", "-0.271789", "-0.014085"):
        assert token in coda
    for token in (
        "All three, .050",
        "MCCA, .100",
        "MCCP, .100",
        "class-unif., .132",
        "MCCA, .132",
        ".584",
        "1.158",
        "$+1.273$",
    ):
        assert token in soft
    assert soft_all.count("class-unif. &") == 7
    assert soft_all.count("MCCA &") == 7
    assert soft_all.count("MCCP &") == 7
    for token in ("1.584", "$+3.724$", ".203", "Final pp"):
        assert token in soft_all
    for token in (
        "NarrativeQA (F1)",
        "$+3.021$ [$2.766,3.282$]",
        "WMT14 fr--en",
        "$+.139$ [$+.102,.177$]",
        "WMT14 ru--en",
        "$+.250$ [$+.154,.350$]",
        "NaturalQA closed",
        "All rows were frozen and retained",
    ):
        assert token in reference_free
    for token in (
        "AlpacaEval$^\\dagger$",
        "$.321$ [$ .194,.453$]",
        "Bingo$^\\dagger$",
        "$.498$ [$ .374,.623$]",
        "Arena-Hard",
        "$-.370$ [$-.531,-.213$]",
        "Flickr30k",
        "MEDIQA",
        "MT-Bench",
    ):
        assert token in second_selector
    for token in (
        "AG News",
        "50,817",
        "24.1--26.3\\%",
        "100.0/55.2\\%",
        "$+.547$ [$+.502,+.593$]",
        "Pass",
    ):
        assert token in learned_adapter
    for token in (
        "Banking77",
        "50,817",
        ".098--.358",
        "99.8/36.6\\%",
        "$+.032$ [$-.055,+.122$]",
        "$+.0409$ [$+.0129,+.0689$]",
        "20 Newsgroups",
        "205,245",
        ".252--.491",
        "100/48.2\\%",
        "$+.120$ [$-.044,+.290$]",
        "$+.0822$ [$+.0385,+.1266$]",
        "No",
    ):
        assert token in source_faithful
    for token in (
        "Locked primary",
        ".875--1.018",
        "99.13/16.23\\%",
        "$+1.631$ [$+1.360,+1.903$]",
        "$+.316$ [$+.285,+.347$]",
        "Conservative top-8 sensitivity",
        ".316--.478",
        "99.43/13.23\\%",
        "$+.915$ [$+.667,+1.173$]",
        "$+.221$ [$+.193,+.250$]",
        "Secondary pass",
    ):
        assert token in step96_terminal
    for token in (
        "Development search",
        "$+1.560$",
        "Development verification",
        "$+.533$ [$+.419,+.648$]",
        "$+.02141$ [$+.01715,+.02560$]",
        "Sealed held-out primary",
        "$+.702$ [$+.615,+.788$]",
        "$+.02795$ [$+.02467,+.03129$]",
        "99.93/60.67",
        "\\texttt{GO}",
    ):
        assert token in step104_imdb
    for token in (
        "Prospective Yelp one-shot",
        "$+.3527$",
        "$[+.3090,+.3977]$",
        "$[+.3076,+.3956]$",
        "$+.017925$",
        "$[+.016163,+.019706]$",
        "$99.90\\%$",
        "\\texttt{NO\\_GO}",
    ):
        assert token in step105_yelp


def directional_terminal_step96_checks() -> None:
    result_path = ROOT / "STEP96_CONFIRMATORY_RESULTS_2026-08-13.json"
    validation_path = ROOT / "STEP96_CONFIRMATORY_INDEPENDENT_VALIDATION_2026-08-13.json"
    sensitivity_path = ROOT / "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_SENSITIVITY_2026-08-13.json"
    sensitivity_raw = ROOT / "STEP96_POSTCONFIRMATORY_COMPLETE_TOP8_RAW_2026-08-13.npz"
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == (
        "41fb12b9797cc592b64a53cb2861b5c39c2e90454e62c6a9e8fd5f3eb60ae901"
    )
    assert hashlib.sha256(validation_path.read_bytes()).hexdigest() == (
        "294af7cf4c850cf053b547d75fde41534ccf48211b63a1986dd127ebcaf77809"
    )
    assert hashlib.sha256(sensitivity_path.read_bytes()).hexdigest() == (
        "9ede07d62bda858c45b1fad503d6735b6fab19b682e012f55a62cc6987aaca51"
    )
    assert hashlib.sha256(sensitivity_raw.read_bytes()).hexdigest() == (
        "abe83b22c879e065114577af36bf93ee43d98cae855f52ebb8ef88affb9e1dad"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    assert result["decision"] == "NO_GO_RETAIN_STEP96_NEGATIVE"
    assert result["failed_gates"] == ["alias_integer_loss_counts_at_most_one_point"]
    assert result["quality"]["loss_counts"] == [93, 100, 88, 86]
    assert result["quality"]["allowed_loss_count"] == 98
    close(result["mechanism"]["path_change_rate"], 0.9913333333333333)
    close(result["mechanism"]["final_root_change_rate"], 0.16233333333333333)
    close(result["inference"]["terminal"]["mean"], 0.01630533333333334)
    close(result["inference"]["terminal"]["bootstrap_95"][0], 0.013596566666666669)
    close(result["inference"]["terminal"]["bootstrap_95"][1], 0.019026)
    close(result["inference"]["cumulative"]["mean"], 0.3161253333333334)
    close(result["inference"]["cumulative"]["bootstrap_95"][0], 0.2854930166666667)
    close(result["inference"]["cumulative"]["bootstrap_95"][1], 0.3465479000000001)
    assert result["mechanism"]["fixed_root_history_exact"] is True
    close(result["mechanism"]["max_abs_fixed_terminal_delta"], 0.0)
    close(result["mechanism"]["max_abs_fixed_cumulative_delta"], 0.0)
    assert validation["status"] == "PASS_STEP96_INDEPENDENT_VALIDATION"
    assert all(validation["checks"].values())
    assert sensitivity["status"] == "POSTCONFIRMATORY_EXPLORATORY_NOT_PRIMARY_CONFIRMATION"
    assert sensitivity["primary_decision_unchanged"] == result["decision"]
    assert sensitivity["scope"]["all_predeclared_top8_conditions_included"] is True
    assert sensitivity["scope"]["nominal_rows"] == 8
    assert sensitivity["scope"]["unique_tau_threshold_conditions"] == 2
    conservative = next(
        row for row in sensitivity["unique_conditions"] if row["loss_cap"] == 0.005
    )
    assert conservative["loss_counts"] == [31, 33, 47, 31]
    assert conservative["allowed_loss_count"] == 98
    assert conservative["all_gates_with_holm"] is True
    close(conservative["path_change_rate"], 0.9943333333333333)
    close(conservative["final_root_change_rate"], 0.13233333333333333)
    close(conservative["inference"]["terminal"]["mean"], 0.009145333333333335)
    close(conservative["inference"]["terminal"]["bootstrap_95"][0], 0.006667966666666669)
    close(conservative["inference"]["terminal"]["bootstrap_95"][1], 0.011730083333333337)
    close(conservative["inference"]["terminal"]["holm_q_across_unique_top8_conditions"], 1.999980000199998e-05)
    close(conservative["inference"]["cumulative"]["mean"], 0.22102200000000002)
    close(conservative["inference"]["cumulative"]["bootstrap_95"][0], 0.19329345000000006)
    close(conservative["inference"]["cumulative"]["bootstrap_95"][1], 0.2500548833333333)


def scitail_step97_stop_checks() -> None:
    result_path = ROOT / "STEP97_STAGEA_ROOT_GEOMETRY_STOP_2026-08-13.json"
    validation_path = ROOT / "STEP97_ROOT_GEOMETRY_STOP_INDEPENDENT_VALIDATION_2026-08-13.json"
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == (
        "6ce784b278c9e710a129c1f119994a049348b056db6d72249aa3435c74a939c7"
    )
    assert hashlib.sha256(validation_path.read_bytes()).hexdigest() == (
        "d26a75ea5f03f22f95e1c5b46b49eadac0c40d6457c3b456a2dd4fbf1a4eb91d"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert result["decision"] == "NO_GO_STEP97_ROOT_GEOMETRY_STOP"
    assert result["failed_gates"] == ["parent_unique_best_by_2_5pp_on_roster_gate"]
    close(result["geometry"]["roster_gate"]["parent_minus_best_challenger"], 0.004)
    assert result["sealed_test_opened"] is False
    assert validation["status"] == "PASS_STEP97_GEOMETRY_STOP_VALIDATION"
    assert all(validation["checks"].values())


def imdb_step104_primary_checks() -> None:
    development_path = ROOT / "STEP103_ROBUST_DUAL_DEVELOPMENT_LEDGER_2026-08-13.json"
    development_validation_path = ROOT / "STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION_2026-08-13.json"
    frozen_config_path = ROOT / "STEP103_ROBUST_HELDOUT_FROZEN_CONFIG_2026-08-13.json"
    lock_path = ROOT / "STEP104_PREOUTCOME_LOCK_2026-08-13.json"
    predictions_path = ROOT / "STEP104_PREOUTCOME_PREDICTIONS_2026-08-13.npz"
    arrays_path = ROOT / "STEP104_PRIMARY_CONFIRMATORY_ARRAYS_2026-08-13.npz"
    result_path = ROOT / "STEP104_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json"
    validation_path = ROOT / "STEP104_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json"
    artifact_reconstruction_path = ROOT / "STEP104_ARTIFACT_RECONSTRUCTION_2026-08-13.json"
    expected_hashes = {
        development_path: "c92841f05a5d642adfd15ef2b20f14b0602e554cbcae5b027f7c1237b7a92d49",
        development_validation_path: "50882f84aab8225a12102e3a876ca220c582497f323eca2deac0192ba5db6ae1",
        frozen_config_path: "1a1a2f3d8c65c40134f5d81d31f8ae2dcebb36bf332f1b2ef156f04a80bca760",
        lock_path: "0d467d13029021caba2bf21eaccdbaedeadc1219922de25ea18f6518e453542a",
        predictions_path: "930591dc4957c15e123a0e1d41f1270cc6952737b35c0315b6ff12ea0bca61a3",
        arrays_path: "6e51edfbd8d52f4e63f945181e8808d542c5914ae569cc8b7853d51f10795c7f",
        result_path: "30ec5cdac98ef04abe605752c38093d65f1950e9f5ff1be7555292fc4310838c",
        validation_path: "abd5e9d2b9cedb2c5626275f8042a01fa543c707e682b7dd0c5b0926455ce5fb",
        artifact_reconstruction_path: "3e18667c56d302f2e73d6fee9d779d0a7371be4a227d2970e01c2c188eb186a4",
    }
    for path, expected in expected_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    development = json.loads(development_path.read_text(encoding="utf-8"))
    development_validation = json.loads(development_validation_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    artifact_reconstruction = json.loads(artifact_reconstruction_path.read_text(encoding="utf-8"))

    assert development["decision"] == "GO_STEP103_TO_HELDOUT_LOCK"
    assert development["unique_condition_count"] == 27
    assert development["expanded_cell_count"] == 81
    assert development["dual_eligible_cell_count"] == 6
    assert development["selected"]["cell_id"] == "l0.0075_t0.025_b5_tau0.050"
    close(development["selected"]["search"]["mean_terminal_delta"], 0.015604999999999982)
    close(development["selected"]["verify"]["mean_terminal_delta"], 0.005331999999999992)
    close(development["selected_verification_inference"]["terminal"]["bootstrap_95"][0], 0.0041853333333333265)
    close(development["selected_verification_inference"]["terminal"]["bootstrap_95"][1], 0.006484033333333323)
    assert development["sealed_outcome_opened"] is False
    assert development_validation["decision"] == "PASS_STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION"
    assert all(development_validation["checks"].values())

    assert result["decision"] == "GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY"
    assert result["failed_gates"] == []
    assert all(result["gates"].values())
    assert result["provenance"]["development_informed_heldout_confirmation"] is True
    assert result["provenance"]["before_all_data_preregistered"] is False
    assert result["rows"] == 13000 and result["runs"] == 3000
    assert result["quality"]["loss_counts"] == [60, 68, 57, 68]
    assert result["quality"]["allowed_loss_count"] == 130
    assert result["quality"]["trigger_counts"] == [116, 126, 108, 136]
    assert result["quality"]["coordinate_wise_nonimproving"] is True
    close(result["effects"]["terminal"]["mean"], 0.0070239999999999895)
    close(result["effects"]["terminal"]["bootstrap_95"][0], 0.00615464999999999)
    close(result["effects"]["terminal"]["bootstrap_95"][1], 0.007881999999999988)
    close(result["effects"]["active_minus_fixed_terminal"]["mean"], 0.0070239999999999895)
    close(result["effects"]["cumulative"]["mean"], 0.02794666666666663)
    close(result["effects"]["cumulative"]["bootstrap_95"][0], 0.02466866666666663)
    close(result["effects"]["cumulative"]["bootstrap_95"][1], 0.03129273333333329)
    close(result["mechanism"]["path_change_rate"], 0.9993333333333333)
    close(result["mechanism"]["query_set_change_rate"], 0.9993333333333333)
    close(result["mechanism"]["final_root_change_rate"], 0.6066666666666667)
    close(result["mechanism"]["parent_to_challenger_rate"], 0.35433333333333333)
    close(result["mechanism"]["challenger_to_parent_rate"], 0.12133333333333333)
    close(result["mechanism"]["max_abs_fixed_terminal_delta"], 0.0)
    close(result["mechanism"]["max_abs_fixed_cumulative_delta"], 0.0)
    assert result["mechanism"]["fixed_root_history_bitwise_equal"] is True

    assert validation["decision"] == "PASS_STEP104_PRIMARY_INDEPENDENT_RECONSTRUCTION"
    assert validation["failed_checks"] == []
    assert all(validation["checks"].values())
    assert all(validation["recomputed_gates"].values())
    close(validation["recomputed"]["terminal_pp"], 0.7023999999999989)
    assert artifact_reconstruction["decision"] == "PASS_STEP104_ARTIFACT_RECONSTRUCTION"
    assert artifact_reconstruction["failed_checks"] == []
    assert all(artifact_reconstruction["checks"].values())
    assert all(artifact_reconstruction["recomputed_gates"].values())


def yelp_step105_one_shot_checks() -> None:
    predata_path = ROOT / "STEP105_PREDATA_LOCK_2026-08-13.json"
    lock_path = ROOT / "STEP105_PREOUTCOME_LOCK_2026-08-13.json"
    predictions_path = ROOT / "STEP105_PREOUTCOME_PREDICTIONS_2026-08-13.npz"
    arrays_path = ROOT / "STEP105_PRIMARY_CONFIRMATORY_ARRAYS_2026-08-13.npz"
    result_path = ROOT / "STEP105_PRIMARY_CONFIRMATORY_LEDGER_2026-08-13.json"
    validation_path = ROOT / "STEP105_PRIMARY_INDEPENDENT_VALIDATION_2026-08-13.json"
    reconstruction_path = ROOT / "STEP105_ARTIFACT_RECONSTRUCTION_2026-08-13.json"
    expected_hashes = {
        predata_path: "11ae70195ab6caabb0c0ad3a0b266a6af306cc1623083f616b8726d481d7a794",
        lock_path: "497db50848a8509013c55463cfac63d11839dbbcf9708da78e0ea677a4cf1610",
        predictions_path: "bc3dcd88cced111d3678dfb85250bd0ba9f47bc1c9fdcfb831450526789bb970",
        arrays_path: "caafd0552132dcdb58b72d27151f4c385c85611551a6165afa7646bc0821c9c5",
        result_path: "e47c83a2b386545a980accea2f755793910e276c86971c7e888967fe3237bd3f",
        validation_path: "0633ca430eafea7f24d30bcfdd006dea186b40ad1df8f28ce9586055c88e1639",
        reconstruction_path: "71b498c266e215484df72275aeaaf633e081d3baca1958530030757c4108defc",
    }
    for path, expected in expected_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    result = json.loads(result_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    assert result["decision"] == "NO_GO_RETAIN_STEP105_PROSPECTIVE_NEGATIVE"
    assert result["failed_gates"] == [
        "terminal_mean_at_least_half_point",
        "active_minus_fixed_mean_at_least_half_point",
    ]
    assert result["target_selector_search_rows"] == 0
    assert result["target_selector_verify_rows"] == 0
    assert result["target_selector_grid_cells"] == 0
    assert result["rows"] == 38000 and result["paired_runs"] == 3000
    assert result["loss_counts"] == [178, 209, 143, 213]
    assert result["allowed_loss_count"] == 380
    assert result["coordinate_wise_nonimproving"] is True
    close(result["effect"]["terminal"]["mean"], 0.003527333333333336)
    close(result["effect"]["terminal"]["bootstrap_95"][0], 0.0030899833333333355)
    close(result["effect"]["terminal"]["bootstrap_95"][1], 0.003977333333333336)
    close(result["effect"]["active_minus_fixed_terminal"]["mean"], 0.003527333333333336)
    close(result["effect"]["active_minus_fixed_terminal"]["bootstrap_95"][0], 0.0030760000000000023)
    close(result["effect"]["active_minus_fixed_terminal"]["bootstrap_95"][1], 0.00395601666666667)
    close(result["effect"]["cumulative"]["mean"], 0.017924666666666682)
    close(result["effect"]["cumulative"]["bootstrap_95"][0], 0.016162583333333345)
    close(result["effect"]["cumulative"]["bootstrap_95"][1], 0.01970603333333335)
    close(result["effect"]["path_change_rate"], 0.999)
    close(result["effect"]["query_set_change_rate"], 0.9986666666666667)
    close(result["effect"]["final_root_change_rate"], 0.417)
    close(result["effect"]["parent_to_challenger_rate"], 0.2653333333333333)
    close(result["effect"]["challenger_to_parent_rate"], 0.09366666666666666)
    close(result["effect"]["max_abs_fixed_terminal_delta"], 0.0)
    close(result["effect"]["max_abs_fixed_cumulative_delta"], 0.0)
    assert result["effect"]["fixed_root_history_exact"] is True
    assert sum(result["gates"].values()) == 14 and len(result["gates"]) == 16

    assert validation["verdict"] == "PASS_STEP105_PRIMARY_INDEPENDENT_RECONSTRUCTION"
    assert validation["failed_checks"] == []
    assert all(validation["checks"].values())
    assert validation["primary_decision"] == result["decision"]
    assert validation["test_rows"] == 38000
    assert validation["test_reference_passed_to_endpoint"] is False
    assert validation["item_lookup_passed_to_endpoint"] is False

    assert reconstruction["decision"] == "PASS_STEP105_ARTIFACT_RECONSTRUCTION"
    assert reconstruction["failed_checks"] == []
    assert all(reconstruction["checks"].values())
    assert reconstruction["recomputed_primary_decision"] == result["decision"]
    close(reconstruction["recomputed"]["terminal_pp"], 0.35273333333333357)


def learned_adapter_step92_checks() -> None:
    result_path = ROOT / "STEP92_LEARNED_ADAPTER_CONFIRMATORY_RESULTS_2026-08-12.json"
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == (
        "8e20db3d7bbeb6fd4dff550ba8a24e2e992d88eb7bcaeb3d1233428deca011e4"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["decision"] == "GO_SCORE_CHANGING_LEARNED_ADAPTER_BRIDGE"
    task = result["task"]
    assert task["n_holdout"] == 7600 and task["paired_runs"] == 1000
    assert task["adapter_parent_accuracy_gaps"] == [0.0, 0.0, 0.0, 0.0]
    close(task["path_change_rate"], 1.0)
    close(task["query_set_change_rate"], 0.993)
    close(task["final_root_change_rate"], 0.552)
    close(task["terminal_regret_delta"]["mean"], 0.005467500000000004)
    close(task["terminal_regret_delta"]["bootstrap_95"][0], 0.005022500000000003)
    close(task["terminal_regret_delta"]["bootstrap_95"][1], 0.0059250000000000014)
    close(task["active_minus_fixed_terminal"]["mean"], 0.005467500000000004)
    close(task["max_abs_fixed_terminal_delta"], 0.0)
    close(task["max_abs_fixed_cumulative_delta"], 0.0)
    assert all(task["gates"].values())
    assert all(row["parameter_count"] == 50817 for row in task["learned_adapter_audits"].values())

    receipt = load("STEP92_INDEPENDENT_VALIDATION_2026-08-12.json")
    assert receipt["verdict"] == "PASS_STEP92_INDEPENDENT_VALIDATION"
    assert receipt["checks"] == 4170
    assert receipt["active_trajectory_replays"] == 2000
    assert receipt["fixed_query_replays"] == 2000
    assert receipt["endpoint_replay"]["sample_size"] == 256
    assert receipt["reconstructed"]["decision"] == result["decision"]

    numeric = load("STEP92_NUMERIC_TIE_ROBUSTNESS_2026-08-12.json")
    assert numeric["decision"] == "PASS_NUMERIC_TIE_ROBUSTNESS"
    assert numeric["all_policies_pass_original_harm_gate"] is True
    means = [row["mean_terminal_delta_pp"] for row in numeric["policies"].values()]
    close(min(means), 0.5457500000000004)
    close(max(means), 0.5480000000000003)


def source_faithful_step93_checks() -> None:
    result_path = ROOT / "STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RESULTS_2026-08-13.json"
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == (
        "bf6fe6dd3d07be47c9aafa534d0bfa1efd8a7bdbef9a702b61f3f9246d229698"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["decision"] == "INVALID_STEP93_IMPLEMENTATION"
    task = result["task"]
    assert task["n_holdout"] == 3076 and task["paired_runs"] == 1000
    close(task["path_change_rate"], 0.998)
    close(task["query_set_change_rate"], 0.991)
    close(task["final_root_change_rate"], 0.366)
    close(task["terminal_regret_delta"]["mean"], 0.00032000000000000106)
    close(task["terminal_regret_delta"]["bootstrap_95"][0], -0.000549999999999999)
    close(task["terminal_regret_delta"]["bootstrap_95"][1], 0.0012150000000000008)
    close(task["terminal_regret_delta"]["one_sided_signflip_p"], 0.23593764062359376)
    close(task["cumulative_regret_delta"]["mean"], 0.04094500000000001)
    close(task["cumulative_regret_delta"]["bootstrap_95"][0], 0.01290481249999998)
    close(task["cumulative_regret_delta"]["bootstrap_95"][1], 0.06891012499999999)
    close(task["active_minus_fixed_terminal"]["mean"], 0.00032000000000000106)
    close(task["max_abs_fixed_terminal_delta"], 0.0)
    close(task["max_abs_fixed_cumulative_delta"], 0.0)
    assert task["gates"]["same_exact_similarity_for_acquisition_and_posterior"] is False
    assert task["gates"]["mean_terminal_delta_at_least_0_5pp"] is False
    assert all(
        row["parameter_count"] == 50817
        for row in task["learned_adapter_audits"].values()
    )

    receipt_path = ROOT / "STEP93_INDEPENDENT_VALIDATION_AND_ADJUDICATION_2026-08-13.json"
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == (
        "5ae09c1e3f6801940183e1e17ef06da4f683b3f2cdd8fed080dbca573b3be1e0"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS_STEP93_INDEPENDENT_RECONSTRUCTION"
    assert receipt["adjudicated_decision"] == "NO_GO_RETAIN_SOURCE_FAITHFUL_NEGATIVE"
    assert receipt["checks_count"] == 82
    close(
        receipt["numeric_validator_adjudication"]["observed_max_abs_difference"],
        2.220446049250313e-16,
    )
    assert receipt["numeric_validator_adjudication"]["corrected_same_s_gate"] is True
    assert receipt["corrected_gates"]["same_exact_similarity_for_acquisition_and_posterior"] is True
    assert receipt["corrected_gates"]["mean_terminal_delta_at_least_0_5pp"] is False
    assert receipt["independent_endpoint_probe"]["samples"] == 128


def source_faithful_step94_checks() -> None:
    result_path = ROOT / "STEP94_CONFIRMATORY_RESULTS_2026-08-13.json"
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == (
        "e93187ecc87341cb24191b46b97372e0e28caf4bb526e3ff8f55319ea0c52ccd"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["decision"] == "NO_GO_RETAIN_FRESH_TASK_NEGATIVE"
    assert result["selected_task"] == "20newsgroups"
    assert result["roster_bank_root_ids"] == [10, 9, 6, 3, 1, 0, 2, 4]
    assert result["parent_bank_root_id"] == 10
    assert result["endpoint"]["integrated_variant_parameter_counts"] == [
        205245,
        205245,
        205245,
        205245,
    ]
    close(result["mechanism"]["path_change_rate"], 1.0)
    close(result["mechanism"]["query_set_change_rate"], 1.0)
    close(result["mechanism"]["final_root_change_rate"], 0.482)
    close(result["effects"]["terminal"]["mean"], 0.0011979999999999998)
    close(
        result["effects"]["terminal"]["bootstrap_95"][0],
        -0.00044402499999999893,
    )
    close(
        result["effects"]["terminal"]["bootstrap_95"][1],
        0.002895049999999998,
    )
    close(
        result["effects"]["terminal"]["one_sided_signflip_p"],
        0.08126918730812692,
    )
    close(result["effects"]["cumulative"]["mean"], 0.08215699999999991)
    close(
        result["effects"]["cumulative"]["bootstrap_95"][0],
        0.03854362499999993,
    )
    close(
        result["effects"]["cumulative"]["bootstrap_95"][1],
        0.12655414999999995,
    )
    close(
        result["effects"]["cumulative"]["one_sided_signflip_p"],
        0.0001999980000199998,
    )
    close(result["effects"]["active_minus_fixed_terminal"]["mean"], 0.0011979999999999998)
    close(result["effects"]["fixed_terminal_max_abs"], 0.0)
    close(result["effects"]["fixed_cumulative_max_abs"], 0.0)
    assert result["effects"]["fixed_root_history_exact"] is True
    assert result["quality"]["coordinate_wise_nonimproving"] is True
    assert max(result["quality"]["alias_losses"]) < 0.005
    assert result["gates"]["terminal_mean_at_least_half_point"] is False
    assert result["gates"]["terminal_bootstrap_lower_positive"] is False
    assert result["gates"]["cumulative_mean_positive"] is True
    assert result["gates"]["cumulative_bootstrap_lower_positive"] is True

    receipt_path = ROOT / "STEP94_INDEPENDENT_VALIDATION_2026-08-13.json"
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == (
        "a3bd46c14edd3ccaf1cfd823c44b2a0d757e51b581a50474a52306563bcd86f7"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["decision"] == result["decision"]
    assert all(receipt["checks"].values())
    close(receipt["terminal_mean"], result["effects"]["terminal"]["mean"])
    close(receipt["cumulative_mean"], result["effects"]["cumulative"]["mean"])
    assert receipt["fixed_root_history_exact"] is True


def main() -> None:
    primary_llm_checks()
    natural_alias_checks()
    operational_significance_checks()
    reference_free_terminal_checks()
    robustness_and_cover_checks()
    coda_checks()
    llm_selector_step88_checks()
    learned_adapter_step92_checks()
    source_faithful_step93_checks()
    source_faithful_step94_checks()
    directional_terminal_step96_checks()
    scitail_step97_stop_checks()
    imdb_step104_primary_checks()
    yelp_step105_one_shot_checks()
    soft_weight_checks()
    rendered_source_checks()
    print("PASS_PAPER_TABLE_SOURCE_CONSISTENCY")


if __name__ == "__main__":
    main()
