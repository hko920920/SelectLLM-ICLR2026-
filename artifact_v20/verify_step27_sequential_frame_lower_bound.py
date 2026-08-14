"""Verify the Step 27 sequential sampling-frame lower bound.

The proof is in STEP27_SEQUENTIAL_FRAME_LOWER_BOUND_2026-08-06.md.  This
checker verifies the binary-channel constants, the integer alias partitions,
the induced product priors, and the direct-product minimax values.  It uses
only Python's standard library.
"""

from __future__ import annotations

import itertools
import json
import math


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def mutual_information(prior_a: float, prob_one_a: float, prob_one_b: float) -> float:
    predictive = prior_a * prob_one_a + (1.0 - prior_a) * prob_one_b
    return (
        binary_entropy(predictive)
        - prior_a * binary_entropy(prob_one_a)
        - (1.0 - prior_a) * binary_entropy(prob_one_b)
    )


def bayes_error(prior_a: float, prob_one_a: float, prob_one_b: float) -> float:
    """Bayes error for identifying A versus B after one binary observation."""

    return min(prior_a * prob_one_a, (1.0 - prior_a) * prob_one_b) + min(
        prior_a * (1.0 - prob_one_a),
        (1.0 - prior_a) * (1.0 - prob_one_b),
    )


def one_coordinate_constants() -> dict:
    c = 0.25
    channels = {
        "q_plus": (1.0, c),
        "q_minus": (1.0 - c, 0.0),
    }
    worlds = {}
    for frame_bit, prior_a in (("A", 2.0 / 3.0), ("B", 1.0 / 3.0)):
        rows = {}
        for query, (prob_one_a, prob_one_b) in channels.items():
            rows[query] = {
                "information_bits": mutual_information(
                    prior_a, prob_one_a, prob_one_b
                ),
                "bayes_error": bayes_error(prior_a, prob_one_a, prob_one_b),
            }
        worlds[frame_bit] = rows

    high_information = worlds["A"]["q_plus"]["information_bits"]
    low_information = worlds["A"]["q_minus"]["information_bits"]
    high_error = worlds["A"]["q_plus"]["bayes_error"]
    low_error = worlds["A"]["q_minus"]["bayes_error"]
    prior_entropy = binary_entropy(2.0 / 3.0)

    assert math.isclose(
        worlds["B"]["q_minus"]["information_bits"], high_information
    )
    assert math.isclose(
        worlds["B"]["q_plus"]["information_bits"], low_information
    )
    assert math.isclose(worlds["B"]["q_minus"]["bayes_error"], high_error)
    assert math.isclose(worlds["B"]["q_plus"]["bayes_error"], low_error)

    blind_first_information = (high_information + low_information) / 2.0
    blind_first_error = (high_error + low_error) / 2.0
    return {
        "c": c,
        "prior_entropy_bits": prior_entropy,
        "worlds": worlds,
        "high_information_bits": high_information,
        "low_information_bits": low_information,
        "wrong_polarity_information_gap_bits": high_information - low_information,
        "frame_aware_information_per_fresh_coordinate_bits": high_information,
        "frame_blind_minimax_information_per_fresh_coordinate_bits": blind_first_information,
        "frame_blind_information_regret_per_coordinate_bits": (
            high_information - blind_first_information
        ),
        "maximum_remaining_information_after_high_query_bits": (
            prior_entropy - high_information
        ),
        "maximum_remaining_information_after_low_query_bits": (
            prior_entropy - low_information
        ),
        "high_query_bayes_error": high_error,
        "low_query_bayes_error": low_error,
        "frame_blind_minimax_bayes_error_per_coordinate": blind_first_error,
        "frame_aware_bayes_error_per_coordinate": high_error,
        "frame_blind_hamming_regret_per_coordinate": blind_first_error - high_error,
        "frame_blind_fresh_coordinate_error_reduction": 1.0 / 3.0
        - blind_first_error,
        "maximum_remaining_error_after_any_first_query": max(high_error, low_error),
    }


def bit_vectors(bits: int):
    return list(itertools.product((0, 1), repeat=bits))


def partition_check(bits: int) -> dict:
    """Check the common leaf registry and all frame-induced root partitions.

    There are 2**bits behavior vectors.  Each behavior has 2**bits public leaf
    IDs.  Under frame omega, the leaves for behavior z are partitioned into
    2**matches(z, omega) roots, each containing 2**mismatches aliases.
    """

    vectors = bit_vectors(bits)
    leaves_per_behavior = 2**bits
    expected_total_roots = 3**bits
    maximum_prior_error = 0.0
    maximum_marginal_error = 0.0
    maximum_factorization_error = 0.0
    minimum_alias_group_size = leaves_per_behavior
    maximum_alias_group_size = 0

    for omega in vectors:
        root_counts = {}
        for z in vectors:
            matches = sum(a == b for a, b in zip(z, omega))
            roots = 2**matches
            group_size = leaves_per_behavior // roots
            assert roots * group_size == leaves_per_behavior
            root_counts[z] = roots
            minimum_alias_group_size = min(minimum_alias_group_size, group_size)
            maximum_alias_group_size = max(maximum_alias_group_size, group_size)

        total_roots = sum(root_counts.values())
        assert total_roots == expected_total_roots
        prior = {z: count / total_roots for z, count in root_counts.items()}

        for z, probability in prior.items():
            matches = sum(a == b for a, b in zip(z, omega))
            expected = (2.0 / 3.0) ** matches * (1.0 / 3.0) ** (bits - matches)
            maximum_prior_error = max(maximum_prior_error, abs(probability - expected))

        for coordinate in range(bits):
            marginal_match = sum(
                probability
                for z, probability in prior.items()
                if z[coordinate] == omega[coordinate]
            )
            maximum_marginal_error = max(
                maximum_marginal_error, abs(marginal_match - 2.0 / 3.0)
            )

        # Product factorization: every two-coordinate joint equals the product
        # of its marginals.  Checking all assignments is cheap for bits <= 8.
        for left in range(bits):
            for right in range(left + 1, bits):
                for left_value in (0, 1):
                    for right_value in (0, 1):
                        joint = sum(
                            probability
                            for z, probability in prior.items()
                            if z[left] == left_value and z[right] == right_value
                        )
                        marginal_left = sum(
                            probability
                            for z, probability in prior.items()
                            if z[left] == left_value
                        )
                        marginal_right = sum(
                            probability
                            for z, probability in prior.items()
                            if z[right] == right_value
                        )
                        maximum_factorization_error = max(
                            maximum_factorization_error,
                            abs(joint - marginal_left * marginal_right),
                        )

    return {
        "bits_or_budget": bits,
        "public_behavior_vectors": 2**bits,
        "public_leaves_per_behavior": leaves_per_behavior,
        "public_candidate_ids": 4**bits,
        "latent_frames": 2**bits,
        "roots_per_frame": expected_total_roots,
        "minimum_alias_group_size": minimum_alias_group_size,
        "maximum_alias_group_size": maximum_alias_group_size,
        "maximum_product_prior_error": maximum_prior_error,
        "maximum_coordinate_marginal_error": maximum_marginal_error,
        "maximum_pairwise_factorization_error": maximum_factorization_error,
    }


def polarity_minimax_check(bits: int, high: float, low: float) -> dict:
    """Enumerate all fixed frames and deterministic polarity vectors."""

    vectors = bit_vectors(bits)
    gap = high - low
    expected_regrets = []
    worst_regrets = []
    for polarity in vectors:
        regrets = []
        for omega in vectors:
            wrong = sum(a != b for a, b in zip(polarity, omega))
            regrets.append(wrong * gap)
        expected_regrets.append(sum(regrets) / len(regrets))
        worst_regrets.append(max(regrets))

    target_average = bits * gap / 2.0
    assert all(math.isclose(value, target_average) for value in expected_regrets)
    return {
        "bits_or_budget": bits,
        "deterministic_policies_checked": len(vectors),
        "frames_checked": len(vectors),
        "average_regret_each_policy_bits": target_average,
        "minimum_worst_frame_regret_bits": min(worst_regrets),
        "fair_randomized_policy_regret_every_frame_bits": target_average,
    }


def bound_table(constants: dict) -> list[dict]:
    info = constants["frame_blind_information_regret_per_coordinate_bits"]
    hamming = constants["frame_blind_hamming_regret_per_coordinate"]
    return [
        {
            "budget": budget,
            "tight_minimax_information_regret_bits": budget * info,
            "tight_minimax_hamming_regret": budget * hamming,
        }
        for budget in (1, 3, 5, 10, 20, 50)
    ]


def exact_match_acquisition(probabilities: list[float], outputs: list[int]) -> float:
    mass_one = sum(
        probability
        for probability, output in zip(probabilities, outputs)
        if output == 1
    )
    return mass_one**2 + (1.0 - mass_one) ** 2


def stable_softmax(values: list[float], tau: float) -> list[float]:
    peak = max(values)
    weights = [math.exp((value - peak) / tau) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def select_llm_linear_amplification(
    budget: int, tau: float, extra_clones: int = 4
) -> dict:
    """Run the fixed-pool construction for Equation (5)/(7) with exact match.

    Base candidate order is A, B, C, D, E.  Every good query partitions
    {C,D,E} from {A,B} and has oracle label 0.  Every bad query isolates C;
    its oracle label alternates 1,0,1,0,... .  Four exact copies of C give the
    C behavior five of nine submitted IDs.
    """

    if budget < 1:
        raise ValueError("budget must be positive")
    base_outputs = []
    labels = []
    kinds = []
    # Good queries first only to make deterministic tie-breaking reproducible;
    # the clean/attacked score inequalities are strict across the two types.
    for _ in range(budget):
        base_outputs.append([0, 0, 1, 1, 1])
        labels.append(0)
        kinds.append("good")
    for step in range(budget):
        base_outputs.append([0, 0, 1, 0, 0])
        labels.append(1 if step % 2 == 0 else 0)
        kinds.append("bad")

    # Transpose to candidates x queries.
    base = [
        [base_outputs[query][candidate] for query in range(2 * budget)]
        for candidate in range(5)
    ]
    base_scores = [
        sum(output == label for output, label in zip(row, labels)) / (2 * budget)
        for row in base
    ]
    best_score = max(base_scores)

    def run(with_clones: bool) -> dict:
        responses = [list(row) for row in base]
        original = list(range(5))
        if with_clones:
            for _ in range(extra_clones):
                responses.append(list(base[2]))
                original.append(2)

        cumulative = [0.0] * len(responses)
        remaining = list(range(2 * budget))
        query_kinds = []
        selected_original = []
        regrets = []
        for _ in range(budget):
            posterior = stable_softmax(cumulative, tau)
            scores = [
                exact_match_acquisition(
                    posterior, [responses[index][query] for index in range(len(responses))]
                )
                for query in remaining
            ]
            location = min(range(len(scores)), key=lambda index: (scores[index], index))
            query = remaining.pop(location)
            query_kinds.append(kinds[query])
            for candidate in range(len(responses)):
                cumulative[candidate] += responses[candidate][query] == labels[query]
            selected = original[max(range(len(cumulative)), key=lambda index: cumulative[index])]
            selected_original.append(selected)
            regrets.append(best_score - base_scores[selected])
        return {
            "query_kinds": query_kinds,
            "selected_original": selected_original,
            "cumulative_regret": sum(regrets),
            "final_regret": regrets[-1],
        }

    clean = run(False)
    refined = run(True)
    expected_gap = base_scores[0] - base_scores[2]
    expected_refined_cumulative = ((budget + 1) // 2) * expected_gap
    assert clean["query_kinds"] == ["good"] * budget
    assert refined["query_kinds"] == ["bad"] * budget
    assert math.isclose(clean["cumulative_regret"], 0.0)
    assert math.isclose(refined["cumulative_regret"], expected_refined_cumulative)
    if budget % 2 == 1:
        assert math.isclose(refined["final_regret"], expected_gap)

    return {
        "budget": budget,
        "tau": tau,
        "base_candidates": 5,
        "extra_exact_C_clones": extra_clones,
        "refined_candidate_ids": 5 + extra_clones,
        "base_scores": base_scores,
        "best_minus_C_score_gap": expected_gap,
        "clean": clean,
        "refined": refined,
        "query_set_overlap": len(
            set(index for index, kind in enumerate(clean["query_kinds"]) if kind == "bad")
            & set(index for index, kind in enumerate(refined["query_kinds"]) if kind == "good")
        ),
        "expected_refined_cumulative_regret": expected_refined_cumulative,
    }


def main() -> None:
    constants = one_coordinate_constants()

    # A new coordinate dominates spending a query on any already-touched
    # coordinate, even if the latter could reveal all its remaining entropy.
    assert (
        constants["maximum_remaining_information_after_low_query_bits"]
        < constants["frame_blind_minimax_information_per_fresh_coordinate_bits"]
    )
    assert (
        constants["maximum_remaining_information_after_high_query_bits"]
        < constants["frame_aware_information_per_fresh_coordinate_bits"]
    )
    assert (
        constants["maximum_remaining_error_after_any_first_query"]
        < constants["frame_blind_fresh_coordinate_error_reduction"]
    )

    partitions = [partition_check(bits) for bits in range(1, 7)]
    for row in partitions:
        assert row["maximum_product_prior_error"] <= 1e-12
        assert row["maximum_coordinate_marginal_error"] <= 1e-12
        assert row["maximum_pairwise_factorization_error"] <= 1e-12

    polarity_checks = [
        polarity_minimax_check(
            bits,
            constants["high_information_bits"],
            constants["low_information_bits"],
        )
        for bits in range(1, 9)
    ]

    amplification_checks = [
        select_llm_linear_amplification(budget, tau)
        for budget in (3, 5, 11, 51)
        for tau in (0.1, 0.5, 1.0, 3.0, 10.0)
    ]

    result = {
        "gate": "STEP27_SEQUENTIAL_FRAME_LOWER_BOUND",
        "one_coordinate_constants": constants,
        "integer_partition_checks": partitions,
        "enumerated_polarity_minimax_checks": polarity_checks,
        "select_llm_exact_refinement_amplification_checks": amplification_checks,
        "scaling": bound_table(constants),
        "status": "PASS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
