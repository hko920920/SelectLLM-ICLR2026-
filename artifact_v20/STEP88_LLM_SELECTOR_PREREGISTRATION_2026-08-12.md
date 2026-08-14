# Step 88 preregistration: second-selector, weak-view-distinct registry refinement

Date locked: 2026-08-12 (Asia/Seoul)

## Purpose

This is a result-blind, confirmatory audit of whether the registry-refinement
mechanism extends to a second official selector family.  The audited method is
LLM Selector, which sequentially chooses prompts using multiple weak pairwise
judges and uses an expensive judge on acquired prompts to select a model.

This audit is deliberately narrower than a production attack claim.  It tests
the causal decision object exposed by the official score matrices: whether four
entries that are exactly utility-equivalent to one root, but distinct in the
weak-judge view consumed by acquisition, can alter acquired prompts and
terminal root regret.  No claim will be made that the synthetic aliases are
independent checkpoints or would pass a production registry's admission rules.

## Result-blind status at lock time

Before this file was written, only the following were inspected:

- official source code and commit metadata;
- file names, byte sizes, SHA-256 hashes;
- JSON key names, array dimensions, baseline names, and source-order model
  names.

No judge outcome, weak-judge outcome, clean/refined trajectory, selected model,
or regret value from these six datasets was printed or used to choose the
design below.  The six datasets have not been used in the project's earlier
selector experiments.  Merely archived copies of the official repository are
not prior empirical use.

## Frozen source and inputs

- Repository: `external/llm-selector`
- Upstream: `https://github.com/RobustML-Lab/llm-selector`
- Commit: `15faa47dab102d7a92124b13c1838b55524fc2bf`
- Official acquisition hyperparameters: `eps_loss=0.2`, `eps_draw=0.4`
- Weak judges: all 10 supplied columns with uniform weight

All six official datasets are confirmatory; none may be omitted after outcomes
are opened.

| dataset | judge SHA-256 | weak-judge SHA-256 | N | original candidates | frozen parent |
|---|---|---|---:|---:|---|
| alpacaeval | `9ee29592ab55d883cad91eee15a3a2607589c43682e624a447a09d8578d275f4` | `84e0faaa2e81993a5ae13d14acda9ca357869d570bd41de1afa9c97baa3d96e0` | 805 | 52 | `llama-2-13b-chat-hf` |
| arena-hard | `94516f0ba7eaf0467943deb12538ad5e473d0d7ef984baaaef2a2a54c90b6a03` | `bd8852a964aef17fb95aa3141e4ba1984272183968f4a85ac3c87cf54e38b1d5` | 500 | 67 | `gpt-4o-2024-08-06` |
| bingo | `d066efca199f4209b284c3c6b0917e1d9b613aa74fa2c506cf7fd2a6f23f5c45` | `0cf1728c7dc6ddddebec71669f7eb4f20259cd0bcbd16cf5203ed8f19725d1f7` | 762 | 30 | `HuggingFaceM4_idefics-9b-instruct` |
| flickr30k | `3a1a2d03a63f5f45acf0a4b9dd85c78885c231a48ce0d9393631206f752433ba` | `1fddbc5c8a37facb5f6d9d845ebf31994b7a6a5e85e7562d45908a44a8a26e06` | 1000 | 50 | `openai_gpt-4o-2024-05-13` |
| medi_qa | `cd76912a786c0ed3f1081b1418b14e8b5827b8e511844a9d292338327fd1b7a5` | `2494bc845b9c16b2d223b48858311a594f95eb304781727e2800acfd90dcaf0a` | 150 | 8 | `openai_gpt-4o-mini-2024-07-18` |
| mt-bench | `8bbe2087b844032c7fd0fd04dcfde3817041b9075233c06352fd601adb20d8f0` | `a00c992bbf4fdd487d76ae277153b315ea677be571a8bd5b50b477f59ade6574` | 80 | 5 | `alpaca-13b` |

The parent is mechanically the first non-null candidate in the upstream JSON
insertion order.  This rule and the resulting names were fixed without reading
any score values.  There is no target search.

## Frozen intervention

For each dataset, add exactly four submitted entries mapped to the frozen
parent root.

1. **Strong-judge/root utility.**  Each alias receives the parent's complete
   strong-judge row exactly.  Hence every alias has zero coordinate-wise and
   aggregate utility difference from the parent.  This copy is performed by
   the evaluator to impose the causal equality; it is not an input to the
   alias-view constructor.
2. **Weak-judge view.**  An alias begins with the parent's weak-judge tensor.
   For coordinate `(dataset, query, weak_judge, alias)`, compute
   `SHA256("step88|dataset|query|weak_judge|alias")`.  If the unsigned first
   eight bytes divided by `2^64` are below `0.20`, change the ternary outcome.
   Aliases 0 and 2 use the forward cycle `-1 -> 0 -> +1 -> -1`; aliases 1 and
   3 use the reverse cycle `-1 -> +1 -> 0 -> -1`.  Otherwise retain the parent
   outcome.  Thus construction reads only the parent weak view plus public
   indices and salt; it never reads another candidate, a strong-judge label,
   a sampled pool, a trajectory, or selector state.
3. **Mass.**  The official selector continues to initialize uniform mass per
   submitted entry.  No acquisition-code correction for the root map is made.
4. **Deployment.**  Report root-level decisions.  Alias entries map back to the
   parent; scores for a root are the common empirical score of its entries.
   Ties are resolved by original root source order, eliminating static
   duplicate tie mass from the primary estimand.

The primary wording is **weak-view-distinct, exactly utility-equivalent
refinement**.  “Executable response wrapper” or “production attack” is not an
allowed description of this experiment.

## Frozen paired protocol

- Seeds: integers 0 through 999.
- For each seed and dataset, draw one ordered realization pool without
  replacement using Python `random.Random(seed).sample`.
- Pool size: `min(400, N)` (the upstream default is 400).
- Budget: 30 acquired prompts.
- Clean and refined runs share the exact ordered pool and all deterministic
  tie rules.
- Acquisition reproduces official `methods/llm_selector.py`: uniform initial
  entry posterior, uniform weak-judge weights, entropy averaged over all ten
  weak judges, and `eps=(0.2, 0.4)`.
- At every step, the deployed root is chosen from mean acquired strong-judge
  outcomes.  Pool-wide root win rate is `(mean outcome + 1)/2`.
- Terminal root regret is the best original root's pool-wide win rate minus
  the deployed root's pool-wide win rate at budget 30.
- Cumulative regret is the sum of root regret over the 30 deployments and is
  secondary.

## Causal controls

For every seed, recompute the refined registry's deployment sequence while
forcing the clean query sequence.  Because the aliases share the parent utility
row and deployment is root-level, the fixed-query terminal and cumulative
regret differences must be exactly zero (absolute tolerance `1e-12`).  Failure
of this invariant invalidates the implementation rather than counting as an
experimental result.

Also report:

- ordered path-change rate;
- unique query-set change rate and Jaccard overlap;
- terminal root-change rate;
- realized alias-to-parent weak-view Hamming distances;
- coordinate-wise and aggregate strong-utility differences.

## Frozen inference and gates

For each dataset, the primary paired statistic is
`terminal_regret_refined - terminal_regret_clean` in win-rate units.

- 50,000 paired percentile-bootstrap resamples, seeded by
  `SHA256("step88-bootstrap|dataset")`;
- 100,000 Monte Carlo paired sign flips, seeded by
  `SHA256("step88-signflip|dataset")`, for the one-sided harmful alternative;
- Benjamini-Hochberg correction across the six primary terminal tests;
- report raw means, medians, 95% intervals, p-values, and q-values for every
  dataset regardless of sign.

A dataset passes the confirmatory terminal-harm gate only if all hold:

1. coordinate-wise strong-utility difference is exactly zero;
2. fixed-query terminal and cumulative differences are exactly zero within
   `1e-12`;
3. ordered path-change rate is at least 0.50;
4. mean terminal-regret increase is at least `0.0025` (0.25 percentage point);
5. the paired-bootstrap 95% lower bound is above zero;
6. BH-adjusted one-sided `q <= 0.05`.

Decision labels are frozen as follows:

- `GO_STRONG_CROSS_SELECTOR`: at least two of six datasets pass;
- `GO_LIMITED_CROSS_SELECTOR`: exactly one of six passes;
- `NO_CONFIRMATORY_CROSS_SELECTOR`: none passes;
- `INVALID_IMPLEMENTATION`: any input hash, utility-equivalence, or fixed-query
  invariant fails.

All six outcomes, including improvements and nulls, remain in the release.
Only `GO_STRONG_CROSS_SELECTOR` warrants a main-text strengthening claim;
`GO_LIMITED_CROSS_SELECTOR` is appendix-only and explicitly limited;
`NO_CONFIRMATORY_CROSS_SELECTOR` is retained as a negative result and does not
replace the already locked paper evidence.

