# Research Sync: Notion ↔ SciSpace ↔ GitHub

_Last synchronized: 2026-08-14 (Asia/Seoul)_

## 1. Purpose

This file is the reproducible bridge between:

- **Notion** — hypotheses, experiment decisions, failure records, and manuscript story;
- **SciSpace** — external literature discovery and prior-art comparison;
- **GitHub** — executable implementation, frozen configurations, results, and claim-to-evidence artifacts.

Primary project page supplied by the researcher:

- https://app.notion.com/p/Select-LLM-ICLR-2026-3bce7b0a7d38808286e5c9f94225568f

Current access note: the exact project page is not yet shared with the active Notion integration. Related pages in the same workspace are discoverable, including the experiment log, the temporal-evidence research story, and the VideoSelector closing roadmap.

## 2. Current Notion-side scientific story

The working causal chain is:

```text
frame-wise transient relevance fluctuations / distractor frames
→ temporal persistence improves evidence precision
→ hard Top-K over-concentrates acquisition in a few temporal subregions
→ persistence benefit versus concentration cost changes with frame budget
→ controlled exploration should restore coverage without discarding relevance
→ downstream Video-LLM QA and counterfactual frame replacement validate causality
```

The latest accessible notes report that:

- the harmful effect of selected Raw-only distractor frames has been tested with fixed-budget counterfactual replacement;
- the direction was reproduced with a second Video-LLM;
- a BLIP-2-based relevance estimator preserved the high-replacement effect direction, but the strongest preregistered confidence-interval criterion remained borderline because the eligible sample was small;
- the safe claim is therefore narrower than a universal failure of all image–text relevance estimators.

## 3. Closest literature surfaced by SciSpace

SciSpace semantic query:

> Which recent papers most closely study query-conditioned frame selection or adaptive visual-token acquisition for long-video question answering, especially temporal relevance noise, harmful distractor frames, persistence versus coverage, stochastic exploration, and frame-budget-dependent behavior?

### A. Query-type and query-adaptive allocation

1. **Divide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding** (2025, arXiv)
   - Separates global and localized questions.
   - Uses uniform sampling for global queries and query-aware selection for localized ones.
   - Main overlap: selection policy changes with query requirements.
   - Main distinction to test: SelectLLM changes the persistence/exploration balance based on evidence behavior and budget, not only a global/local query taxonomy.

2. **Static or Dynamic: Towards Query-Adaptive Token Selection for Video Question Answering** (2025, arXiv)
   - Explores allocations between key-frame tokens and delta-frame tokens.
   - Chooses the allocation using query-aware attention.
   - Main overlap: query-specific token allocation under a budget.
   - Main distinction to test: SelectLLM operates on temporal evidence persistence and distractor causality rather than static-versus-dynamic token types.

3. **Seeing the Forest and the Trees: Query-Aware Tokenizer for Long-Video Multimodal Language Models** (2025, arXiv)
   - Scores visual tokens with cross-attention.
   - Predicts an instance-specific retention budget.
   - Preserves temporal order with a re-encoder.
   - Main overlap: adaptive budget and query-aware retention.
   - Main distinction to test: SelectLLM should establish a budget-dependent reversal and controlled-intervention evidence, not only learned retention efficiency.

### B. Relevance, diversity, coverage, and exploration

4. **FOCUS: Efficient Keyframe Selection for Long Video Understanding** (2025, arXiv)
   - Frames temporal clips as arms in a combinatorial pure-exploration problem.
   - Uses confidence bounds to explore uncertain regions before exploiting high-value regions.
   - Main overlap: exploration–exploitation under a strict frame/token budget.
   - Critical baseline: compare against FOCUS-style uncertainty exploration under matched scoring cost and frame budget.

5. **AdaRD-Key: Adaptive Relevance-Diversity Keyframe Sampling for Long-form Video Understanding** (2025, arXiv)
   - Optimizes a relevance–diversity max-volume objective.
   - Falls back toward diversity when the query–video relevance distribution is weak.
   - Main overlap: preventing redundant concentration while retaining query relevance.
   - Critical baseline: determine whether temporal persistence supplies value beyond generic diversity and relevance-aware gating.

6. **End-to-End Video Question Answering with Frame Scoring Mechanisms and Adaptive Sampling** (2024, arXiv)
   - Scores frames using question relevance and inter-frame similarity.
   - Uses differentiable adaptive frame sampling.
   - Main overlap: relevance plus redundancy-aware adaptive selection.
   - Main distinction to test: training-free/frozen evidence analysis and causal frame-replacement tests.

### C. Hierarchical, iterative, or keyframe-conditioned long-video reasoning

7. **MIST: Multi-modal Iterative Spatial-Temporal Transformer for Long-form Video Question Answering** (2022, arXiv)
   - Iteratively selects question-relevant segments and regions.
   - Supports multi-event reasoning through repeated selection and attention.
   - Baseline role: learned iterative spatial-temporal selection.

8. **Koala: Key frame-conditioned long video-LLM** (2024, arXiv)
   - Conditions learnable tokenizers on sparse key frames to adapt short-video models to long videos.
   - Baseline role: learned keyframe-conditioned representation rather than external frame acquisition.

9. **Too Many Frames, not all Useful: Efficient Strategies for Long-Form Video QA** (2024, preprint)
   - Uses hierarchical keyframe selection and sequence-aware captioning.
   - Baseline role: coarse-to-fine keyframe reduction and sequence-aware evidence serialization.

10. **NeuS-QA: Grounding Long-Form Video Understanding in Temporal Logic and Neuro-Symbolic Reasoning** (2025, arXiv)
    - Converts questions into temporal logic and submits logic-verified segments to the VLM.
    - Main overlap: evidence adequacy and temporal structure.
    - Main distinction to test: empirical persistence/distractor mechanism versus explicit symbolic event constraints.

## 4. Provisional novelty boundary

The literature already covers:

- query-aware frame or token selection;
- relevance–diversity objectives;
- adaptive retention budgets;
- exploration–exploitation;
- hierarchical and iterative retrieval;
- temporal-order-aware or logic-aware evidence selection.

Therefore, **“query-aware adaptive frame selection” alone is not a defensible novelty claim**.

The strongest potentially independent contribution is the complete mechanism and evidence package:

```text
transient relevance can select specifically harmful distractors
+ persistence suppresses those distractors
+ persistence induces temporal concentration under hard Top-K
+ the net benefit reverses with frame budget
+ controlled exploration addresses the concentration cost
+ fixed-budget counterfactual replacement verifies that selected distractors,
  rather than frame count alone, cause downstream answer failures
```

This contribution remains credible only if each link is established separately and compared with generic diversity, uncertainty exploration, query-type routing, and adaptive-budget baselines.

## 5. Required comparison matrix

| Question | Required evidence | Closest prior-art families |
|---|---|---|
| Are transient high-relevance frames specifically harmful? | Same-budget Raw-only versus random replacement, answer-flip analysis, multiple relevance estimators | VidF4, AdaRD-Key |
| Does persistence improve evidence precision? | Raw versus persistent scores, retrieval metrics, selected-frame evidence quality | query-aware scoring methods |
| Does persistence create concentration? | span, temporal-bin coverage, event-hit rate, frames-per-hit-region | diversity and coverage selectors |
| Is the trade-off budget dependent? | frozen candidate pool; K sweep; interaction test or paired bootstrap | QTSplus, adaptive-budget methods |
| Is stochastic exploration the correct remedy? | quality–coverage frontier at matched cost and K | FOCUS and stochastic/differentiable samplers |
| Does the mechanism improve actual reasoning? | at least two Video-LLMs, multiple datasets/question types, fixed prompts and answer parsing | DIG, MIST, Koala, LVNet |
| Is the effect scorer-specific? | MobileCLIP2 plus independent matching/relevance models | all retrieval-style selectors |

## 6. GitHub synchronization contract

Use the Notion experiment identifier everywhere.

Example for experiment `M5.27`:

```text
configs/M5.27.yaml
scripts/run_M5_27.sh
results/M5.27/per_item.jsonl
results/M5.27/aggregate.json
artifacts/M5.27/figure.pdf
docs/experiments/M5.27.md
```

Each experiment record should contain:

```yaml
experiment_id: M5.27
notion_source: <direct page URL>
hypothesis: <one falsifiable statement>
primary_endpoint: <single preregistered endpoint>
fixed_factors:
  frame_budget: 8
  replacement_rates: [0.25, 0.50, 0.75]
  prompt_version: <hash or filename>
  answer_parser_version: <hash or filename>
models:
  scorer: <model and revision>
  downstream_vlm: <model and revision>
random_seeds: []
result_files: []
verdict: pending | supported | partial | rejected
claim_scope_after_result: <exact allowed claim>
```

## 7. Current integration status

- [x] GitHub repository access verified.
- [x] Initial GitHub README committed.
- [x] SciSpace semantic literature search executed.
- [x] Relevant Notion research notes and experiment-log pages discovered.
- [x] Literature-to-claim comparison recorded in GitHub.
- [ ] Exact supplied Notion project page shared with the active Notion integration.
- [ ] This synchronization map copied or linked from the exact Notion project page.
- [ ] Code, configurations, and result artifacts committed under stable experiment IDs.
