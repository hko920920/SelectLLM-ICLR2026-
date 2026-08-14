# SelectLLM — Research Integration Hub

This repository is the implementation and reproducibility hub for the SelectLLM video-frame acquisition project.

## Connected research workspace

- **Notion project:** https://app.notion.com/p/Select-LLM-ICLR-2026-3bce7b0a7d38808286e5c9f94225568f
- **GitHub repository:** https://github.com/hko920920/SelectLLM-ICLR2026-
- **Literature discovery:** SciSpace semantic searches on query-conditioned frame selection, temporal evidence acquisition, distractor frames, relevance–coverage trade-offs, and budget-adaptive sampling

> Automation note: the linked Notion page must be explicitly shared with the ChatGPT/Notion integration before it can be fetched or updated directly. Related research-note database pages are already discoverable through the integration.

## Current research statement

The working hypothesis is not merely that query-aware frame selection is better than uniform sampling. The project studies the following causal mechanism:

1. Frame-wise vision–language relevance can contain transient fluctuations and harmful distractor frames.
2. Temporally persistent relevance can improve evidence precision.
3. Hard Top-K selection can over-concentrate the frame budget in a small temporal subregion.
4. The balance between instantaneous and persistent evidence may change with the available frame budget.
5. Controlled exploration should mitigate concentration without discarding query relevance.

The intended downstream validation includes actual Video-LLM question answering, fixed-budget comparisons, independent-model replication, and counterfactual replacement of selected frames.

## Closest literature identified through SciSpace

The following papers are immediate comparison targets for novelty and experimental design:

- **Do Video Language Models Really Know Where to Look? Diagnosing Attention Failures in Video Language Models** — questions whether common vision encoders identify the frames a Video-LLM actually needs.
- **FOCUS: Efficient Keyframe Selection for Long Video Understanding** — frames selection as exploration–exploitation under a strict token budget.
- **AdaRD-Key: Adaptive Relevance-Diversity Keyframe Sampling for Long-form Video Understanding** — combines query relevance and diversity with relevance-aware gating.
- **Adaptive Keyframe Sampling for Long Video Understanding (AKS)** — optimizes prompt relevance and temporal coverage under a fixed frame budget.
- **MDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs** — jointly models query relevance, diversity, and sequentiality.
- **HFS: Holistic Query-Aware Frame Selection for Efficient Video Reasoning** — set-level relevance, coverage, and redundancy optimization with Gumbel-Softmax.
- **FrameOracle: Learning What to See and How Much to See in Videos** — predicts both which frames and how many frames are needed.
- **Temporal Chain of Thought: Long-Video Understanding by Thinking in Frames** — iteratively curates relevant visual context and addresses irrelevant distractors.
- **From Frames to Clips: Efficient Key Clip Selection for Long-Form Video Understanding** — preserves temporal coherence by selecting clips rather than isolated frames.
- **Select Less, Reason More: Prioritizing Evidence Purity for Video Reasoning** — emphasizes high-purity visual evidence and localized temporal resampling.

## Differentiation to preserve

The strongest potential distinction is a mechanism-level account of **transient distractor relevance → persistence gain → concentration cost → frame-budget-dependent reversal**, supported by controlled interventions rather than only endpoint accuracy improvements.

This distinction must be tested explicitly against relevance–diversity, coverage, exploration–exploitation, adaptive-budget, and clip-selection baselines.

## Suggested repository convention

```text
configs/       frozen experiment configurations
src/           selector and scoring implementations
scripts/       dataset preparation, inference, and evaluation entry points
results/       machine-readable aggregate results
artifacts/     figures and compact reproducibility artifacts
docs/          claim-to-evidence map and experiment decisions
```

Each experiment should preserve the same identifier used in Notion (for example, `M5.27`) across configuration files, result directories, and documentation.

## Integration status

- [x] GitHub connection and write access verified
- [x] SciSpace literature search verified
- [x] Related Notion research pages discovered and read
- [ ] Exact linked Notion project page shared with the ChatGPT/Notion integration
- [ ] Implementation and experiment artifacts committed
