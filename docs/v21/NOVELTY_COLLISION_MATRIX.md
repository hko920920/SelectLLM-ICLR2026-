# V21 closest-work novelty collision matrix

Date: 2026-08-16 (Asia/Seoul)  
Question: does a prior work subsume the paper's central object, or only one component?

## The central object being defended

The paper is not claiming the first clone effect or the first strategic-multiplicity attack. Its candidate object is the conjunction:

1. submitted entries map to latent or externally authenticated accountable roots;
2. one newly acquired label is reusable evidence for every candidate entry;
3. entry-level mass changes the future adaptive acquisition transcript;
4. the irreversibly changed transcript changes final root-level model selection cost; and
5. a fixed-query counterfactual isolates acquisition mediation.

A prior work subsumes the paper only if it already treats essentially this complete object. Sharing “clones,” “multiplicity,” “active,” or “ranking” is not sufficient.

## Collision matrix

| Work | Unit being counted | Root/owner supplied? | Feedback structure | Adaptive acquisition? | Final object | Clone/refinement type | Remedy | Collision risk | Binding distinction |
|---|---|---:|---|---:|---|---|---|---:|---|
| [Multi-armed Bandit Algorithm against Strategic Replication](https://proceedings.mlr.press/v151/shin22a.html) | Arms registered by agents | Yes | Arm-local stochastic rewards | Yes | Agent reward/regret | Duplicate arms | Agent-first mechanism | **High conceptual, low subsumption** | Ownership is given; rewards are arm-local; there is no shared label updating every candidate and no latent root-level model-selection endpoint. |
| [Replication-proof Bandit Mechanism Design with Bayesian Agents](https://arxiv.org/abs/2312.16896) | Arms/models owned by strategic agents | Yes | Bandit rewards | Yes | Agent utility/regret | Strategic replication | Replication-proof mechanism | **High conceptual, low subsumption** | Same ownership and feedback distinction as above; our problem is an evaluation-unit ambiguity under reusable labels. |
| [Clone-Robust AI Alignment](https://openreview.net/forum?id=vbw9hSlOaG) | Response alternatives in preference data | No accountable-root layer | Pairwise preference observations used to fit a reward function | No future label-acquisition transcript | Learned reward function | Approximate response clones | Similarity-weighted MLE | **Highest static-method collision** | It studies static learning from an unbalanced alternative set. Our effect operates through which future labels are acquired and is evaluated at final root selection; post-hoc reweighting cannot recover unqueried labels. |
| [Clone-Robust Weights in Metric Spaces](https://doi.org/10.65109/MJWJ6521) / [Breaking the Illusion of Artificial Consensus](https://arxiv.org/abs/2602.24024) | Points/agents in a metric aggregation problem | Not necessarily | Static consensus/aggregation | No | Aggregate score/decision | Approximate metric clones | Clone-robust weights | **High defense collision** | Useful response-only baseline, but it neither authenticates lineage nor analyzes shared-label adaptive acquisition. Our experiments show the clean/attack trade-off of fixed metric weights. |
| [Strategic Candidacy in Generative AI Arenas](https://arxiv.org/abs/2603.26891) | Models submitted by known producers | Producer identity supplied | Pairwise arena votes | Votes accumulate, but the paper targets ranking under statistical noise rather than active label choice | Arena rank / producer utility | Producer-submitted clones | Producer self-ranking (YRWR) | **High threat-framing collision** | Their gain is a ranking/estimation lottery under known producer sets. Ours is a label-allocation distortion under latent roots, with irreversible missing evidence and fixed-query mediation. |
| [The Leaderboard Illusion](https://papers.nips.cc/paper_files/paper/2025/hash/70a93f260a51123b3c0e33ecd1b4de97-Abstract-Datasets_and_Benchmarks_Track.html) | Models and benchmark records | Model identities known | Fixed benchmark outcomes | No | Leaderboard reliability/rank | Model/benchmark multiplicity and evaluation choices | Reliability analysis | **Medium** | Static leaderboard inference, not candidate-indexed adaptive acquisition. |
| [Dropping Just a Handful of Preferences Can Change Top LLM Rankings](https://openreview.net/forum?id=jNiEMDsRgc) | Pairwise preference records | Not a root ambiguity problem | Preference observations | No label acquisition policy under candidate refinement | Top rank | Data removal/perturbation | Robustness diagnosis | **Medium significance comparator** | It offers a very direct real-data ranking instability result. Our novelty is deeper causal structure, but our practical story is less immediate; presentation must compensate. |
| [How Reliable is Language Model Micro-Benchmarking?](https://arxiv.org/abs/2510.08730) | Evaluation examples | Model identity known | Fixed benchmark labels | Subset selection/meta-evaluation, not strategic candidate refinement | Pairwise ranking preservation | Small evaluation subsets | Sample-size guidance | **Medium significance comparator** | It asks whether small selected subsets reproduce full rankings. We ask whether the candidate evidence frame itself changes the selected subset and root decision. |
| [UNREAL: Unique Rashomon Sets for Active Learning](https://arxiv.org/abs/2503.06770) | Predictive models in an ensemble/Rashomon set | Candidate identities available | Labels train a downstream predictor/committee | Yes | Learned predictor / active-learning performance | Redundant committee members | Unique/reduced model set | **Highest active-learning collision** | UNREAL removes redundant predictors to improve active learning. Our task is active *model selection*, one label scores every submitted candidate, same-root status can be latent, and the endpoint is selected-root regret with strategic refinement and fixed-query mediation. |
| [Active Evaluation of General Agents](https://arxiv.org/abs/2601.07651) | Agents/policies under active evaluation | Candidate identity supplied | Candidate-specific performance observations | Yes | Evaluation efficiency/ranking | Perturbed or similar agents | Baselines/quotients | **High adjacent-object collision** | It establishes sequential clone/perturbation sensitivity in active evaluation; our remaining novelty must be stated as latent-root shared-label refinement and the complete acquisition-to-root-cost causal chain. |
| [All Models Are Wrong, Some Are Useful: Model Selection with Limited Labels](https://proceedings.mlr.press/v258/okanovic25a.html) | Candidate models | Model entries are the unit | A label evaluates all models | Yes | Selected model | Not a strategic clone paper | Active model-selection algorithm | **Base-method, not novelty collision** | Supplies the active-selection setting; it does not audit whether the candidate list encodes the correct experimental unit. |
| [Consensus-Driven Active Model Selection](https://openaccess.thecvf.com/content/ICCV2025/html/Kay_Consensus-Driven_Active_Model_Selection_ICCV_2025_paper.html) | Candidate models | Entries are the unit | Shared labels | Yes | Selected model | Not a clone paper | Consensus acquisition | **Base-method and transfer comparator** | Our CODA audit tests whether candidate multiplicity affects another acquisition family; CODA itself does not define root-refinement invariance. |
| Large Language Model Selection with Limited Annotations / Select-LLM | Candidate LLM entries | Entries are the unit | One reference scores all candidates through one similarity | Yes | Selected LLM | Not originally a clone paper | Bayesian active selection | **Mechanism under audit** | Our exact-match specialization must remain source-faithful and should not be generalized to all Select-LLM similarities without evidence. |
| [Quotient Semivalues for False-Name-Resistant Data Attribution](https://arxiv.org/abs/2605.07663) | Data owners/contributions | Ownership/coalitions define quotient | Static attribution utility | No | Attribution value | False names | Ownership quotient | **Medium remedy collision** | Supports the principle that false-name resistance needs an authenticated quotient; it does not analyze adaptive model-selection labels. |
| Model-lineage attestation work | Model checkpoints/ancestry | Provides lineage evidence | No active selection by itself | No | Lineage/identity | Derived models | Attestation | **Low algorithmic, high operational relevance** | Lineage is missing input for our root map, not a substitute for a refinement-invariant selector. |

## Subsumption verdicts

### No single identified work subsumes the complete object

The closest works cover separate pieces:

- strategic replication: sequentiality and duplicate registration, but supplied ownership and arm-local rewards;
- clone-robust alignment/weights: response multiplicity and robust aggregation, but static learning;
- active evaluation/UNREAL: sequential acquisition and redundancy, but not latent same-root shared-label model selection with root-level cost;
- arena/leaderboard work: strategic candidacy and real ranking consequences, but not adaptive label allocation;
- active model-selection papers: shared labels and sequential acquisition, but no audited experimental unit.

The paper's novelty is therefore a **meaningful sequential extension and synthesis**, not an isolated first principle. This matches a realistic human-review characterization: novel object and causal identification, but not a new field from scratch.

## Reviewer failure modes the manuscript must preempt

1. **“This is just clone-robust alignment applied to evaluation.”**  
   Answer before the contribution list: static reward fitting versus irreversible future-label acquisition; final reward model versus root-selection cost; similarity weighting versus authenticated root evidence.

2. **“This is strategic replication bandits with LLMs.”**  
   Answer: ownership is not supplied, one label updates all candidates, and the experimental claim is about selecting an accountable model root.

3. **“UNREAL already removes redundant models in active learning.”**  
   Answer: active learning trains a target predictor; active model selection chooses among submitted predictors. Same-root status is the missing claim unit, not merely predictive redundancy.

4. **“The result is another leaderboard-instability paper.”**  
   Answer: leaderboard papers perturb fixed evidence; here the roster changes which evidence exists at the end of the experiment. Fixed-query equality identifies that difference.

5. **“The effect is cherry-picked.”**  
   Answer: show the complete evidence ladder, disclose negative families compactly in main, and state that material terminal transfer is selective.

## V21 novelty sentence

> Prior work studies duplicate arms with known owners, approximate clones in static preference learning, redundant committees in active learning, or clones in fixed rankings. We study the remaining shared-label sequential object: submitted entries may hide a smaller set of accountable roots, entry mass changes which reusable labels are acquired, and the resulting irreversible transcript changes final root selection; a fixed-query counterfactual isolates this mediation.

## Remaining novelty risk

A skeptical reviewer can reasonably classify the contribution as a strong extension rather than a foundationally new paradigm. The paper should not fight that classification. It should argue that the combination creates a previously unmeasured failure mode and a precise invariance obligation, then win on causal rigor and clarity.
