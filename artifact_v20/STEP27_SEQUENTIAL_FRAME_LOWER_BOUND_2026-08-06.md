# STEP 27 — Sequential Sampling-Frame Lower Bound

Date: 2026-08-06  
Target paper: **Candidate Lists Are Priors: Registry-Refinement Attacks on Active Model Selection**

## 0. Decision first

**FORMAL GATE: PASS.**

Step 19 had only a one-query, two-frame minimax gap of `0.040852` bits. That result was correct but too easy to carry the paper. The formal core can now be strengthened in two complementary ways:

1. a **tight sequential frame-unidentifiability lower bound** that grows linearly with annotation budget `B`;
2. a **constant-size exact-refinement construction for Select-LLM** in which four exact clones change all `B` acquired queries, cause `Omega(B)` cumulative selection regret, and—for odd `B`—force a nearly `1/2` final performance gap.

The first theorem establishes that response-only inference cannot recover the right evidence frame. The second establishes that the latest concrete per-candidate acquisition rule can amplify the missing frame into a full active-path failure.

This does not turn the work into a pure theory paper. The proof techniques are elementary direct-product, Bayes-risk, and mass-counting arguments. They are, however, now strong enough to support the real-response paper rather than merely motivate it.

## 1. Sequential product-frame construction

### 1.1 One public registry, exponentially many valid latent frames

Fix an annotation budget `B`. Let a canonical candidate behavior be a bit vector

\[
z=(z_1,\ldots,z_B)\in\{A,B\}^B.
\]

The public registry contains `2^B` exact submitted copies of every behavior vector `z`. Hence it contains

\[
2^B\cdot 2^B=4^B
\]

candidate IDs. Their IDs and all query-conditioned likelihoods are public and identical under every latent frame.

A frame is indexed by

\[
\omega=(\omega_1,\ldots,\omega_B)\in\{A,B\}^B.
\]

For behavior `z`, define

\[
m_\omega(z)=|\{t:z_t=\omega_t\}|.
\]

Frame `Phi_omega` partitions the `2^B` public copies of `z` into

\[
r_\omega(z)=2^{m_\omega(z)}
\]

accountable roots. Each root therefore contains exactly

\[
2^{B-m_\omega(z)}
\]

aliases. These are valid integer partitions of the same public leaves.

The total number of roots in every frame is

\[
\sum_z 2^{m_\omega(z)}=3^B.
\]

If the best root is uniform over accountable roots, the induced prior over canonical behavior vectors is

\[
P_\omega(F=z)
=\frac{2^{m_\omega(z)}}{3^B}
=\prod_{t=1}^B
\begin{cases}
2/3,&z_t=\omega_t,\\
1/3,&z_t\neq\omega_t.
\end{cases}
\]

Thus the coordinates are independent, and `P(F_t=omega_t)=2/3`. The public candidate registry is unchanged; only the latent alias partition changes.

### 1.2 Two queries per coordinate

For every coordinate `t`, the fixed query pool contains `q_{t,+}` and `q_{t,-}`. With `c=1/4`, their binary oracle channels are

\[
q_{t,+}:\quad
P(Y=1\mid F_t=A)=1,\qquad
P(Y=1\mid F_t=B)=c,
\]

\[
q_{t,-}:\quad
P(Y=1\mid F_t=A)=1-c,\qquad
P(Y=1\mid F_t=B)=0.
\]

Outcomes are conditionally independent across coordinates given `F`.

For prior `P(F_t=A)=2/3`:

\[
I(F_t;Y_{t,+})=0.540852,
\qquad
I(F_t;Y_{t,-})=0.459148.
\]

The values reverse for prior `1/3`. Therefore the frame-aware query is `q_{t,+}` when `omega_t=A` and `q_{t,-}` when `omega_t=B`.

Let

\[
I_{hi}=0.540852,\qquad
I_{lo}=0.459148,\qquad
g=I_{hi}-I_{lo}=0.081704.
\]

## 2. Main theorem: tight sequential frame regret

### Theorem 1 — Sequential sampling-frame lower bound

For every integer budget `B >= 1`, the construction above has the following properties.

1. A frame-aware active selector obtains total mutual information

\[
V^*_{\omega,B}=B I_{hi}.
\]

2. For every possibly randomized prediction-only selector `pi`, there exists a fixed latent frame `omega` such that

\[
V^*_{\omega,B}-V^{\pi}_{\omega,B}
\geq \frac{Bg}{2}
=0.040852B\text{ bits}.
\]

3. The bound is tight. A frame-blind selector that queries every coordinate once and independently randomizes between `q_+` and `q_-` has regret exactly `Bg/2` under every frame.

4. Under coordinate-wise Hamming loss for the selected canonical behavior, the tight minimax excess Bayes risk is

\[
\frac{B}{24}.
\]

The frame-aware expected Hamming error is `B/12`; the optimal frame-blind minimax error is `B/8`.

### Proof

Place a uniform prior over the `2^B` latent frames. Before coordinate `t` is queried, `omega_t` is independent of all observations made on other coordinates. A prediction-only selector therefore chooses the correct polarity for a fresh coordinate with probability at most `1/2` on average. Its expected first-query information is

\[
\frac{I_{hi}+I_{lo}}{2}=0.5\text{ bits}.
\]

Repeating an already touched coordinate cannot improve this value. The prior entropy of one coordinate is

\[
h(2/3)=0.918296.
\]

After even the lower-information first query, all information that could remain in that coordinate is at most

\[
h(2/3)-I_{lo}=0.459148<0.5.
\]

Thus one query on a fresh coordinate dominates spending a query on any previously touched coordinate, even if the latter query could reveal all remaining uncertainty. The frame-aware policy similarly prefers a fresh coordinate because

\[
h(2/3)-I_{hi}=0.377444<I_{hi}.
\]

Both optimal policies consequently query every coordinate exactly once. The frame-aware value is `B I_hi`, whereas every frame-blind policy has average value at most `B/2` bits. Hence its average regret over frames is at least

\[
B(I_{hi}-1/2)=Bg/2.
\]

At least one fixed frame attains that regret. Fair polarity randomization attains equality for every frame, establishing tightness.

For Hamming loss, the Bayes error after the correct-polarity query is `1/12`, and after the wrong-polarity query it is `1/6`. A fair frame-blind query therefore has error `1/8`. An untouched coordinate has prior error `1/3`, so a fresh query reduces error by `5/24`; all error remaining after any first query is at most `1/6 < 5/24`. The same fresh-coordinate exchange argument applies, giving excess error

\[
B(1/8-1/12)=B/24.
\]

### Why this is genuinely sequential

This is not the Step 19 one-query result multiplied by assertion.

- The candidate registry and query pool are fixed before acquisition.
- A policy may select coordinates and query polarities adaptively.
- Feedback can reveal the frame bit of the coordinate already queried.
- It cannot reveal the independently hidden frame bits of untouched coordinates.
- The entropy and Bayes-error exchange arguments prove that revisiting a coordinate cannot beat moving to a fresh one.

The lower bound therefore applies to the complete adaptive `B`-query policy.

## 3. Mechanism-specific amplification for Select-LLM

The product construction proves a necessity result but uses exponentially many candidates. A second theorem shows linear active amplification with only five canonical behaviors and nine submitted IDs.

### 3.1 Fixed-pool instance

Let `B` be odd. There are five base candidates `A,B,C,D,E` and `2B` binary exact-match queries.

For every one of `B` **good** queries:

| Candidate | A | B | C | D | E | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| response | 0 | 0 | 1 | 1 | 1 | 0 |

For the `t`-th of `B` **bad** queries:

| Candidate | A | B | C | D | E | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| response | 0 | 0 | 1 | 0 | 0 | `1` for odd `t`, `0` for even `t` |

The registry refinement adds four exact copies of `C`. It adds no response function and changes no oracle score.

Equation 5 with exact-match similarity minimizes

\[
S(x)=x^2+(1-x)^2,
\]

where `x` is posterior mass predicting response 1. Equation 7 is the exact-match exponential posterior update with arbitrary finite `tau>0`.

### Theorem 2 — Full-path exact-refinement amplification

For every odd `B >= 3` and every `tau>0`:

1. the clean five-candidate registry selects all `B` good queries;
2. the nine-ID refined registry selects all `B` bad queries;
3. the two acquired query sets are disjoint;
4. `A` and `B` are the unchanged true-best candidates;
5. clean cumulative selection regret is zero;
6. refined cumulative selection regret is at least

\[
\frac{B^2-1}{4B}=\Omega(B);
\]

7. the refined final selected behavior is `C`, with final regret

\[
\Delta_B=\frac{B-1}{2B}\longrightarrow\frac12.
\]

### Proof

After the clean selector has annotated `s` good queries, candidates `A,B` have likelihood weight `a=e^{s/tau}`, while `C,D,E` have weight one. Let `d=2a+3`.

An unqueried good query assigns response-1 mass `3/d`; an unqueried bad query assigns mass `1/d`. Direct calculation gives

\[
S(1/d)-S(3/d)=\frac{4(d-4)}{d^2}>0.
\]

Hence every remaining good query has a strictly lower acquisition score, so the clean selector exhausts the good queries. `A` and `B` are correct on all of them and remain selected.

After four exact copies are added, behavior `C` owns five of nine submitted IDs. Along the bad-query path its aggregate posterior mass is

\[
x=5/9
\]

after every even number of bad labels, and

\[
x=\frac{5e^{1/\tau}}{5e^{1/\tau}+4}
\]

after every odd number. In either case `1/2<x<1`.

A bad query has response-1 mass `x`. A good query groups `C` with `D,E`; since all four non-`C` candidates have equal posterior weight on this path, its response-1 mass is `(1+x)/2`. The bad query is strictly closer to one half because

\[
x-1/2<x/2
\quad\text{for all }x<1.
\]

Therefore the refined selector exhausts the bad queries.

Over the complete `2B`-query pool, the true score gap between `A/B` and `C` is

\[
\Delta_B=\frac{B-1}{2B}.
\]

After every odd acquired bad query, `C` has one more observed correct answer than every non-`C` candidate, so the selector must choose `C`. There are `(B+1)/2` such steps. Even if tie-breaking is maximally favorable on all even steps, cumulative regret is at least

\[
\frac{B+1}{2}\Delta_B
=\frac{B^2-1}{4B}.
\]

Since the final step is odd, final regret is `Delta_B`.

### Interpretation

The refinement is semantically null at the function level: four exact copies add no prediction and the true ranking is unchanged. Under a uniform per-entry prior, however, they implement a prior-odds intervention that redirects every label acquisition. Active feedback then converts registry bookkeeping into linear cumulative harm.

This is an existence theorem, not a claim that every exact clone harms every dataset. The real-response experiments correctly show mixed signs on natural tasks.

## 4. Numerical and combinatorial verification

The standard-library checker is:

`verify_step27_sequential_frame_lower_bound.py`

It verifies:

- the information values `0.540852` and `0.459148`;
- the one-coordinate Bayes errors `1/12` and `1/6`;
- every integer root partition for `B=1,...,6`;
- the induced product prior and all coordinate marginals;
- all deterministic polarity vectors against all frames for `B=1,...,8`;
- the exact Select-LLM construction for budgets `3,5,11,51` and temperatures `0.1,0.5,1,3,10`.

All assertions pass. Representative scaling values are:

| Budget `B` | Tight frame-blind information regret | Tight excess Hamming mistakes |
|---:|---:|---:|
| 1 | 0.040852 bits | 0.041667 |
| 10 | 0.408521 bits | 0.416667 |
| 20 | 0.817042 bits | 0.833333 |
| 50 | 2.042604 bits | 2.083333 |

For the constant-size Select-LLM construction at `B=51`, all tested temperatures produce:

- clean cumulative regret: `0`;
- refined cumulative regret: `12.745098`;
- refined final regret: `0.490196`;
- 51 of 51 query positions redirected to the disjoint bad set.

## 5. Reviewer-facing novelty boundary

### What can now be claimed

- The latent candidate sampling frame can carry `B` independently consequential bits that are absent from the response registry.
- Missing frame information causes tight annotation-budget-linear information and structured-selection regret.
- For the newest acquisition/update equations, four exact clones can cause full-path divergence and `Omega(B)` cumulative selection regret with a constant number of base behaviors.
- Fixed root mass prevents exact refinement only when the roots are known; the first theorem shows why predictions cannot identify those roots in general.

### What still cannot be claimed

- Direct-product/Yao reasoning is itself new.
- Every duplicate is harmful.
- The exponential product instance is realistic in scale.
- Exact-clone vulnerability alone defeats exact deduplication.
- The theorem authenticates provider ownership or model independence.
- This is sufficient as a standalone theory paper without the real-response experiments and defense frontier.

The formal novelty is the active-model-selection object and its composition:

> one public response registry, incompatible valid evidence frames, a tight sequential acquisition lower bound, and an explicit current-method path-amplification theorem.

## 6. Effect on the paper decision

The principal formal risk identified in Step 26 is resolved:

| Formal concern | Before Step 27 | After Step 27 |
|---|---|---|
| Same-view lemma is nearly tautological | High risk | Retained only as setup |
| Bound covers one query only | High risk | Tight `Omega(B)` sequential bound |
| No decision loss | High risk | `B/24` excess Hamming risk |
| No direct link to current algorithm | High risk | Constant-size Eq. 5/7 amplification theorem |
| Theorem technique alone carries novelty | Not credible | Explicitly not claimed |

**Result: the topic remains GO and the formal component advances from minimum-pass to paper-usable PASS.**

The next step is not more theorem invention. It is to freeze the three threat models—accidental exact refinement, response-only strategic refinement, and benchmark-aware adaptive refinement—and ensure that every experiment and claim is assigned to exactly one of them.

## 7. Primary method source

The concrete acquisition and update rules audited in Theorem 2 are Equations 5 and 7 of:

- Durmazkeser et al., *Large Language Model Selection with Limited Annotations*, arXiv:2605.24981v1: https://arxiv.org/html/2605.24981v1

The paper defines best-model identification under limited annotations, uses a uniform prior over candidate models, performs sequential information-based acquisition, and updates candidate posterior mass after every annotated response.
