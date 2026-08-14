# Manuscript review closure — 2026-08-14

## Objective

Strengthen the current ICLR 2027 manuscript against the repeated reviewer concerns about dense presentation, unclear evidence tiers, closest-prior framing, and conflation of causal identification with operational attack realism. This closure changes presentation and claim organization only; it does not change any locked experimental decision.

## Scientific status retained

- Step 104 IMDB remains `GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY`.
- Step 105 Yelp remains `NO_GO_RETAIN_STEP105_PROSPECTIVE_NEGATIVE`; its positive directional effect is not promoted to a material terminal success.
- Later local development attempts through Step 112 remain STOP/NO_GO or post-stop diagnostics as originally adjudicated. They are not promoted into the submission evidence and no sealed outcome is opened contrary to its protocol.
- The manuscript claims targeted evidence-frame dependence, not universal instability, live-registry admission, hidden-label compromise, or production prevalence.

## Manuscript changes

1. Rewrote the abstract around the missing scientific object, causal result, held-out learned bridge, and exact non-claims.
2. Reorganized the introduction around three questions: frame necessity, causal mediation, and outcome-free executability.
3. Replaced protocol chronology with an evidence-ladder table distinguishing the natural anchor, T2 causal test, T1 transfer, learned endpoints, and cross-selector audit.
4. Reorganized results into occurrence, causal identification, outcome-free transfer, and bounded generality; added a result-ladder table retaining all material negative outcomes.
5. Replaced a single dense limitations paragraph with explicit “established / not established / implication” paragraphs.
6. Preserved the direct strategic-replication bandit comparison and the exact shared-label / latent-root / terminal-root distinction.
7. Removed internal Step and GO/NO_GO language from the scientific narrative except where needed for reproducibility and retained-status documentation.

## Validation

- LaTeX build: 28 pages; scientific main text ends on page 9; references begin on page 10.
- Evidence-to-source validator: `PASS_PAPER_TABLE_SOURCE_CONSISTENCY`.
- Anonymous artifact build: `PASS_STEP105_V20_BUILD`.
- Portable integrity validation: `PASS_STEP105_ANONYMOUS_RELEASE`.
- Portable full validation: paper rebuild plus Step 104 and Step 105 reconstructions all pass.
- ZIP stream audit: `PASS_STEP105_V20_ZIP_STREAM_AUDIT`, 851 manifest records, zero failures.

## Release bindings

- Paper PDF SHA-256: `5774968956ba2144414756a8c0cdba6d4ad7ac494b1cc9088d7a99c0fb5cff4b`
- Release manifest SHA-256: `b405b9d72a7795415e143a273e292df1dd07fc48ebb35b1dd434cdb790bdc0fc`
- V20 ZIP SHA-256: `14ce5abfd32091e4219c89308670ce64c1d8effcb59606db54c0729382ba8388`

## Remaining irreducible boundary

The current evidence does not contain a material terminal pass from a completely untouched task/registry under a one-shot source-faithful pipeline, nor a live admission study. This is an empirical boundary, not a writing defect. The revised manuscript makes it easy to see without allowing it to erase the distinct causal and evaluation-integrity contribution that is already established.
