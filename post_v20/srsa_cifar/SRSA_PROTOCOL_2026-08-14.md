# SRSA-CIFAR: Structured-Response Sybil Admission Audit

Date: 2026-08-14 (Asia/Seoul)
Status: frozen design document; no CIFAR input, label, endpoint prediction, selector path, or outcome has been accessed by this project at the time of writing.

## 1. Scientific purpose

The previous Prospective Registry Admission Audit (PRAA) is closed literally as `STOP_PRAA_STAGE3_ERROR_HEAD_AND_THRESHOLD`. Emotion passed, but Language-ID failed the already-frozen cross-split trigger-rate and one-point utility-loss gates. Its primary and deployment inputs and labels remain unopened. Nothing in this document reinterprets, repairs, or reuses that stopped family.

SRSA-CIFAR is a new, separately registered study family. It targets the same score-changing review gap without learned abstention heads or transferred score thresholds:

> Can a completely new task and real checkpoint registry exhibit material root-level selection harm when one root submits independently admitted, semantically identical structured-response variants, while an authenticated root-aware admission policy is exactly invariant?

The design jointly targets prospective cross-task transfer and operational admission realism.

## 2. Development disclosure

Before this protocol was written, only previously opened NLP safety blocks were used for mechanism diagnostics. Deterministic raw-input-hash variant rates of 1% and 5% were examined. No primary or deployment input, label, prediction, selector path, or effect from PRAA was opened. The 5% rate used below is therefore development-informed, not confirmatory evidence.

To keep the new test prospective:

- CIFAR-100 is the single prespecified primary task.
- CIFAR-10 is a simultaneous replication and can never replace a failed primary.
- no additional task may be added after either CIFAR outcome is opened;
- no canary rate, alias count, roster, budget, temperature, seed, or gate may change after this document;
- all prior attempts (Steps 105, 113, 114, PRAA, and the NLP safety diagnostics) remain disclosed.

## 3. Fixed public task family

### 3.1 Datasets

The official torchvision `v0.21.0` CIFAR test archives are used.

- CIFAR-100 primary: `cifar-100-python.tar.gz`, official MD5 `eb9058c3a382ffc7106e4002c42a8d85`.
- CIFAR-10 replication: `cifar-10-python.tar.gz`, official MD5 `c58f30108f718f92721af3b95e74349a`.

Each official 10,000-example test set is ordered by

`SHA256("srsa-cifar-partition-v1" || raw_RGB_bytes)`

with the raw archive position used only as a deterministic collision tie-break. The first 2,000 rows are `safety`, the next 6,000 are `primary_outcome`, and the final 2,000 are `deployment`. The partition rule reads no label.

### 3.2 Registry roster

Model source repository: `chenyaofo/pytorch-cifar-models` at commit
`786c16252c0fc58ee9adac063f8337cc4a7a497a`.

The roster rule is metadata-only: select the smallest released checkpoint from each of four architecture families that is available under the same model name on both CIFAR-10 and CIFAR-100. Published model-zoo metrics are not used for root selection or experimental tuning.

The four root names for each task are:

1. `{task}_resnet20`
2. `{task}_vgg11_bn`
3. `{task}_mobilenetv2_x0_5`
4. `{task}_shufflenetv2_x0_5`

Every checkpoint is executed from raw RGB input under its pinned source, weight, normalization, and runtime manifest. No fine-tuning, calibration, or learned adapter is permitted.

Normalization is pinned to the public training configuration at `chenyaofo/image-classification-codebase` commit `610e6523ab462351700aaf629e6c739a9a30e903`:

- CIFAR-10 mean `[0.4914, 0.4822, 0.4465]`, std `[0.2023, 0.1994, 0.2010]`.
- CIFAR-100 mean `[0.5070, 0.4865, 0.4409]`, std `[0.2673, 0.2564, 0.2761]`.

## 4. Structured response and the single similarity

A clean endpoint returns

`Response(label = y_hat, variant = "base")`.

For an attacked root, four independently admitted wrapper endpoints execute the exact same checkpoint. Wrapper `j` returns the same semantic label `y_hat` on every input. Its optional variant field is determined solely from raw input bytes:

`canary_j(x) = 1[uint64(SHA256("srsa-canary-v1" || j || raw_RGB_bytes)) mod 20 = 0]`.

It returns `variant = "v{j}"` on its canary inputs and `variant = "base"` otherwise. The wrapper has no label, item identifier, peer output, selector state, or model-quality access.

A human annotation is represented as

`Reference(label = y, variant = "*")`.

The same typed compatibility function is used for candidate grouping and acquired-label feedback:

`s((y,t),(y',t')) = 1[y = y'] * 1[t = t' or t = "*" or t' = "*"]`.

Candidate endpoints never emit `"*"`; therefore candidate-candidate compatibility is an equivalence relation. A reference wildcard makes feedback depend only on the semantic label. Consequently, for every item and every wrapper:

- semantic prediction equals the parent exactly;
- task accuracy and loss equal the parent exactly;
- acquired-label feedback equals the parent exactly;
- any clean-query fixed counterfactual is exactly unchanged.

There is no learned error head, transferred threshold, abstention code, or quality gate to estimate.

## 5. Admission conditions

For each of the four fixed roots `r`, the study evaluates a separately frozen attack condition.

- `E-clean`: entry-indexed admission with the four clean endpoints.
- `E-structured-r`: the same registry plus four structured-response wrappers around root `r`, each receiving independent entry mass.
- `E-exact-r`: secondary anchor with four exact wrappers around `r` that always emit `variant="base"`.
- `R-clean`: authenticated root-aware canonical admission with fixed mass per root.
- `R-structured-r`: all wrappers are admitted for audit but mapped to the authenticated parent root and receive no independent root mass.

The operational manifest binds endpoint ID, checkpoint SHA-256, code SHA-256, container/runtime versions, declared root ID, and wrapper schema. The vulnerable policy trusts entry identities. The remedy policy verifies root linkage and factors through one canonical root response.

## 6. Frozen selector and runs

The selector is the equation-faithful Select-LLM exact-group specialization, generalized only to the structured compatibility function above. The same `s` is called by the candidate grouping and reference feedback paths.

Common settings for both tasks and every attacked root:

- pool size: 500;
- annotation budget: 2;
- temperature: 0.05;
- paired runs: 3,000;
- structured wrappers per attacked root: 4;
- per-wrapper canary probability: 1/20;
- CIFAR-100 seed range: 1,410,000 through 1,412,999;
- CIFAR-10 seed range: 1,413,000 through 1,415,999;
- bootstrap replicates: 10,000;
- one-sided paired sign flips: 100,000.

Clean and attacked runs share the exact pool, semantic predictions, references, budget, temperature, and tie rules. Query ties use the smallest frozen raw-image digest. Root ties use the smallest frozen root-manifest digest. Registry input order is not a decision variable.

## 7. Outcome and estimands

Root loss is `1 - accuracy` on the complete 6,000-row primary block. The oracle best-root set contains every clean root attaining minimum complete-block loss. Every member of that set has zero regret.

For each attacked root, paired runs report:

- ordered acquisition-path change;
- final-root change;
- attacked-root final selection-probability change;
- terminal-regret difference `E-structured-r - E-clean`;
- cumulative-regret difference through budget two;
- fixed-query terminal, cumulative, and root-history differences;
- active-minus-fixed terminal difference;
- exact-wrapper anchor effects;
- root-aware clean/structured differences.

The security estimand is adversarial and frozen before outcomes:

`M_terminal = max_r mean(Delta_terminal,r)`

over the four fixed root attacks. This is not post-hoc attacker selection: every root attack is executed and reported, and familywise inference is applied across all four.

## 8. Primary decision and multiplicity

CIFAR-100 receives `GO_SRSA_PRIMARY` only if at least one fixed attacked root passes all of the following under four-root familywise control:

1. every dataset, source, checkpoint, runtime, endpoint, response, seed, and prediction binding passes;
2. all four wrappers are semantic-prediction-identical to their parent on every safety, primary, and deployment row;
3. each wrapper's realized canary fraction on the primary block is between 4% and 6%;
4. ordered acquisition paths change in at least 50% of paired runs;
5. mean terminal harm is at least +0.5 percentage points;
6. mean active-minus-fixed terminal harm is at least +0.5 percentage points;
7. the one-sided Bonferroni-simultaneous 95% lower confidence bound for terminal harm exceeds zero;
8. the one-sided paired sign-flip p-value passes Holm correction across the four attacked roots;
9. cumulative-regret mean and its Bonferroni-simultaneous lower bound are positive;
10. fixed-query root histories are bitwise identical, with exactly zero terminal and cumulative effects;
11. `R-clean` and `R-structured-r` query, posterior-root, terminal, and cumulative histories are bitwise identical.

Because fixed-query effects are structurally zero, active-minus-fixed and active terminal contrasts must be numerically identical; both are still materialized and checked.

CIFAR-10 is always executed and fully reported under the same four-root family. It is replication evidence only and cannot rescue a CIFAR-100 `NO_GO`.

## 9. Deployment endpoint

The 2,000-row deployment block is not used for safety, query acquisition, inference gates, or root selection. After primary labels are authorized, map every run's final selected root to fresh deployment accuracy and negative log-likelihood from that root's pinned endpoint. Report paired structured-minus-clean deployment loss, simultaneous intervals, and attacked-root selection gain. Deployment is prespecified secondary evidence and cannot convert primary `NO_GO` to `GO`.

## 10. Execution order and information boundary

1. `Stage 0`: download only public source, model weights, and dataset archives; bind SHA-256, sizes, source commits, model parameter counts, and normalization. No dataset row is decoded.
2. `Stage 1`: decode raw test images once, create hash partitions, and place labels in a separate sealed artifact. Public artifacts contain only raw inputs, digests, partition markers, and manifests.
3. `Stage 2`: independently execute all eight clean endpoints on public inputs; no labels are available.
4. `Stage 3`: construct deterministic wrappers from raw inputs, certify semantic equality, bind all primary/deployment predictions, and write a pre-outcome lock. No label or selector is available.
5. `Stage 4`: one outcome job opens the sealed labels once, executes all fixed attack and remedy conditions, performs inference, and writes the literal decision.

No primary or deployment label may be opened before the pre-outcome lock. A technical defect may be repaired only under unchanged scientific settings and with a narrow defect ledger.

## 11. Literal closure

- `GO`: CIFAR-100 passes every primary gate; promote the result and report CIFAR-10 and deployment evidence.
- `NO_GO`: the bound experiment executes validly but the CIFAR-100 gate fails; retain every outcome and add no task.
- `STOP`: a scientific pre-outcome gate fails; primary and deployment labels remain sealed.
- `INVALID`: a post-opening binding or implementation defect invalidates the run; only an exact technical rerun is allowed.

A scientific `NO_GO` or `STOP` is written as a successful workflow completion with an explicit decision artifact. GitHub Actions failure status is reserved for technical `INVALID`, preventing expected scientific outcomes from generating misleading CI-failure email.
