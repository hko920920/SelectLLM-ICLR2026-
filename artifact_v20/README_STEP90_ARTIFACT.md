# Anonymous Step 90 review artifact

This release binds the current 21-page anonymous paper to the inherited causal,
prospective, defense, and cross-selector evidence, plus the sealed executable
response-adapter audit reported in Appendix G.

The promoted Step 90 result is deliberately narrow.  Four confidence-gated
response adapters share a hash-pinned public classification parent, accept raw
text, and preserve its hard answer.  They are independently executable variants,
not independently trained full checkpoints and not evidence of live registry
admission.  MNLI and QQP pass the preregistered terminal and active-minus-fixed
gates; the earlier QNLI nonconfirmation is retained with its complete record.

## Validation

From the extracted artifact root:

```powershell
python validate_step90_anonymous_release.py --integrity-only
```

checks every bundled byte, manifest binding, PDF binding, and anonymity rule
without network access.  With the Python/LaTeX dependencies installed:

```powershell
python validate_step90_anonymous_release.py --full
```

verifies the inherited full-validated V11 manifest and its Step 80/88 receipt
hashes, reconstructs 4,000 Step 90 active and 2,000 fixed-query trajectories in 12,077 checks, and
rebuilds the paper while enforcing that the main scientific text ends on page 9.
Historical validators remain bundled, but this incremental V12 command does not
redownload and rerun every V11 dependency.

The Step 90 independent validator imports no Step 89/90 experimental runner.
Its raw archive contains the frozen official score views, parent labels,
confidence views, pools, paths, selected roots, and regret arrays needed for
offline reconstruction.  Public parent weights and third-party repositories are
not bundled.

## Raw-input endpoint

After installing `torch`, `transformers`, `huggingface_hub`, and
`safetensors`, one adapter can be executed as follows:

```powershell
python step90_executable_adapter_endpoint.py --task mnli --adapter 0 `
  --text-a "A person is outdoors." --text-b "Someone is outside."
```

The endpoint fetches the exact public parent revision, verifies its weight hash,
loads the hash-bound local adapter, and emits a parsed hard label plus an
adapter-specific style only when the frozen confidence threshold is met.  Its
runtime interface has no item identifier, reference label, peer row, pool, or
selector state.
