# Step 97 / V18 anonymous review artifact

This release extends the inherited V17 artifact with the complete Step 96 SNLI
learned single-similarity audit, the post-confirmatory complete-top-eight
sensitivity, and the prospectively stopped Step 97 SciTail replication.  It
preserves the literal Step 96 primary `NO_GO_RETAIN_STEP96_NEGATIVE` decision:
one alias lost 100 rather than at most 98 correct test examples.  It separately
labels the conservative top-eight row as secondary, post-confirmatory evidence.

The bundled learned adapter weights are the four independently seeded Step 96
error adapters.  Public base-model weights and third-party repositories are not
bundled.  The raw predictions, sealed outcome, acquisition trajectories,
effect arrays, protocol locks, hashes, and independent validators needed to
reconstruct the reported decisions are included.

Run from the extracted artifact root:

```bash
python validate_step97_anonymous_release.py --integrity-only
python validate_step97_anonymous_release.py --full
```

The full mode checks every manifest record, anonymity, PDF page boundaries,
manuscript-to-source numerical bindings, reconstructs the Step 96 primary
decision in an isolated temporary directory, regenerates the complete
top-eight sensitivity, reconstructs the Step 97 geometry stop, and builds the
paper.  Python requires `numpy`, `pypdf`, and the dependencies already listed
in the inherited artifact; the PDF build additionally requires `latexmk`.

Interpretation boundary: Step 96 is evidence for a learned raw-input,
same-similarity terminal mechanism.  It is not a live-registry admission, a
full independently trained base checkpoint, or a universal attack result.
Step 97 opened no SciTail test outcomes because its preregistered roster gate
failed before adapter training.
