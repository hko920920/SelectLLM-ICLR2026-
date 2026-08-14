# Step 104 / V19 anonymous review artifact

This release extends the inherited V18 artifact with the complete Step 98--99
ANLI negative trail and the Step 100--104 IMDB development-informed held-out
primary. The literal primary decision is:

`GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY`

The release includes the disjoint development and held-out bindings, complete
81-cell development grid, robust selection, four independently seeded learned
adapter weights, label-free pre-outcome predictions, sealed 13,000-row outcome,
3,000-pair trajectory arrays, inference, gates, and two independent validation
paths. Earlier no-go decisions remain bound and are not replaced.

From the extracted artifact root, run:

```bash
python validate_step104_anonymous_release.py --integrity-only
python validate_step104_anonymous_release.py --full
```

Integrity mode checks every manifest byte, parent-release binding, anonymity,
adapter map, primary status, and PDF boundary. Full mode additionally rebuilds
the paper, validates every manuscript number, independently reconstructs the
complete Step 103 robust development selection, and independently reimplements
all Step 104 trajectories, inference, and gates.

If the four pinned public Hugging Face roots are available in the local cache
or can be downloaded, the strongest optional check is:

```bash
python validate_step104_anonymous_release.py --raw-input
```

That mode additionally re-infers all four roots and all four learned aliases on
13,000 raw reviews before reconstructing the primary. It is slower and may use
a GPU. Public base-model weights and third-party repositories are deliberately
not bundled; the four independently learned adapter files are bundled.

Interpretation boundary: Step 104 is a development-informed, disjoint held-out,
no-heldout-reference, raw-input, same-similarity terminal confirmation. It is
not a before-all-data preregistration, an independently trained full base model,
a live registry admission, or evidence of universal or production compromise.
