# Anonymous validation-closure artifact (V10 / Step 87)

This archive supports **Candidate Lists Are Priors: Registry-Refinement Attacks
on Active Model Selection**.

V10 adds the outcome-blind, reference-free terminal audit on eight previously
untouched HELM-Lite scenarios. It contains the preregistration and three execution
locks, the frozen attacks, all 8,000 paired raw outcome arrays, the complete result,
the independent reconstruction, the revised manuscript, and source-consistency
checks. NarrativeQA and WMT14 fr--en/ru--en pass the preregistered terminal gate;
all five null or improving scenarios are retained.

Public HELM response files, the 18.4 MB calibration package, the 16.5 MB sealed
holdout package, and per-parent blind inputs are deliberately not bundled. Their
URLs and hashes are locked, and the preparation script recreates them from the
public source.

Offline byte-integrity and anonymity check:

```text
python validate_step87_anonymous_release.py --integrity-only
```

Full replay (network access is needed only for the public dependencies):

```text
python fetch_step68_public_dependencies.py
python validate_step87_anonymous_release.py --full
```

The full mode first runs the inherited validation closure, recreates the Step 87
packages from the locked public URLs, then verifies every substantive source and
preparation hash and independently reconstructs canonicalization, active
and fixed acquisition paths, regret arrays, 10,000-repetition inference, BH
adjustments, all scenario gates, the 3/8 decision, the manuscript table, and the
nine-page main-text boundary. Download-versus-cache provenance is deliberately
ignored during regeneration; URL, byte count, content SHA-256, and every derived
package hash must match.

Three inherited runners had one workstation-specific `ROOT` assignment replaced
by a file-relative assignment for anonymity. The portable Step 87B adapter uses
the inherited audited redaction map: it returns a historical lock hash only after
the current portable bytes match their bound portable hash. All unlisted files
are hashed normally, and the separately locked Step 87B validator performs the
numerical reconstruction unchanged.

The historical manifest filename `STEP82_RELEASE_MANIFEST.json` is retained for
backward compatibility; its V10 refresh block and Step 87 bindings identify this
release.
