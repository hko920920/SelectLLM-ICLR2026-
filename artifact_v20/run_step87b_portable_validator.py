"""Run the locked Step 87B validator across audited anonymous path rewrites.

The anonymous release changes one workstation-specific ``ROOT`` assignment in
three historical dependencies to a file-relative assignment. The inherited
Step 68 redaction map binds both the historical and portable hashes. Its hash
adapter returns a historical hash only after the current portable bytes match
the map; every unlisted file is hashed normally. The Step 87B reconstruction
and all numerical checks remain in the separately locked validator.
"""

from __future__ import annotations

import validate_step87b_sealed_holdout as validator
from step68_portable_hash import historical_sha256_file


def main() -> None:
    validator.file_sha256 = historical_sha256_file
    validator.main()


if __name__ == "__main__":
    main()
