from __future__ import annotations

import json

from step95_stageb_confirmatory import LOCK_PATH, create_execution_lock
import step95_common as common


def main() -> None:
    lock = create_execution_lock()
    print(
        json.dumps(
            {
                "lock": LOCK_PATH.name,
                "lock_sha256": common.sha256_path(LOCK_PATH),
                "parent": lock["selected_parent"]["key"],
                "seed_count": lock["confirmatory"]["seed_count"],
                "pre_outcome_assertions": lock["pre_outcome_assertions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
