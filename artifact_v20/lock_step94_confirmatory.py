from __future__ import annotations

import json

import step94_common as common
from step94_stageb_confirmatory import LOCK_PATH, create_execution_lock


def main() -> None:
    lock = create_execution_lock()
    print(
        json.dumps(
            {
                "lock": LOCK_PATH.name,
                "lock_sha256": common.sha256_path(LOCK_PATH),
                "selected_task": lock["selected_task"]["task_key"],
                "seed_count": lock["confirmatory"]["seed_count"],
                "pre_outcome_assertions": lock["pre_outcome_assertions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
