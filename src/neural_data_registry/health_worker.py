from __future__ import annotations

import argparse

from neural_data_registry.health import run_health_checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Neural dataset health worker")
    parser.add_argument("--all", action="store_true", dest="all_datasets")
    parser.add_argument("--dataset-id")
    parser.add_argument("--history-id")
    arguments = parser.parse_args()

    if arguments.all_datasets:
        completed = run_health_checks()
    elif arguments.dataset_id and arguments.history_id:
        completed = run_health_checks(
            pending_history=(arguments.dataset_id, arguments.history_id)
        )
    else:
        parser.error("provide --all or both --dataset-id and --history-id")
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
