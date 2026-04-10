"""
evaluation/setup_assets.py

Manual asset bootstrapper for datasets, KB files, and baseline models.
"""

import argparse

from assets import ensure_task_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", help="all | entity_linking | sts | nli")
    parser.add_argument("--model", default=None, help="optional model to pre-download")
    args = parser.parse_args()

    ensure_task_assets(task=args.task, model_name=args.model)
    print("asset setup complete")


if __name__ == "__main__":
    main()
