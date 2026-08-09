"""Stage 0 -- download every upstream source into data/raw.

Usage:
    python scripts/fetch_data.py [--force]
"""

from __future__ import annotations

import argparse

from pandemic.config import get_logger
from pandemic.data import fetch_all

log = get_logger("fetch_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    paths = fetch_all(force=args.force)
    log.info("%d/%d sources available", len(paths), 5)
    for key, path in paths.items():
        log.info("  %-18s -> %s", key, path.name)


if __name__ == "__main__":
    main()
