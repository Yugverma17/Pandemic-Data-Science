"""Run the whole pipeline end to end.

Usage:
    python scripts/run_all.py [--fast] [--skip fetch,forecast]

Stages are independent and each caches its own expensive work, so a failure in
one does not invalidate the others. The runner reports per-stage timings and
exits non-zero if any stage failed, which is what makes it usable in CI or a
scheduled job rather than only interactively.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from pandemic.config import ROOT, get_logger

log = get_logger("run_all")

STAGES = [
    ("fetch", "fetch_data.py", []),
    ("forensics", "run_forensics.py", []),
    ("forecast", "run_forecast.py", []),
    ("causal", "run_causal.py", ["--fast"]),
    ("capacity", "run_capacity.py", []),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="fewer resampling draws where a stage supports it")
    parser.add_argument("--skip", default="",
                        help="comma-separated stage names to skip")
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    scripts_dir = Path(__file__).parent

    results: list[tuple[str, bool, float]] = []
    for name, script, fast_args in STAGES:
        if name in skip:
            log.info("skipping %s", name)
            continue

        cmd = [sys.executable, str(scripts_dir / script)]
        if args.fast:
            cmd += fast_args

        log.info("=" * 60)
        log.info("stage: %s", name)
        started = time.perf_counter()
        completed = subprocess.run(cmd, cwd=ROOT, check=False)
        elapsed = time.perf_counter() - started

        ok = completed.returncode == 0
        results.append((name, ok, elapsed))
        log.info("stage %s %s in %.1fs", name, "succeeded" if ok else "FAILED", elapsed)

    log.info("=" * 60)
    for name, ok, elapsed in results:
        log.info("  %-12s %-8s %6.1fs", name, "ok" if ok else "FAILED", elapsed)

    failures = [n for n, ok, _ in results if not ok]
    if failures:
        log.error("failed stages: %s", ", ".join(failures))
        return 1
    log.info("pipeline complete; figures in reports/figures, tables in reports/tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
