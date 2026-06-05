#!/usr/bin/env python3
"""Large-scale variant of the entity boost benchmark.

Runs the same recall-quality comparison as benchmark_entity_boost.py but with
a much larger synthetic dataset (10 000 memories, 100 queries) to measure
entity boost behaviour at scale.

Usage:
    uv run python scripts/benchmark_entity_boost_large.py
"""

import os
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
sys.path.insert(0, str(_scripts_dir))

from benchmark_entity_boost import run_benchmark

# ── Large-scale defaults ────────────────────────────────────────────
NUM_MEMORIES = int(os.environ.get("BENCH_LARGE_NUM_MEMORIES", "10000"))
NUM_QUERIES = int(os.environ.get("BENCH_LARGE_NUM_QUERIES", "100"))
TOP_K = int(os.environ.get("BENCH_LARGE_TOP_K", "5"))

_default_dir = Path(__file__).resolve().parent
RESULTS_FILE = Path(
    os.environ.get(
        "BENCH_LARGE_RESULTS_FILE",
        _default_dir / "benchmark_entity_boost_large_results.json",
    )
)
PNG_FILE = Path(
    os.environ.get(
        "BENCH_LARGE_PNG_FILE",
        _default_dir / "benchmark_entity_boost_large_results.png",
    )
)


def main() -> int:
    run_benchmark(
        num_memories=NUM_MEMORIES,
        num_queries=NUM_QUERIES,
        top_k=TOP_K,
        results_file=RESULTS_FILE,
        png_file=PNG_FILE,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
