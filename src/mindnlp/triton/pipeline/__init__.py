"""
Triton Optimization Pipeline.

Phases:
    1. profiling - Analyze model performance data
    2. test - Numerical accuracy validation
    3. benchmark - Single-operator performance comparison
    4. e2e - End-to-end MLP validation
    5. report - Generate summary report

Usage:
    from mindnlp.triton.pipeline import run_pipeline

    config = {"model": "qwen2-0.5b", "device": "cpu"}
    results = run_pipeline(config, ["profiling", "test", "benchmark"])

    # Or run all phases:
    from mindnlp.triton.pipeline import run_all
    results = run_all(config)
"""

from .runner import run_pipeline, PHASES

ALL_PHASES = list(PHASES.keys())


def run_all(config: dict) -> dict:
    """Run all pipeline phases.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Dictionary containing results from all phases
    """
    return run_pipeline(config, ALL_PHASES)


__all__ = [
    "run_pipeline",
    "run_all",
    "PHASES",
    "ALL_PHASES",
    "profiling",
    "testing",
    "benchmark",
    "e2e",
    "report",
]