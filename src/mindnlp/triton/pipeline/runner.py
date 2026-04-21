"""
Pipeline dispatcher: schedules phases in order and collects results.
"""

import traceback
from datetime import datetime

from . import profiling, testing, benchmark, e2e, report

PHASES = {
    "profiling": profiling.run,
    "test": testing.run,
    "benchmark": benchmark.run,
    "e2e": e2e.run,
    "report": report.run,
}


def _reset_ms_generator():
    """Reset MindSpore default generator to fix Step tensor issue."""
    try:
        from mindtorch._C import default_generator
        import torch
        default_generator._seed = torch.Tensor([0])
        default_generator._offset = torch.Tensor([0])
    except Exception:
        pass


def run_pipeline(config: dict, phases: list) -> dict:
    """Run the optimization pipeline for specified phases.

    Args:
        config: Pipeline configuration dictionary
        phases: List of phase names to execute

    Returns:
        Dictionary containing results from all executed phases
    """
    _reset_ms_generator()

    results = {
        "meta": {
            "model": config.get("model", "unknown"),
            "device": config.get("device", "cpu"),
            "timestamp": datetime.now().isoformat(),
            "config_phases": phases,
        }
    }

    phase_results = results

    for phase in phases:
        if phase not in PHASES:
            print(f"[runner] Unknown phase '{phase}', skipping.")
            continue

        print(f"[runner] Running phase: {phase} ...")
        try:
            if phase == "report":
                phase_results[phase] = PHASES[phase](config, phase_results)
            else:
                phase_results[phase] = PHASES[phase](config)
            print(f"[runner] Phase '{phase}' completed.")
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[runner] Phase '{phase}' FAILED: {exc}")
            phase_results[phase] = {"error": str(exc), "traceback": tb}

    return results