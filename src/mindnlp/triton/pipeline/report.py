"""
Phase 5: Generate summary report from all phase results.
"""

from datetime import datetime


def run(config: dict, phase_results: dict) -> dict:
    """Generate a summary report from all phase results.

    Args:
        config: Pipeline configuration
        phase_results: Dictionary containing results from all phases

    Returns:
        Dictionary containing summary report
    """
    profiling = phase_results.get("profiling", {})
    test_results = phase_results.get("test", {})
    benchmark = phase_results.get("benchmark", {})
    e2e = phase_results.get("e2e", {})

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": config.get("model", "unknown"),
        "device": config.get("device", "unknown"),
        "phases_completed": list(phase_results.keys()),
    }

    if profiling:
        summary["profiling"] = {
            "total_ms": profiling.get("total_ms", 0),
            "bottleneck": profiling.get("summary", {}).get("bottleneck", "unknown"),
            "mlp_ratio": profiling.get("summary", {}).get("mlp_ratio", 0),
            "act_fn_ratio": profiling.get("summary", {}).get("act_fn_ratio", 0),
        }

    if test_results:
        summary["testing"] = {
            "all_passed": test_results.get("all_passed", False),
            "results": {k: v for k, v in test_results.items() if k != "all_passed"},
        }

    if benchmark and isinstance(benchmark, dict) and "error" not in benchmark:
        summary["benchmark"] = {}
        for op_name, op_results in benchmark.items():
            if op_results and isinstance(op_results, list):
                best_speedup = max(r.get("speedup", 0) for r in op_results)
                summary["benchmark"][op_name] = {"best_speedup": best_speedup}

    if e2e and isinstance(e2e, dict) and "error" not in e2e:
        summary["e2e"] = {}
        for mlp_type, mlp_results in e2e.items():
            if mlp_results and isinstance(mlp_results, list):
                best_speedup = max(r.get("speedup", 0) for r in mlp_results)
                summary["e2e"][mlp_type] = {"best_speedup": best_speedup}

    recommendation = []
    benchmark_data = summary.get("benchmark", {})
    if benchmark_data and isinstance(benchmark_data, dict):
        gelu_data = benchmark_data.get("gelu")
        swiglu_data = benchmark_data.get("swiglu")
        if gelu_data and isinstance(gelu_data, dict):
            speedup = gelu_data.get("best_speedup", 0)
            if speedup > 1.0:
                recommendation.append(f"GELU: Use Triton (up to {speedup}x speedup)")
        if swiglu_data and isinstance(swiglu_data, dict):
            speedup = swiglu_data.get("best_speedup", 0)
            if speedup > 1.0:
                recommendation.append(f"SwiGLU: Use Triton (up to {speedup}x speedup)")
    if not recommendation:
        recommendation.append("No clear winner: use native implementations")

    summary["recommendation"] = recommendation

    return summary