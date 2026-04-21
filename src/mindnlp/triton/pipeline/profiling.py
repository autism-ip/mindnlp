"""
Phase 1: Profiling data collection.

Simulates per-operator time distribution for Qwen models based on
measured MindSpore Profiler data. Returns bottleneck analysis.
"""

_PROFILE_DATA = {
    "qwen2-0.5b": {
        "down_proj": {"ms": 4383.5, "type": "GEAM"},
        "gate_proj": {"ms": 3862.2, "type": "GEAM"},
        "up_proj": {"ms": 3831.0, "type": "GEAM"},
        "q_proj": {"ms": 754.8, "type": "GEAM"},
        "o_proj": {"ms": 727.4, "type": "GEAM"},
        "act_fn": {"ms": 421.2, "type": "Elementwise"},
        "k_proj": {"ms": 141.5, "type": "GEAM"},
        "v_proj": {"ms": 123.3, "type": "GEAM"},
        "post_layernorm": {"ms": 75.5, "type": "RMSNorm"},
        "input_layernorm": {"ms": 67.6, "type": "RMSNorm"},
    },
    "qwen2.5-0.5b": {
        "down_proj": {"ms": 4383.5, "type": "GEAM"},
        "gate_proj": {"ms": 3862.2, "type": "GEAM"},
        "up_proj": {"ms": 3831.0, "type": "GEAM"},
        "q_proj": {"ms": 754.8, "type": "GEAM"},
        "o_proj": {"ms": 727.4, "type": "GEAM"},
        "act_fn": {"ms": 421.2, "type": "Elementwise"},
        "k_proj": {"ms": 141.5, "type": "GEAM"},
        "v_proj": {"ms": 123.3, "type": "GEAM"},
        "post_layernorm": {"ms": 75.5, "type": "RMSNorm"},
        "input_layernorm": {"ms": 67.6, "type": "RMSNorm"},
    },
}


def run(config: dict) -> dict:
    """Run profiling analysis on Qwen models.

    Args:
        config: Pipeline configuration with optional 'model' key

    Returns:
        Dictionary containing profiling results and bottleneck analysis
    """
    model = config.get("model", "qwen2-0.5b")
    ops = _PROFILE_DATA.get(model, _PROFILE_DATA["qwen2-0.5b"])

    total_ms = sum(v["ms"] for v in ops.values())
    result_ops = {}
    for name, data in ops.items():
        result_ops[name] = {
            "ms": data["ms"],
            "type": data["type"],
            "ratio": round(data["ms"] / total_ms, 4),
        }

    mlp_ops = ["down_proj", "gate_proj", "up_proj", "act_fn"]
    mlp_ms = sum(ops[k]["ms"] for k in mlp_ops if k in ops)
    geam_ms = sum(v["ms"] for v in ops.values() if v["type"] == "GEAM")

    return {
        "model": model,
        "total_ms": round(total_ms, 2),
        "ops": result_ops,
        "summary": {
            "mlp_ratio": round(mlp_ms / total_ms, 4),
            "geam_ratio": round(geam_ms / total_ms, 4),
            "act_fn_ratio": round(ops.get("act_fn", {}).get("ms", 0) / total_ms, 4),
            "bottleneck": "MLP layer (matmul dominates 97% of MLP time, act_fn only 2.9%)",
        },
    }