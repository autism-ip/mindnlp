#!/usr/bin/env python3
"""
Triton Qwen Operator Optimization Pipeline - CLI Entry Point

Usage:
    python -m mindnlp.triton.pipeline --config config.yaml
    python -m mindnlp.triton.pipeline --config config.yaml --phase benchmark
    python -m mindnlp.triton.pipeline --config config.yaml --phase e2e,report
    python -m mindnlp.triton.pipeline --model qwen2.5-0.5b --phase all
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Triton Qwen Operator Optimization Pipeline"
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config file (e.g. config.yaml)"
    )
    parser.add_argument(
        "--model",
        default="qwen2-0.5b",
        help="Model name (qwen2-0.5b or qwen2.5-0.5b)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (cpu, npu, cuda)"
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Phases to run, comma-separated or 'all'. "
             "Options: profiling,test,benchmark,e2e,report"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path (optional)"
    )

    args = parser.parse_args()

    config = {"model": args.model, "device": args.device}

    if args.config:
        import yaml
        with open(args.config) as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)

    phases = args.phase
    if phases is None:
        phases = config.get("phases", "all")
    if phases == "all":
        phases = ["profiling", "test", "benchmark", "e2e", "report"]
    elif isinstance(phases, str):
        phases = [p.strip() for p in phases.split(",") if p.strip()]

    print(f"[pipeline] Model:  {config.get('model')}")
    print(f"[pipeline] Device: {config.get('device')}")
    print(f"[pipeline] Phases: {phases}")
    print()

    from mindnlp.triton.pipeline import run_pipeline
    results = run_pipeline(config, phases)

    print()
    print("[pipeline] Pipeline complete.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[pipeline] Results saved to: {args.output}")
    else:
        print("[pipeline] Use --output to save results to JSON file")

    return 0


if __name__ == "__main__":
    sys.exit(main())