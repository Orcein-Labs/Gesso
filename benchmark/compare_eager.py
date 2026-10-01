#!/usr/bin/env python3
# G2 harness (Phase 10 — SPEED FLOOR): eager PyTorch reference timings for
# the same checkpoint / prompt / max_new_tokens the Gesso CUDA Session rows
# measure (benchmark/runbenchmarks.jl, G2 block).
#
# Contract (docs/goals/PHASE10_SPEED_FLOOR.md, G2):
#   * eager only — transformers LlamaForCausalLM.generate, greedy
#     (do_sample=False), batch 1, NO torch.compile. torch.compile / vLLM
#     are later additional rows, not the G2 gate.
#   * local weights only: from_pretrained(local_files_only=True) on
#     --model-dir (or $GESSO_SMOLLM2_DIR). NEVER downloads.
#   * two clocks: --mode first (single shot; kernel compile/load allowed
#     inside — that is what first-token means) and --mode warmed (one
#     untimed warmup, then the median of --samples runs).
#   * external binary: this script is invoked as `python3`, never imported
#     from Julia, and torch/transformers are NEVER added to any
#     Project.toml (§VII dependency law).
#   * machine-readable output: by default `key: value` lines (what the
#     Julia harness in runbenchmarks.jl parses — the bench env has no JSON
#     package and gains none); `--json` emits one JSON object with the same
#       keys instead (for external tooling):
#       {"mode", "seconds", "tok_s", "max_new_tokens", "samples",
#        "device", "compute_dtype", "prompt_tokens"}
#     Exit codes: 0 measured · 3 torch/transformers unavailable ·
#     4 no model dir · 5 measurement failed.
#   * `--help` never imports torch: the harness itself is testable on a
#     box without PyTorch (test/test_speed_floor_harness.jl).
#
# Arithmetic stamp (printed in the JSON as compute_dtype): the model loads
# with from_pretrained defaults — no torch_dtype override — so a BF16
# checkpoint is upcast to FP32 by the loader. The Gesso side computes F32.
# Both sides' dtype stamps are recorded in the bench rows' bench_note;
# mixing silent casts is a lie, so both stamps travel with the numbers.

import argparse
import json
import os
import sys
import time


def build_parser():
    p = argparse.ArgumentParser(
        prog="compare_eager.py",
        description=(
            "Eager PyTorch reference timings (first-token and warmed) for "
            "the G2 SPEED FLOOR comparison. Local weights only; never "
            "downloads; never imports torch on --help."
        ),
    )
    p.add_argument(
        "--model-dir",
        default=os.environ.get("GESSO_SMOLLM2_DIR"),
        help="local snapshot dir (default: $GESSO_SMOLLM2_DIR)",
    )
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--mode", choices=["first", "warmed"], default="warmed")
    p.add_argument("--samples", type=int, default=5, help="warmed-mode sample count")
    p.add_argument("--json", action="store_true", help="emit one JSON object on stdout")
    return p


def _emit(args, payload):
    if args.json:
        print(json.dumps(payload))
    else:
        for k in sorted(payload):
            print(f"{k}: {payload[k]}")
    return 0


def _fail(args, code, kind, detail):
    if args.json:
        print(json.dumps({"error": kind, "detail": detail}), file=sys.stderr)
    else:
        print(f"{kind}: {detail}", file=sys.stderr)
    return code


def main():
    args = build_parser().parse_args()

    # torch is imported only AFTER argument handling — --help and bad args
    # work on a box with no PyTorch at all.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # ImportError and any loader-side import failure
        return _fail(args, 3, "torch_unavailable", str(e))

    if not args.model_dir or not os.path.isdir(args.model_dir):
        return _fail(
            args,
            4,
            "model_dir_missing",
            "pass --model-dir or set GESSO_SMOLLM2_DIR to a local snapshot",
        )

    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, local_files_only=True
        )
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

        def one_generate():
            t0 = time.perf_counter()
            model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.eos_token_id,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter() - t0

        dtype = next(model.parameters()).dtype
        stamp = {
            "device": device,
            "compute_dtype": str(dtype).replace("torch.", ""),
            "max_new_tokens": args.max_new_tokens,
            "prompt_tokens": int(ids.shape[1]),
        }
        if args.mode == "first":
            dt = one_generate()  # single shot: first-token clock INCLUDES one-shot kernel compile/load
            return _emit(
                args,
                {**stamp, "mode": "first", "seconds": dt, "tok_s": args.max_new_tokens / dt, "samples": 1},
            )
        one_generate()  # untimed warmup (§XXXIII hygiene: warmed excludes compile)
        times = sorted(one_generate() for _ in range(max(args.samples, 1)))
        dt = times[len(times) // 2]
        return _emit(
            args,
            {
                **stamp,
                "mode": "warmed",
                "seconds": dt,
                "tok_s": args.max_new_tokens / dt,
                "samples": len(times),
            },
        )
    except Exception as e:
        return _fail(args, 5, "measurement_failed", str(e))


if __name__ == "__main__":
    sys.exit(main())
