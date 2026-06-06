#!/usr/bin/env python3
"""HunyuanImage-3-Instruct i2i editing from editing_prompts.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_pipeline import (
    HUNYUAN_MODEL,
    build_device_map,
    extract_json,
    log,
    run_hunyuan_edit,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hunyuan i2i editing with prompts from editing_prompts.json"
    )
    parser.add_argument("--image-a", type=Path, required=True, help="Input image (start state)")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
        help="editing_prompts.json from LLM step",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./edits_out"))
    parser.add_argument("--hunyuan-model", default=HUNYUAN_MODEL)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0). -1 for auto.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diff-infer-steps", type=int, default=50)
    parser.add_argument("--moe-impl", choices=["eager", "flashinfer"], default="eager")
    parser.add_argument(
        "--chain-mode",
        choices=["sequential", "from_a"],
        default="sequential",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image_a.is_file():
        raise SystemExit(f"image not found: {args.image_a}")
    if not args.prompts_file.is_file():
        raise SystemExit(f"prompts file not found: {args.prompts_file}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_text = args.prompts_file.read_text(encoding="utf-8")
    prompts_data = extract_json(prompts_text)
    save_json(out_dir / "editing_prompts.json", prompts_data)

    manifest = run_hunyuan_edit(
        image_a=args.image_a.resolve(),
        prompts_data=prompts_data,
        out_dir=out_dir,
        model_path=args.hunyuan_model,
        seed=args.seed,
        diff_infer_steps=args.diff_infer_steps,
        moe_impl=args.moe_impl,
        chain_mode=args.chain_mode,
        device_map=build_device_map(args.gpu),
    )

    log(f"Edited {len(manifest)} steps. Outputs in {out_dir / 'edits'}")


if __name__ == "__main__":
    main()
