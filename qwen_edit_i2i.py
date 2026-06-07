#!/usr/bin/env python3
"""Qwen-Image-Edit-2511 i2i editing from editing_prompts.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_pipeline import (
    QWEN_EDIT_MODEL,
    extract_json,
    log,
    require_edit_model_dir,
    run_qwen_edit,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen-Image-Edit with prompts from editing_prompts.json"
    )
    parser.add_argument("--image-a", type=Path, required=True, help="Input image (start state)")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
        help="editing_prompts.json from LLM step",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./edits_out"))
    parser.add_argument("--edit-model", default=QWEN_EDIT_MODEL)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=" ")
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

    edit_model = require_edit_model_dir(args.edit_model, "Qwen-Image-Edit")
    manifest = run_qwen_edit(
        image_a=args.image_a.resolve(),
        prompts_data=prompts_data,
        out_dir=out_dir,
        model_path=edit_model,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        gpu=args.gpu,
    )

    log(f"Edited {len(manifest)} steps. Outputs in {out_dir / 'edits'}")


if __name__ == "__main__":
    main()
