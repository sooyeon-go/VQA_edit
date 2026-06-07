#!/usr/bin/env python3
"""Batch runner: circular image pairs x multiple seeds."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_PIPELINE = SCRIPT_DIR / "run_pipeline.py"

MODEL_ROOT = "/data/shared-vilab/pretrained_models"
VQA_MODEL = f"{MODEL_ROOT}/Qwen3-VL-8B-Instruct"
LLM_MODEL = f"{MODEL_ROOT}/qwen3-32b-weights"
EDIT_MODEL = f"{MODEL_ROOT}/Qwen-Image-Edit-2511"


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def sort_key(path: Path) -> tuple[str, int]:
    match = re.match(r"^(cat|dog)_(\d+)$", path.stem)
    if not match:
        return path.stem, 0
    return match.group(1), int(match.group(2))


def collect_images(dataset_dir: Path) -> list[Path]:
    images = [
        p
        for p in dataset_dir.glob("*.png")
        if p.stem.startswith(("cat_", "dog_"))
    ]
    cats = sorted([p for p in images if p.stem.startswith("cat_")], key=sort_key)
    dogs = sorted([p for p in images if p.stem.startswith("dog_")], key=sort_key)
    return cats + dogs


def circular_pairs(images: list[Path]) -> list[tuple[Path, Path]]:
    if len(images) < 2:
        return []
    pairs: list[tuple[Path, Path]] = []
    for i in range(len(images)):
        pairs.append((images[i], images[(i + 1) % len(images)]))
    return pairs


def pair_name(image_a: Path, image_b: Path) -> str:
    return f"{image_a.stem}__to__{image_b.stem}"


def copy_prompt_artifacts(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("vqa_delta.json", "editing_prompts.json", "vqa_delta_raw.txt", "editing_prompts_raw.txt"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
    src_inputs = src_dir / "inputs"
    dst_inputs = dst_dir / "inputs"
    if src_inputs.is_dir():
        if dst_inputs.exists():
            shutil.rmtree(dst_inputs)
        shutil.copytree(src_inputs, dst_inputs)


def run_pipeline(args: list[str], env: dict[str, str]) -> None:
    cmd = [sys.executable, str(RUN_PIPELINE), *args]
    log(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VQA+LLM once per pair, then edit with multiple seeds (cat/dog circular pairs)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=SCRIPT_DIR / "image_dataset",
    )
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "out_batch")
    parser.add_argument("-n", "--num-prompts", type=int, default=5)
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds (seed0 .. seed{N-1})")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", help="Skip pair/seed if edits already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_base = args.out_dir.resolve()

    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset dir not found: {dataset_dir}")

    images = collect_images(dataset_dir)
    pairs = circular_pairs(images)
    if not pairs:
        raise SystemExit(f"need at least 2 cat/dog images in {dataset_dir}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", str(args.gpu))

    common = [
        "--vqa-model", VQA_MODEL,
        "--llm-model", LLM_MODEL,
        "--edit-model", EDIT_MODEL,
        "--gpu", "0",
    ]

    log(f"dataset: {dataset_dir}")
    log(f"images:  {len(images)} (cat+dog, lion excluded)")
    log(f"pairs:   {len(pairs)} (circular)")
    log(f"seeds:   {args.num_seeds} (seed0 .. seed{args.num_seeds - 1})")
    log(f"out:     {out_base}")

    for idx, (image_a, image_b) in enumerate(pairs, start=1):
        name = pair_name(image_a, image_b)
        pair_dir = out_base / name
        log(f"\n[{idx}/{len(pairs)}] pair: {name}")

        prompts_ready = (pair_dir / "editing_prompts.json").exists()
        if not prompts_ready:
            run_pipeline(
                [
                    "--image-a", str(image_a),
                    "--image-b", str(image_b),
                    "-n", str(args.num_prompts),
                    "--out-dir", str(pair_dir),
                    "--skip-edit",
                    *common,
                ],
                env,
            )
        else:
            log("  prompts exist, skipping VQA+LLM")

        for seed_idx in range(args.num_seeds):
            seed_dir = pair_dir / f"seed{seed_idx}"
            edits_done = (seed_dir / "edits" / "manifest.json").exists()
            if args.skip_existing and edits_done:
                log(f"  seed{seed_idx}: skip (exists)")
                continue

            copy_prompt_artifacts(pair_dir, seed_dir)
            log(f"  seed{seed_idx}: editing")
            run_pipeline(
                [
                    "--image-a", str(image_a),
                    "--image-b", str(image_b),
                    "-n", str(args.num_prompts),
                    "--out-dir", str(seed_dir),
                    "--skip-vqa",
                    "--skip-llm",
                    "--seed", str(seed_idx),
                    *common,
                ],
                env,
            )

    log(f"\nDone. Results in {out_base}")


if __name__ == "__main__":
    main()
