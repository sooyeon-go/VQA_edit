#!/usr/bin/env python3
"""Batch runner: circular image pairs x multiple seeds (multi-GPU parallel)."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_PIPELINE = SCRIPT_DIR / "run_pipeline.py"

MODEL_ROOT = "/data/shared-vilab/pretrained_models"
VQA_MODEL = f"{MODEL_ROOT}/Qwen3-VL-8B-Instruct"
LLM_MODEL = f"{MODEL_ROOT}/qwen3-32b-weights"
EDIT_MODEL = f"{MODEL_ROOT}/Qwen-Image-Edit-2511"

COMMON_ARGS = [
    "--vqa-model", VQA_MODEL,
    "--llm-model", LLM_MODEL,
    "--edit-model", EDIT_MODEL,
    "--gpu", "0",
]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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


def run_pipeline(args: list[str], env: dict[str, str], label: str = "") -> None:
    cmd = [sys.executable, str(RUN_PIPELINE), *args]
    prefix = f"[{label}] " if label else ""
    log(f"{prefix}$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="", file=sys.stderr)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        raise RuntimeError(
            f"pipeline failed (exit {proc.returncode})"
            + (f"\n--- last output ---\n{tail}" if tail else "")
        )


def gpu_env(gpu_id: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def prompts_path(pair_dir: Path) -> Path:
    return pair_dir / "editing_prompts.json"


def edits_manifest_path(seed_dir: Path) -> Path:
    return seed_dir / "edits" / "manifest.json"


def require_prompts(pair_dir: Path) -> None:
    path = prompts_path(pair_dir)
    if not path.is_file():
        raise RuntimeError(f"missing prompts file: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"empty prompts file: {path}")


def require_edits(seed_dir: Path) -> None:
    manifest = edits_manifest_path(seed_dir)
    if not manifest.is_file():
        raise RuntimeError(f"missing edit manifest: {manifest}")
    edits_dir = seed_dir / "edits"
    pngs = list(edits_dir.glob("step_*.png"))
    if not pngs:
        raise RuntimeError(f"no step images in {edits_dir}")


@dataclass(frozen=True)
class PromptJob:
    label: str
    gpu: int
    image_a: str
    image_b: str
    pair_dir: str
    num_prompts: int


@dataclass(frozen=True)
class EditJob:
    label: str
    gpu: int
    image_a: str
    image_b: str
    pair_dir: str
    seed_dir: str
    seed_idx: int
    num_prompts: int


def run_prompt_job(job: PromptJob) -> str:
    env = gpu_env(job.gpu)
    log(f"[GPU {job.gpu}] {job.label}: VQA+LLM")
    run_pipeline(
        [
            "--image-a", job.image_a,
            "--image-b", job.image_b,
            "-n", str(job.num_prompts),
            "--out-dir", job.pair_dir,
            "--skip-edit",
            *COMMON_ARGS,
        ],
        env,
        label=job.label,
    )
    require_prompts(Path(job.pair_dir))
    return job.label


def run_edit_job(job: EditJob) -> str:
    env = gpu_env(job.gpu)
    pair_dir = Path(job.pair_dir)
    seed_dir = Path(job.seed_dir)
    require_prompts(pair_dir)
    copy_prompt_artifacts(pair_dir, seed_dir)
    require_prompts(seed_dir)
    log(f"[GPU {job.gpu}] {job.label}: editing -> {seed_dir / 'edits'}")
    run_pipeline(
        [
            "--image-a", job.image_a,
            "--image-b", job.image_b,
            "-n", str(job.num_prompts),
            "--out-dir", job.seed_dir,
            "--skip-vqa",
            "--skip-llm",
            "--seed", str(job.seed_idx),
            *COMMON_ARGS,
        ],
        env,
        label=job.label,
    )
    require_edits(seed_dir)
    return job.label


def assign_gpus(jobs: list, gpus: list[int]) -> list:
    return [type(job)(**{**job.__dict__, "gpu": gpus[i % len(gpus)]}) for i, job in enumerate(jobs)]


def run_parallel(jobs: list, worker, gpus: list[int]) -> None:
    if not jobs:
        return
    assigned = assign_gpus(jobs, gpus)
    workers = min(len(gpus), len(assigned))
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, job): job for job in assigned}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors.append(f"{job.label} (GPU {job.gpu}): {exc}")
    if errors:
        raise SystemExit("Failed jobs:\n  " + "\n  ".join(errors))


def run_sequential(jobs: list, worker, gpus: list[int]) -> None:
    for job in assign_gpus(jobs, gpus):
        worker(job)


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
    parser.add_argument("--gpu", type=int, default=0, help="Single GPU (used when --gpus is not set)")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids for parallel runs, e.g. 0,1,2,3 (one job per GPU)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip pair/seed if edits already exist")
    parser.add_argument(
        "--skip-prompts",
        action="store_true",
        help="Skip VQA+LLM; only run edit jobs (requires existing editing_prompts.json per pair)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run jobs one at a time (easier debugging)",
    )
    return parser.parse_args()


def parse_gpu_list(args: argparse.Namespace) -> list[int]:
    if args.gpus:
        return [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    return [args.gpu]


def run_jobs(jobs: list, worker, gpus: list[int], sequential: bool) -> None:
    if sequential or len(gpus) == 1:
        run_sequential(jobs, worker, gpus)
    else:
        run_parallel(jobs, worker, gpus)


def main() -> None:
    args = parse_args()
    gpus = parse_gpu_list(args)
    dataset_dir = args.dataset_dir.resolve()
    out_base = args.out_dir.resolve()

    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset dir not found: {dataset_dir}")

    images = collect_images(dataset_dir)
    pairs = circular_pairs(images)
    if not pairs:
        raise SystemExit(f"need at least 2 cat/dog images in {dataset_dir}")

    log(f"dataset: {dataset_dir}")
    log(f"images:  {len(images)} (cat+dog, lion excluded)")
    log(f"pairs:   {len(pairs)} (circular)")
    log(f"seeds:   {args.num_seeds} (seed0 .. seed{args.num_seeds - 1})")
    log(f"gpus:    {gpus} ({'sequential' if args.sequential or len(gpus) == 1 else 'parallel'})")
    log(f"out:     {out_base}")
    log(f"edits save to: {{pair}}/seedN/edits/step_XX.png")

    prompt_jobs: list[PromptJob] = []
    edit_jobs: list[EditJob] = []

    for image_a, image_b in pairs:
        name = pair_name(image_a, image_b)
        pair_dir = out_base / name
        prompts_ready = prompts_path(pair_dir).is_file()

        if not args.skip_prompts and not prompts_ready:
            prompt_jobs.append(
                PromptJob(
                    label=name,
                    gpu=0,
                    image_a=str(image_a),
                    image_b=str(image_b),
                    pair_dir=str(pair_dir),
                    num_prompts=args.num_prompts,
                )
            )
        elif args.skip_prompts and not prompts_ready:
            log(f"warning: skipping edits for {name} (no editing_prompts.json)")
            continue

        for seed_idx in range(args.num_seeds):
            seed_dir = pair_dir / f"seed{seed_idx}"
            if args.skip_existing and edits_manifest_path(seed_dir).exists():
                continue
            edit_jobs.append(
                EditJob(
                    label=f"{name}/seed{seed_idx}",
                    gpu=0,
                    image_a=str(image_a),
                    image_b=str(image_b),
                    pair_dir=str(pair_dir),
                    seed_dir=str(seed_dir),
                    seed_idx=seed_idx,
                    num_prompts=args.num_prompts,
                )
            )

    log(f"\nPhase 1: {len(prompt_jobs)} prompt job(s)")
    run_jobs(prompt_jobs, run_prompt_job, gpus, args.sequential)

    for job in edit_jobs:
        require_prompts(Path(job.pair_dir))

    log(f"\nPhase 2: {len(edit_jobs)} edit job(s)")
    if not edit_jobs:
        log("No edit jobs scheduled.")
    else:
        run_jobs(edit_jobs, run_edit_job, gpus, args.sequential)

    log(f"\nDone. Results in {out_base}")


if __name__ == "__main__":
    main()
