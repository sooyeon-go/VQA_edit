#!/usr/bin/env python3
"""Qwen-Image-Edit on first frame of each video + instruction from prompt.yaml."""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from pathlib import Path

import torch
from PIL import Image

from run_pipeline import QWEN_EDIT_MODEL, log, require_edit_model_dir, save_json

DEFAULT_PROMPT_YAML = Path(
    "/mnt/sy/VIVA_project/VIVA/data/sy_prompt/prompt.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit the first frame of each src_video using Qwen-Image-Edit + instruction"
    )
    parser.add_argument(
        "--prompt-yaml",
        type=Path,
        default=DEFAULT_PROMPT_YAML,
        help="YAML with instruction / src_video / ref_img entries",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./viva_first_frame_edits"),
        help="Output directory",
    )
    parser.add_argument("--edit-model", default=QWEN_EDIT_MODEL)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N entries (0 = all)")
    return parser.parse_args()


def load_prompt_entries(prompt_yaml: Path) -> list[dict]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc

    if not prompt_yaml.is_file():
        raise SystemExit(f"prompt yaml not found: {prompt_yaml}")

    data = yaml.safe_load(prompt_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"expected a YAML list in {prompt_yaml}")

    entries: list[dict] = []
    yaml_dir = prompt_yaml.parent.resolve()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise SystemExit(f"entry {idx} is not a mapping")
        instruction = str(item.get("instruction", "")).strip()
        src_video = str(item.get("src_video", "")).strip()
        if not instruction or not src_video:
            raise SystemExit(f"entry {idx} missing instruction or src_video")

        video_path = (yaml_dir / src_video).resolve()
        ref_img = str(item.get("ref_img", "")).strip()
        ref_path = (yaml_dir / ref_img).resolve() if ref_img else None
        entries.append(
            {
                "index": idx,
                "instruction": instruction,
                "video_path": video_path,
                "ref_path": ref_path,
                "src_video": src_video,
                "ref_img": ref_img,
            }
        )
    return entries


def entry_slug(entry: dict) -> str:
    if entry["ref_path"] is not None:
        return entry["ref_path"].stem
    video_stem = entry["video_path"].stem
    digest = hashlib.md5(entry["instruction"].encode("utf-8")).hexdigest()[:8]
    return f"{video_stem}_{entry['index']:03d}_{digest}"


def extract_first_frame(video_path: Path) -> Image.Image:
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    try:
        from torchvision.io import read_video

        frames, _, _ = read_video(str(video_path), pts_unit="sec")
        if frames.numel() == 0:
            raise RuntimeError(f"video has no frames: {video_path}")
        frame = frames[0].numpy()
        if frame.dtype != "uint8":
            frame = frame.clip(0, 255).astype("uint8")
        return Image.fromarray(frame).convert("RGB")
    except Exception as torchvision_exc:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                f"failed to read {video_path} with torchvision ({torchvision_exc}); "
                "install opencv-python or torchvision"
            ) from exc

        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read first frame: {video_path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).convert("RGB")


def load_pipeline(model_path: str, gpu: int):
    try:
        from diffusers import QwenImageEditPlusPipeline
    except ImportError as exc:
        raise SystemExit(
            "QwenImageEditPlusPipeline not found in diffusers.\n"
            "Install: pip install -U git+https://github.com/huggingface/diffusers.git"
        ) from exc

    device = f"cuda:{gpu}"
    log(f"loading Qwen-Image-Edit on {device}...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=None)
    return pipeline


def run_edit(
    pipeline,
    image: Image.Image,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    negative_prompt: str,
) -> Image.Image:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        output = pipeline(
            image=[image],
            prompt=prompt,
            generator=generator,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
        )
    return output.images[0]


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    prompt_yaml = args.prompt_yaml.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_prompt_entries(prompt_yaml)
    if args.limit > 0:
        entries = entries[: args.limit]

    log(f"prompt yaml: {prompt_yaml}")
    log(f"entries:     {len(entries)}")
    log(f"out:         {out_dir}")

    edit_model = require_edit_model_dir(args.edit_model, "Qwen-Image-Edit")
    pipeline = load_pipeline(edit_model, args.gpu)

    frame_cache: dict[str, Path] = {}
    manifest: list[dict] = []

    for entry in entries:
        slug = entry_slug(entry)
        item_dir = out_dir / slug
        edited_path = item_dir / "edited.png"
        if args.skip_existing and edited_path.is_file():
            log(f"[skip] {slug}")
            continue

        video_path = entry["video_path"]
        video_key = str(video_path)
        item_dir.mkdir(parents=True, exist_ok=True)

        if video_key not in frame_cache:
            log(f"extract first frame: {video_path.name}")
            frame = extract_first_frame(video_path)
            frame_path = out_dir / "frames" / f"{video_path.stem}_frame0.png"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame.save(frame_path)
            frame_cache[video_key] = frame_path
        else:
            frame = Image.open(frame_cache[video_key]).convert("RGB")

        input_path = item_dir / "input_frame.png"
        frame.save(input_path)

        log(f"[edit] {slug}: {entry['instruction']}")
        edited = run_edit(
            pipeline=pipeline,
            image=frame,
            prompt=entry["instruction"],
            seed=args.seed + entry["index"],
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
        )
        edited.save(edited_path)

        record = {
            "index": entry["index"],
            "slug": slug,
            "instruction": entry["instruction"],
            "src_video": entry["src_video"],
            "video_path": str(video_path),
            "ref_img": entry["ref_img"],
            "ref_path": str(entry["ref_path"]) if entry["ref_path"] else None,
            "input_frame": str(input_path),
            "cached_frame": str(frame_cache[video_key]),
            "output_image": str(edited_path),
            "seed": args.seed + entry["index"],
        }
        save_json(item_dir / "meta.json", record)
        manifest.append(record)
        log(f"      saved: {edited_path}")

    save_json(out_dir / "manifest.json", {"items": manifest})
    log(f"done: {len(manifest)} edits in {out_dir}")

    del pipeline
    free_gpu()


if __name__ == "__main__":
    main()
