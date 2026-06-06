#!/usr/bin/env python3
"""Download all models used by run_pipeline.py."""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download

MODELS = {
    "vqa": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "local_dir": "/data/shared-vilab/pretrained_models/Qwen3-VL-8B-Instruct",
        "desc": "Qwen3-VL (VQA step)",
    },
    "llm": {
        "repo_id": "Qwen/Qwen3-32B",
        "local_dir": "/data/shared-vilab/pretrained_models/qwen3-32b-weights",
        "desc": "Qwen3-32B (LLM prompt step)",
    },
    "hunyuan": {
        "repo_id": "tencent/HunyuanImage-3.0-Instruct",
        "local_dir": "/data/shared-vilab/pretrained_models/HunyuanImage-3-Instruct",
        "desc": "HunyuanImage-3-Instruct (i2i edit step)",
    },
    "qwen-edit": {
        "repo_id": "Qwen/Qwen-Image-Edit-2511",
        "local_dir": "/data/shared-vilab/pretrained_models/Qwen-Image-Edit-2511",
        "desc": "Qwen-Image-Edit-2511 (diffusers i2i edit, ~58GB)",
    },
}


def download_one(name: str, repo_id: str, local_dir: str, desc: str) -> None:
    print(f"\n[{name}] {desc}")
    print(f"  repo:  {repo_id}")
    print(f"  path:  {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        max_workers=1,
    )
    print(f"  done:  {local_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pipeline models from Hugging Face")
    parser.add_argument(
        "models",
        nargs="*",
        choices=[*MODELS.keys(), "all"],
        default=["all"],
        help="Which model(s) to download (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(MODELS.keys()) if "all" in args.models else args.models

    print(f"Downloading {len(selected)} model(s): {', '.join(selected)}")
    failed: list[str] = []

    for name in selected:
        cfg = MODELS[name]
        try:
            download_one(name, cfg["repo_id"], cfg["local_dir"], cfg["desc"])
        except Exception as e:
            print(f"  error: {e}", file=sys.stderr)
            print("  다시 실행하면 이어받기 됩니다.", file=sys.stderr)
            failed.append(name)

    print()
    if failed:
        print(f"실패: {', '.join(failed)}")
        raise SystemExit(1)
    print("모든 다운로드 완료.")


if __name__ == "__main__":
    main()
