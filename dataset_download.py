#!/usr/bin/env python3
"""Download Hugging Face datasets."""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download

DATASETS = {
    "kiwi": {
        "repo_id": "linyq/kiwi_edit_training_data",
        "local_dir": "/hdd/sy/datasets/kiwi_edit_training_data",
        "desc": "Kiwi edit training data (~51GB, image/video edit metadata)",
        "subdirs": {
            "image": "image_edit_metadata",
            "video": "video_edit_metadata",
            "refvie": "refvie_477k",
        },
    },
}


def download_dataset(
    repo_id: str,
    local_dir: str,
    allow_patterns: list[str] | None = None,
) -> None:
    kwargs = dict(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        max_workers=1,
    )
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns

    snapshot_download(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face datasets (default: kiwi edit training data)"
    )
    parser.add_argument(
        "name",
        nargs="?",
        default="kiwi",
        choices=list(DATASETS.keys()),
        help="Dataset to download (default: kiwi)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Override default local directory",
    )
    parser.add_argument(
        "--part",
        choices=["all", "image", "video", "refvie"],
        default="all",
        help="Download only a subfolder (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DATASETS[args.name]

    local_dir = args.local_dir or cfg["local_dir"]
    allow_patterns = None
    if args.part != "all":
        subdir = cfg["subdirs"][args.part]
        allow_patterns = [f"{subdir}/*", f"{subdir}/**"]
        print(f"Partial download: {subdir}/")

    print(f"Dataset: {cfg['desc']}")
    print(f"  repo:  {cfg['repo_id']}")
    print(f"  path:  {local_dir}")

    try:
        download_dataset(cfg["repo_id"], local_dir, allow_patterns)
        print(f"다운로드 완료: {local_dir}")
    except Exception as e:
        print(f"에러: {e}", file=sys.stderr)
        print("다시 실행하면 이어받기 됩니다.", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
