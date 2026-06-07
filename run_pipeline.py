#!/usr/bin/env python3
"""VQA -> LLM -> Qwen-Image-Edit i2i interpolation pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
from pathlib import Path

import torch

MODEL_ROOT = "/data/shared-vilab/pretrained_models"
VQA_MODEL = f"{MODEL_ROOT}/Qwen3-VL-8B-Instruct"
LLM_MODEL = f"{MODEL_ROOT}/qwen3-32b-weights"
QWEN_EDIT_MODEL = f"{MODEL_ROOT}/Qwen-Image-Edit-2511"

VQA_PROMPT = """You are a precise visual analyst for image interpolation.

You will receive TWO images of the SAME object.
Image A = start state, Image B = end state.

First, pick ONE stable, easily-identifiable landmark feature of the object
(e.g. for a cat: the head; for a chair: the backrest; for a car: the front).
This landmark will be tracked across the whole transition.

Then compare the two images and describe the differences in
landmark direction, pose, size, and angle using NATURAL, PLAIN language.
Do NOT use precise degrees or percentages. Describe it the way a person would
casually describe it (e.g. "the cat is sitting and facing right",
"the chair's backrest faces forward").
Do NOT describe color, texture, background, or identity.

For landmark direction, use simple words:
- horizontal: facing left / facing forward / facing right
- pose: sitting / standing / lying down / head up / head down, etc.

OUTPUT FORMAT (JSON only, no explanation):
{
  "object_name": "<object>",
  "landmark": "<the single tracked feature, e.g. 'cat head', 'chair backrest'>",
  "state_A": "<plain-language description of the object's pose, size, and which way the landmark faces in A>",
  "state_B": "<plain-language description of the object's pose, size, and which way the landmark faces in B>",
  "main_changes": "<what changes between A and B, in plain words>"
}"""

LLM_PROMPT_TEMPLATE = """You are an image editing prompt engineer specializing in object-level visual transitions.

The SAME object will be transformed step-by-step from state A to state B.
IMPORTANT: These prompts are applied SEQUENTIALLY. Each prompt edits the
OUTPUT of the previous step, NOT the original image. Each prompt is a SMALL
change relative to the immediately preceding state.

VISUAL DELTA:
{vqa_output}

---

Generate {n} editing prompts that gradually move the object from A toward B.

CRITICAL DIRECTION RULES:
- You MUST use explicit directional words in EVERY prompt: "left", "right",
  "forward" (toward camera), or "away" (back to camera). NEVER use vague words
  like "to the side", "sideways", or "around" without saying which side.
- The landmark must move in ONE consistent direction across all steps.
  Determine the start direction and end direction from the VISUAL DELTA, then
  move the landmark step by step from start to end, passing through the
  in-between directions in order. Example ordering for left -> right:
  facing left -> facing forward -> facing right. NEVER skip or reverse.
- The "prompt" and "landmark_after" fields MUST agree. The instruction in
  "prompt" must result in exactly the direction stated in "landmark_after".
- Each consecutive step's "landmark_after" must show clear progress from the
  previous step. Two steps must NOT have the same direction.

OTHER RULES:
- Plain, natural language. No degrees or percentages.
  GOOD: "the cat turns its head a little further to the right"
  BAD:  "rotate 30°", "the cat turns to the side"
- Each prompt edits the PREVIOUS result, so keep each step small.
- Keep prompts short and concrete.

OUTPUT FORMAT (JSON only, no explanation):
{{
  "landmark": "<the tracked feature>",
  "direction_path": "<the full ordered direction sequence, e.g. 'left -> forward -> right'>",
  "prompts": [
    {{
      "step": 1,
      "focus": "<pose | direction | size | combined>",
      "landmark_after": "<explicit direction the landmark faces after this step, using left/right/forward/away>",
      "prompt": "<short instruction using an explicit direction word>"
    }}
  ]
}}"""


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Could not parse JSON from model output")


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_rgb_image(path: Path, label: str):
    from PIL import Image

    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"{label} is empty: {path}")
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        raise SystemExit(f"{label} is not a valid image: {path}") from exc


def _image_is_readable(path: Path) -> bool:
    from PIL import Image

    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def sync_input_image(src: Path, dst: Path, label: str, force: bool = False) -> None:
    if not src.is_file():
        raise SystemExit(f"{label} source not found: {src}")
    if force or not _image_is_readable(dst):
        shutil.copy2(src, dst)
    load_rgb_image(dst, label)


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_device_map(gpu: int) -> str | dict:
    if gpu < 0:
        return "auto"
    return {"": gpu}


def require_model_dir(path: str, name: str) -> str:
    model_dir = Path(path)
    if not model_dir.is_dir():
        raise SystemExit(f"{name} model not found: {path}")
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"{name} model invalid (missing config.json): {path}")
    return str(model_dir)


def require_edit_model_dir(path: str, name: str) -> str:
    model_dir = Path(path)
    if not model_dir.is_dir():
        raise SystemExit(f"{name} model not found: {path}")
    if not (model_dir / "model_index.json").exists():
        raise SystemExit(f"{name} model invalid (missing model_index.json): {path}")
    return str(model_dir)


def run_vqa(
    image_a: Path,
    image_b: Path,
    out_dir: Path,
    model_path: str,
    max_new_tokens: int,
    device_map: str | dict,
) -> str:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    log("[1/3] VQA: loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map=device_map
    )
    processor = AutoProcessor.from_pretrained(model_path)

    pil_a = load_rgb_image(image_a, "VQA image A")
    pil_b = load_rgb_image(image_b, "VQA image B")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_a},
                {"type": "image", "image": pil_b},
                {"type": "text", "text": VQA_PROMPT},
            ],
        }
    ]

    log("[1/3] VQA: comparing images...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    raw_path = out_dir / "vqa_delta_raw.txt"
    save_text(raw_path, output_text)
    log(f"      saved: {raw_path}")

    try:
        parsed = extract_json(output_text)
        save_json(out_dir / "vqa_delta.json", parsed)
        log(f"      saved: {out_dir / 'vqa_delta.json'}")
    except ValueError:
        save_text(out_dir / "vqa_delta.json", output_text)
        log("      warning: VQA output is not valid JSON; saved raw text to vqa_delta.json")

    del model, processor, inputs, generated_ids
    free_gpu()
    return output_text


def run_llm(
    vqa_output: str,
    num_prompts: int,
    out_dir: Path,
    model_path: str,
    max_new_tokens: int,
    thinking: bool,
    device_map: str | dict,
) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("[2/3] LLM: loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype="auto", device_map=device_map
    )

    prompt = LLM_PROMPT_TEMPLATE.format(vqa_output=vqa_output, n=num_prompts)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    log("[2/3] LLM: generating editing prompts...")
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    thinking_content = ""
    if thinking:
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        if thinking_content:
            save_text(out_dir / "llm_thinking.txt", thinking_content)
    else:
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    raw_path = out_dir / "editing_prompts_raw.txt"
    save_text(raw_path, content)
    log(f"      saved: {raw_path}")

    try:
        parsed = extract_json(content)
        save_json(out_dir / "editing_prompts.json", parsed)
        log(f"      saved: {out_dir / 'editing_prompts.json'}")
    except ValueError:
        save_text(out_dir / "editing_prompts.json", content)
        log("      warning: LLM output is not valid JSON; saved raw text to editing_prompts.json")

    del model, tokenizer, model_inputs, generated_ids
    free_gpu()
    return content


def load_editing_steps(prompts_data: dict) -> list[dict]:
    steps = prompts_data.get("prompts", [])
    if not steps:
        raise ValueError("editing_prompts.json has no 'prompts' array")
    return sorted(steps, key=lambda x: x.get("step", 0))


def run_qwen_edit(
    image_a: Path,
    prompts_data: dict,
    out_dir: Path,
    model_path: str,
    seed: int,
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    negative_prompt: str,
    gpu: int,
) -> list[dict]:
    from PIL import Image

    try:
        from diffusers import QwenImageEditPlusPipeline
    except ImportError as exc:
        raise SystemExit(
            "QwenImageEditPlusPipeline not found in diffusers.\n"
            "Install the latest diffusers from git:\n"
            "  pip install -U git+https://github.com/huggingface/diffusers.git"
        ) from exc

    edits_dir = out_dir / "edits"
    edits_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if gpu < 0 else f"cuda:{gpu}"
    log(f"[3/3] Qwen-Image-Edit: loading model on {device}...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=None)

    steps = load_editing_steps(prompts_data)
    manifest: list[dict] = []
    current_image = image_a  # step 1 input = image A; step k>1 input = step k-1 output

    for item in steps:
        step_num = int(item.get("step", len(manifest) + 1))
        focus = item.get("focus", "")
        landmark_after = item.get("landmark_after", "")
        prompt = item.get("prompt", "").strip()
        if not prompt:
            raise ValueError(f"Step {step_num} has empty prompt")

        input_image = current_image
        out_name = f"step_{step_num:02d}.png"
        out_path = edits_dir / out_name

        log(
            f"[3/3] Qwen-Image-Edit: step {step_num}/{len(steps)} ({focus}) "
            f"input={input_image}"
        )
        pil_image = Image.open(input_image).convert("RGB")
        generator = torch.Generator(device="cpu").manual_seed(seed + step_num)

        with torch.inference_mode():
            output = pipeline(
                image=[pil_image],
                prompt=prompt,
                generator=generator,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
            )

        output.images[0].save(out_path)

        entry = {
            "step": step_num,
            "focus": focus,
            "landmark_after": landmark_after,
            "prompt": prompt,
            "input_image": str(input_image),
            "output_image": str(out_path),
            "chain_mode": "sequential",
        }
        manifest.append(entry)
        current_image = out_path  # next step uses this output as input
        log(f"      saved: {out_path}")

    save_json(edits_dir / "manifest.json", {"steps": manifest})
    log(f"      saved: {edits_dir / 'manifest.json'}")

    del pipeline
    free_gpu()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQA -> LLM -> Qwen-Image-Edit i2i interpolation pipeline"
    )
    parser.add_argument("--image-a", type=Path, required=True, help="Start image (A)")
    parser.add_argument("--image-b", type=Path, required=True, help="End image (B)")
    parser.add_argument("-n", "--num-prompts", type=int, required=True, help="Editing steps")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./out"),
        help="Output directory (default: ./out)",
    )

    parser.add_argument("--skip-vqa", action="store_true", help="Skip VQA; use existing vqa_delta.json")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM; use existing editing_prompts.json")
    parser.add_argument("--skip-edit", action="store_true", help="Skip image editing step")

    parser.add_argument("--vqa-model", default=VQA_MODEL)
    parser.add_argument("--llm-model", default=LLM_MODEL)
    parser.add_argument("--edit-model", default=QWEN_EDIT_MODEL)

    parser.add_argument("--vqa-max-tokens", type=int, default=512)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-thinking", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (default: 0). Set -1 for auto multi-GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image_a.is_file():
        raise SystemExit(f"image A not found: {args.image_a}")
    if not args.image_b.is_file():
        raise SystemExit(f"image B not found: {args.image_b}")

    out_dir = args.out_dir.resolve()
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    image_a = inputs_dir / "image_a.png"
    image_b = inputs_dir / "image_b.png"
    # Always refresh inputs before VQA; stale/corrupt cached copies caused load errors.
    sync_input_image(args.image_a, image_a, "image A", force=not args.skip_vqa)
    sync_input_image(args.image_b, image_b, "image B", force=not args.skip_vqa)

    vqa_path = out_dir / "vqa_delta.json"
    prompts_path = out_dir / "editing_prompts.json"
    device_map = build_device_map(args.gpu)
    if args.gpu >= 0:
        log(f"Using single GPU: cuda:{args.gpu}")
    else:
        log("Using multi-GPU: device_map=auto")

    if args.skip_vqa:
        if not vqa_path.exists():
            raise SystemExit(f"--skip-vqa but missing {vqa_path}")
        vqa_output = vqa_path.read_text(encoding="utf-8").strip()
        log("[1/3] VQA: skipped (using existing vqa_delta.json)")
    else:
        vqa_model = require_model_dir(args.vqa_model, "VQA")
        vqa_output = run_vqa(
            image_a, image_b, out_dir, vqa_model, args.vqa_max_tokens, device_map
        )

    if args.skip_llm:
        if not prompts_path.exists():
            raise SystemExit(f"--skip-llm but missing {prompts_path}")
        llm_output = prompts_path.read_text(encoding="utf-8").strip()
        log("[2/3] LLM: skipped (using existing editing_prompts.json)")
    else:
        llm_model = require_model_dir(args.llm_model, "LLM")
        llm_output = run_llm(
            vqa_output,
            args.num_prompts,
            out_dir,
            llm_model,
            args.llm_max_tokens,
            args.llm_thinking,
            device_map,
        )

    if args.skip_edit:
        log("[3/3] Qwen-Image-Edit: skipped")
        log(f"Done. Outputs in {out_dir}")
        return

    try:
        prompts_data = extract_json(llm_output)
    except ValueError:
        try:
            prompts_data = extract_json(prompts_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise SystemExit(
                f"Could not parse editing prompts from {prompts_path}. "
                "Re-run LLM or fix the JSON before editing."
            ) from exc
    try:
        load_editing_steps(prompts_data)
    except ValueError as exc:
        raise SystemExit(f"Invalid editing prompts in {prompts_path}: {exc}") from exc

    edit_model = require_edit_model_dir(args.edit_model, "Qwen-Image-Edit")
    run_qwen_edit(
        image_a=image_a,
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
    log(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
