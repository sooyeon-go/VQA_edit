import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = """You are an image editing prompt engineer specializing in object-level visual transitions.

The SAME object will be transformed step-by-step from state A to state B.
IMPORTANT: These prompts are applied SEQUENTIALLY. Each prompt edits the
OUTPUT of the previous step, NOT the original image. Each step is a SMALL
incremental change relative to the immediately preceding state.

VISUAL DELTA:
{vqa_output}

---

Generate {n} editing prompts that gradually move the object from A toward B.

CRITICAL RULES:
1. DIRECTION: Use explicit words in every prompt — "left", "right", "forward", "away".
   NEVER use vague terms like "to the side" or "sideways".
2. CONSISTENCY: The landmark must move in ONE direction only, never jump or reverse.
   If the landmark goes left -> right, it must pass through forward in between.
3. AGREEMENT: "prompt" and "landmark_after" MUST describe the same state.
   The instruction must result in exactly what "landmark_after" states.
4. PROGRESS: Each step's "pose_after" must show clear progress from the previous step.
   Two consecutive steps must NOT have the same "pose_after".
5. LANGUAGE: Plain, natural language only. No degrees or percentages.
   GOOD: "gently turn the cat's head to the right"
         "tilt the cat's body slightly to the right"
         "fully roll the cat onto its side while keeping its head turned right"
   BAD:  "rotate 30°", "turn to the side"

OUTPUT FORMAT (JSON only, no explanation):
{{
  "landmark": "<the tracked feature>",
  "direction_path": "<full ordered direction sequence, e.g. 'forward -> right'>",
  "prompts": [
    {{
      "step": 1,
      "focus": "<pose | direction | size | combined>",
      "landmark_after": "<explicit direction the landmark faces: left/forward/right/away>",
      "pose_after": "<plain description of body pose after this step>",
      "progress": "<rough fraction toward B, e.g. '1/{n}', '3/{n}'>",
      "prompt": "<short, plain-language incremental editing instruction>"
    }}
  ]
}}"""

parser = argparse.ArgumentParser(
    description="Generate progressive editing prompts from VQA visual delta"
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--vqa-file", type=Path, help="Path to VQA JSON/text output")
group.add_argument("--vqa-text", type=str, help="VQA output as inline string")
parser.add_argument("-n", "--num-prompts", type=int, required=True, help="Number of editing steps")
parser.add_argument("--max-new-tokens", type=int, default=4096)
parser.add_argument(
    "--thinking",
    action="store_true",
    help="Enable Qwen3 thinking mode (default: off for JSON output)",
)
args = parser.parse_args()

if args.vqa_file:
    vqa_output = args.vqa_file.read_text(encoding="utf-8").strip()
else:
    vqa_output = args.vqa_text.strip()

prompt = PROMPT_TEMPLATE.format(vqa_output=vqa_output, n=args.num_prompts)

model_name = "/hdd/sy/models/qwen3-32b-weights/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=args.thinking,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

if args.thinking:
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print("thinking content:", thinking_content)
    print("content:", content)
else:
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    print(content)
