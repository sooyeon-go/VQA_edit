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

2. WAYPOINTS: If the landmark needs to cross from one side to the opposite
   (e.g. right -> left, or away -> forward), it MUST pass through a natural
   intermediate state first. Do not jump directly across.
   Required waypoints by transition type:
   - right -> left : must pass through "facing forward"
   - left -> right : must pass through "facing forward"
   - away -> forward : must pass through "facing slightly to one side"
   - sitting -> lying : must pass through "leaning to the side"
   Spend at least ONE full step at each waypoint before continuing.

3. CONSISTENCY: The landmark moves in one smooth continuous arc. Never reverse
   direction. Every step must be closer to B than the previous step.

4. AGREEMENT: "prompt" and "landmark_after" MUST describe the same state.
   The instruction must result in exactly what "landmark_after" states.

5. PROGRESS: Each step's "pose_after" must show clear progress from the previous
   step. Two consecutive steps must NOT have the same "pose_after".

6. LANGUAGE: Plain, natural language only. No degrees or percentages.
   GOOD: "the cat's head now faces forward"
         "the cat settles into a forward-facing sit"
         "the cat's head begins to turn toward the left"
   BAD:  "rotate 30°", "turn to the side"

OUTPUT FORMAT (JSON only, no explanation):
{{
  "landmark": "<the tracked feature>",
  "direction_path": "<full ordered path including waypoints, e.g. 'right -> forward -> left'>",
  "prompts": [
    {{
      "step": 1,
      "focus": "<pose | direction | size | combined>",
      "landmark_after": "<explicit direction: left/forward/right/away>",
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
