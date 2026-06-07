import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = """You are an image editing prompt engineer specializing in object-level visual transitions.

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
