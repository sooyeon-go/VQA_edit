import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = """You are an image editing prompt engineer specializing in object-level visual transitions.

The SAME object will be transformed step-by-step from state A to state B.
IMPORTANT: These prompts are applied SEQUENTIALLY. Each prompt edits the
OUTPUT of the previous step, NOT the original image. So every prompt must
describe a SMALL change relative to the immediately preceding state.

Below is the visual delta extracted by a VQA model comparing the two images.

VISUAL DELTA:
{vqa_output}

---

Generate {n} editing prompts that gradually move the object from A toward B.

RULES:
- Write each prompt in NATURAL, PLAIN language, the way a person would describe
  a pose or direction. NO precise degrees or percentages.
  GOOD: "the cat lifts its head slightly", "the cat is now sitting and facing right",
        "the chair's backrest turns to face forward"
  BAD:  "rotate the cat 23° clockwise", "scale to 47% of the frame"
- Each prompt edits the PREVIOUS step's result, so describe only a SMALL step of change
- The tracked landmark must change direction GRADUALLY and in ONE consistent direction.
  Never let it jump or reverse. If the landmark goes from facing left to facing right,
  it must pass through facing-forward in between — do not skip straight across.
- In EVERY prompt, clearly state the landmark's current direction/pose after this step
  (e.g. "the cat's head now faces forward")
- Steps do NOT need to be evenly spaced; add more steps where the change is biggest
- Keep prompts short and concrete

OUTPUT FORMAT (JSON only, no explanation):
{{
  "landmark": "<the tracked feature>",
  "prompts": [
    {{
      "step": 1,
      "focus": "<pose | direction | size | combined>",
      "landmark_after": "<plain description of where the landmark faces/sits after this step>",
      "prompt": "<short, plain-language editing instruction for this small step>"
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
