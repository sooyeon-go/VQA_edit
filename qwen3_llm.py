import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = """You are an image editing prompt engineer specializing in object-level visual transitions.

The SAME object transitions from state A to state B.
Below is the visual delta extracted by a VQA model comparing the two images.

VISUAL DELTA:
{vqa_output}

---

Generate {n} editing prompts that progressively move the object from A toward B.

RULES:
- Each prompt is a standalone image editing instruction
- Steps do NOT need to be evenly spaced; cluster more steps where the change is largest or most complex
- Each prompt must describe the object's ABSOLUTE state at that step
  (e.g. "rotated ~30° clockwise from upright" — not "rotate it a bit more")
- Use directional and approximate degree language
  (e.g. "facing ~45° to the right", "occupies roughly 60% of the frame", "tilted ~20° forward")

OUTPUT FORMAT (JSON only, no explanation):
{{
  "prompts": [
    {{
      "step": 1,
      "focus": "<pose | size | angle | combined>",
      "prompt": "<standalone editing instruction describing absolute state>"
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
