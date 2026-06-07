import argparse
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

INTERPOLATION_PROMPT = """You are a precise visual analyst for image interpolation.

You will receive TWO images of the SAME object.
Image A = start state, Image B = end state.

First, pick ONE stable, easily-identifiable landmark feature of the object
(e.g. for a cat: the head; for a chair: the backrest; for a car: the front).

Then compare the two images and describe the differences using NATURAL, PLAIN language.
Do NOT use precise degrees or percentages.
Do NOT describe color, texture, background, or identity.

For landmark direction, use only these words:
- horizontal: left / forward / right / away (back to camera)
- pose: sitting / standing / lying on side / head up / head down / crouching, etc.

OUTPUT FORMAT (JSON only, no explanation):
{
  "object_name": "<object>",
  "landmark": "<the single tracked feature, e.g. 'cat head', 'chair backrest'>",
  "state_A": {
    "landmark_direction": "<which way the landmark faces in A: left/forward/right/away>",
    "body_pose": "<plain description of the body pose in A>"
  },
  "state_B": {
    "landmark_direction": "<which way the landmark faces in B: left/forward/right/away>",
    "body_pose": "<plain description of the body pose in B>"
  },
  "main_changes": "<what changes between A and B in plain words, mentioning direction explicitly>"
}"""

parser = argparse.ArgumentParser(description="Qwen3-VL pose/size/angle diff between two images")
parser.add_argument("--image-a", required=True, help="Start state image (Image A)")
parser.add_argument("--image-b", required=True, help="End state image (Image B)")
parser.add_argument("--max-new-tokens", type=int, default=512)
args = parser.parse_args()

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/hdd/sy/models/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": args.image_a},
            {"type": "image", "image": args.image_b},
            {"type": "text", "text": INTERPOLATION_PROMPT},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
