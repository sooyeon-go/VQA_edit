import argparse
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

INTERPOLATION_PROMPT = """You are a precise visual analyst for image interpolation.

You will receive TWO images of the SAME object.
Image A = start state, Image B = end state.

Compare them and describe ONLY the differences in pose, size, and angle.
Do NOT describe color, texture, background, or identity.

OUTPUT FORMAT (JSON only, no explanation):
{
  "object_name": "<object>",
  "pose_change": {
    "from": "<pose description of A>",
    "to": "<pose description of B>",
    "delta": "<what specifically changed and in which direction>"
  },
  "size_change": {
    "from": "<size/frame coverage of A>",
    "to": "<size/frame coverage of B>",
    "delta": "<larger/smaller, approximate degree>"
  },
  "angle_change": {
    "from": "<camera angle or object rotation in A>",
    "to": "<camera angle or object rotation in B>",
    "delta": "<direction and degree of angular change>"
  }
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
