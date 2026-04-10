import argparse
import os
import random

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_lora_weights(weights_path: str):
    if os.path.isfile(weights_path):
        return os.path.dirname(weights_path) or ".", os.path.basename(weights_path)

    if os.path.isdir(weights_path):
        weight_name = "pytorch_lora_weights.safetensors"
        if os.path.isfile(os.path.join(weights_path, weight_name)):
            return weights_path, weight_name

    raise FileNotFoundError(f"Could not find SD3 LoRA weights from: {weights_path}")


base_model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
seed_num = 2025
num_inference_steps = 28
guidance_scale = 5.0


parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, required=True, help="Path to the SD3 LoRA weights file or directory")
parser.add_argument("--prompt", type=str, required=True, help="Prompt used for image generation")
parser.add_argument("--output_path", type=str, default=None, help="Path to save the generated image")
args = parser.parse_args()

set_seed(seed_num)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

script_name = os.path.splitext(os.path.basename(__file__))[0]
out_root = script_name
os.makedirs(out_root, exist_ok=True)

lora_dir, weight_name = resolve_lora_weights(args.weights)

if args.output_path is None:
    run_name = os.path.basename(os.path.normpath(lora_dir))
    output_path = os.path.join(out_root, f"{run_name}.png")
else:
    output_path = args.output_path

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

pipe = StableDiffusion3Pipeline.from_pretrained(base_model_id, torch_dtype=dtype)
pipe.load_lora_weights(lora_dir, weight_name=weight_name)
pipe = pipe.to(device)

generator = torch.Generator(device=device).manual_seed(seed_num)

with torch.no_grad():
    image = pipe(
        args.prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

image.save(output_path)
print(f"Image saved at {output_path}")
