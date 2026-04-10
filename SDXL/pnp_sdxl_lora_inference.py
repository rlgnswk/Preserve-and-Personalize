import argparse
import os
import random

import numpy as np
import torch
from diffusers import DiffusionPipeline


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_lora_weights(weights_path: str) -> str:
    if os.path.isfile(weights_path):
        return weights_path

    if os.path.isdir(weights_path):
        root_file = os.path.join(weights_path, "pytorch_lora_weights.safetensors")
        if os.path.isfile(root_file):
            return root_file

    raise FileNotFoundError(f"Could not find LoRA weights from: {weights_path}")


base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
seed_num = 2025
num_inference_steps = 50
guidance_scale = 7.5


parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, required=True, help="Path to the SDXL LoRA weights file or directory")
parser.add_argument("--prompt", type=str, required=True, help="Prompt used for image generation")
parser.add_argument("--output_path", type=str, default=None, help="Path to save the generated image")
args = parser.parse_args()

set_seed(seed_num)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

script_name = os.path.splitext(os.path.basename(__file__))[0]
out_root = script_name
os.makedirs(out_root, exist_ok=True)

lora_weights = resolve_lora_weights(args.weights)

if args.output_path is None:
    if os.path.basename(lora_weights) == "pytorch_lora_weights.safetensors":
        run_name = os.path.basename(os.path.dirname(lora_weights))
    else:
        run_name = os.path.splitext(os.path.basename(lora_weights))[0]
    output_path = os.path.join(out_root, f"{run_name}.png")
else:
    output_path = args.output_path

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

pipe_kwargs = {"torch_dtype": dtype}
if device == "cuda":
    pipe_kwargs["variant"] = "fp16"

pipe = DiffusionPipeline.from_pretrained(base_model_id, **pipe_kwargs)


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        return images, [False] * images.shape[0]
    return images, False


pipe.safety_checker = disabled_safety_checker
pipe.load_lora_weights(lora_weights)
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
