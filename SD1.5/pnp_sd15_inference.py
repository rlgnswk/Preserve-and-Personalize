import argparse
import os
import random

import numpy as np
import torch
from diffusers import StableDiffusionPipeline


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Hyperparameters
base_model_id = "runwayml/stable-diffusion-v1-5"
seed_num = 2025
num_inference_steps = 50
guidance_scale = 7.5


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the fine-tuned UNet checkpoint")
parser.add_argument("--test_prompt", type=str, required=True, help="Test prompt used for image generation")
parser.add_argument("--output_path", type=str, default=None, help="Path to save the generated image")
args = parser.parse_args()

set_seed(seed_num)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

script_name = os.path.splitext(os.path.basename(__file__))[0]
out_root = script_name
os.makedirs(out_root, exist_ok=True)

if args.output_path is None:
    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    output_path = os.path.join(out_root, f"{ckpt_stem}.png")
else:
    output_path = args.output_path

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

if not os.path.exists(args.ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=dtype)


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        return images, [False] * images.shape[0]
    return images, False


pipe.safety_checker = disabled_safety_checker

state_dict = torch.load(args.ckpt_path, map_location="cpu")
pipe.unet.load_state_dict(state_dict)
pipe = pipe.to(device)

generator = torch.Generator(device=device).manual_seed(seed_num)

with torch.no_grad():
    image = pipe(
        prompt=args.test_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

image.save(output_path)
print(f"Image saved at {output_path}")
