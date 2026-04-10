import argparse
import os
import random
import sys

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler


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
parser.add_argument("--weights", type=str, required=True, help="Path to the Custom Diffusion delta.bin file")
parser.add_argument("--prompt", type=str, required=True, help="Prompt used for image generation")
parser.add_argument("--output_path", type=str, default=None, help="Path to save the generated image")
args = parser.parse_args()

set_seed(seed_num)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from src.diffusers_model_pipeline import CustomDiffusionPipeline

script_name = os.path.splitext(os.path.basename(__file__))[0]
out_root = script_name
os.makedirs(out_root, exist_ok=True)

if not os.path.exists(args.weights):
    raise FileNotFoundError(f"Weights not found: {args.weights}")

if args.output_path is None:
    weight_stem = os.path.splitext(os.path.basename(args.weights))[0]
    if weight_stem == "delta":
        run_name = os.path.basename(os.path.dirname(os.path.abspath(args.weights))) or weight_stem
    else:
        run_name = weight_stem
    output_path = os.path.join(out_root, f"{run_name}.png")
else:
    output_path = args.output_path

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

pipe = CustomDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=dtype)


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        return images, [False] * images.shape[0]
    return images, False


pipe.safety_checker = disabled_safety_checker
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_model(args.weights, compress=False)
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
