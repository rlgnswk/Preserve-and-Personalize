import argparse
import os
import random

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline, UNet2DConditionModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_unet_dir(weights_path: str) -> str:
    if os.path.isdir(weights_path) and os.path.isfile(os.path.join(weights_path, "config.json")):
        return weights_path

    if os.path.isdir(weights_path):
        unet_dir = os.path.join(weights_path, "unet")
        if os.path.isfile(os.path.join(unet_dir, "config.json")):
            return unet_dir

    raise FileNotFoundError(f"Could not find a saved UNet directory from: {weights_path}")


base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
seed_num = 2025
num_inference_steps = 50
guidance_scale = 5.0


parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, required=True, help="Path to the saved SDXL Custom Diffusion UNet")
parser.add_argument("--prompt", type=str, required=True, help="Prompt used for image generation")
parser.add_argument("--output_path", type=str, default=None, help="Path to save the generated image")
args = parser.parse_args()

set_seed(seed_num)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

script_name = os.path.splitext(os.path.basename(__file__))[0]
out_root = script_name
os.makedirs(out_root, exist_ok=True)

unet_dir = resolve_unet_dir(args.weights)

if args.output_path is None:
    if os.path.basename(os.path.normpath(unet_dir)) == "unet":
        run_name = os.path.basename(os.path.dirname(os.path.normpath(unet_dir)))
    else:
        run_name = os.path.basename(os.path.normpath(unet_dir))
    output_path = os.path.join(out_root, f"{run_name}.png")
else:
    output_path = args.output_path

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

pipe_kwargs = {"torch_dtype": dtype}
if device == "cuda":
    pipe_kwargs["variant"] = "fp16"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, **pipe_kwargs)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = lambda images, **kwargs: (images, False)
pipe.unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=dtype).to(device)
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
