import argparse
import os
import random

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from peft import PeftModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_lora_dir(weights_path: str) -> str:
    if os.path.isfile(weights_path):
        return os.path.dirname(weights_path)

    if os.path.isdir(weights_path):
        if os.path.isfile(os.path.join(weights_path, "adapter_config.json")):
            return weights_path

        unet_dir = os.path.join(weights_path, "unet")
        if os.path.isfile(os.path.join(unet_dir, "adapter_config.json")):
            return unet_dir

    raise FileNotFoundError(f"Could not find a LoRA adapter directory from: {weights_path}")


# Hyperparameters
base_model_id = "runwayml/stable-diffusion-v1-5"
seed_num = 2025
num_inference_steps = 50
guidance_scale = 7.5


parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, required=True, help="Path to the LoRA adapter directory or file")
parser.add_argument("--prompt", type=str, required=True, help="Prompt used for image generation")
parser.add_argument("--output_path", type=str, default=None, help="Path to save the generated image")
args = parser.parse_args()

set_seed(seed_num)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

script_name = os.path.splitext(os.path.basename(__file__))[0]
out_root = script_name
os.makedirs(out_root, exist_ok=True)

lora_dir = resolve_lora_dir(args.weights)

if args.output_path is None:
    if os.path.basename(os.path.normpath(lora_dir)) == "unet":
        run_name = os.path.basename(os.path.dirname(os.path.normpath(lora_dir)))
    else:
        run_name = os.path.basename(os.path.normpath(lora_dir))
    output_path = os.path.join(out_root, f"{run_name}.png")
else:
    output_path = args.output_path

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=dtype)


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        return images, [False] * images.shape[0]
    return images, False


pipe.safety_checker = disabled_safety_checker
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir, adapter_name="default")
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
