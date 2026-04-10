import argparse
import os
import random

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from safetensors.torch import load_file


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_transformer_weights(weights_path: str) -> str:
    if os.path.isfile(weights_path):
        return weights_path

    if os.path.isdir(weights_path):
        candidate = os.path.join(weights_path, "transformer_trainable.safetensors")
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"Could not find SD3 Custom Diffusion weights from: {weights_path}")


def load_trainable_only(module: torch.nn.Module, weights_path: str, strict: bool = False):
    state_dict = load_file(weights_path, device="cpu")
    incompat = module.load_state_dict(state_dict, strict=False)
    if strict and (incompat.missing_keys or incompat.unexpected_keys):
        raise RuntimeError(
            f"missing={incompat.missing_keys[:8]}..., unexpected={incompat.unexpected_keys[:8]}..."
        )


base_model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
seed_num = 2025
num_inference_steps = 28
guidance_scale = 5.0


parser = argparse.ArgumentParser()
parser.add_argument(
    "--weights",
    type=str,
    required=True,
    help="Path to transformer_trainable.safetensors or a directory containing it",
)
parser.add_argument("--prompt", type=str, required=True, help="Prompt used for image generation")
parser.add_argument("--output_path", type=str, default=None, help="Path to save the generated image")
args = parser.parse_args()

set_seed(seed_num)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

script_name = os.path.splitext(os.path.basename(__file__))[0]
out_root = script_name
os.makedirs(out_root, exist_ok=True)

weights_file = resolve_transformer_weights(args.weights)

if args.output_path is None:
    run_name = os.path.basename(os.path.dirname(weights_file)) if os.path.dirname(weights_file) else "output"
    output_path = os.path.join(out_root, f"{run_name}.png")
else:
    output_path = args.output_path

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

pipe = StableDiffusion3Pipeline.from_pretrained(base_model_id, torch_dtype=dtype)
load_trainable_only(pipe.transformer, weights_file, strict=False)
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
