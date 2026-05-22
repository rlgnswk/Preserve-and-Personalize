import argparse
import random

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image


SEED = 2025
UNIQUE_TOKEN = "<new1>"
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate one generated image with CLIP-T.")
    parser.add_argument("--gen_image_path", type=str, required=True, help="Path to one generated image.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt corresponding to the generated image.")
    parser.add_argument("--score_csv", type=str)
    return parser


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine(a, b):
    a = a / a.norm()
    b = b / b.norm()
    return torch.dot(a, b).item()


def load_model():
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    return clip_model, clip_preprocess


def get_clip_image_feat(image, clip_model, clip_preprocess):
    return clip_model.encode_image(clip_preprocess(image).unsqueeze(0).to(device)).squeeze(0)


def get_clip_text_feat(text, clip_model):
    return clip_model.encode_text(clip.tokenize([text]).to(device)).squeeze(0)


def score_clip_t(image, prompt, clip_model, clip_preprocess):
    clip_feat = get_clip_image_feat(image, clip_model, clip_preprocess)
    clip_text_feat = get_clip_text_feat(prompt.replace(UNIQUE_TOKEN, "").strip(), clip_model)
    clip_t_score = cosine(clip_feat, clip_text_feat)
    return round(clip_t_score, 4)


def score_clip_t_path(gen_image_path, prompt, clip_model=None, clip_preprocess=None):
    seed_everything()
    if clip_model is None or clip_preprocess is None:
        clip_model, clip_preprocess = load_model()

    image = Image.open(gen_image_path).convert("RGB")
    return score_clip_t(image, prompt, clip_model, clip_preprocess)


def main():
    args = build_arg_parser().parse_args()
    score = score_clip_t_path(args.gen_image_path, args.prompt)
    print("clip_t:", score)

    if args.score_csv:
        row = {
            "img_path": args.gen_image_path,
            "prompt": args.prompt,
            "clip_t": score,
        }
        pd.DataFrame([row]).to_csv(args.score_csv, index=False)
        print(f"Saved score CSV to {args.score_csv}")


if __name__ == "__main__":
    main()
