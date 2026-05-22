import argparse
import os
import random

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image


SEED = 2025
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate one generated image with CLIP-I.")
    parser.add_argument("--gen_image_path", type=str, required=True, help="Path to one generated image.")
    parser.add_argument("--ref_images_dir", type=str, required=True, help="Path to reference images directory.")
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt saved to CSV; not used for CLIP-I.")
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


def load_ref_images(ref_images_dir):
    ref_files = [f for f in os.listdir(ref_images_dir) if f.lower().endswith(("jpg", "png"))]
    return [Image.open(os.path.join(ref_images_dir, f)).convert("RGB") for f in ref_files]


def get_clip_image_feat(image, clip_model, clip_preprocess):
    return clip_model.encode_image(clip_preprocess(image).unsqueeze(0).to(device)).squeeze(0)


def get_ref_feats(ref_images, clip_model, clip_preprocess):
    return [get_clip_image_feat(img, clip_model, clip_preprocess) for img in ref_images]


def score_clip_i(image, clip_ref_feats, clip_model, clip_preprocess):
    clip_feat = get_clip_image_feat(image, clip_model, clip_preprocess)
    clip_i_score = np.mean([cosine(clip_feat, ref) for ref in clip_ref_feats])
    return round(clip_i_score, 4)


def score_clip_i_path(gen_image_path, ref_images_dir, clip_model=None, clip_preprocess=None):
    seed_everything()
    if clip_model is None or clip_preprocess is None:
        clip_model, clip_preprocess = load_model()

    ref_images = load_ref_images(ref_images_dir)
    clip_ref_feats = get_ref_feats(ref_images, clip_model, clip_preprocess)
    image = Image.open(gen_image_path).convert("RGB")
    return score_clip_i(image, clip_ref_feats, clip_model, clip_preprocess)


def main():
    args = build_arg_parser().parse_args()
    score = score_clip_i_path(args.gen_image_path, args.ref_images_dir)
    print("clip_i:", score)

    if args.score_csv:
        row = {
            "img_path": args.gen_image_path,
            "ref_images_dir": args.ref_images_dir,
            "prompt": args.prompt,
            "clip_i": score,
        }
        pd.DataFrame([row]).to_csv(args.score_csv, index=False)
        print(f"Saved score CSV to {args.score_csv}")


if __name__ == "__main__":
    main()
