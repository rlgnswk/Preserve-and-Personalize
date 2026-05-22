import argparse
import os
import random

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms as T


SEED = 2025
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate one generated image with DINO.")
    parser.add_argument("--gen_image_path", type=str, required=True, help="Path to one generated image.")
    parser.add_argument("--ref_images_dir", type=str, required=True, help="Path to reference images directory.")
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt saved to CSV; not used for DINO.")
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
    return timm.create_model("vit_small_patch16_224_dino", pretrained=True).to(device).eval()


def load_ref_images(ref_images_dir):
    ref_files = [f for f in os.listdir(ref_images_dir) if f.lower().endswith(("jpg", "png"))]
    return [Image.open(os.path.join(ref_images_dir, f)).convert("RGB") for f in ref_files]


dino_preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])


def get_dino_feat(image, dino_model):
    img_tensor = dino_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = dino_model.forward_features(img_tensor)
        if isinstance(outputs, dict):
            cls_embedding = outputs["x_norm_clstoken"]
        else:
            cls_embedding = outputs[:, 0]
    return cls_embedding.squeeze(0)


def get_ref_feats(ref_images, dino_model):
    return [get_dino_feat(img, dino_model) for img in ref_images]


def score_dino(image, dino_ref_feats, dino_model):
    dino_feat = get_dino_feat(image, dino_model)
    dino_score = np.mean([cosine(dino_feat, ref) for ref in dino_ref_feats])
    return round(dino_score, 4)


def score_dino_path(gen_image_path, ref_images_dir, dino_model=None):
    seed_everything()
    if dino_model is None:
        dino_model = load_model()

    ref_images = load_ref_images(ref_images_dir)
    dino_ref_feats = get_ref_feats(ref_images, dino_model)
    image = Image.open(gen_image_path).convert("RGB")
    return score_dino(image, dino_ref_feats, dino_model)


def main():
    args = build_arg_parser().parse_args()
    score = score_dino_path(args.gen_image_path, args.ref_images_dir)
    print("dino:", score)

    if args.score_csv:
        row = {
            "img_path": args.gen_image_path,
            "ref_images_dir": args.ref_images_dir,
            "prompt": args.prompt,
            "dino": score,
        }
        pd.DataFrame([row]).to_csv(args.score_csv, index=False)
        print(f"Saved score CSV to {args.score_csv}")


if __name__ == "__main__":
    main()
