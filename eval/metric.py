import argparse
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import clipi
import clipt
import dino


SEED = 2025
UNIQUE_TOKEN = "<new1>"


object_classes = [
    "backpack", "backpack_dog", "bear_plushie", "berry_bowl", "can", "candle",
    "clock", "colorful_sneaker", "duck_toy", "fancy_boot", "grey_sloth_plushie",
    "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon",
    "robot_toy", "shiny_sneaker", "teapot", "vase", "wolf_plushie",
]


subject_to_class = {
    "backpack": "backpack",
    "backpack_dog": "backpack",
    "bear_plushie": "stuffed animal",
    "berry_bowl": "bowl",
    "can": "can",
    "candle": "candle",
    "cat": "cat",
    "cat2": "cat",
    "clock": "clock",
    "colorful_sneaker": "sneaker",
    "dog": "dog",
    "dog2": "dog",
    "dog3": "dog",
    "dog5": "dog",
    "dog6": "dog",
    "dog7": "dog",
    "dog8": "dog",
    "duck_toy": "toy",
    "fancy_boot": "boot",
    "grey_sloth_plushie": "stuffed animal",
    "monster_toy": "toy",
    "pink_sunglasses": "glasses",
    "poop_emoji": "toy",
    "rc_car": "toy",
    "red_cartoon": "cartoon",
    "robot_toy": "toy",
    "shiny_sneaker": "sneaker",
    "teapot": "teapot",
    "vase": "vase",
    "wolf_plushie": "stuffed animal",
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate generated images with CLIP and DINO metrics.")
    parser.add_argument("--generated_images_dir", type=str, required=True, help="Path to the generated images directory.")
    parser.add_argument("--ref_data_root", type=str, required=True, help="Path to reference dataset root.")
    parser.add_argument("--score_csv", type=str)
    return parser


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt_list(unique_token, class_token, is_object=True):
    base = [
        "a {0} in the jungle",
        "a {0} in the snow",
        "a {0} on the beach",
        "a {0} on a cobblestone street",
        "a {0} on top of pink fabric",
        "a {0} on top of a wooden floor",
        "a {0} with a city in the background",
        "a {0} with a mountain in the background",
        "a {0} with a blue house in the background",
        "a {0} on top of a purple rug in a forest",
    ]
    if is_object:
        base += [
            "a {0} with a wheat field in the background",
            "a {0} with a tree and autumn leaves in the background",
            "a {0} with the Eiffel Tower in the background",
            "a {0} floating on top of water",
            "a {0} floating in an ocean of milk",
            "a {0} on top of green grass with sunflowers around it",
            "a {0} on top of a mirror",
            "a {0} on top of the sidewalk in a crowded street",
            "a {0} on top of a dirt road",
            "a {0} on top of a white rug",
        ]
    else:
        base += [
            "a {0} wearing a red hat",
            "a {0} wearing a santa hat",
            "a {0} wearing a rainbow scarf",
            "a {0} wearing a black top hat and a monocle",
            "a {0} in a chef outfit",
            "a {0} in a firefighter outfit",
            "a {0} in a police outfit",
            "a {0} wearing pink glasses",
            "a {0} wearing a yellow shirt",
            "a {0} in a purple wizard outfit",
        ]
    base += [
        "a red {0}",
        "a purple {0}",
        "a shiny {0}",
        "a wet {0}",
        "a cube shaped {0}",
    ]
    return [prompt.format(class_token) for prompt in base]


def default_score_csv(generated_images_dir):
    last_folder = os.path.basename(os.path.normpath(generated_images_dir))
    return last_folder + "classToken.csv"


def average_csv_path(score_csv):
    base_name, ext = os.path.splitext(score_csv)
    return base_name + "_ave" + ext


def evaluate_all(generated_images_dir, ref_data_root):
    seed_everything()

    clip_model, clip_preprocess = clipi.load_model()
    dino_model = dino.load_model()

    results = []
    for subject_name in tqdm(os.listdir(generated_images_dir), desc="Evaluating subjects"):
        subject_path = os.path.join(generated_images_dir, subject_name)
        sub_dir = subject_path
        subname = subject_name
        if not os.path.isdir(subject_path):
            continue
        if not os.path.isdir(sub_dir):
            continue

        subject_name = re.sub(r"\d+$", "", subname)
        class_token = subject_to_class[subject_name]
        is_obj = subject_name in object_classes
        prompt_list = build_prompt_list(UNIQUE_TOKEN, class_token, is_obj)

        ref_images_dir = os.path.join(ref_data_root, subname)
        if not os.path.exists(ref_images_dir):
            print(f"Reference images for subject '{subname}' not found. Skipping evaluation for this subject.")
            continue

        ref_images = clipi.load_ref_images(ref_images_dir)
        clip_ref_feats = clipi.get_ref_feats(ref_images, clip_model, clip_preprocess)
        dino_ref_feats = dino.get_ref_feats(ref_images, dino_model)

        for filename in os.listdir(sub_dir):
            if not filename.lower().endswith(("jpg", "png")):
                continue
            m = re.match(r"prompt_(\d+)_img_(\d+)\.png", filename)
            if not m:
                continue

            p_idx = int(m.group(1))
            img_id = int(m.group(2))
            prompt = prompt_list[p_idx]
            img_path = os.path.join(sub_dir, filename)
            image = Image.open(img_path).convert("RGB")

            clip_i_score = clipi.score_clip_i(image, clip_ref_feats, clip_model, clip_preprocess)
            dino_score = dino.score_dino(image, dino_ref_feats, dino_model)
            clip_t_score = clipt.score_clip_t(image, prompt, clip_model, clip_preprocess)

            results.append({
                "subject": subject_name,
                "subname": subname,
                "prompt_id": p_idx,
                "img_id": img_id,
                "prompt": prompt,
                "clip_i": clip_i_score,
                "dino": dino_score,
                "clip_t": clip_t_score,
                "img_path": img_path,
            })

    return pd.DataFrame(results)


def main():
    args = build_arg_parser().parse_args()
    score_csv = args.score_csv or default_score_csv(args.generated_images_dir)
    print("Using score_csv:", score_csv)

    df = evaluate_all(args.generated_images_dir, args.ref_data_root)
    df.to_csv(score_csv, index=False)
    print(f"Saved score CSV to {score_csv}")

    avg_df = pd.DataFrame({
        "clip_i": [round(df["clip_i"].mean(), 4)],
        "clip_t": [round(df["clip_t"].mean(), 4)],
        "dino": [round(df["dino"].mean(), 4)],
    })
    ave_csv = average_csv_path(score_csv)
    avg_df.to_csv(ave_csv, index=False)
    print(f"Saved average scores CSV to {ave_csv}")


if __name__ == "__main__":
    main()
