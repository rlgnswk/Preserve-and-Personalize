# Evaluation Scripts

These scripts evaluate generated DreamBooth-style images with the same metric details used in the SD15/SDXL experiment code.

## Single-Image Metrics

Run from the repository root:

```bash
cd /home/s2/gihoonkim/gihoon/shared_gihoon/project_github/Preserve-and-Personalize
```

CLIP-I compares one generated image with the reference images for the same subject:

```bash
python eval/clipi.py \
  --gen_image_path /path/to/generated/duck_toy/prompt_12_img_02.png \
  --ref_images_dir data/dreambooth/dataset/duck_toy \
  --score_csv /path/to/output/clipi_one.csv
```

CLIP-T compares one generated image with the corresponding class-token prompt:

```bash
python eval/clipt.py \
  --gen_image_path /path/to/generated/duck_toy/prompt_12_img_02.png \
  --prompt "a toy with the Eiffel Tower in the background" \
  --score_csv /path/to/output/clipt_one.csv
```

DINO compares one generated image with the reference images for the same subject:

```bash
python eval/dino.py \
  --gen_image_path /path/to/generated/duck_toy/prompt_12_img_02.png \
  --ref_images_dir data/dreambooth/dataset/duck_toy \
  --score_csv /path/to/output/dino_one.csv
```

## Full-Folder Metric

This is provided as a reference workflow for generated image folders that already follow the expected prompt and file naming convention. The generation step must save images in the matching layout before `metric.py` can evaluate them.

- `clip_i`: CLIP image-image similarity between one generated image and its reference images.
- `clip_t`: CLIP image-text similarity between one generated image and its class-token prompt.
- `dino`: DINO image-image similarity between one generated image and its reference images.
- `metric.py`: full-folder evaluation that imports `clipi.py`, `clipt.py`, and `dino.py`, then writes per-image scores and average scores.

`metric.py` expects generated images to be arranged as:

```text
<generated_images_dir>/<subject>/prompt_XX_img_YY.png
```

Reference images are expected at:

```text
<ref_data_root>/<subject>/*.jpg|*.png
```

Use `--ref_data_root` to point to the reference dataset. If the dataset is in this repository, a typical value is `data/dreambooth/dataset`.

Example for one generated evaluation folder:

```bash
python eval/metric.py \
  --generated_images_dir /path/to/generated_eval_images \
  --ref_data_root data/dreambooth/dataset \
  --score_csv /path/to/generated_eval_images/classToken.csv
```

This writes:

```text
classToken.csv
classToken_ave.csv
```

`classToken.csv` columns:

```text
subject,subname,prompt_id,img_id,prompt,clip_i,dino,clip_t,img_path
```

`classToken_ave.csv` columns:

```text
clip_i,clip_t,dino
```
