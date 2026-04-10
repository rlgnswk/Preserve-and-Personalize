# [ICLR2026] Preserve and Personalize: Personalized Text-to-Image Diffusion Models without Distributional Drift

[Paper](https://openreview.net/forum?id=2ge1Y6DWPw) | [Project Page](https://rlgnswk.github.io/PreserveAndPersonalize_ProjectPage/)


## Data

The data required for this project can be obtained from
[google/dreambooth](https://github.com/google/dreambooth).

Please download the DreamBooth data from the official repository and place
the downloaded files under `data/` in this repository.

Expected location:

```bash
Preserve-and-Personalize/data/
```


## Toy Experiments

The toy experiments are located in `toy/` and include three variants:

- `toy_naive.py`: naive personalization loss
- `toy_db.py`: prior preservation loss (DreamBooth)
- `toy_ours.py`: our method

Run all three experiments in sequence:

```bash
conda activate pnp
cd /home/s2/gihoonkim/gihoon/shared_gihoon/diffusion/Preserve-and-Personalize/toy
python toy_naive.py
python toy_db.py
python toy_ours.py
```
Each script saves its figures in a folder with the same name as the script:

- `toy_naive/`
- `toy_db/`
- `toy_ours/`

The saved figures are:

- `data_distribution.png`
- `target_data.png`
- `pretrained_samples.png`
- `personalized_samples.png`

## BibTeX

```bibtex
@inproceedings{kim2026preserveandpersonalize,
  title     = {Preserve and Personalize: Personalized Text-to-Image Diffusion Models without Distributional Drift},
  author    = {Gihoon Kim and Hyungjin Park and Taesup Kim},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
