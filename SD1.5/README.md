# SD1.5

## SD1.5 Full-Finetune

### Training

The full-finetuning script is [pnp_sd15_full.py](./pnp_sd15_full.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SD1.5
python pnp_sd15_full.py \
  --subject_folder path/to/Preserve-and-Personalize/data/dreambooth/dataset/cat \
  --prompt "a photo of a sks cat"
```

The trained checkpoint is saved to:

```bash
pnp_sd15_full/pnp_sd15_full.ckpt
```

### Inference

The inference script is [pnp_sd15_inference.py](./pnp_sd15_inference.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SD1.5
python pnp_sd15_inference.py \
  --ckpt_path path/to/Preserve-and-Personalize/SD1.5/pnp_sd15_full/pnp_sd15_full.ckpt \
  --test_prompt "a sks cat in the snow" \
  --output_path path/to/output/cat_in_the_snow.png
```

## SD1.5 Custom Diffusion

### Training

The Custom Diffusion training script is [pnp_sd15_custom_diffusion.py](./pnp_sd15_custom_diffusion.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SD1.5
accelerate launch pnp_sd15_custom_diffusion.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --instance_data_dir path/to/Preserve-and-Personalize/data/dreambooth/dataset/cat \
  --instance_prompt "photo of a <new1> cat" \
  --train_batch_size 2 \
  --learning_rate 3e-5 \
  --lr_warmup_steps 0 \
  --max_train_steps 250 \
  --gradient_checkpointing \
  --scale_lr \
  --hflip \
  --seed 2025 \
  --modifier_token "<new1>" \
  --l2_weight 175
```

The trained delta is saved to:

```bash
pnp_sd15_custom_diffusion/delta.bin
```

### Inference

The Custom Diffusion inference script is [pnp_sd15_custom_diffusion_inference.py](./pnp_sd15_custom_diffusion_inference.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SD1.5
python pnp_sd15_custom_diffusion_inference.py \
  --weights path/to/Preserve-and-Personalize/SD1.5/pnp_sd15_custom_diffusion/delta.bin \
  --prompt "a <new1> cat in the snow" \
  --output_path path/to/output/cat_in_the_snow.png
```

## SD1.5 LoRA

### Training

The LoRA training script is [pnp_sd15_lora.py](./pnp_sd15_lora.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SD1.5
accelerate launch pnp_sd15_lora.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --instance_data_dir path/to/Preserve-and-Personalize/data/dreambooth/dataset/cat \
  --instance_prompt "a photo of a sks cat" \
  --lr_warmup_steps 0 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 27 \
  --learning_rate 1e-4 \
  --gradient_checkpointing \
  --max_train_steps 500 \
  --adam_weight_decay 0.0 \
  --l2_weight 1e-7 \
  --seed 0 \
  --no_tracemalloc
```

The trained LoRA adapter is saved to:

```bash
pnp_sd15_lora/unet/
```

### Inference

The LoRA inference script is [pnp_sd15_lora_inference.py](./pnp_sd15_lora_inference.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SD1.5
python pnp_sd15_lora_inference.py \
  --weights path/to/Preserve-and-Personalize/SD1.5/pnp_sd15_lora/unet \
  --prompt "a sks cat in the snow" \
  --output_path path/to/output/cat_in_the_snow.png
```
