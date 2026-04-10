# SDXL

## SDXL Custom Diffusion

### Training

The Custom Diffusion training script is [pnp_sdxl_custom_diffusion.py](./pnp_sdxl_custom_diffusion.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SDXL
accelerate launch pnp_sdxl_custom_diffusion.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --instance_data_dir path/to/Preserve-and-Personalize/data/dreambooth/dataset/cat \
  --instance_prompt "photo of a <new1> cat" \
  --resolution 512 \
  --train_batch_size 2 \
  --learning_rate 5e-5 \
  --lr_warmup_steps 0 \
  --max_train_steps 250 \
  --gradient_checkpointing \
  --scale_lr \
  --hflip \
  --l2_reg_weight 50 \
  --seed 2025 \
  --modifier_token "<new1>"
```

The trained UNet is saved to:

```bash
pnp_sdxl_custom_diffusion/unet/
```

### Inference

The Custom Diffusion inference script is [pnp_sdxl_custom_diffusion_inference.py](./pnp_sdxl_custom_diffusion_inference.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SDXL
python pnp_sdxl_custom_diffusion_inference.py \
  --weights path/to/Preserve-and-Personalize/SDXL/pnp_sdxl_custom_diffusion/unet \
  --prompt "a <new1> cat in the snow" \
  --output_path path/to/output/cat_in_the_snow.png
```

## SDXL LoRA

### Training

The LoRA training script is [pnp_sdxl_lora.py](./pnp_sdxl_lora.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SDXL
accelerate launch pnp_sdxl_lora.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
  --instance_data_dir path/to/Preserve-and-Personalize/data/dreambooth/dataset/cat \
  --instance_prompt "a photo of a sks cat" \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --max_train_steps 500 \
  --mixed_precision fp16 \
  --l2_reg_weight 1e-6 \
  --seed 0
```

The trained LoRA weights are saved to:

```bash
pnp_sdxl_lora/pytorch_lora_weights.safetensors
```

### Inference

The LoRA inference script is [pnp_sdxl_lora_inference.py](./pnp_sdxl_lora_inference.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SDXL
python pnp_sdxl_lora_inference.py \
  --weights path/to/Preserve-and-Personalize/SDXL/pnp_sdxl_lora/pytorch_lora_weights.safetensors \
  --prompt "a sks cat in the snow" \
  --output_path path/to/output/cat_in_the_snow.png
```
