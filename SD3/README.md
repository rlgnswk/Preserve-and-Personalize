# SD3

## SD3 Custom Diffusion

### Training

The Custom Diffusion training script is [pnp_sd3_custom_diffusion.py](./pnp_sd3_custom_diffusion.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SD3
accelerate launch pnp_sd3_custom_diffusion.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
  --instance_data_dir path/to/Preserve-and-Personalize/data/dreambooth/dataset/cat \
  --instance_prompt "a photo of a sks cat" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --max_train_steps 500 \
  --mixed_precision fp16 \
  --l2_reg_weight 10.1 \
  --seed 0
```

The trained Custom Diffusion weights are saved to:

```bash
pnp_sd3_custom_diffusion/transformer_trainable.safetensors
```

### Inference

The Custom Diffusion inference script is [pnp_sd3_custom_diffusion_inference.py](./pnp_sd3_custom_diffusion_inference.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SD3
python pnp_sd3_custom_diffusion_inference.py \
  --weights path/to/Preserve-and-Personalize/SD3/pnp_sd3_custom_diffusion/transformer_trainable.safetensors \
  --prompt "a sks cat in the snow" \
  --output_path path/to/output/cat_in_the_snow.png
```

## SD3 LoRA

### Training

The LoRA training script is [pnp_sd3_lora.py](./pnp_sd3_lora.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SD3
accelerate launch pnp_sd3_lora.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
  --instance_data_dir path/to/Preserve-and-Personalize/data/dreambooth/dataset/cat \
  --instance_prompt "a photo of a sks cat" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 4e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --max_train_steps 500 \
  --mixed_precision fp16 \
  --l2_reg_weight 15.0 \
  --seed 0
```

The trained LoRA weights are saved to:

```bash
pnp_sd3_lora/pytorch_lora_weights.safetensors
```

### Inference

The LoRA inference script is [pnp_sd3_lora_inference.py](./pnp_sd3_lora_inference.py).

Example:

```bash
conda activate pnp
cd path/to/Preserve-and-Personalize/SD3
python pnp_sd3_lora_inference.py \
  --weights path/to/Preserve-and-Personalize/SD3/pnp_sd3_lora/pytorch_lora_weights.safetensors \
  --prompt "a sks cat in the snow" \
  --output_path path/to/output/cat_in_the_snow.png
```
