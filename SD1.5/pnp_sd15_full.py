import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline  # Use the appropriate pipeline
import numpy as np 
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import random
import copy
import argparse

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
from accelerate import Accelerator

accelerator = Accelerator() # mixed_precision="fp16"
device = accelerator.device


# Hyperparameters
seed_num = 2025
num_train_steps = 1000
train_noise_steps = 1000
particle4dmd = 1
batch_size = 1
learning_rate = 2e-6
weight = 500

set_seed(seed_num)


# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--subject_folder", type=str, required=True, help="Path to the subject data folder")
parser.add_argument("--prompt", type=str, required=True, help="Prompt used for training and final generation")
args = parser.parse_args()

subject_folder = args.subject_folder
prompt = args.prompt
data_folder = subject_folder
script_name = os.path.splitext(os.path.basename(__file__))[0]
ckpt_name = f"{script_name}.ckpt"
out_root = script_name
os.makedirs(out_root, exist_ok=True)
out_dir = out_root

ckpt_path = os.path.join(out_dir, ckpt_name)
if os.path.exists(ckpt_path):
    print(f"Checkpoint already exists for subject '{subject_folder}', skipping...")
else:
    # Load the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

    def disabled_safety_checker(images, clip_input):
        if len(images.shape)==4:
            num_images = images.shape[0]
            return images, [False]*num_images
        else:
            return images, False
        
    pipe.safety_checker = disabled_safety_checker

    image_z_T_list = []

    # Load all image files in the data folder
    image_files = [f for f in os.listdir(data_folder) if f.endswith(("jpg", "png", "jpeg"))]

    for img_file in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(data_folder, img_file)
        # Extract image latents
        with torch.no_grad():
            target_image = np.array(Image.open(img_path).resize((512,512)))[:, :, :3]    
            target_image = torch.from_numpy(target_image).float().permute(2, 0, 1) / 127.5 - 1
            image_tensor = target_image.unsqueeze(0).to(device)
            latents = pipe.vae.encode(image_tensor).latent_dist.sample() * 0.18215  # [1, 4, 64, 64]
            image_z_T_list.append(latents.to("cpu"))

    pipe.to("cpu")
        
    unet = pipe.unet.to(device)
    teacher_unet = copy.deepcopy(unet).to(device).eval()
    for param in teacher_unet.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Build the data loader

    # Start the training loop
    text_input = pipe.tokenizer([prompt], padding="max_length", truncation=True, return_tensors="pt").input_ids
    text_embeddings = pipe.text_encoder(text_input).last_hidden_state.to(device)  # Used as encoder_hidden_states

    class DreamBoothDataset(Dataset):
        def __init__(self, image_z_T_list):
            self.image_z_T_list = image_z_T_list  # list of torch.Tensor

        def __len__(self):
            return len(self.image_z_T_list)

        def __getitem__(self, idx):
            return {
                "image_latent": self.image_z_T_list[idx]
            }

    dataset = DreamBoothDataset(image_z_T_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    unet.enable_gradient_checkpointing()  # Enable gradient checkpointing
    if device.type == "cuda":
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Skipping xformers memory efficient attention: {e}")
    
    unet, teacher_unet, optimizer, dataloader = accelerator.prepare(unet, teacher_unet, optimizer, dataloader)
    
    unet.train()
    
    dataloader_iter = iter(dataloader)
    
    for step in tqdm(range(num_train_steps), desc="Training"):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
                
        image_z_T = batch["image_latent"].to(device).squeeze(1)
        b_size = image_z_T.shape[0]
        # Backpropagation and parameter update
                
        optimizer.zero_grad()
        
        pipe.scheduler.set_timesteps(train_noise_steps)
        # Add noise as part of diffusion training
        
        image_z_T_rep = image_z_T.repeat_interleave(particle4dmd, dim=0)  

        noise_base = torch.randn_like(image_z_T)              
        noise = noise_base.repeat_interleave(particle4dmd, dim=0)       

        # Repeat each timestep P times: [B] -> [B * P]
        timesteps_base = torch.randint(0, 1000, (b_size,), device=device).long()  
        timesteps = timesteps_base.repeat_interleave(particle4dmd)      

        # Apply the noise schedule
        noisy_latents = pipe.scheduler.add_noise(image_z_T_rep, noise, timesteps).to(device)

        # Predict the denoising target with the UNet
        pred_noise = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings.expand(particle4dmd * b_size, -1, -1)).sample

        # Compute the MSE loss
        loss_reg = 0.0
        for student_param, teacher_param in zip(unet.parameters(), teacher_unet.parameters()):
            loss_reg += F.mse_loss(student_param, teacher_param)
        loss_reg = loss_reg * weight
        
        accelerator.backward(loss_reg, retain_graph=False)
        
        loss_recon = loss_fn(pred_noise, noise.detach())
        accelerator.backward(loss_recon, retain_graph=True)
        optimizer.step()
            
    if accelerator.is_main_process:
        # 1) Unwrap the underlying model
        unwrapped_unet = accelerator.unwrap_model(unet)
        # 2) Save the state_dict with torch.save
        torch.save(unwrapped_unet.state_dict(), os.path.join(out_dir, ckpt_name))        
    
    del unet, optimizer, dataloader, pipe 
    torch.cuda.empty_cache()    
