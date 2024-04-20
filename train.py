import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from models import UNet, Diffusion
from dataset import TalkingFaceDataset
from torch import optim
import os

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup Dataset and DataLoader
transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

video_files = []
dataset = TalkingFaceDataset(video_files, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model Setup
image_size = 128
in_channels = 3
model_channels = 64
out_channels = 3
num_res_blocks = 2
attention_resolutions = [32, 16, 8]
dropout = 0.1
channel_mult = (1, 2, 4, 8)
num_heads = 4
num_head_channels = 64
resblock_updown = True
n_timesteps = 1000
id_condition_type = 'simple'
audio_condition_type = 'complex'

unet = UNet(image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, num_heads, num_head_channels, resblock_updown, id_condition_type, audio_condition_type).to(device)
diffusion = Diffusion(unet, n_timesteps, in_channels, image_size, out_channels).to(device)

# Optimizer
optimizer = optim.Adam(diffusion.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

# Training loop
n_epochs = 10
for epoch in range(n_epochs):
    for batch_idx, (x, x_cond, motion_frames, audio_emb, landmarks) in enumerate(dataloader):
        x = x.to(device)
        x_cond = x_cond.to(device)
        motion_frames = motion_frames.to(device)
        audio_emb = audio_emb.to(device)
        landmarks = landmarks.to(device) if landmarks is not None else None

        optimizer.zero_grad()
        losses = diffusion(x, x_cond, motion_frames=motion_frames, audio_emb=audio_emb, landmarks=landmarks)
        loss = losses['simple']
        if 'vlb' in losses:
            loss += losses['vlb']
        if 'lip' in losses:
            loss += losses['lip']
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

    scheduler.step()

# Note: Validation loop not included, add if necessary based on your setup.
