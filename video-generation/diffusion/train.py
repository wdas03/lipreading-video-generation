import torch
import torch.nn as nn

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unet_audio import UNetAudio, Wav2Vec2Encoder, AudioFeatureTransformer
from linear_noise_scheduler import LinearNoiseScheduler
from dataset import TalkingFaceFrameDataset, FrameItem
from transformers import Wav2Vec2Processor

import pickle

import random

from utils import *

set_visible_devices()

def manual_train_val_test_split(dataset_size, train_frac=0.8, val_frac=0.1):
    indices = list(range(dataset_size))
    random.shuffle(indices)  # Shuffle the indices to ensure random splitting

    # Calculate split sizes
    train_end = int(train_frac * dataset_size)
    val_end = train_end + int(val_frac * dataset_size)

    # Split the indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return train_indices, val_indices, test_indices

def train():
    # Noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=100,
        beta_start=0.00085,
        beta_end=0.012
    )

    # Dataset setup
    device = "cuda"
    print(f"Training on {device}.")
    
    print("Loading dataset...")
    # Load dataset
    video_dataset_frame_items = pickle.load(open("/proj/vondrick/aa4870/lipreading-data/video_dataset_list.pkl", "rb"))
    video_dataset_frame_items = np.array(video_dataset_frame_items)
    
    # Splitting data
    train_indices, val_indices, test_indices = manual_train_val_test_split(len(video_dataset_frame_items))
    
    train_frames = video_dataset_frame_items[train_indices[:5000]]
    val_frames = video_dataset_frame_items[val_indices[:1000]]
    
    # Transformations
    frame_transforms = transforms.Compose([
        transforms.ToPILImage(),        
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = TalkingFaceFrameDataset(train_frames, frame_transforms=frame_transforms)
    val_dataset = TalkingFaceFrameDataset(val_frames, frame_transforms=frame_transforms)
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    print("Initializing dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, num_workers=4, shuffle=False)

    print("Creating model...")
    
    # Initialize UNetAudio model
    model = UNetAudio(
        image_size=128,
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(1, 2, 4),
        audio_feature_dim=768,  
        projected_audio_dim=128,
        use_fp16=False).to(device)
    # model = nn.DataParallel(model)
    model.train()

    # Optimizer and loss function
    optimizer = Adam(model.parameters(), 1e-2)
    criterion = torch.nn.MSELoss()

    print("Training model...")
    # Training loop
    for epoch_idx in range(10):
        for data in tqdm(train_dataloader):
            optimizer.zero_grad()

            if data is None:
                continue

            input_frame, output_frame, audio_segment = data
            input_frame, output_frame = input_frame.to(device), output_frame.to(device)

            # Process audio using Wav2Vec2Processor
            # audio_segment = audio_processor(audio_segment.squeeze(0).squeeze(0).squeeze(0), return_tensors="pt", padding="longest", sampling_rate=16000)
            audio_segment = {k: v.squeeze(1).to(device) for k, v in audio_segment.items()}

            # Sample noise
            noise_to_add = torch.randn_like(output_frame)

            # Sample random timestep
            t = torch.randint(0, 500, (output_frame.shape[0],)).to(device)

            # Add noise to frames according to timestep
            noisy_frame = scheduler.add_noise(output_frame, noise_to_add, t)

            # Forward pass
            noise_pred = model(noisy_frame, input_frame, audio_segment, t)
            loss = criterion(noise_pred, noise_to_add)
            loss.backward()
            optimizer.step()

        print(f'Finished epoch {epoch_idx + 1} | Loss: {loss.item()}')
        torch.save(model.state_dict(), "best_diffusion.pth")

    print('Done Training ...')

if __name__ == '__main__':
    train()