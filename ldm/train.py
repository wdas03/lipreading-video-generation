import torch
import torch.nn as nn

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim import Adam
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

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define configurations directly in the script for simplicity
config = {
    'diffusion_params': {
        'num_timesteps': 500,
        'beta_start': 0.0001,
        'beta_end': 0.02
    },
    'train_params': {
        'ldm_batch_size': 4,
        'ldm_epochs': 10,
        'ldm_lr': 1e-4,
        'ldm_ckpt_name': '/proj/vondrick/aa4870/ldm_model_checkpoint.pth'
    },
    'dataset_params': {
        'im_path': '/path/to/data',
        'im_size': 128,
        'im_channels': 3,
        'frame_rate': 30
    },
    'ldm_params': {
        'model_channels': 64,
        'num_res_blocks': 2,
        'attention_resolutions': {1, 2, 4}
    },
    'autoencoder_params': {
        'z_channels': 3
    }
}

def train():
    # Noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=config['diffusion_params']['num_timesteps'],
        beta_start=config['diffusion_params']['beta_start'],
        beta_end=config['diffusion_params']['beta_end'])

    # Dataset setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading dataset...")
    # Load dataset
    video_dataset_frame_items = pickle.load(open("../preprocessing/video_dataset_list.pkl", "rb"))
    video_dataset_frame_items = np.array(video_dataset_frame_items)
    
    # Splitting data
    train_indices, val_indices, test_indices = manual_train_val_test_split(len(video_dataset_frame_items))
    
    train_frames = video_dataset_frame_items[train_indices[:5000]]
    val_frames = video_dataset_frame_items[val_indices[:2000]]
    
    # Transformations
    frame_transforms = transforms.Compose([
        transforms.ToPILImage(),        
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = TalkingFaceFrameDataset(train_frames[:2000], frame_transforms=frame_transforms)
    val_dataset = TalkingFaceFrameDataset(val_frames[:500], frame_transforms=frame_transforms)
    
    print("Initializing dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, num_workers=4, shuffle=False)

    print("Creating model...")
    # Initialize UNetAudio model
    model = UNetAudio(
        image_size=config['dataset_params']['im_size'],
        in_channels=config['autoencoder_params']['z_channels'],
        model_channels=config['ldm_params']['model_channels'],
        out_channels=config['dataset_params']['im_channels'],
        num_res_blocks=config['ldm_params']['num_res_blocks'],
        attention_resolutions=config['ldm_params']['attention_resolutions'],
        audio_feature_dim=768,  # Example dimensions
        projected_audio_dim=128).to(device)
    model = nn.DataParallel(model)
    model.train()

    # Audio processing setup
    # audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/proj/vondrick/aa4870/hf-model-checkpoints")

    # Optimizer and loss function
    optimizer = Adam(model.parameters(), lr=config['train_params']['ldm_lr'])
    criterion = torch.nn.MSELoss()

    print("Training model...")
    # Training loop
    for epoch_idx in range(config['train_params']['ldm_epochs']):
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
            t = torch.randint(0, config['diffusion_params']['num_timesteps'], (output_frame.shape[0],)).to(device)

            # Add noise to frames according to timestep
            noisy_frame = scheduler.add_noise(output_frame, noise_to_add, t)

            # Forward pass
            noise_pred = model(noisy_frame, input_frame, audio_segment, t)
            loss = criterion(noise_pred, noise_to_add)
            loss.backward()
            optimizer.step()

        print(f'Finished epoch {epoch_idx + 1} | Loss: {loss.item()}')
        torch.save(model.state_dict(), config['train_params']['ldm_ckpt_name'])

    print('Done Training ...')

if __name__ == '__main__':
    train()