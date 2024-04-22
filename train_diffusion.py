import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from diffusion import Diffusion
from unet import UNet

import numpy as np

import torchvision.transforms as transforms

import pickle

from dataset import *


def list_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Total number of available GPU(s): {num_gpus}\n")
        
        for i in range(num_gpus):
            print(f"GPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}")
            # print(f"  Maximum Threads per Block: {torch.cuda.get_device_properties(i).max_threads_per_block}")
            print("-" * 40)
    else:
        print("No CUDA-enabled GPU is available.")

list_gpus()

# video_dataset_frame_items = pickle.load(open("preprocessing/video_dataset_list.pkl", "rb"))

# video_dataset_frame_items = np.array(video_dataset_frame_items)

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

# train_indices, val_indices, test_indices = manual_train_val_test_split(len(video_dataset_frame_items))

# train_frames = video_dataset_frame_items[train_indices]
# val_frames = video_dataset_frame_items[val_indices]
# test_frames = video_dataset_frame_items[test_indices]

# print(len(train_frames))
# print(len(val_frames))
# print(len(test_frames))

# frame_transforms = transforms.Compose([
#     transforms.ToPILImage(),        # Convert the tensor to PIL image
#     transforms.Resize((128, 128)),  # Resize to 128x128 for uniformity
#     transforms.ToTensor(),          # Convert the PIL image back to tensor
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
# ])

# train_dataset = TalkingFaceFrameDataset(train_frames[:5], frame_transforms=frame_transforms)
# val_dataset = TalkingFaceFrameDataset(val_frames, frame_transforms=frame_transforms)
# test_dataset = TalkingFaceFrameDataset(test_frames, frame_transforms=frame_transforms)

# del video_dataset_frame_items, train_frames, val_frames, test_frames

# device = 'cuda'

# image_size = 128
# in_channels = 3
# model_channels = 64
# out_channels = 3
# num_res_blocks = 1
# attention_resolutions = (8, 4, 2)
# dropout = 0.1
# channel_mult = (1, 2, 3)
# num_heads = 4
# num_head_channels = -1
# resblock_updown = True

# unet = UNet(image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=dropout, channel_mult=channel_mult, num_heads=num_heads, num_head_channels=num_head_channels, resblock_updown=resblock_updown, id_condition_type='frame', precision=32)
# unet.to(device)

# diffusion = Diffusion(unet, 10, in_channels, image_size, out_channels, 32)
# diffusion.to(device)

# # Assuming 'diffusion' is your model instance
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs.")
#     # This wrapper will enable your model to run on multiple GPUs
#     diffusion = nn.DataParallel(diffusion)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# diffusion.to(device)

# # Optimizer setup
# train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
# optimizer = optim.Adam(diffusion.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

# # Training loop
# n_epochs = 10
# for epoch in range(n_epochs):
#     torch.cuda.empty_cache()

#     epoch_loss = 0.0

#     for batch_idx, (x, x_cond) in enumerate(tqdm(train_dataloader)):
#         x = x.to(device)
#         x_cond = x_cond.to(device)

#         optimizer.zero_grad()

#         losses = diffusion(x, x_cond)
#         loss = losses['simple']
#         if 'vlb' in losses:
#             loss += losses['vlb']
#         if 'lip' in losses:
#             loss += losses['lip']
        
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#         if batch_idx % 10 == 0:  # Adjust print frequency as needed
#             print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

#     scheduler.step()

#     avg_epoch_loss = epoch_loss / len(train_dataloader)
#     print(f"Epoch {epoch + 1}/{n_epochs} completed, Average Loss: {avg_epoch_loss}")


def validate_model(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, x_cond in dataloader:
            x, x_cond = x.to(device), x_cond.to(device)
            losses = model(x, x_cond)
            val_loss += losses['simple'].mean().item()
    return val_loss / len(dataloader)

def train_and_validate():
    # GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    video_dataset_frame_items = pickle.load(open("preprocessing/video_dataset_list.pkl", "rb"))
    video_dataset_frame_items = np.array(video_dataset_frame_items)
    
    # Splitting data
    train_indices, val_indices, test_indices = manual_train_val_test_split(len(video_dataset_frame_items))
    
    train_frames = video_dataset_frame_items[train_indices]
    val_frames = video_dataset_frame_items[val_indices]
    
    # Transformations
    frame_transforms = transforms.Compose([
        transforms.ToPILImage(),        
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = TalkingFaceFrameDataset(train_frames[:2000], frame_transforms=frame_transforms)
    val_dataset = TalkingFaceFrameDataset(val_frames[:500], frame_transforms=frame_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, num_workers=4, shuffle=False)
    
    # Model setup
    image_size = 128
    in_channels = 3
    model_channels = 64
    out_channels = 3
    num_res_blocks = 1
    attention_resolutions = (8, 4, 2)
    dropout = 0.1
    channel_mult = (1, 2, 3)
    num_heads = 4
    num_head_channels = -1
    resblock_updown = True

    unet = UNet(image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=dropout, channel_mult=channel_mult, num_heads=num_heads, num_head_channels=num_head_channels, resblock_updown=resblock_updown, id_condition_type='frame', precision=32)
    # unet = nn.DataParallel(unet)

    diffusion = Diffusion(unet, 10, in_channels, image_size, out_channels, 32)
    diffusion.to(device)

    # diffusion = nn.DataParallel(diffusion)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        diffusion = nn.DataParallel(diffusion)
        
    optimizer = optim.Adam(diffusion.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    
    n_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        diffusion.train()
        epoch_loss = 0.0
        
        for x, x_cond in tqdm(train_dataloader):
            x, x_cond = x.to(device), x_cond.to(device)
            optimizer.zero_grad()
            
            losses = diffusion(x, x_cond)
            loss = losses['simple'].mean()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        val_loss = validate_model(diffusion, val_dataloader, device)
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss / len(train_dataloader)}, Val Loss: {val_loss}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            print("Saving model...")
            torch.save(diffusion.state_dict(), '/proj/vondrick/aa4870/lipreading_model.pth')
        
        scheduler.step()

train_and_validate()

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, DistributedSampler
# import torch.optim as optim
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torchvision.transforms import transforms
# from tqdm import tqdm
# import numpy as np
# import random
# import pickle

# import os

# from dataset import *
# from diffusion import Diffusion
# from unet import UNet

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)

# def cleanup():
#     dist.destroy_process_group()

# def train(rank, world_size, epochs):
#     setup(rank, world_size)

#     device = torch.device(f"cuda:{rank}")

#     # Model setup
#     image_size = 128
#     in_channels = 3
#     model_channels = 64
#     out_channels = 3
#     num_res_blocks = 1
#     attention_resolutions = (8, 4, 2)
#     dropout = 0.1
#     channel_mult = (1, 2, 3)
#     num_heads = 4
#     num_head_channels = -1
#     resblock_updown = True

#     unet = UNet(image_size, in_channels, model_channels, out_channels, num_res_blocks,
#                 attention_resolutions, dropout=dropout, channel_mult=channel_mult,
#                 num_heads=num_heads, num_head_channels=num_head_channels,
#                 resblock_updown=resblock_updown, id_condition_type='frame', precision=32).to(device)
#     diffusion = Diffusion(unet, 10, in_channels, image_size, out_channels, 32)
#     diffusion = nn.parallel.DistributedDataParallel(diffusion, device_ids=[rank])

#     # Data loading
#     video_dataset_frame_items = pickle.load(open("preprocessing/video_dataset_list.pkl", "rb"))
#     video_dataset_frame_items = np.array(video_dataset_frame_items)
#     random.shuffle(video_dataset_frame_items)

#     # Split dataset
#     split_train = int(0.8 * len(video_dataset_frame_items))
#     split_val = int(0.9 * len(video_dataset_frame_items))

#     train_data = video_dataset_frame_items[:split_train]
#     val_data = video_dataset_frame_items[split_train:split_val]

#     # Transform setup
#     frame_transforms = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])

#     train_dataset = TalkingFaceFrameDataset(train_data, frame_transforms=frame_transforms)
#     val_dataset = TalkingFaceFrameDataset(val_data, frame_transforms=frame_transforms)

#     train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
#     train_loader = DataLoader(train_dataset, batch_size=2, sampler=train_sampler)

#     # Optimizer and scheduler setup
#     optimizer = optim.Adam(diffusion.parameters(), lr=0.001)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

#     # Training loop
#     for epoch in range(epochs):
#         train_sampler.set_epoch(epoch)
#         total_loss = 0

#         for batch_idx, (x, x_cond) in enumerate(tqdm(train_loader, desc=f"Rank {rank} Epoch {epoch+1}")):
#             x = x.to(device)
#             x_cond = x_cond.to(device)

#             optimizer.zero_grad()
#             losses = diffusion(x, x_cond)
#             loss = losses['simple']
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         if rank == 0:
#             print(f"Epoch {epoch + 1}: Avg Loss = {total_loss / len(train_loader)}")

#         scheduler.step()

#     cleanup()

# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()
#     epochs = 10
#     mp.spawn(train, args=(world_size, epochs), nprocs=world_size, join=True)


