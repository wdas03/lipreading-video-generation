import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os
import random
import numpy as np
import torchvision.transforms as transforms
import pickle
from diffusion import Diffusion
from unet import UNet
from dataset import *

def setup(rank, world_size):
    os.environ['TORCH_IPV6'] = '0'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    torch.distributed.destroy_process_group()

def train_and_validate(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    video_dataset_frame_items = pickle.load(open("preprocessing/video_dataset_list.pkl", "rb"))
    video_dataset_frame_items = np.array(video_dataset_frame_items)

    train_indices, val_indices, _ = manual_train_val_test_split(len(video_dataset_frame_items))

    frame_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = TalkingFaceFrameDataset([video_dataset_frame_items[i] for i in train_indices], frame_transforms=frame_transforms)
    val_dataset = TalkingFaceFrameDataset([video_dataset_frame_items[i] for i in val_indices], frame_transforms=frame_transforms)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=8, sampler=val_sampler, num_workers=4)

    model = Diffusion(UNet(32, 3, 64, 3, 1, (8, 4, 2), 0.1, (1, 2, 3), 4, -1, True, 'frame', 32), 10, 3, 32, 3, 32).to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for x, x_cond in tqdm(train_dataloader):
            x, x_cond = x.to(device), x_cond.to(device)
            optimizer.zero_grad()
            losses = model(x, x_cond)
            loss = losses['simple'].mean()  # Ensure loss is a scalar
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        val_loss = validate_model(model, val_dataloader, device)
        if rank == 0:
            print(f'Epoch {epoch+1}, Train Loss: {epoch_loss / len(train_dataloader)}, Val Loss: {val_loss}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), '/proj/vondrick/aa4870/lipreading_model.pth')
        scheduler.step()

    cleanup()

def manual_train_val_test_split(dataset_size, train_frac=0.8, val_frac=0.1):
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_end = int(train_frac * dataset_size)
    val_end = train_end + int(val_frac * dataset_size)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train_and_validate, args=(world_size,), nprocs=world_size, join=True)
