import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pickle

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from unet_audio import UNetAudio
from linear_noise_scheduler import LinearNoiseScheduler, LinearNoiseSchedulerV2
from noise_scheduler import CosineNoiseScheduler

from dataset import TalkingFaceFrameDataset, FrameItem

from utils import *

set_visible_devices()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# Load the model configuration and weights
config = {
    'dataset_params': {
        'im_path': '/path/to/data',
        'im_size': 128,
        'im_channels': 3,
        'frame_rate': 30
    },
    'ldm_params': {
        'model_channels': 64,
        'num_res_blocks': 2,
        'attention_resolutions': (1, 2, 4),
        'z_channels': 3  # Channels in the latent space
    },
    'train_params': {
        'ldm_ckpt_name': '/proj/vondrick/aa4870/ldm_model_checkpoint_3.pth' 
    }
}

def sample_images(model, scheduler, img_cond, audio_cond, n_timesteps=500):
    model.eval()

    xt = torch.randn((1, 3, config['dataset_params']['im_size'], config['dataset_params']['im_size'])).to(device)
    xt = xt.detach()  # Ensure no gradient is computed for initial noise
    
    for i in tqdm(reversed(range(n_timesteps))):
        torch.cuda.empty_cache()

        t = torch.tensor([i]).long().to(device)
        xt = xt.detach()  # Detach xt at each step to ensure no gradient accumulation
        noise_pred_cond = model(xt, img_cond, audio_cond, t).detach()
        
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred_cond, t)
        xt = xt.detach()  # Optionally detach if keeping in GPU memory
        
        if (i + 1) % 50 == 0 or i == 0:  # Save images less frequently
            # ims = torch.clamp(x0_pred, -1., 1.).detach().cpu()
            # ims = (ims + 1) / 2
            
            ims = x0_pred.detach().cpu()
            # First, check if the tensor needs scaling
            if ims.dtype == torch.float32 and ims.max() > 1:
                ims = ims / 255.0

            ims = (ims + 1) / 2 

            img = transforms.ToPILImage()(ims[0])
            img_path = os.path.join('lipreading_generated_images', f'x0_{i}.png')
            img.save(img_path)
            img.close()

    print("All images have been processed and saved.")


def load_model_and_scheduler(config):
    model = UNetAudio(
        image_size=config['dataset_params']['im_size'],
        in_channels=config['ldm_params']['z_channels'],
        model_channels=config['ldm_params']['model_channels'],
        out_channels=config['dataset_params']['im_channels'],
        num_res_blocks=config['ldm_params']['num_res_blocks'],
        attention_resolutions=config['ldm_params']['attention_resolutions'],
        audio_feature_dim=768,  # Assuming feature dimension
        projected_audio_dim=128
    ).to(device)
    # model = torch.nn.DataParallel(model)

    # Load the checkpoint
    model.load_state_dict(torch.load(config['train_params']['ldm_ckpt_name']))
    model = torch.nn.DataParallel(model)

    # Initialize the scheduler
    # scheduler = LinearNoiseScheduler(
    #     num_timesteps=2000,  
    #     beta_start=0.0001,
    #     beta_end=0.02
    # )
    # scheduler = CosineNoiseScheduler(num_timesteps=2000)

    scheduler = LinearNoiseSchedulerV2(num_timesteps=500, beta_start=0.00005, beta_end=0.015)

    return model, scheduler

# Load the model and the scheduler
model, scheduler = load_model_and_scheduler(config)
# print(model)

print("Processing dataset...")
video_dataset_frame_items = pickle.load(open("/proj/vondrick/aa4870/lipreading-data/video_dataset_list.pkl", "rb"))
video_dataset_frame_items = np.array(video_dataset_frame_items)

# Transformations
frame_transforms = transforms.Compose([
    transforms.ToPILImage(),        
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_dataset = TalkingFaceFrameDataset(video_dataset_frame_items[:10000], frame_transforms=frame_transforms)

test_instance = full_dataset[5000]
img_cond = test_instance[0].unsqueeze(0).to(device)
audio_cond = test_instance[2].to(device)

img_cond_pil = test_instance[0]

if img_cond_pil.dtype == torch.float32 and img_cond_pil.max() > 1:
    img_cond_pil = img_cond_pil / 255.0

img_cond_pil = (img_cond_pil + 1) / 2  

# Convert to PIL Image
img_cond_pil = transforms.ToPILImage()(img_cond_pil)

img_cond_pil.save(os.path.join('lipreading_generated_images', f'image_cond.png'))
img_cond_pil.close()

print(img_cond.shape, audio_cond)

img = sample_images(model, scheduler, img_cond, audio_cond, n_timesteps=500)