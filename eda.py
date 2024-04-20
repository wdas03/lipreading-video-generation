import torch
import torchaudio

import numpy as np

from dataset import *

print(torch.__version__)
print(torch.version.cuda)
print(torchaudio.__version__)

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

# Call the function to list GPUs
list_gpus()

import os
import glob
import pickle

DATA_DIR = "/proj/vondrick/aa4870/lipreading-data/mvlrs_v1"

def get_mp4_files(directory):
    # Use a glob pattern to match all .mp4 files
    pattern = os.path.join(directory, '**', '*.mp4')
    mp4_files = glob.glob(pattern, recursive=True)
    return mp4_files

if not os.path.exists("lrs_video_files.pkl"):
    mp4_files = get_mp4_files(DATA_DIR)
    pickle.dump(mp4_files, open("lrs_video_files.pkl", "wb+"))
else:
    mp4_files = pickle.load(open("lrs_video_files.pkl", "rb"))

print("Loaded video files.")

train_files, val_files, test_files = split_data(mp4_files)

video_transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = TalkingFaceDataset(train_files, transform=video_transform)
val_dataset = TalkingFaceDataset(val_files, transform=video_transform)
test_dataset = TalkingFaceDataset(test_files, transform=video_transform)

for t in train_dataset[0]:
    if isinstance(t, list):
        print(len(t))
    else:
        print(np.array(t).shape)

