import torch

import torchaudio
from torchaudio.transforms import Resample, Vad

from torch.utils.data import Dataset, DataLoader

from torchvision.io import read_video
from torchvision.transforms import Resize, ToTensor, Compose, Normalize

from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC

import cv2
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip

import torchaudio

import numpy as np

from sklearn.model_selection import train_test_split

DATA_DIR = "/home/whd2108/mvlrs_v1"
CACHE_DIR = "/home/whd2108/hf-model-checkpoints"

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

buffer_frames = 5

def high_pass_filter(waveform, sr, cutoff_freq=300):
    waveform = torchaudio.functional.highpass_biquad(waveform, sr, cutoff_freq)
    return waveform

def normalize_waveform(waveform):
    mean = waveform.mean()
    std = waveform.std()
    return (waveform - mean) / std

def voice_activity_detection(waveform, sr):
    vad = Vad(sample_rate=sr)
    return vad(waveform)

class FrameItem:
    def __init__(self, video_path, frame_start, frame_end):
        self.video_path = video_path
        self.frame_start = frame_start
        self.frame_end = frame_end

audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/proj/vondrick/aa4870/hf-model-checkpoints")

def process_audio(audio_segment, target_length, target_sr):
    # Resample the audio segment if the sample rate is different from the target sample rate
    if audio_segment.size(0) != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=audio_segment.size(0), new_freq=target_sr)
        audio_segment = resampler(audio_segment)
    
    # Pad or truncate the audio segment to the target length
    if audio_segment.size(1) < target_length:
        # Pad the audio segment if it's shorter than the target length
        padding = torch.zeros(audio_segment.size(0), target_length - audio_segment.size(1))
        audio_segment = torch.cat([audio_segment, padding], dim=1)
    elif audio_segment.size(1) > target_length:
        # Truncate the audio segment if it's longer than the target length
        audio_segment = audio_segment[:, :target_length]
    
    return audio_segment

class TalkingFaceFrameDataset(Dataset):
    def __init__(self, frame_items, frame_transforms=None, frame_rate=30, audio_transforms=None):
        self.frame_items = frame_items
        self.frame_transforms = frame_transforms
        self.frame_rate = frame_rate
        self.audio_transforms = audio_transforms

    def __len__(self):
        return len(self.frame_items)

    def __getitem__(self, idx):
        frame_item = self.frame_items[idx]
        video_path = frame_item.video_path
        frame_start = frame_item.frame_start
        frame_end = frame_item.frame_end

        try:
            # Initialize the VideoReader with the CPU context
            vr = VideoReader(video_path, ctx=cpu(0))  # Use CPU context for reading videos
            video_fps = vr.get_avg_fps()
            total_frames = len(vr)

            if video_fps == 0:
                raise ValueError("FPS is zero, which may indicate an issue with the video file or codec.")

            # Calculate the step size to simulate an effective FPS of 30 if needed
            step = max(1, int(video_fps / self.frame_rate))

            # Get the input and output frame indices based on the frame_start and frame_end
            # input_frame_idx = max(0, frame_start)
            input_frame_idx = 0
            output_frame_idx = min(frame_end, total_frames - 1)

            # Read the input and output frames from the video
            vr.seek(input_frame_idx)
            input_frame = vr.next().asnumpy()

            vr.seek(output_frame_idx)
            output_frame = vr.next().asnumpy()

            # Apply frame transforms if provided
            if self.frame_transforms is not None:
                input_frame = self.frame_transforms(input_frame)
                output_frame = self.frame_transforms(output_frame)
            
            # # Calculate exact second intervals for audio corresponding to these frames
            frame_duration = 1 / video_fps
            start_sec = max(0, (output_frame_idx - buffer_frames) * frame_duration)
            end_sec = output_frame_idx * frame_duration

            waveform, sr = torchaudio.load(video_path)
            start_sample = int(sr * start_sec)
            end_sample = int(sr * end_sec)

            audio_segment = waveform[:, start_sample:end_sample]
            audio_segment = normalize_waveform(high_pass_filter(audio_segment, sr))

            target_length = 4000  # Set the target length (e.g., 16000 samples for 1 second of audio)
            target_sr = 16000  # Set the target sample rate
            audio_segment = process_audio(audio_segment, target_length, target_sr)

            # Process audio using Wav2Vec2Processor
            audio_segment_processed = audio_processor(audio_segment.squeeze(0), return_tensors="pt", sampling_rate=target_sr)

            # Apply audio transforms if provided (e.g., resampling, filtering)
            if self.audio_transforms is not None:
                audio_segment = self.audio_transforms(audio_segment)

            return input_frame, output_frame, audio_segment_processed
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None, None

class TalkingFaceDataset(Dataset):
    def __init__(self, video_files, transform=None, frame_rate=30, audio_model="facebook/wav2vec2-base"):
        """
        Args:
        video_files (list): List of paths to video files.
        transform (callable, optional): Optional transform to be applied on a video frame.
        frame_rate (int): Number of frames per second to sample from the video.
        """
        self.video_files = video_files
        self.transform = transform
        self.frame_rate = frame_rate

        # Initialize wav2vec 2.0 processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(audio_model, cache_dir=CACHE_DIR)
        self.model = Wav2Vec2Model.from_pretrained(audio_model, cache_dir=CACHE_DIR).to(device)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        text_path = video_path.replace('.mp4', '.txt')  # Assuming the text file has the same name but .txt extension

        # Load transcription text
        with open(text_path, 'r') as file:
            lines = file.readlines()
            transcription = lines[0].strip().split('Text:')[1].strip()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Cannot open video file: {}".format(video_path))

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            raise ValueError("FPS is zero, which may indicate an issue with the video file or codec.")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB

        # Ensure there are frames to process
        if not frames or len(frames) < 5:
            raise ValueError("Little to no frames were read from the video, check the video file and codec.")

        step = max(1, int(video_fps // 30))
        frames = frames[::step]
        cap.release()

        # Transform video frames if applicable
        # if self.transform:
        #     frames = [self.transform(torch.permute(frame, (2, 0, 1))) for frame in frames]

        clip = VideoFileClip(video_path)
        
        # Extract frames
        audio_frames = [frame for frame in clip.iter_frames(fps=self.frame_rate, dtype="uint8")]
        clip.reader.close()

        # Extract audio
        audio = clip.audio.to_soundarray(fps=16000, nbytes=2)
        waveform = torch.tensor(audio, dtype=torch.float32).mean(dim=1, keepdim=True).t()  # Convert to mono and transpose
        waveform = waveform.squeeze(0).squeeze(0)
        clip.audio.reader.close_proc()

        # # Audio processing
        # audio_sample_rate = info['audio_fps']
        # waveform = aframes.squeeze(0)  # Assuming audio is mono or first channel is used

        # # Resample audio to 16kHz (required by wav2vec 2.0)
        # resampler = torchaudio.transforms.Resample(orig_freq=audio_sample_rate, new_freq=16000)
        # waveform = resampler(waveform)

        # Prepare waveform for wav2vec 2.0
        inputs = self.processor(waveform, return_tensors="pt", sampling_rate=16000)

        # Move processed inputs to GPU just before model inference
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Extract embeddings with minimal GPU memory usage
        with torch.no_grad():
            audio_embeddings = self.model(**inputs, output_hidden_states=True).last_hidden_state
            audio_embeddings = audio_embeddings.squeeze(0).cpu()  # Move embeddings back to CPU if not immediately used

        # Make sure frames are tensors before returning
        frames = [torch.tensor(frame).to(torch.float32) for frame in frames]  # Ensure frames are tensors

        del inputs
        torch.cuda.empty_cache()

        clip.close()

        # Return first frame, other frames, audio embeddings, and the transcription text
        return frames[0], frames[1:], audio_embeddings, transcription


def split_data(mp4_files, train_size=0.8, test_size=0.1, random_seed=42):
    # First, split into train and remaining (test + validation)
    train_files, remaining_files = train_test_split(
        mp4_files, train_size=train_size, test_size=(1 - train_size), random_state=random_seed)

    # Calculate the proportion of remaining data to allocate to test to maintain overall test_size percentage
    test_proportion = test_size / (1 - train_size)
    # Split remaining into validation and test
    val_files, test_files = train_test_split(
        remaining_files, test_size=test_proportion, random_state=random_seed)

    return train_files, val_files, test_files

# Example usage
# directory_path = '/path/to/your/directory'
# mp4_files = ['file1.mp4', 'file2.mp4', 'file3.mp4', 'file4.mp4', 'file5.mp4', 'file6.mp4', 'file7.mp4', 'file8.mp4', 'file9.mp4', 'file10.mp4']
# train_files, val_files, test_files = split_data(mp4_files)

# print("Training Files:", train_files)
# print("Validation Files:", val_files)
# print("Testing Files:", test_files)


# # Example transforms for video
# # video_transform = Compose([
# #     Resize((128, 128)),
# #     ToTensor(),
# #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# # ])

# # # Usage
# # video_files = ['path/to/video1.mp4', 'path/to/video2.mp4']
# # dataset = TalkingFaceDataset(video_files, transform=video_transform)
# # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# import torch
# import torchaudio
# from torch.utils.data import Dataset, DataLoader
# from torchvision.io import read_video
# from torchvision.transforms import Resize, ToTensor, Compose, Normalize
# from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC

# import cv2
# from moviepy.editor import VideoFileClip

# import numpy as np

# from sklearn.model_selection import train_test_split

# DATA_DIR = "/proj/vondrick/aa4870/lipreading-data/mvlrs_v1"
# CACHE_DIR = "/proj/vondrick/aa4870/hf-model-checkpoints"

# # device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# class TalkingFaceDataset(Dataset):
#     def __init__(self, video_files, transform=None, frame_rate=30, audio_model="facebook/wav2vec2-base"):
#         """
#         Args:
#         video_files (list): List of paths to video files.
#         transform (callable, optional): Optional transform to be applied on a video frame.
#         frame_rate (int): Number of frames per second to sample from the video.
#         """
#         self.video_files = video_files
#         self.transform = transform
#         self.frame_rate = frame_rate

#         # Initialize wav2vec 2.0 processor and model
#         self.processor = Wav2Vec2Processor.from_pretrained(audio_model, cache_dir=CACHE_DIR)
#         self.model = Wav2Vec2Model.from_pretrained(audio_model, cache_dir=CACHE_DIR).to(device)

#     def __len__(self):
#         return len(self.video_files)

#     def __getitem__(self, idx):
#         video_path = self.video_files[idx]
#         text_path = video_path.replace('.mp4', '.txt')  # Assuming the text file has the same name but .txt extension

#         # Load transcription text
#         with open(text_path, 'r') as file:
#             lines = file.readlines()
#             transcription = lines[0].strip().split('Text:')[1].strip()

#         cap = cv2.VideoCapture(video_path)

#         if not cap.isOpened():
#             raise IOError("Cannot open video file: {}".format(video_path))

#         video_fps = cap.get(cv2.CAP_PROP_FPS)
#         if video_fps == 0:
#             raise ValueError("FPS is zero, which may indicate an issue with the video file or codec.")

#         frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB

#         # Ensure there are frames to process
#         if not frames:
#             raise ValueError("No frames were read from the video, check the video file and codec.")

#         step = max(1, int(video_fps // self.frame_rate))
#         frames = frames[::step]
#         cap.release()

#         # Transform video frames if applicable
#         if self.transform:
#             frames = [self.transform(frame) for frame in frames]

#         # Load audio using moviepy
#         clip = VideoFileClip(video_path)
#         audio = clip.audio.to_soundarray(fps=16000, nbytes=4)  # Extract audio and resample to 16 kHz

#         # Ensure audio is a 2D array with shape [channels, length]
#         waveform = torch.tensor(audio, dtype=torch.float32).transpose(0, 1)
#         if waveform.ndim == 2 and waveform.shape[0] != 1:
#             waveform = waveform.unsqueeze(0)  # Add a batch dimension if only one channel is present

#         # Process waveform with the Wav2Vec2 processor
#         inputs = self.processor(waveform, return_tensors="pt", sampling_rate=16000)
#         inputs = {key: val.to(device) for key, val in inputs.items()}

#         # Extract embeddings with minimal GPU memory usage
#         with torch.no_grad():
#             audio_embeddings = self.model(**inputs, output_hidden_states=True).last_hidden_state
#             audio_embeddings = audio_embeddings.squeeze(0).cpu()  # Move embeddings back to CPU if not immediately used

#         return frames[0], frames[1:], audio_embeddings, transcription

# def split_data(mp4_files, train_size=0.8, test_size=0.1, random_seed=42):
#     # First, split into train and remaining (test + validation)
#     train_files, remaining_files = train_test_split(
#         mp4_files, train_size=train_size, test_size=(1 - train_size), random_state=random_seed)

#     # Calculate the proportion of remaining data to allocate to test to maintain overall test_size percentage
#     test_proportion = test_size / (1 - train_size)
#     # Split remaining into validation and test
#     val_files, test_files = train_test_split(
#         remaining_files, test_size=test_proportion, random_state=random_seed)

#     return train_files, val_files, test_files

# # Example usage
# # directory_path = '/path/to/your/directory'
# # mp4_files = ['file1.mp4', 'file2.mp4', 'file3.mp4', 'file4.mp4', 'file5.mp4', 'file6.mp4', 'file7.mp4', 'file8.mp4', 'file9.mp4', 'file10.mp4']
# # train_files, val_files, test_files = split_data(mp4_files)

# # print("Training Files:", train_files)
# # print("Validation Files:", val_files)
# # print("Testing Files:", test_files)


# # # Example transforms for video
# # # video_transform = Compose([
# # #     Resize((128, 128)),
# # #     ToTensor(),
# # #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# # # ])

# # # # Usage
# # # video_files = ['path/to/video1.mp4', 'path/to/video2.mp4']
# # # dataset = TalkingFaceDataset(video_files, transform=video_transform)
# # # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
