import os
import cv2
from preprocess import landmark_crop, contrast_boost
import numpy as np

def get_files(folder='/kaggle/input/lrs-pta/mvlrs_v1/pretrain'):
    files = {'.mp4': [], '.txt': []}
    for dirname, _, filenames in os.walk(folder):
        for filename in filenames:
            files[filename[-4:]].append(os.path.join(dirname, filename))
    return files

def get_timestamps(filename):
    file = open(filename)
    text_data = file.readlines()[4:]
    timestamps = {}
    for line in text_data:
        line = line.split()
        timestamps[(float(line[1]), float(line[2]))] = line[0]
    return timestamps

def get_frames(filename, timestamps):
    vid = cv2.VideoCapture(filename)
    fps = vid.get(cv2.CAP_PROP_FPS)

    data = []
    check = True
    i = 0
    
    top_crop = 0.2
    bottom_crop = 0.2
    left_crop = 0.2
    right_crop = 0.2
    
    ds_factor = 0.5

    while check:
        check, arr = vid.read()
        if check and not i % 1:  # This line is to subsample (i.e. keep one frame every 5)
            image = landmark_crop(arr)
            if image is None:
                SKIPPED[0] += 1
                return None
            try:
                image = cv2.resize(image, dsize=(48,48), interpolation=cv2.INTER_CUBIC)
                data.append(contrast_boost(image))
            except:
                return None
        i += 1

    SKIPPED[1] += 1
    frames = {}
    
    for start, end in timestamps:
        start_frame = round(fps * start)
        end_frame = round(fps * end)
        
        frames[(start,end)] = data[start_frame:end_frame+1]
    
    return frames

def get_vocab(folder = "/kaggle/input/lrs-pta/mvlrs_v1/pretrain"):
    vocab_set = set()
    for dirname, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename[-4:] != ".txt": continue
            file = open(os.path.join(dirname, filename))
            line = file.readlines()[0]
            line = line.split()
            words = line[1:]
            vocab_set.update(words)
    return vocab_set