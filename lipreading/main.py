from transformers import VivitForVideoClassification, VivitImageProcessor, VivitModel, VivitConfig
import numpy as np
import pandas as pd
import cv2
import os
from tqdm.auto import tqdm
from copy import deepcopy
import pickle
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow_docs.vis import embed
import imageio
import torch
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import jellyfish
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from preprocess import * 
from get_data import *
from feature_extraction import *
from keras_vivit_model import *
from huggingface_vivit_model import *
from sentence_eval import *

files = get_files()
timestamps = {}
frames = {}

MAX_SEQ_LENGTH = 5 # 15
NUM_FEATURES = 1024
IMG_SIZE_X = 32
IMG_SIZE_Y = 48

for file in tqdm(files['.txt']):
    prefix = file[:-4]
    temp_timestamps = get_timestamps(file)
    temp_frame = get_frames(prefix + '.mp4', temp_timestamps)
    if temp_frame!= None:
        frames[prefix] = temp_frame
        timestamps[prefix] = temp_timestamps
vocab_set = get_vocab()
vocab_list = sorted(list(vocab_set))

feature_extractor = build_feature_extractor()
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(vocab_list), mask_token=None
)
frame_features, labels, test_train_cut_idx, sentence_start_idx = prepare_all_videos(timestamps, frames)

X_test, X_train, Y_test, Y_train = train_test_split(frame_features, labels, random_state=42)


vivit_image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
# vivit_model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
vivit_config = VivitConfig(image_size = 32, num_frames = 15, num_channels=1, hidden_size = 256, num_attention_heads=8, num_attention_layers=8)#.from_pretrained("google/vivit-b-16x2-kinetics400")
VIVIT_model = ViViT(VivitModel(vivit_config), len(vocab_list), 5)

trained_VIVIT_model = train_huggingface_model(VIVIT_model)