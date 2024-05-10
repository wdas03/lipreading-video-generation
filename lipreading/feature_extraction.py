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

MAX_SEQ_LENGTH = 5
NUM_FEATURES = 1024
IMG_SIZE_X = 32
IMG_SIZE_Y = 48

# Reference: https://keras.io/examples/vision/video_transformers/
def build_feature_extractor():
    feature_extractor = DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE_X, IMG_SIZE_Y, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((IMG_SIZE_X, IMG_SIZE_Y, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

# Reference: https://keras.io/examples/vision/video_transformers/
def prepare_all_videos(timestamps, frames):
    num_samples = sum([len(file_frames) for _, file_frames in frames.items()])
    words = [d.values() for d in timestamps.values()]
    words = [word for subwords in words for word in subwords]
    labels = pd.Series(words)
    labels = label_processor(labels).numpy()[..., None]

    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    num_sentences = len(timestamps)
    nth_sentence = 0
    test_train_cut_idx = -1
    idx=0
    sentence_start_idx = []
    for file, file_frames in tqdm(frames.items()):
        nth_sentence += 1
        if nth_sentence == int(num_sentences * 0.2 + 1):
            test_train_cut_idx = idx
        if test_train_cut_idx == -1:
            sentence_start_idx.append(idx)
        for time, ind_frames in file_frames.items():
            ind_frames = np.array(ind_frames)
            
            if len(ind_frames) < MAX_SEQ_LENGTH:
                diff = MAX_SEQ_LENGTH - len(ind_frames)
                padding = np.zeros((diff, IMG_SIZE_X, IMG_SIZE_Y, 3))
                try:
                    ind_frames = np.concatenate((ind_frames, padding))
                except:
                    continue
            
            
            ind_frames = ind_frames[None, ...]

            temp_frame_features = np.zeros(
                shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
            )

            for i, batch in enumerate(ind_frames):
                video_length = batch.shape[0]
                length = min(MAX_SEQ_LENGTH, video_length)
                for j in range(length):
                    if np.mean(batch[j, :]) > 0.0:
                        temp_frame_features[i, j, :] = feature_extractor.predict(
                            batch[None, j, :], verbose=0
                        )

                    else:
                        temp_frame_features[i, j, :] = 0.0

            frame_features[idx,] = temp_frame_features.squeeze()
            idx+=1

    return frame_features, labels, test_train_cut_idx, sentence_start_idx