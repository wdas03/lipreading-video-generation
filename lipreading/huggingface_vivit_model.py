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

class ViViT(torch.nn.Module):
    def __init__(self, vivit_model, num_classes, num_frames):
        super(ViViT, self).__init__()
        self.num_frames = num_frames
        self.vit = vivit_model
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, _, _, height,width = x.shape
        # x = x.view(batch_size * self.num_frames, -1, height, width)  # flatten frames into batch dimension
        x = self.vit(x).last_hidden_state
        # print(x.shape)
        # x = x.view(batch_size, self.num_frames, -1)  # reshape to (batch_size, num_frames, dim)
        x = torch.mean(x, dim=1)  # average pooling over frames
        x = self.fc(x)
        return x

def train_huggingface_model(VIVIT):
    DEVICE = "cuda"
    VIVIT = VIVIT.to(DEVICE)
    X_train_huggingface = torch.from_numpy(X_train.reshape(len(X_train), 5, 1, 32,32))
    X_test_huggingface = torch.from_numpy(X_test.reshape(len(X_test), 5, 1, 32,32))
    Y_train_huggingface = torch.from_numpy(Y_train_p.reshape(len(Y_train)))
    Y_test_huggingface = torch.from_numpy(Y_test_p.reshape(len(Y_test)))
    num_epochs = 10
    batch_size = 16
    loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(VIVIT.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
    best_acc = 0
    
    for i in range(num_epochs):
        VIVIT.train()
        running_loss = 0.0
        running_corrects = 0
        for idx in tqdm(range(0, len(X_train_huggingface), batch_size)):
            data = X_train_huggingface[idx:idx+batch_size].to(DEVICE)
            labels = Y_train_huggingface[idx:idx+batch_size].to(DEVICE)
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                outputs = VIVIT(data)
                _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            running_corrects += sum([pred==label for pred,label in zip(preds, labels)])
            
        epoch_loss = running_loss / len(X_train_huggingface)
        epoch_acc = running_corrects.double() / len(X_train_huggingface)
        print(f"train loss is {epoch_loss}, epoch_acc is {epoch_acc}")
        
        VIVIT.eval()
        running_loss = 0.0
        running_corrects = 0
        for idx in tqdm(range(0, len(X_test_huggingface), batch_size)):
            data = X_test_huggingface[idx:idx+batch_size].to(DEVICE)
            labels = Y_test_huggingface[idx:idx+batch_size].to(DEVICE)
            with torch.set_grad_enabled(False):
                # optimizer.zero_grad()
                outputs = VIVIT(data)
                _, preds = torch.max(outputs, 1)
            # loss = loss_fn(outputs, labels)
            # loss.backward()
            # optimizer.step()
            running_loss += loss.item() * len(labels)
            running_corrects += sum([pred==label for pred,label in zip(preds, labels)])
            
        epoch_loss = running_loss / len(X_test_huggingface)
        epoch_acc = running_corrects.double() / len(X_test_huggingface)
        print(f"val loss is {epoch_loss}, epoch_acc is {epoch_acc}")
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(VIVIT.state_dict())
        
        scheduler.step()
    
    VIVIT.load_state_dict(best_model_wts)
    
    return VIVIT