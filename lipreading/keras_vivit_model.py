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

# Reference: https://keras.io/examples/vision/video_transformers/
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def build(self, input_shape):
        self.position_embeddings.build(input_shape)

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        inputs = keras.ops.cast(inputs, self.compute_dtype)
        length = keras.ops.shape(inputs)[1]
        positions = keras.ops.arange(start=0, stop=length, step=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

# Reference: https://keras.io/examples/vision/video_transformers/
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        
        # Define four multi-headed attention layers
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.attention_3 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.attention_4 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation=keras.activations.gelu),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        # Compute four attention outputs
        attention_output_1 = self.attention_1(inputs, inputs, attention_mask=mask)
        attention_output_2 = self.attention_2(attention_output_1, attention_output_1, attention_mask=mask)
        #attention_output_3 = self.attention_3(attention_output_2, attention_output_2, attention_mask=mask)
        #attention_output_4 = self.attention_4(attention_output_3, attention_output_3, attention_mask=mask)
        
        # Layer normalization and residual connections for each attention layer
        proj_input = self.layernorm_1(inputs + attention_output_1)
        proj_input = self.layernorm_1(proj_input + attention_output_2)
        #proj_input = self.layernorm_1(proj_input + attention_output_3)
        #proj_input = self.layernorm_1(proj_input + attention_output_4)
        
        # Final projection
        proj_output = self.dense_proj(proj_input)
        
        return self.layernorm_2(proj_input + proj_output)

# Reference: https://keras.io/examples/vision/video_transformers/
def get_compiled_model(shape):
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 4
    num_heads = 2
    classes = len(label_processor.get_vocabulary())

    inputs = keras.Input(shape=shape)
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Reference: https://keras.io/examples/vision/video_transformers/
def train_keras_model(X_train, X_test, Y_train, Y_test):
    model = get_compiled_model(X_train.shape[1:])
    history = model.fit(
        X_train,
        Y_train,
        validation_split=0.15,
        epochs=20,
    )

    _, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model
