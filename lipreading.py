#!/usr/bin/env python
# coding: utf-8

# In[2]:

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import cv2
    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm
    from copy import deepcopy


    # In[3]:


    import sys


    # In[4]:


    with open('output.txt', 'w') as f:
        # Redirect stdout to the file
        sys.stdout = f
        sys.stderr = f
        
        for i in tqdm(range(5)):
            pass
        
    # Restore stdout to default
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


    # In[5]:


    def get_files(folder='/proj/vondrick/aa4870/lipreading-data/mvlrs_v1/pretrain'):
        files = {'.mp4': [], '.txt': []}
        for dirname, _, filenames in os.walk(folder):
            for filename in filenames:
                if len(files['.mp4']) > 1500:
                    return files
                files[filename[-4:]].append(os.path.join(dirname, filename))
        return files


    # In[6]:


    def get_timestamps(filename):
        file = open(filename)
        text_data = file.readlines()[4:]
        timestamps = {}
        for line in text_data:
            line = line.split()
            timestamps[(float(line[1]), float(line[2]))] = line[0]
        return timestamps


    # In[7]:


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
        
    #     face_cascade = cv2.CascadeClassifier('/kaggle/input/haarcascade-frontalface-default-xml/haarcascade_frontalface_default.xml')
        mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
        # mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_mouth.xml")
        ds_factor = 0.5
        
        skipped_frames = []
        skipped_flag = None

        while check:
            check, arr = vid.read()
            if check and not i % 1:  # This line is to subsample (i.e. keep one frame every 5)
                height, width, _ = arr.shape
                arr = arr[int(height*0.3):,:,:]
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) 

                mouths = mouth_cascade.detectMultiScale(gray, 1.01, minNeighbors=4)
                dist_to_center = height
                x, y, w, h = (0, 0, 0, 0)
                for (mx,my,mw,mh) in mouths:
                    if abs(my-(height//2 - int(height*0.3))) < dist_to_center:
                        x, y, w, h = mx, my, mw, mh
                        dist_to_center = abs(my-(height//2 - int(height*0.3)))
                
                temp_arr = deepcopy(arr)
                try:
                    arr = cv2.resize(arr, dsize=(48, 32), interpolation=cv2.INTER_CUBIC)
                    data.append(arr)
                except:
                    return None
        
            i += 1
        
        frames = {}
        
        for start, end in timestamps:
            start_frame = round(fps * start)
            end_frame = round(fps * end)
            
            frames[(start,end)] = data[start_frame:end_frame+1]
        
        return frames


    # In[8]:


    f = open('real_model_output.txt', 'w')
    # Redirect stdout to the file
    sys.stdout = f
    sys.stderr = f


    # In[ ]:


    files = get_files()
    timestamps = {}
    frames = {}
    for file in tqdm(files['.txt'][:1000]):
        prefix = file[:-4]
        temp_timestamps = get_timestamps(file)
        temp_frame = get_frames(prefix + '.mp4', temp_timestamps)
        if temp_frame!= None:
            frames[prefix] = temp_frame
            timestamps[prefix] = temp_timestamps


    # In[ ]:


    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications.densenet import DenseNet121

    from tensorflow_docs.vis import embed

    import imageio


    # In[ ]:


    MAX_SEQ_LENGTH = 5
    NUM_FEATURES = 1024
    IMG_SIZE_X = 32
    IMG_SIZE_Y = 48

    EPOCHS = 5


    # In[ ]:


    device = "cuda"



    # In[ ]:


    import collections
    words = [d.values() for d in timestamps.values()]
    words = [word for subwords in words for word in subwords]

    test = collections.Counter(words)



    # In[ ]:


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


    feature_extractor = build_feature_extractor()


    # Label preprocessing with StringLookup.
    label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(pd.Series(words)), mask_token=None
    )
    print(label_processor.get_vocabulary())


    # In[ ]:


    def prepare_all_videos(timestamps, frames):
        num_samples = sum([len(file_frames) for _, file_frames in frames.items()])
        labels = pd.Series(words)
        labels = label_processor(labels).numpy()[..., None]

        # `frame_features` are what we will feed to our sequence model.
        frame_features = np.zeros(
            shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # For each video.
        #for start, end in enumerate(optical_flow_outputs):
        idx=0
        for file, file_frames in tqdm(frames.items()):
            for time, ind_frames in file_frames.items():
                ind_frames = np.array(ind_frames)
                
                # print(ind_frames)
                # Pad shorter videos.
                if len(ind_frames) < MAX_SEQ_LENGTH:
                    diff = MAX_SEQ_LENGTH - len(ind_frames)
                    padding = np.zeros((diff, IMG_SIZE_X, IMG_SIZE_Y, 3))
                    try:
                        ind_frames = np.concatenate((ind_frames, padding))
                    except:
                        continue
                
                
                ind_frames = ind_frames[None, ...]

                # Initialize placeholder to store the features of the current video.
                temp_frame_features = np.zeros(
                    shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
                )

                # Extract features from the frames of the current video.
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

        return frame_features, labels


    # In[ ]:


    frame_features, labels = prepare_all_videos(timestamps, frames)


    # In[ ]:


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


    # In[ ]:


    class TransformerEncoder(layers.Layer):
        def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.dense_dim = dense_dim
            self.num_heads = num_heads
            self.attention = layers.MultiHeadAttention(
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
            attention_output = self.attention(inputs, inputs, attention_mask=mask)
            proj_input = self.layernorm_1(inputs + attention_output)
            proj_output = self.dense_proj(proj_input)
            return self.layernorm_2(proj_input + proj_output)


    # In[ ]:


    def get_compiled_model(shape):
        sequence_length = MAX_SEQ_LENGTH
        embed_dim = NUM_FEATURES
        dense_dim = 4
        num_heads = 1
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


    # In[ ]:


    def run_experiment(train_data, train_labels):
        filepath = "video_classifier.weights.h5"
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, save_weights_only=True, save_best_only=True, verbose=1
        )

        model = get_compiled_model(train_data.shape[1:])
        history = model.fit(
            train_data,
            train_labels,
            validation_split=0.15,
            epochs=50,
            callbacks=[checkpoint],
        )

        model.load_weights(filepath)
        _, accuracy = model.evaluate(test_data, test_labels)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

        return model


    # In[ ]:


    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    # In[ ]:


    trained_model = run_experiment(frame_features, labels)


    # In[ ]:


    trained_model.save_weights('model1000')


# In[ ]:




