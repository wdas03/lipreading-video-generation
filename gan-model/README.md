# Talking Face Repo
[![Demo Video](./results/thumbnail.png)](./results/result_voice.mp4)

## Prerequisites

- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`
- Face detection pre-trained model should be downloaded to `face_detection/detection/sfd/s3fd.pth`. 
- Install AV-Hubert by following the installation: https://github.com/facebookresearch/av_hubert

## Lip-syncing videos using the pre-trained models (Inference)

The result is saved (by default) in `results/result_voice.mp4`.

You can lip-sync any video to any audio:

```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source>
```

## Training
### Preprocess the dataset for fast training
```python
python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/
```
### Download weights for lipreading model, audio-visual synchronization model, and face detection model
```python
python train_lipreading.py --data_root lrs2_preprocessed/ --checkpoint_dir $folder_to_save_checkpoints --syncnet_checkpoint_path $syncnet_weights --avhubert_root $root_to_avhubert_model --avhubert_path $path_to_avhubert_weights
```

This repo combines code from a few different repos. It uses code from Wav2Lip for preprocessing LRS2, SyncNet, inference, and part of the training, it uses the AV-HuBERT repo from Meta for lipreading (along with util functions from TalkLip that act as a wrapper around AV-HuBERT), and it uses the S3FD face detection model

