# Talking Face Repo
[<img src="./results/thumbnail.png" width="300" alt="Demo Video">](./results/result_voice.mp4)



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

### Weights
[Checkpoint Step 9000](https://drive.google.com/file/d/1FohGonbtcrCaE1oGo_EVcolxRGxUdS79/view?usp=sharing)

[Face detection](https://drive.google.com/file/d/1_DuKk_q3YsmwitfYic6akMRXR857X8ZL/view?usp=sharing)

[SyncNet](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Flipsync%5Fexpert%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1)

[Lip reading](https://drive.google.com/file/d/1XAVhWXjd77UHsfna9O8cASHr3iGiQBQU/view)
## Acknowledgements
This repository combines code from a few different repositories. It uses code from Wav2Lip for preprocessing LRS2, SyncNet, inference, and part of the training, it uses the AV-HuBERT repo from Meta for lipreading (along with util functions from TalkLip that act as a wrapper around AV-HuBERT), and it uses the S3FD face detection model

