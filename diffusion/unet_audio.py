# Adapted from: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
# Incorporates additional modality layers, which are concatenated with inputs

import torch as th
import torch.nn as nn
from unet import UNetModel
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Trainable audio encoder
class Wav2Vec2Encoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h",
                 cache_dir="/proj/vondrick/aa4870/hf-model-checkpoints"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, cache_dir=cache_dir)

    def forward(self, audio_input):
        outputs = self.wav2vec2(**audio_input, output_hidden_states=True).last_hidden_state
        return outputs

# Audio feature processor
class AudioFeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.transform(x)

class UNetAudio(UNetModel):
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, image_cond=True, im_cond_input_ch=3, im_cond_output_ch=64, dropout=0.1, channel_mult=(1, 2, 4),
                 conv_resample=True, dims=2, num_classes=None, use_checkpoint=False,
                 use_fp16=False, num_heads=1, num_head_channels=-1, num_heads_upsample=-1,
                 use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False,
                 audio_feature_dim=512, projected_audio_dim=256):
        super().__init__(image_size, in_channels + projected_audio_dim + (im_cond_output_ch if image_cond else 0), model_channels, out_channels, num_res_blocks,
                         attention_resolutions, dropout, channel_mult, conv_resample, dims,
                         num_classes, use_checkpoint, use_fp16, num_heads, num_head_channels,
                         num_heads_upsample, use_scale_shift_norm, resblock_updown, use_new_attention_order)
        self.audio_encoder = Wav2Vec2Encoder()
        self.audio_transformer = AudioFeatureTransformer(audio_feature_dim, projected_audio_dim)
        self.projected_audio_dim = projected_audio_dim
        self.image_size = image_size
        self.image_cond = image_cond
        if self.image_cond:
            self.cond_conv_in = nn.Conv2d(in_channels=im_cond_input_ch, out_channels=im_cond_output_ch, kernel_size=1, bias=False)

    def forward(self, image, cond_image, audio, timesteps, y=None):
        audio_features = self.audio_encoder(audio)  # Encode audio
        audio_features = self.audio_transformer(audio_features.mean(dim=1))  # Transform and prepare for fusion

        # Reshape audio features to match image feature maps
        audio_features = audio_features.view(-1, self.projected_audio_dim, 1, 1).expand(-1, -1, self.image_size, self.image_size)

        if self.image_cond:
            im_cond = nn.functional.interpolate(cond_image, size=image.shape[-2:])
            im_cond = self.cond_conv_in(im_cond)
            image_features = th.cat([image, im_cond, audio_features], dim=1)
        else:
            image_features = th.cat([image, audio_features], dim=1)

        # Process combined features through the UNet
        return super().forward(image_features, timesteps, y)
    
# UNet with audio features
# class UNetAudio(UNetModel):
#     def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks,
#                  attention_resolutions, image_cond=True, im_cond_input_ch=3, im_cond_output_ch=3, dropout=0.1, channel_mult=(1, 2, 4, 8),
#                  conv_resample=True, dims=2, num_classes=None, use_checkpoint=False,
#                  use_fp16=False, num_heads=1, num_head_channels=-1, num_heads_upsample=-1,
#                  use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False,
#                  audio_feature_dim=512, projected_audio_dim=256):
#         super().__init__(image_size, in_channels + im_cond_output_ch + projected_audio_dim, model_channels, out_channels, num_res_blocks,
#                          attention_resolutions, dropout, channel_mult, conv_resample, dims,
#                          num_classes, use_checkpoint, use_fp16, num_heads, num_head_channels,
#                          num_heads_upsample, use_scale_shift_norm, resblock_updown, use_new_attention_order)
#         self.audio_encoder = Wav2Vec2Encoder()
#         self.audio_transformer = AudioFeatureTransformer(audio_feature_dim, projected_audio_dim)
#         self.projected_audio_dim = projected_audio_dim
#         self.image_size = image_size

#         self.image_cond = image_cond
#         if self.image_cond:
#             self.cond_conv_in = nn.Conv2d(in_channels=im_cond_input_ch, out_channels=im_cond_output_ch, kernel_size=1, bias=False)
#             self.conv_in = nn.Conv2d(in_channels + im_cond_output_ch + projected_audio_dim, model_channels, kernel_size=3, padding=1)
#         else:
#             self.conv_in = nn.Conv2d(in_channels + projected_audio_dim, model_channels, kernel_size=3, padding=1)


#     def forward(self, image, cond_image, audio, timesteps, y=None):
#         audio_features = self.audio_encoder(audio)  # Encode audio
#         audio_features = self.audio_transformer(audio_features.mean(dim=1))  # Transform and prepare for fusion

#         # Reshape audio features to match image feature maps
#         audio_features = audio_features.view(-1, self.projected_audio_dim, 1, 1).expand(-1, -1, self.image_size, self.image_size)

#         if self.image_cond:
#             im_cond = cond_image
#             im_cond = nn.functional.interpolate(im_cond, size=image.shape[-2:])
#             im_cond = self.cond_conv_in(im_cond)

#             assert im_cond.shape[-2:] == image.shape[-2:]

#             image = th.cat([image, im_cond, audio_features], dim=1)
#             image_features = self.conv_in(image)
#         else:
#             image_features = th.cat([image, audio_features], dim=1)
#             image_features = self.conv_in(image_features)

#         # Early fusion: concatenate audio features with image features
#         # image_features = th.cat([image, audio_features], dim=1)

#         # Process combined features through the UNet
#         return super().forward(image_features, timesteps, y)

if __name__ == "__main__":
    device = 'cuda'

    model = UNetAudio(image_size=128, in_channels=3, model_channels=64, out_channels=3, num_res_blocks=2,
                      attention_resolutions={1, 2, 4}, audio_feature_dim=768, projected_audio_dim=256)
    model.to(device)

    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/proj/vondrick/aa4870/hf-model-checkpoints")

    image = th.randn(1, 3, 128, 128).to(device)
    raw_audio = th.randn(16000)  # Simulated raw audio
    audio = audio_processor(raw_audio.squeeze(0), return_tensors="pt", sampling_rate=16000).to(device)
    # audio = {'input_values': th.randn(1, 16000).to(device)}  # Simulating processed audio input

    print("Processing inputs...")
    timesteps = th.tensor([1]).to(device)
    output = model(image, image, audio, timesteps)
    print(f"Image shape: {image.shape}")
    print(f"Output shape: {output.shape}")
