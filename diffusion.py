import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose
from tqdm import tqdm, trange

from losses import gaussian_kl, discretized_gaussian_nll


class Diffusion(nn.Module):
    def __init__(self, nn_backbone, n_timesteps, in_channels, image_size, out_channels, precision=32, motion_transforms=None):
        super(Diffusion, self).__init__()
        self.nn_backbone = nn_backbone
        self.n_timesteps = n_timesteps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x_shape = (image_size, image_size)
        self.dtype = torch.float32 if precision == 32 else torch.float16
        self.motion_transforms = motion_transforms if motion_transforms else Compose([])

        self.timesteps = torch.arange(n_timesteps, device=self.device)
        self.beta = self.get_beta_schedule().type(self.dtype)
        self.set_params()

    def forward(self, x0, x_cond, motion_frames=None, audio_emb=None, landmarks=None):
        timesteps = torch.randint(self.n_timesteps, (x0.shape[0],), device=self.device)
        eps, xt = self.forward_diffusion(x0, timesteps)
        nn_out = self.nn_backbone(xt, timesteps, x_cond, motion_frames=motion_frames, audio_emb=audio_emb)

        losses = {}
        if self.out_channels == self.in_channels:
            eps_pred = nn_out
        else:
            eps_pred, nu = nn_out.chunk(2, 1)
            nn_out_frozen = torch.cat([eps_pred.detach(), nu], dim=1)
            losses['vlb'] = self.vlb_loss(x0, xt, timesteps, nn_out_frozen).mean()

        if landmarks is not None:
            losses['lip'] = self.lip_loss(eps, eps_pred, landmarks)

        losses['simple'] = ((eps - eps_pred) ** 2).mean()
        return losses

    def forward_diffusion(self, x0, timesteps):
        eps = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[timesteps]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

        xt = self.broadcast(sqrt_alpha_bar_t) * x0 + self.broadcast(sqrt_one_minus_alpha_bar_t) * eps
        return eps, xt

    def sample(self, x_cond, bsz, audio_emb=None, device='cuda', mode=None, stabilize=False, segment_len=None):
        with torch.no_grad():
            n_frames = audio_emb.shape[1]
            xT = torch.randn(x_cond.shape[0], n_frames, self.in_channels, *self.x_shape, device=self.device)

            audio_ids = [0] * self.nn_backbone.n_audio_motion_embs
            for i in range(self.nn_backbone.n_audio_motion_embs + 1):
                audio_ids += [i]

            motion_frames = [self.motion_transforms(x_cond) for _ in range(self.nn_backbone.n_motion_frames)]
            motion_frames = torch.cat(motion_frames, dim=1)

            samples = []
            for i in trange(n_frames, desc=f'Sampling {mode}'):
                if stabilize and i > 0 and i % segment_len == 0:
                    motion_frames = torch.cat([self.motion_transforms(x_cond), motion_frames[:, self.nn_backbone.motion_channels:, :]], dim=1)
                sample_frame = self.sample_loop(xT[:, i], x_cond, motion_frames=motion_frames, audio_emb=audio_emb[:, audio_ids], i_batch=i, n_batches=n_frames)
                samples.append(sample_frame.unsqueeze(1))
                motion_frames = torch.cat([motion_frames[:, self.nn_backbone.motion_channels:, :], self.motion_transforms(sample_frame)], dim=1)
                audio_ids = audio_ids[1:] + [min(i + self.nn_backbone.n_audio_motion_embs + 1, n_frames - 1)]
            return torch.cat(samples, dim=1)

    def get_beta_schedule(self, max_beta=0.999):
        alpha_bar = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        betas = []
        for i in range(self.n_timesteps):
            t1 = i / self.n_timesteps
            t2 = (i + 1) / self.n_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, device=self.device)

    def set_params(self):
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = torch.cat([torch.ones(1, device=self.device), self.alpha_bar[:-1]])

        self.beta_tilde = self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.log_beta_tilde_clipped = torch.log(torch.cat([self.beta_tilde[1:2], self.beta_tilde[1:]]))

        self.coef1_x0 = torch.sqrt(1.0 / self.alpha_bar)
        self.coef2_x0 = torch.sqrt(1.0 / self.alpha_bar - 1)

        self.coef1_q = self.beta * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.coef2_q = (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alpha) / (1.0 - self.alpha_bar)

    def broadcast(self, arr, dim=4):
        while arr.dim() < dim:
            arr = arr[:, None]
        return arr.to(self.device)

    @property
    def device(self):
        return next(self.nn_backbone.parameters()).device


if __name__ == '__main__':
    print(torch.cuda.is_available())

    from unet import UNet

    image_size = 32
    in_channels = 3
    model_channels = 64
    out_channels = 3
    num_res_blocks = 1
    attention_resolutions = (8, 4, 2)
    dropout = 0.1
    channel_mult = (1, 2, 3)
    num_heads = 4
    num_head_channels = -1
    resblock_updown = True
    
    unet = UNet(image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=dropout, channel_mult=channel_mult, num_heads=num_heads, num_head_channels=num_head_channels, resblock_updown=resblock_updown, id_condition_type='frame', precision=32).to('cpu')
    
    diffusion = Diffusion(unet, 10, in_channels, image_size, out_channels, 32)
    print(diffusion.device)

    x = torch.randn(25, 3, image_size, image_size, device=diffusion.device)
    print(diffusion(x, x)['simple'].shape)  # Display the shape of one of the losses for example
