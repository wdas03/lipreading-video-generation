import torch
import math

class CosineNoiseScheduler:
    def __init__(self, num_timesteps, s=0.008):
        self.num_timesteps = num_timesteps
        self.s = s  # Smoothing factor
        self.timesteps = torch.arange(num_timesteps, dtype=torch.float32) / num_timesteps
        self.alphas_cumprod = torch.cos(((self.timesteps + self.s) / (1 + self.s)) * math.pi * 0.5) ** 2
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def sample_prev_timestep(self, xt, noise_pred, t):
        sqrt_alpha = self.sqrt_alphas_cumprod.to(xt.device)[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(xt.device)[t]

        # Mean of xt given xt+1
        mean = (xt - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        
        # Variance is constant in the original formulation, but often adjusted in practice
        variance = 1e-5
        if t > 0:
            variance = self.alphas_cumprod.to(xt.device)[t - 1] * (1 - self.alphas_cumprod.to(xt.device)[t]) / (1 - self.alphas_cumprod.to(xt.device)[t - 1])

        sigma = torch.sqrt(torch.tensor(variance)).to(xt.device)
        z = torch.randn_like(xt)
        
        sampled_xt = mean + sigma * z if t > 0 else mean
        return sampled_xt, mean

# # Example of setting up the model and scheduler
# scheduler = CosineNoiseScheduler(num_timesteps=1000)

# # Assuming 'model', 'img_cond', and 'audio_cond' are defined and loaded appropriately
# sample_images(model, scheduler, img_cond, audio_cond, n_timesteps=1000)
