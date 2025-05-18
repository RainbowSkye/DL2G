import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# 扩散模型
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        loss_type_mask='l1',
        conditional=True,
        schedule_opt=None,
        reverse_times=0,
        n_iter = 0,
        m_items=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.loss_type_mask = loss_type_mask
        self.conditional = conditional
        self.reverse = reverse_times
        self.n_iter = n_iter
        self.num_iter = 0
        if m_items is not None:
            self.m_items = m_items
        else:
            self.m_items = None
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_loss_mask(self, device):
        if self.loss_type_mask == 'l1':
            self.loss_func_mask = nn.L1Loss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod_prev',
                             to_torch(np.sqrt(1. - alphas_cumprod_prev)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def predict_start_from_noise_ddim(self, x_t, t, noise):
        return torch.sqrt(self.alphas_cumprod_prev[t]) / torch.sqrt(self.alphas_cumprod[t]) * (
            x_t - self.sqrt_one_minus_alphas_cumprod[t]*noise
        )

    # inference
    def q_posterior(self, et_1, x_t, t):
        mid = (x_t - self.sqrt_one_minus_alphas_cumprod[t] * et_1) / self.sqrt_alphas_cumprod[t]
        xt_1 = self.sqrt_alphas_cumprod_prev[t] * mid + self.sqrt_one_minus_alphas_cumprod_prev[t] * et_1
        return xt_1

    def q_posterior_sr(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped


    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            et_1, x_2, _, _, _, = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level,
                                             train=False, keys=self.m_items, eye_x=condition_x)
        else:
            et_1, x_2, _, _, _ = self.denoise_fn(x, noise_level, train=False, keys=self.m_items, eye_x=condition_x)

        # ddim
        eta = 0.0
        sigma = (
            eta * self.sqrt_one_minus_alphas_cumprod_prev[t] / self.sqrt_one_minus_alphas_cumprod[t]
            * torch.sqrt(1. - self.alphas_cumprod[t] / self.alphas_cumprod_prev[t])
        )
        x_recon = self.predict_start_from_noise_ddim(x, t=t, noise=et_1)
        mean_pred = (
                x_recon
                + torch.sqrt(1 - self.sqrt_alphas_cumprod_prev[t] - sigma ** 2) * et_1
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        xt_1 = mean_pred + sigma * noise
        return xt_1, x_2


    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        # ddim
        xt_1, x_2 = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        return xt_1, x_2


    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in['Input']
            mask = x_in['Mask']
            zt = torch.randn(x.shape, device=device)
            zt_img = zt
            ret_img = x
            d1_img = zt
            for i in reversed(range(0, self.num_timesteps)):
                t = i
                xt_1, x_2 = self.p_sample(zt, t, condition_x=x)
                zt = xt_1
                if i == 0:
                    x1 = self.denoise_fn.DualFusion(xt_1, x_2, mask)
                    x2 = self.denoise_fn.DualFusion(x_in['Input'], x1, mask)
                    d1_img = x2
        if continous:
            return [d1_img]
        else:
            return [d1_img[-1]]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None, train=True):
        x_start = x_in['GT']
        condition_x = x_in['Input']
        mask = x_in['Mask']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)

        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start,
                            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
        if not self.conditional:
            x_recon, x_2, m_items, gathering_loss, spreading_loss = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon, x_2, m_items, gathering_loss, spreading_loss = self.denoise_fn(torch.cat([condition_x, x_noisy], dim=1),
                                      continuous_sqrt_alpha_cumprod, train=True, keys=self.m_items, eye_x=condition_x)
        self.m_items = m_items
        x_re = (x_noisy - torch.mul((1 - continuous_sqrt_alpha_cumprod[0] ** 2).sqrt(), x_recon)) / continuous_sqrt_alpha_cumprod[0]
        x_fusion_1 = self.denoise_fn.DualFusion(x_re, x_2, mask)
        x_fusion = self.denoise_fn.DualFusion(x_in['Input'], x_fusion_1, mask)
        # fusion loss
        loss_fu = self.loss_func(x_start, x_fusion)
        loss_fu = loss_fu.sum() / int(b * c * h * w)
        # noise loss
        loss_noise = self.loss_func(noise, x_recon)  # sr3. loss for x_recon and noise, x_recon is et_1
        loss_noise = loss_noise.sum() / int(b * c * h * w)
        # stream2 loss
        loss_s2 = self.loss_func(x_start, x_2)
        loss_s2 = loss_s2.sum() / int(b * c * h * w)
        loss = loss_noise + gathering_loss + spreading_loss * 0.0001 + loss_fu*0.5 + loss_s2*0.5

        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
