import math
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from inspect import isfunction
from model.sr3_modules.EFA_Module import EFA
from model.sr3_modules.DualStreamAttentionFusion import DualStreamFeatureFusion


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        # self.resblock = ResBlock(dim)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)

        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, Channels, dilate=1):
        super(ResBlock, self).__init__()
        Ch = Channels
        self.relu  = nn.ReLU()

        self.conv1 = nn.Conv2d(Ch, Ch, 3, dilation=1*dilate, padding=1*dilate, stride=1)
        self.conv2 = nn.Conv2d(Ch, Ch, 3, dilation=1*dilate, padding=1*dilate, stride=1)

        self.conv3 = nn.Conv2d(Ch, Ch, 3, dilation=2*dilate, padding=2*dilate, stride=1)
        self.conv4 = nn.Conv2d(Ch, Ch, 3, dilation=2*dilate, padding=2*dilate, stride=1)

        self.conv5 = nn.Conv2d(Ch, Ch, 3, dilation=2*dilate, padding=2*dilate, stride=1)
        self.conv6 = nn.Conv2d(Ch, Ch, 3, dilation=4*dilate, padding=4*dilate, stride=1)

    def forward(self, x, prev_x, is_the_second):
        if is_the_second==1:
            x = x + self.relu(self.conv2(self.relu(self.conv1(x)))) + 0.1*self.relu(self.conv4(self.relu(self.conv3(x)))) + self.relu(self.conv6(self.relu(self.conv5(x))))*0.1 + prev_x
        else:
            x = x + self.relu(self.conv2(self.relu(self.conv1(x)))) + self.relu(self.conv4(self.relu(self.conv3(x))))*0.1 + self.relu(self.conv6(self.relu(self.conv5(x))))*0.1
        return x


class RefineModule(nn.Module):
    def __init__(self, growRate0, nConvLayers, kSize=3):
        super(RefineModule, self).__init__()
        G0 = growRate0      # 64
        C  = nConvLayers    # 2
        self.conv_1 = nn.Conv2d(3, 64, 3, padding=1, stride=1)
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.res1 = ResBlock(G0, 1)
        self.res2 = ResBlock(G0, 2)
        self.deconv_1 = nn.Conv2d(128+128, 64, 3, padding=1, stride=1)
        self.deconv_2 = nn.Conv2d(64+64, 3, 3, padding=1, stride=1)
        self.deconv_3 = nn.Conv2d(3, 3, 1, padding=0, stride=1)
        self.C = C

    def forward(self, x):
        feats = []
        x = self.conv_1(x)
        feats.append(x)
        x = self.conv_2(x)
        feats.append(x)
        x = self.res1(x, [], 0)
        x = self.res2(x, [], 0)
        x = self.deconv_1(torch.cat((x, feats.pop()), dim=1))        # skip
        x = self.deconv_2(torch.cat((x, feats.pop()), dim=1))
        x = self.deconv_3(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel     # 64
        feat_channels = [pre_channel]
        now_res = image_size            # 128

        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):    # 5
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]   # 64*[1,2,4,8,8]=[64,128,256,512,512]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        # ori
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for index in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2
        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        # Stream 2
        # ########### eye feature aggregation module ---- memory module #############
        self.EFA = EFA(3, 512)

        self.RefineModule = RefineModule(128, 2)

        # Dual Stream Feature Fusion Module
        self.DualFusion = DualStreamFeatureFusion(3, 512)

    def forward(self, x, time, train=False, keys=None, eye_x=None):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        # Stream 2
        if not keys.is_cuda:
            keys = keys.to(x.device)
        updated_feat, m_items, gathering_loss, spreading_loss = self.EFA(eye_x, keys, train)
        stream_2_out = self.RefineModule(updated_feat)

        # Stream 1
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
        stream_1_out = self.final_conv(x)
        return stream_1_out, stream_2_out, m_items, gathering_loss, spreading_loss

