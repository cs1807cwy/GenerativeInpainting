import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union

from Util import extract_image_patches, same_padding, mask_image


# Elementary Networks
class GatedConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple[int, int]],
                 stride: Union[int, tuple[int, int]] = 1,
                 padding: Union[str, Union[int, tuple[int, int]]] = 0,
                 dilation: Union[int, tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 activation: nn.Module = nn.ELU,
                 **actkwargs
                 ):
        super(GatedConv2d, self).__init__()
        latent_channels = out_channels * 2
        self.conv2d_all_feature = nn.Conv2d(in_channels,
                                            latent_channels,
                                            kernel_size,
                                            stride,
                                            padding,
                                            dilation,
                                            groups,
                                            bias,
                                            padding_mode,
                                            device,
                                            dtype)
        self.incomplete_feature_activation = activation(**actkwargs)
        self.soft_gating_S_normalizer = nn.Sigmoid()

    def forward(self, x):
        all_feature: torch.Tensor = self.conv2d_all_feature(x)

        # do channel split
        # the former: represents incomplete feature (not gated)
        # the latter: represents soft gating raw weight (later normalized to [0,1] using Sigmoid)
        incomplete_feature, soft_gating_raw_weight = \
            all_feature.split(split_size=(all_feature.size(1) + 1) // 2, dim=1)

        incomplete_feature_activated = \
            self.incomplete_feature_activation(incomplete_feature)

        soft_gating_S_normalized_weight = \
            self.soft_gating_S_normalizer(soft_gating_raw_weight)

        gated_feature = \
            soft_gating_S_normalized_weight * incomplete_feature_activated

        return gated_feature


class UpSampler2x2(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 padding_mode: str = 'zeros',
                 up_sample_mode: str = 'nearest',
                 **kwargs
                 ):
        super(UpSampler2x2, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=2, mode=up_sample_mode)
        self.conv2d_fine = GatedConv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       padding_mode=padding_mode,
                                       **kwargs)

    def forward(self, x):
        upscaled_feature: torch.Tensor = self.upsampler(x)
        fine_feature = self.conv2d_fine(upscaled_feature)
        return fine_feature


class SpectralNormConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple[int, int]],
                 stride: Union[int, tuple[int, int]] = 1,
                 padding: Union[str, Union[int, tuple[int, int]]] = 0,
                 dilation: Union[int, tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 activation: nn.Module = nn.LeakyReLU,
                 **actkwargs
                 ):
        super(SpectralNormConv2d, self).__init__()
        conv2d_std = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               groups,
                               bias,
                               padding_mode,
                               device,
                               dtype)
        self.conv2d_sn = nn.utils.parametrizations.spectral_norm(conv2d_std)
        self.feature_activation = activation(**actkwargs)

    def forward(self, x):
        feature: torch.Tensor = self.conv2d_sn(x)
        feature_activated = self.feature_activation(feature)
        return feature_activated


class ContextualAttention(nn.Module):
    def __init__(self, kernel_size=3, stride=1, dilation=1, fuse_k=3, softmax_scale=10, fuse=False):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            kernel_size: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            dilation: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        super(ContextualAttention, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse

    def forward(self, f, b, mask=None):
        # get shapes
        raw_fs = list(f.size())  # b*c*h*w
        raw_bs = list(b.size())  # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.dilation
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, kernel_size=(kernel, kernel),
                                      strides=(self.dilation * self.stride,
                                               self.dilation * self.stride),
                                      dilation=(1, 1))  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_bs[0], raw_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1. / self.dilation, mode='nearest')
        b = F.interpolate(b, scale_factor=1. / self.dilation, mode='nearest')
        if mask is not None:
            mask = F.interpolate(mask, scale_factor=1. / self.dilation, mode='nearest')

        fs = list(f.size())  # b*c*h*w
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension

        bs = list(b.size())
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, kernel_size=(self.kernel_size, self.kernel_size),
                                  strides=(self.stride, self.stride),
                                  dilation=(1, 1))
        # w shape: [N, C, k, k, L]
        w = w.view(bs[0], bs[1], self.kernel_size, self.kernel_size, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([1, 1, bs[2], bs[3]]).type_as(f)  # [N=1, C=1, H, W]
        int_ms = list(mask.size())
        # m shape: [1, C*k*k, L]
        m = extract_image_patches(mask, kernel_size=(self.kernel_size, self.kernel_size),
                                  strides=(self.stride, self.stride),
                                  dilation=(1, 1))
        # m shape: [1, C, k, k, L]
        m = m.view(1, 1, self.kernel_size, self.kernel_size, -1)
        m = m.permute(0, 4, 1, 2, 3)  # m shape: [1, L, C, k, k]
        m = m[0]  # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = torch.mean(m, dim=[1, 2, 3], keepdim=True).eq(0.).float()
        mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

        y = []
        k = self.fuse_k  # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).type_as(f).view(1, 1, k, k)  # 1*1*k*k

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.maximum(
                torch.sqrt(torch.sum(torch.pow(wi, 2), dim=[1, 2, 3], keepdim=True)),
                torch.as_tensor(1e-4).type_as(wi))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, (self.kernel_size, self.kernel_size), (1, 1), (1, 1))  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, bs[2] * bs[3], fs[2] * fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, (k, k), (1, 1), (1, 1))
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, bs[2], bs[3], fs[2], fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, bs[2] * bs[3], fs[2] * fs[3])
                yi = same_padding(yi, (k, k), (1, 1), (1, 1))
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, bs[3], bs[2], fs[3], fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, bs[2] * bs[3], fs[2], fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi * self.softmax_scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            # deconv for patch pasting
            wi_center = raw_wi[0]

            # conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.dilation,
                                    padding=(self.dilation + 1) // 2) / 4.  # (B=1, C=128, H=64, W=64)
            output_padding = self.dilation % 2
            yi = F.pad(yi, (0, output_padding, 0, output_padding))
            y.append(yi)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_fs)

        return y


# Inpaint Contextual-Attention Generator Sub-Networks
class Generator_CoarseNet_StageI(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_channels_basis: int = 24  # also output_channels of GatedConv2d
                 # they are the same: here use latent_channels_basis=24 <=> official implementation use cnum=48
                 # latent_channels_basis should be set to cnum//2 to get the same structure as the official implementation
                 # this is because:
                 # official implementation use 48 as the channels of the first Conv2d layer in gen_conv(GatedConv2d), while the ouput of gen_conv is cnum//2
                 # here use latent_channels_basis*2=24*2=48 (namely the output channels) as the channels of the first Conv2d layer in GatedConv2d
                 ):
        super(Generator_CoarseNet_StageI, self).__init__()
        # region Module: downsample_extractor
        self.downsample_extractor = nn.Sequential(
            GatedConv2d(in_channels,
                        latent_channels_basis,
                        kernel_size=5, stride=1, padding=2),
            GatedConv2d(latent_channels_basis,
                        latent_channels_basis * 2,
                        kernel_size=3, stride=2, padding=1),
            GatedConv2d(latent_channels_basis * 2,
                        latent_channels_basis * 2,
                        kernel_size=3, stride=1, padding=1),
            GatedConv2d(latent_channels_basis * 2,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=2, padding=1),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1),
        )
        # endregion
        # region Module: atrous_receptor
        self.atrous_receptor = nn.Sequential(
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=2, dilation=2),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=4, dilation=4),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=8, dilation=8),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=16, dilation=16),
        )
        # endregion
        # region Module: upsample_reconstractor
        self.upsample_reconstractor = nn.Sequential(
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1),
            UpSampler2x2(latent_channels_basis * 4,
                         latent_channels_basis * 2),
            GatedConv2d(latent_channels_basis * 2,
                        latent_channels_basis * 2,
                        kernel_size=3, stride=1, padding=1),
            UpSampler2x2(latent_channels_basis * 2,
                         latent_channels_basis),
            GatedConv2d(latent_channels_basis,
                        latent_channels_basis // 2,
                        kernel_size=3, stride=1, padding=1),
            nn.Conv2d(latent_channels_basis // 2,
                      3,
                      kernel_size=3, stride=1, padding=1),
        )
        # endregion
        # normalized to [-1,1] consistent to input value range
        self.coarse_result_tanh_normalizer = nn.Tanh()

    def forward(self, x):
        feature: torch.Tensor = self.downsample_extractor(x)
        atrous_feature = self.atrous_receptor(feature)
        reconstructed_raw_stage1 = self.upsample_reconstractor(atrous_feature)
        reconstructed_normalized_stage1 = self.coarse_result_tanh_normalizer(reconstructed_raw_stage1)
        return reconstructed_normalized_stage1


class Generator_2BranchRefinementNet_StageII(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_channels_basis: int = 24  # also output_channels of GatedConv2d
                 # they are the same: here use latent_channels_basis=24 <=> official implementation use cnum=48
                 ):
        super(Generator_2BranchRefinementNet_StageII, self).__init__()
        # region Branch: Feature Extraction
        # region Module: feature_extraction_branch_downsample_extractor
        self.feature_extraction_branch_downsample_extractor = nn.Sequential(
            GatedConv2d(in_channels,
                        latent_channels_basis,
                        kernel_size=5, stride=1, padding=2),
            GatedConv2d(latent_channels_basis,
                        latent_channels_basis,
                        kernel_size=3, stride=2, padding=1),
            GatedConv2d(latent_channels_basis,
                        latent_channels_basis * 2,
                        kernel_size=3, stride=1, padding=1),
            GatedConv2d(latent_channels_basis * 2,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=2, padding=1),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1),
        )
        # endregion
        # region Module: feature_extraction_branch_atrous_receptor
        self.feature_extraction_branch_atrous_receptor = nn.Sequential(
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=2, dilation=2),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=4, dilation=4),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=8, dilation=8),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=16, dilation=16),
        )
        # endregion
        # endregion
        # region Branch: Contextural-Attention
        # region Module: ca_branch_downsample_extractor
        self.ca_branch_downsample_extractor = nn.Sequential(
            GatedConv2d(in_channels,
                        latent_channels_basis,
                        kernel_size=5, stride=1, padding=2),
            GatedConv2d(latent_channels_basis,
                        latent_channels_basis,
                        kernel_size=3, stride=2, padding=1),
            GatedConv2d(latent_channels_basis,
                        latent_channels_basis * 2,
                        kernel_size=3, stride=1, padding=1),
            GatedConv2d(latent_channels_basis * 2,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=2, padding=1),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1,
                        activation=nn.ReLU),
        )
        # endregion
        self.ca_branch_contextual_attention = \
            ContextualAttention(kernel_size=3, stride=1, dilation=2, fuse_k=3, softmax_scale=10, fuse=True)
        # region Module: ca_branch_feature_refiner
        self.ca_branch_feature_refiner = nn.Sequential(
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1),
        )
        # endregion
        # endregion
        # region Module: aggregation_reconstructor
        self.aggregation_reconstructor = nn.Sequential(
            GatedConv2d(latent_channels_basis * 8,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1),
            GatedConv2d(latent_channels_basis * 4,
                        latent_channels_basis * 4,
                        kernel_size=3, stride=1, padding=1),
            UpSampler2x2(latent_channels_basis * 4,
                         latent_channels_basis * 2),
            GatedConv2d(latent_channels_basis * 2,
                        latent_channels_basis * 2,
                        kernel_size=3, stride=1, padding=1),
            UpSampler2x2(latent_channels_basis * 2,
                         latent_channels_basis),
            GatedConv2d(latent_channels_basis,
                        latent_channels_basis // 2,
                        kernel_size=3, stride=1, padding=1),
            nn.Conv2d(latent_channels_basis // 2,
                      3,
                      kernel_size=3, stride=1, padding=1),
        )
        # endregion
        # normalized to [-1,1] consistent to input value range
        self.refined_result_tanh_normalizer = nn.Tanh()

    def forward(self, x, mask):
        # Feature Extraction Branch
        fe_branch_feature: torch.Tensor = self.feature_extraction_branch_downsample_extractor(x)
        fe_branch_atrous_feature = self.feature_extraction_branch_atrous_receptor(fe_branch_feature)
        # Contextural-Attention Branch
        ca_branch_feature: torch.Tensor = self.ca_branch_downsample_extractor(x)
        _, _, h, w = ca_branch_feature.size()
        downsampled_mask: torch.Tensor = F.interpolate(mask, size=(h, w))
        ca_branch_contextual_attention = self.ca_branch_contextual_attention(
            ca_branch_feature, ca_branch_feature, downsampled_mask)
        ca_branch_refined_feature = self.ca_branch_feature_refiner(ca_branch_contextual_attention)
        # Aggregation Reconstructor
        # channel-dim concatenate 2-branches' feature
        merged_feature: torch.Tensor = torch.cat([fe_branch_atrous_feature, ca_branch_refined_feature], dim=1)
        reconstructed_raw_stage2 = self.aggregation_reconstructor(merged_feature)
        reconstructed_normalized_stage2 = self.refined_result_tanh_normalizer(reconstructed_raw_stage2)
        return reconstructed_normalized_stage2


# SN-PatchGAN Generator: Inpaint Contextual-Attention Generator
class InpaintContextualAttentionGenerator(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 latent_channels_basis: int = 24  # also output_channels of GatedConv2d
                 # they are the same: here use latent_channels_basis=24 <=> official implementation use cnum=48
                 # latent_channels_basis should be set to cnum//2 to get the same structure as the official implementation
                 # this is because:
                 # official implementation use 48 as the channels of the first Conv2d layer in gen_conv(GatedConv2d), while the ouput of gen_conv is cnum//2
                 # here use latent_channels_basis*2=24*2=48 (namely the output channels) as the channels of the first Conv2d layer in GatedConv2d
                 ):
        super(InpaintContextualAttentionGenerator, self).__init__()
        self.coarse_stage1: Generator_CoarseNet_StageI = Generator_CoarseNet_StageI(
            in_channels + 2,  # include latent_channel & mask_channel
            latent_channels_basis
        )
        self.refinement_stage2: Generator_2BranchRefinementNet_StageII = Generator_2BranchRefinementNet_StageII(
            in_channels,  # original 3-channels, do not use latent_channel or mask_channel
            latent_channels_basis
        )

    def forward(self,
                incomplete_image_with_potential_guidance: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        x = incomplete_image_with_potential_guidance

        # mask: 1 as invalid, 0 as valid
        if mask is None:
            mask = torch.zeros(1, 1, x.size(2), x.size(3)).type_as(x)
        # extend latent dimension for gating
        latent_feature_placeholder = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        # broadcast tensor dimension for pixel-wise masking
        pixel_wise_mask = latent_feature_placeholder * mask

        # StageI: Generate Coarse Result
        stage1_input: torch.Tensor = torch.cat(
            [x, latent_feature_placeholder, pixel_wise_mask],
            dim=1
        )
        coarse_result = self.coarse_stage1.forward(stage1_input)

        # StageII: Generate Refined Result
        # mask: 1 as invalid (incomplete), 0 as valid (original pixels)
        stage2_input: torch.Tensor = coarse_result * mask + stage1_input[:, 0:3, :, :] * (1. - mask)
        refined_result = self.refinement_stage2.forward(stage2_input, mask)

        return coarse_result, refined_result


# SN-PatchGAN Discriminator
class SpectralNormMarkovianDiscriminator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_channels_basis: int = 64):
        super(SpectralNormMarkovianDiscriminator, self).__init__()
        # region Module: feature_extractor
        self.feature_extractor = nn.Sequential(
            SpectralNormConv2d(in_channels,
                               latent_channels_basis,
                               kernel_size=5, stride=2, padding=2),
            SpectralNormConv2d(latent_channels_basis,
                               latent_channels_basis * 2,
                               kernel_size=5, stride=2, padding=2),
            SpectralNormConv2d(latent_channels_basis * 2,
                               latent_channels_basis * 4,
                               kernel_size=5, stride=2, padding=2),
            SpectralNormConv2d(latent_channels_basis * 4,
                               latent_channels_basis * 4,
                               kernel_size=5, stride=2, padding=2),
            SpectralNormConv2d(latent_channels_basis * 4,
                               latent_channels_basis * 4,
                               kernel_size=5, stride=2, padding=2),
            SpectralNormConv2d(latent_channels_basis * 4,
                               latent_channels_basis * 4,
                               kernel_size=5, stride=2, padding=2),
        )
        # endregion
        self.flatten_operator = nn.Flatten()

    def forward(self, x):
        feature: torch.Tensor = self.feature_extractor(x)
        reality_flattened = self.flatten_operator(feature)
        return reality_flattened


def Test_GatedConv2d():
    print('[Test] GatedConv2d:')
    x = torch.ones(2, 2, 2, 3)
    print(f'\tinput\t -> \tshape: {x.shape}, dtype: {x.dtype}, device: {x.device}\n\t{x}')
    net = GatedConv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
    y = net.forward(x)
    print(f'\toutput\t -> \tshape: {y.shape}, dtype: {y.dtype}, device: {y.device}\n\t{y}')


def Test_UpSampler2x2():
    print('[Test] UpSampler2x2:')
    x = torch.ones(1, 2, 2, 3)
    print(f'\tinput\t -> \tshape: {x.shape}, dtype: {x.dtype}, device: {x.device}\n\t{x}')
    net = UpSampler2x2(in_channels=2, out_channels=1)
    y = net.forward(x)
    print(f'\toutput\t -> \tshape: {y.shape}, dtype: {y.dtype}, device: {y.device}\n\t{y}')


def Test_SpectralNormConv2d():
    print('[Test] SpectralNormConv2d:')
    x = torch.ones(2, 3, 8, 16)
    print(f'\tinput\t -> \tshape: {x.shape}, dtype: {x.dtype}, device: {x.device}')
    net = SpectralNormConv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1)
    y = net.forward(x)
    print(f'\toutput\t -> \tshape: {y.shape}, dtype: {y.dtype}, device: {y.device}')


def Test_ContextualAttention():
    print('[Test] ContextualAttention:')
    x = torch.ones(2, 3, 128, 128)
    x, mask = mask_image(x, None, (64, 64), (8, 8), (8, 8))
    print(f'\tinput\t -> \tshape: {x.shape}, dtype: {x.dtype}, device: {x.device}')
    print(f'\tmask\t -> \tshape: {mask.shape}, dtype: {mask.dtype}, device: {mask.device}')
    net = ContextualAttention(kernel_size=3, stride=1, dilation=1, fuse_k=3, softmax_scale=10, fuse=True)
    y = net.forward(x, x, mask)
    print(f'\toutput\t -> \tshape: {y.shape}, dtype: {y.dtype}, device: {y.device}')
    # test dilation
    x = torch.ones(2, 3, 128, 128)
    x, mask = mask_image(x, None, (64, 64), (8, 8), (8, 8))
    print(f'\tinput\t -> \tshape: {x.shape}, dtype: {x.dtype}, device: {x.device}')
    print(f'\tmask\t -> \tshape: {mask.shape}, dtype: {mask.dtype}, device: {mask.device}')
    net = ContextualAttention(kernel_size=3, stride=1, dilation=2, fuse_k=3, softmax_scale=10, fuse=True)
    y = net.forward(x, x, mask)
    print(f'\toutput\t -> \tshape: {y.shape}, dtype: {y.dtype}, device: {y.device}')


def Test_Generator_CoarseNet_StageI():
    print('[Test] Generator_CoarseNet_StageI:')
    x = torch.ones(2, 3, 128, 128)
    print(f'\tinput\t -> \tshape: {x.shape}, dtype: {x.dtype}, device: {x.device}')
    net = Generator_CoarseNet_StageI(in_channels=3,
                                     latent_channels_basis=24)  # the same when using cnum=48 in official implementation
    y = net.forward(x)
    print(f'\toutput\t -> \tshape: {y.shape}, dtype: {y.dtype}, device: {y.device}')


def Test_Generator_2BranchRefinementNet_StageII():
    print('[Test] Generator_2BranchRefinementNet_StageII:')
    x = torch.ones(2, 3, 128, 128)
    x, mask = mask_image(x, None, (64, 64), (8, 8), (8, 8))
    print(f'\tinput\t -> \tshape: {x.shape}, dtype: {x.dtype}, device: {x.device}')
    print(f'\tmask\t -> \tshape: {mask.shape}, dtype: {mask.dtype}, device: {mask.device}')
    net = Generator_2BranchRefinementNet_StageII(in_channels=3, latent_channels_basis=24)
    y = net.forward(x, mask)
    print(f'\toutput\t -> \tshape: {y.shape}, dtype: {y.dtype}, device: {y.device}')


def Test_InpaintContextualAttentionGenerator():
    print('[Test] InpaintContextualAttentionGenerator:')
    x = torch.ones(2, 3, 128, 128)
    x, mask = mask_image(x, None, (64, 64), (8, 8), (8, 8))
    print(f'\tinput\t -> \tshape: {x.shape}, dtype: {x.dtype}, device: {x.device}')
    print(f'\tmask\t -> \tshape: {mask.shape}, dtype: {mask.dtype}, device: {mask.device}')
    net = InpaintContextualAttentionGenerator(in_channels=3, latent_channels_basis=24)
    coarse_result, refined_result = net.forward(x, mask)
    print(f'\tcoarse_result\t -> \tshape: {coarse_result.shape}, dtype: {coarse_result.dtype}, '
          f'device: {coarse_result.device}')
    print(f'\trefined_result\t -> \tshape: {refined_result.shape}, dtype: {refined_result.dtype}, '
          f'device: {refined_result.device}')
    print(f'\tAs auto-encoder:')
    print(f'\tinput\t -> \tshape: {x.shape}, dtype: {x.dtype}, device: {x.device}')
    print(f'\tmask\t -> \tshape: {mask.shape}, dtype: {mask.dtype}, device: {mask.device}')
    coarse_result, refined_result = net.forward(x)
    print(f'\tcoarse_result\t -> \tshape: {coarse_result.shape}, dtype: {coarse_result.dtype}, '
          f'device: {coarse_result.device}')
    print(f'\trefined_result\t -> \tshape: {refined_result.shape}, dtype: {refined_result.dtype}, '
          f'device: {refined_result.device}')


def Test_SpectralNormMarkovianDiscriminator():
    print('[Test] SpectralNormMarkovianDiscriminator:')
    x = torch.ones(2, 3, 128, 128)
    print(f'\tinput\t -> \tshape: {x.shape}, dtype: {x.dtype}, device: {x.device}')
    net = SpectralNormMarkovianDiscriminator(in_channels=3, latent_channels_basis=64)
    output = net.forward(x)
    print(f'\toutput\t -> \tshape: {output.shape}, dtype: {output.dtype}, '
          f'device: {output.device}')


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # Test_GatedConv2d()
    # Test_UpSampler2x2()
    # Test_SpectralNormConv2d()
    # Test_ContextualAttention()
    # Test_Generator_CoarseNet_StageI()
    # Test_Generator_2BranchRefinementNet_StageII()
    # Test_InpaintContextualAttentionGenerator()
    Test_SpectralNormMarkovianDiscriminator()
