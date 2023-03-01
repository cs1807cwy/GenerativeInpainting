import os

import cv2
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from Net import InpaintContextualAttentionGenerator, SpectralNormMarkovianDiscriminator
from Util import mask_image
import numpy as np
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union


class SNPatchGAN(LightningModule):
    def __init__(
            self,
            image_height: int,
            image_width: int,
            image_channel: int,
            mask_height: int,
            mask_width: int,
            max_delta_height: int,
            max_delta_width: int,
            vertical_margin: int,
            horizontal_margin: int,
            guided: bool = False,
            batch_size: int = 16,
            l1_loss: bool = True,
            l1_loss_alpha: float = 1.,
            gan_loss_alpha: float = 1.,
            gan_with_mask: bool = True,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
            save_dir: str = 'Experiment/SN_PatchGAN_logs/saved_images',
            prefix: str = 'gen_',
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        # This property activates manual optimization
        self.automatic_optimization = False

        guided_channel_expand = 1 if guided else 0
        generator_in_channels = image_channel + guided_channel_expand
        mask_channel_expand = 1 if gan_with_mask else 0
        discriminator_in_channels = image_channel + guided_channel_expand + mask_channel_expand

        # networks
        self.generator: InpaintContextualAttentionGenerator = \
            InpaintContextualAttentionGenerator(in_channels=generator_in_channels)
        self.discriminator = SpectralNormMarkovianDiscriminator(in_channels=discriminator_in_channels)
        self.example_input_array = \
            [torch.zeros(batch_size, image_channel + guided_channel_expand, image_height, image_width),
             torch.zeros(1, 1, image_height, image_width)]

        # make dirs
        self.validate_save_dir = os.path.join(self.hparams.save_dir, 'validate')
        self.test_save_dir = os.path.join(self.hparams.save_dir, 'test')
        self.predict_save_dir = os.path.join(self.hparams.save_dir, 'predict')
        os.makedirs(self.validate_save_dir, exist_ok=True)
        os.makedirs(self.test_save_dir, exist_ok=True)
        os.makedirs(self.predict_save_dir, exist_ok=True)

    def forward(self, x, mask):
        return self.generator(x, mask)

    def training_step(self, batch, batch_idx: int):

        # region 1. get optimizers
        g_opt, d_opt = self.optimizers()
        # endregion

        # region 2. extract input for net
        if self.hparams.guided:
            name, ground_truth, edge = batch
        else:
            name, ground_truth = batch

        real_batch_size = ground_truth.size()
        # endregion

        # region 3. prepare incomplete(masked)_image & mask
        incomplete, mask = mask_image(
            ground_truth,
            image_height_width=(self.hparams.image_height, self.hparams.image_width),
            mask_height_width=(self.hparams.mask_height, self.hparams.mask_width),
            margin_vertical_horizontal=(self.hparams.vertical_margin, self.hparams.horizontal_margin),
            max_delta_height_width=(self.hparams.max_delta_height, self.hparams.max_delta_width)
        )
        # endregion

        # region 4. concatenate the guide map for generator
        if self.hparams.guided:
            masked_edge = edge * mask
            incomplete = torch.cat([incomplete, masked_edge], dim=1)
        # endregion

        # region 5. generate images
        coarse_result, refined_result = self(incomplete, mask)
        complete_result = refined_result * mask + ground_truth * (1. - mask)
        # endregion

        # region 6. wrapping generator's output for discriminator
        complete_for_discrimination = complete_result
        if self.hparams.gan_with_mask:
            complete_for_discrimination = \
                torch.cat([complete_for_discrimination,
                           torch.tile(mask, (real_batch_size[0], 1, 1, 1))],
                          dim=1)
        if self.hparams.guided:
            complete_for_discrimination = \
                torch.cat([complete_for_discrimination, masked_edge], dim=1)
        # endregion

        # region 7. generator hinge loss & l1 loss
        g_hinge_loss = -torch.mean(self.discriminator(complete_for_discrimination))
        g_loss = self.hparams.gan_loss_alpha * g_hinge_loss
        if self.hparams.l1_loss:
            g_l1_loss = (F.l1_loss(ground_truth, coarse_result) +
                         F.l1_loss(ground_truth, refined_result))
            g_loss += self.hparams.l1_loss_alpha * g_l1_loss
        # endregion

        # region 8. optimize generator
        g_opt.zero_grad()

        # note: force no NaN to train GAN, may not be a good practice
        # if not torch.any(torch.isnan(g_loss)):
        #     self.manual_backward(g_loss)
        #     g_opt.step()

        # note: currently use gradient norm clipping
        self.manual_backward(g_loss)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1., 2., False)
        g_opt.step()
        # endregion

        # region 9. log generator losses
        self.log("train_g_loss", g_loss, prog_bar=True)
        if self.hparams.l1_loss:
            self.log("train_g_l1_loss", g_l1_loss, prog_bar=True)
        self.log("train_g_hinge_loss", g_hinge_loss, prog_bar=True)
        # endregion

        # region 10. log generated images
        # broad_mask = torch.ones_like(ground_truth).type_as(ground_truth) * mask
        # if self.hparams.guided:
        #     broad_mask = torch.cat([broad_mask, masked_edge], dim=3)
        # broad_mask.mul_(2.).add_(-1.)
        # sample_imgs: torch.Tensor = torch.cat(
        #     [incomplete[:, 0:3, :, :], broad_mask,
        #      coarse_result, refined_result,
        #      complete_result, ground_truth], dim=3)
        # sample_imgs.add_(1.).mul_(0.5)
        # grid = torchvision.utils.make_grid(sample_imgs, nrow=1)
        # self.logger.experiment.add_image("train_generated_images", grid, self.current_epoch)
        # endregion

        # region 11. concatenate positive-negtive pairs for discriminator
        pos_neg_pair = torch.cat([ground_truth, complete_result], dim=0)
        if self.hparams.gan_with_mask:
            pos_neg_pair = torch.cat([pos_neg_pair,
                                      torch.tile(mask, (real_batch_size[0] * 2, 1, 1, 1))],
                                     dim=1)
        # endregion

        # region 12. concatenate the guide map for discriminator
        if self.hparams.guided:
            pos_neg_pair = torch.cat([pos_neg_pair,
                                      torch.tile(masked_edge, (2, 1, 1, 1))],
                                     dim=1)
        # endregion

        # region 13. classify result output by discriminator
        classify_result = self.discriminator(pos_neg_pair.detach())  # detach to train discriminator alone
        # fairly extract positive-negative reality result
        pos, neg = torch.split(classify_result, classify_result.size(0) // 2)
        # endregion

        # region 14. discriminator hinge loss
        d_loss = 0.5 * (torch.mean(F.relu(1. - pos)) + torch.mean(F.relu(1. + neg)))
        # endregion

        # region 15. optimize discriminator
        d_opt.zero_grad()

        # note: force no NaN to train GAN, may not be a good practice
        # if not torch.any(torch.isnan(d_loss)):
        #     self.manual_backward(d_loss)
        #     d_opt.step()
        # endregion

        # note: currently use gradient norm clipping
        self.manual_backward(d_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1., 2., False)
        d_opt.step()

        # region 16. log discriminator losses
        self.log("train_d_loss", d_loss, prog_bar=True)
        # endregion

        # region log something global
        # self.log('iter', float(self.global_step), on_step=True, prog_bar=True)
        # endregion

    def validation_step(self, batch, batch_idx: int):

        # region 1. extract input for net
        if self.hparams.guided:
            name, ground_truth, edge = batch
        else:
            name, ground_truth = batch

        real_batch_size = ground_truth.size()
        # endregion

        # region 2. prepare incomplete(masked)_image & mask
        incomplete, mask = mask_image(
            ground_truth,
            image_height_width=(self.hparams.image_height, self.hparams.image_width),
            mask_height_width=(self.hparams.mask_height, self.hparams.mask_width),
            margin_vertical_horizontal=(self.hparams.vertical_margin, self.hparams.horizontal_margin),
            max_delta_height_width=(self.hparams.max_delta_height, self.hparams.max_delta_width)
        )
        # endregion

        # region 3. concatenate the guide map for generator
        if self.hparams.guided:
            masked_edge = edge * mask
            incomplete = torch.cat([incomplete, masked_edge], dim=1)
        # endregion

        # region 4. generate images
        coarse_result, refined_result = self(incomplete, mask)
        complete_result = refined_result * mask + ground_truth * (1. - mask)
        # endregion

        # region 5. wrapping generator's output for discriminator
        complete_for_discrimination = complete_result
        if self.hparams.gan_with_mask:
            complete_for_discrimination = \
                torch.cat([complete_for_discrimination,
                           torch.tile(mask, (real_batch_size[0], 1, 1, 1))],
                          dim=1)
        if self.hparams.guided:
            complete_for_discrimination = \
                torch.cat([complete_for_discrimination, masked_edge], dim=1)
        # endregion

        # region 6. generator hinge loss & l1 loss
        g_hinge_loss = -torch.mean(self.discriminator(complete_for_discrimination))
        g_loss = self.hparams.gan_loss_alpha * g_hinge_loss
        if self.hparams.l1_loss:
            g_l1_loss = (F.l1_loss(ground_truth, coarse_result) +
                         F.l1_loss(ground_truth, refined_result))
            g_loss += self.hparams.l1_loss_alpha * g_l1_loss
        # endregion

        # region 7. log generator losses
        self.log("val_g_loss", g_loss, prog_bar=True, sync_dist=True)
        if self.hparams.l1_loss:
            self.log("val_g_l1_loss", g_l1_loss, prog_bar=True, sync_dist=True)
        self.log("val_g_hinge_loss", g_hinge_loss, prog_bar=True, sync_dist=True)
        # endregion

        # region 8. log generated images
        broad_mask = torch.ones_like(ground_truth).type_as(ground_truth) * mask
        if self.hparams.guided:
            broad_mask = torch.cat([broad_mask, masked_edge], dim=3)
        broad_mask.mul_(2.).add_(-1.)
        sample_imgs: torch.Tensor = torch.cat(
            [incomplete[:, 0:3, :, :], broad_mask,
             coarse_result, refined_result,
             complete_result, ground_truth], dim=3)
        sample_imgs.add_(1.).mul_(0.5)
        grid = torchvision.utils.make_grid(sample_imgs, nrow=1)
        self.logger.experiment.add_image(f"validate_generated_images_{batch_idx}", grid, self.current_epoch)
        for i in range(sample_imgs.size(0)):
            img: np.ndarray = sample_imgs[i].cpu().numpy()
            img = img * 255.
            img = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.validate_save_dir,
                                     self.hparams.prefix + f'{self.current_epoch}_' + name[i]), img)
        # endregion

        # region 9. metrics l1 loss & l2 loss
        # normalize value to [0,1]
        normalized_ground_truth = ground_truth.add(1.).mul(0.5)
        normalized_complete_result = complete_result.add(1.).mul(0.5)
        metric_l1_err = F.l1_loss(normalized_complete_result, normalized_ground_truth, reduction='sum') / (
                torch.sum(normalized_ground_truth * mask) + 1e-8)
        metric_l2_err = F.mse_loss(normalized_complete_result, normalized_ground_truth, reduction='sum') / (
                torch.sum(normalized_ground_truth * mask) + 1e-8)

        # region 10. log metric losses
        self.log("val_metric_l1_err", metric_l1_err, prog_bar=True, sync_dist=True)
        self.log("val_metric_l2_err", metric_l2_err, prog_bar=True, sync_dist=True)
        # endregion

    def test_step(self, batch, batch_idx: int):

        # region 1. extract input for net
        if self.hparams.guided:
            name, ground_truth, edge = batch
        else:
            name, ground_truth = batch
        # endregion

        # region 2. prepare incomplete(masked)_image & mask
        incomplete, mask = mask_image(
            ground_truth,
            image_height_width=(self.hparams.image_height, self.hparams.image_width),
            mask_height_width=(self.hparams.mask_height, self.hparams.mask_width),
            margin_vertical_horizontal=(self.hparams.vertical_margin, self.hparams.horizontal_margin),
            max_delta_height_width=(self.hparams.max_delta_height, self.hparams.max_delta_width)
        )
        # endregion

        # region 3. concatenate the guide map for generator
        if self.hparams.guided:
            masked_edge = edge * mask
            incomplete = torch.cat([incomplete, masked_edge], dim=1)
        # endregion

        # region 4. generate images
        coarse_result, refined_result = self(incomplete, mask)
        complete_result = refined_result * mask + ground_truth * (1. - mask)
        # endregion

        # region 5. metrics l1 loss & l2 loss
        # normalize value to [0,1]
        normalized_ground_truth = ground_truth.add(1.).mul(0.5)
        normalized_complete_result = complete_result.add(1.).mul(0.5)
        metric_l1_err = F.l1_loss(normalized_complete_result, normalized_ground_truth, reduction='sum') / (
                torch.sum(normalized_ground_truth * mask) + 1e-8)
        metric_l2_err = F.mse_loss(normalized_complete_result, normalized_ground_truth, reduction='sum') / (
                torch.sum(normalized_ground_truth * mask) + 1e-8)

        # region 6. log metric losses
        self.log("test_metric_l1_err", metric_l1_err, prog_bar=True, sync_dist=True)
        self.log("test_metric_l2_err", metric_l2_err, prog_bar=True, sync_dist=True)
        # endregion

        # region 7. log generated images
        broad_mask = torch.ones_like(ground_truth).type_as(ground_truth) * mask
        if self.hparams.guided:
            broad_mask = torch.cat([broad_mask, masked_edge], dim=3)
        broad_mask.mul_(2.).add_(-1.)
        sample_imgs: torch.Tensor = torch.cat(
            [incomplete, broad_mask,
             coarse_result, refined_result,
             complete_result, ground_truth], dim=3)
        sample_imgs.add_(1.).mul_(0.5)
        grid = torchvision.utils.make_grid(sample_imgs, nrow=1)
        self.logger.experiment.add_image("test_generated_images", grid, batch_idx)
        for i in range(sample_imgs.size(0)):
            img: np.ndarray = sample_imgs[i].cpu().numpy()
            img = img * 255.
            img = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.test_save_dir, self.hparams.prefix + name[i]), img)
        # endregion

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # region 1. extract input for net
        if self.hparams.guided:
            name, incomplete, mask, edge = batch
        else:
            name, incomplete, mask = batch
        # endregion

        # region 2. concatenate the guide map for generator
        if self.hparams.guided:
            masked_edge = edge * mask
            incomplete = torch.cat([incomplete, masked_edge], dim=1)
        # endregion

        # region 3. generate images
        coarse_result, refined_result = self(incomplete, mask)
        complete_result = refined_result * mask + incomplete * (1. - mask)
        # endregion

        # region 4. log generated images
        broad_mask = torch.ones_like(incomplete).type_as(incomplete) * mask
        if self.hparams.guided:
            broad_mask = torch.cat([broad_mask, masked_edge], dim=3)
        broad_mask.mul_(2.).add_(-1.)
        sample_imgs: torch.Tensor = torch.cat(
            [incomplete, broad_mask,
             coarse_result, refined_result,
             complete_result], dim=3)
        sample_imgs.add_(1.).mul_(0.5)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("predict_generated_images", grid, batch_idx)
        os.makedirs(os.path.join(self.hparams.save_dir, 'predict'), exist_ok=True)
        for i in range(complete_result.size(0)):
            img: np.ndarray = complete_result[i].cpu().numpy()
            img = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            img = (img + 1.) * 127.5
            cv2.imwrite(os.path.join(self.predict_save_dir, self.hparams.prefix + name[i]), img)
        # endregion

    def configure_optimizers(self):

        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
