import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from Net import InpaintContextualAttentionGenerator, SpectralNormMarkovianDiscriminator
from Util import mask_image


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

    def forward(self, x, mask):
        return self.generator(x, mask)

    def training_step(self, batch, batch_idx):

        # region 1. get optimizers
        g_opt, d_opt = self.optimizers()
        # endregion

        # region 2. extract input for net
        if self.hparams.guided:
            ground_truth, edge = batch
        else:
            ground_truth = batch
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

        # region 6. hinge loss & l1 loss
        g_hinge_loss = -torch.mean(self.discriminator(complete_result))
        g_loss = self.hparams.gan_loss_alpha * g_hinge_loss
        if self.hparams.l1_loss:
            g_l1_loss = (F.l1_loss(ground_truth, coarse_result) +
                         F.l1_loss(ground_truth, refined_result))
            g_loss += self.hparams.l1_loss_alpha * g_l1_loss
        # endregion

        # region 7. optimize generator
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()
        # endregion

        # region 8. log generator losses
        self.log("train_g_loss", g_loss, prog_bar=True)
        if self.hparams.l1_loss:
            self.log("train_g_l1_loss", g_l1_loss, prog_bar=True)
        self.log("train_g_hinge_loss", g_hinge_loss, prog_bar=True)
        # endregion

        # region 9. log generated images
        # broad_mask = torch.ones_like(ground_truth).type_as(ground_truth) * mask
        # broad_mask.mul_(2.).add_(-1.)
        # sample_imgs: torch.Tensor = torch.cat(
        #     [incomplete, broad_mask,
        #      coarse_result, refined_result,
        #      complete_result, ground_truth], dim=3)
        # sample_imgs.add_(1.).mul_(0.5)
        # grid = torchvision.utils.make_grid(sample_imgs, nrow=1)
        # self.logger.experiment.add_image("training_generated_images", grid, self.current_epoch)
        # endregion

        # region 10. concatenate positive-negtive pairs for discriminator
        pos_neg_pair = torch.cat([ground_truth, complete_result], dim=1)
        if self.hparams.gan_with_mask:
            pos_neg_pair = torch.cat([pos_neg_pair,
                                      torch.tile(mask, (self.hparams.batch_size * 2, 1, 1, 1))],
                                     dim=1)
        # endregion

        # region 11. concatenate the guide map for discriminator
        if self.hparams.guided:
            pos_neg_pair = torch.cat([pos_neg_pair,
                                      torch.tile(masked_edge, (2, 1, 1, 1))],
                                     dim=1)
        # endregion

        # region 12. classify result output by discriminator
        classify_result = self.discriminator(pos_neg_pair.detach())
        # fairly extract positive-negative reality result
        pos, neg = torch.split(classify_result, classify_result.size(0) // 2)
        # endregion

        # region 13. discriminator loss is hinge loss
        d_loss = 0.5 * (torch.mean(F.relu(1. - pos)) + torch.mean(F.relu(1. + neg)))
        # endregion

        # region 14. optimize discriminator
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()
        # endregion

        # region 15. log discriminator losses
        self.log("train_d_loss", d_loss, prog_bar=True)
        # endregion

    def validation_step(self, batch, batch_idx):

        # region 1. extract input for net
        if self.hparams.guided:
            ground_truth, edge = batch
        else:
            ground_truth = batch
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

        # region 5. hinge loss & l1 loss
        g_hinge_loss = -torch.mean(self.discriminator(complete_result))
        g_loss = self.hparams.gan_loss_alpha * g_hinge_loss
        if self.hparams.l1_loss:
            g_l1_loss = (F.l1_loss(ground_truth, coarse_result) +
                         F.l1_loss(ground_truth, refined_result))
            g_loss += self.hparams.l1_loss_alpha * g_l1_loss
        # endregion

        # region 6. log generator losses
        self.log("val_g_loss", g_loss, prog_bar=True)
        if self.hparams.l1_loss:
            self.log("val_g_l1_loss", g_l1_loss, prog_bar=True)
        self.log("val_g_hinge_loss", g_hinge_loss, prog_bar=True)
        # endregion

        # region 7. log generated images
        # broad_mask = torch.ones_like(ground_truth).type_as(ground_truth) * mask
        # broad_mask.mul_(2.).add_(-1.)
        # sample_imgs: torch.Tensor = torch.cat(
        #     [incomplete, broad_mask,
        #      coarse_result, refined_result,
        #      complete_result, ground_truth], dim=3)
        # sample_imgs.add_(1.).mul_(0.5)
        # grid = torchvision.utils.make_grid(sample_imgs, nrow=1)
        # self.logger.experiment.add_image("validating_generated_images", grid, self.current_epoch)
        # endregion

    def configure_optimizers(self):

        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
