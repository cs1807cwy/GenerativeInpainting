import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from .Net import InpaintContextualAttentionGenerator, SpectralNormMarkovianDiscriminator
from .Util import mask_image


# TODO: This Module is a placeholder, currently not implemented
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
            batch_size: int = 16,
            train_save_point_epoches: int = 4000,
            max_iteration: int = 100000000,
            visualization_max_out: int = 10,
            validation_period_step: int = 2000,
            validation: bool = False,
            ae_loss: bool = True,
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

        generator_in_channels = image_channel
        discriminator_in_channels = image_channel

        # networks
        self.generator: InpaintContextualAttentionGenerator = \
            InpaintContextualAttentionGenerator(in_channels=generator_in_channels)
        self.discriminator = SpectralNormMarkovianDiscriminator(in_channels=discriminator_in_channels)
        self.example_input_array = torch.zeros(batch_size, 3, 256, 256)

        # losses
        self.l1_loss = torch.nn.L1Loss()
        self.hinge_loss = torch.nn.HingeEmbeddingLoss()

    def forward(self, x, mask):
        return self.generator.forward(x, mask)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.ground_truth = batch

        # train generator
        if optimizer_idx == 0:
            # prepare masked_image & mask
            self.incomplete, self.mask = mask_image(
                self.ground_truth,
                image_height_width=(self.hparams.image_height, self.hparams.image_width),
                mask_height_width=(self.hparams.mask_height, self.hparams.mask_width),
                margin_vertical_horizontal=(self.hparams.vertical_margin, self.hparams.horizontal_margin),
                max_delta_height_width=(self.hparams.max_delta_height, self.hparams.max_delta_width)
            )
            # generate images
            self.coarse_result, self.refined_result = self.forward(self.incomplete, self.mask)
            self.complete_result = self.refined_result * self.mask + self.ground_truth * (1. - self.mask)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("training_generated_images", grid, self.current_epoch)

            # pair for comparation
            self.negative_statement = self.discriminator.forward(self.complete_result)
            self.valid = torch.ones_like(self.negative_statement).type_as(self.negative_statement)

            # adversarial loss is binary cross-entropy
            g_l1_loss = self.hparams.l1_loss_alpha * \
                        (self.l1_loss(self.ground_truth, self.coarse_result) +
                         self.l1_loss(self.ground_truth, self.refined_result))
            g_hinge_loss = -torch.mean(self.discriminator.forward(self.complete_result))
            g_loss = g_l1_loss + g_hinge_loss
            self.log("g_loss", g_loss, prog_bar=True)

            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # pair for comparation
            d_loss = 0.5 * (torch.mean(F.relu(1. - self.discriminator.forward(self.ground_truth))) +
                            torch.mean(F.relu(1. + self.discriminator.forward(self.complete_result))))

            # discriminator loss is the average of these
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("validation_generated_images", grid, self.current_epoch)
