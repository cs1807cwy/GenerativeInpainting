import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from .DataModule import CelebAMaskHQ, ILSVRC2012_Task1_2, ILSVRC2012_Task3
from .Model import SNPatchGAN

# hparams for Model
image_height = 256
image_width = 256
image_channel = 3
mask_height = 128
mask_width = 128
max_delta_height = 32
max_delta_width = 32
vertical_margin = 0
horizontal_margin = 0
guided = False

batch_size = 16

l1_loss = True
l1_loss_alpha = 1.
gan_loss_alpha = 1.
gan_with_mask = True
lr = 1e-4
b1 = 0.5
b2 = 0.999
save_dir = 'Experiment/SN_PatchGAN_logs/saved_images'
prefix = 'gen_'

# hparams for DataModule
train_data_dir: str = 'Example/ILSVRC2012/ILSVRC2012_img_train'
validation_data_dir: str = 'Example/ILSVRC2012/ILSVRC2012_img_val'
test_data_dir: str = 'Example/ILSVRC2012/ILSVRC2012_img_test_v10102019'
out_shape = (image_height, image_width)
num_workers: int = 4

# hparams for Trainer
max_iteration: int = 100000000
validation_period_step: int = 2000
train_save_point_epoches: int = 4000
validation: bool = False
device = 4


def train():
    data_module = ILSVRC2012_Task1_2(
        train_data_dir=train_data_dir,
        validation_data_dir=validation_data_dir,
        test_data_dir=test_data_dir,
        out_shape=out_shape, batch_size=batch_size, num_workers=num_workers)
    model = SNPatchGAN(image_height, image_width, image_channel,
                       mask_height, mask_width,
                       max_delta_height, max_delta_width,
                       vertical_margin, horizontal_margin,
                       guided, batch_size,
                       l1_loss, l1_loss_alpha, gan_loss_alpha, gan_with_mask,
                       lr, b1, b2,
                       save_dir, prefix)
    tensorboard = TensorBoardLogger(save_dir='Experiment', name='SN_PatchGAN_logs', version='tensorboard_train_val')
    csv = CSVLogger(save_dir='Experiment', name='SN_PatchGAN_logs', version='csv_train_val')
    # ddp_strategy = DDPStrategy(find_unused_parameters=False)
    checkpoint_callback_regular = ModelCheckpoint(
        save_last=True,
        every_n_epochs=train_save_point_epoches,
        filename='snpatchgan_{epoch:02d}',
        save_top_k=-1
    )
    checkpoint_callback_best_l1_err = ModelCheckpoint(
        monitor='val_metric_l1_err',
        filename='snpatchgan_best_l1_{epoch:02d}_{val_metric_l1_err:.4f}_{val_metric_l2_err:.4f}'
    )
    checkpoint_callback_best_l2_err = ModelCheckpoint(
        monitor='val_metric_l2_err',
        filename='snpatchgan_best_l2_{epoch:02d}_{val_metric_l1_err:.4f}_{val_metric_l2_err:.4f}'
    )

    # initialize the Trainer
    trainer = Trainer(
        logger=[tensorboard, csv],
        default_root_dir='Experiment/SN_PatchGAN_logs',
        accelerator='auto',
        devices=device if torch.cuda.is_available() else 1,
        max_steps=max_iteration,
        callbacks=[TQDMProgressBar(refresh_rate=20),
                   checkpoint_callback_regular,
                   checkpoint_callback_best_l1_err,
                   checkpoint_callback_best_l2_err],
        strategy='ddp',  # use build-in default DDPStrategy, it casts FLAG find_unused_parameters=True
        check_val_every_n_epoch=None,
        val_check_interval=validation_period_step,
        limit_val_batches=1. if validation else 0,
        log_every_n_steps=100,
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    train()
