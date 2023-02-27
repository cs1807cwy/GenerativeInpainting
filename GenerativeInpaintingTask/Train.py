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
from DataModule import CelebAMaskHQ, ILSVRC2012_Task1_2, ILSVRC2012_Task3
from Model import SNPatchGAN

max_iteration: int = 100000000
validation_period_step: int = 50
train_save_point_epoches: int = 4000
validation: bool = True
batch_size = 4
device = [1,7]


def train():
    data_module = ILSVRC2012_Task3(out_shape=(256, 256), batch_size=batch_size, num_workers=4)
    model = SNPatchGAN(256, 256, 3, 128, 128, 32, 32, 0, 0, batch_size=batch_size)
    # tensorboard = TensorBoardLogger(save_dir='.', name='SN_PatchGAN_logs')
    csv = CSVLogger(save_dir='.', name='SN_PatchGAN_logs')
    # ddp_strategy = DDPStrategy(find_unused_parameters=False)
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        every_n_epochs=train_save_point_epoches,
        save_on_train_epoch_end=True,
    )
    trainer = Trainer(
        logger=csv,
        default_root_dir='./SN_PatchGAN_logs',
        accelerator="gpu",
        devices=device if torch.cuda.is_available() else 1,
        # max_epochs=10,
        max_steps=150,
        # max_steps=max_iteration,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback],
        strategy='ddp', # use build-in default DDPStrategy, it casts FLAG find_unused_parameters=True
        check_val_every_n_epoch=None,
        val_check_interval=validation_period_step,
        limit_val_batches=1. if validation else 0,
        log_every_n_steps=1,
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    train()
