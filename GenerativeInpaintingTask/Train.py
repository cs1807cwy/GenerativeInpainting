import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from DataModule import CelebAMaskHQ
from Model import SNPatchGAN, S1PatchGAN
from Testbench import GAN


def train():
    batch_size = 8
    dm = CelebAMaskHQ(out_shape=(256, 256), batch_size=batch_size)
    model = S1PatchGAN(256, 256, 3, 128, 128, 32, 32, 0, 0, batch_size)
    tensorboard = TensorBoardLogger(save_dir='.', name='SN_PatchGAN_logs')
    checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor='epoch', mode='max')
    trainer = Trainer(
        logger=tensorboard,
        default_root_dir='./SN_PatchGAN_logs',
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=10,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback],
    )
    trainer.fit(model, dm)


if __name__ == '__main__':
    train()
