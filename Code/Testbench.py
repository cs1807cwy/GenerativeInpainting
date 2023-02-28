import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from DataModule import CelebAMaskHQ, ILSVRC2012_Task1_2, ILSVRC2012_Task3
from Model import SNPatchGAN

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

save_dir = 'Experiment/SN_PatchGAN_logs/saved_images'
prefix = 'gen_'

# hparams for DataModule
train_data_dir: str = 'ILSVRC2012/ILSVRC2012_img_train'
validation_data_dir: str = 'ILSVRC2012/ILSVRC2012_img_val'
test_data_dir: str = 'ILSVRC2012/ILSVRC2012_img_test_v10102019'
out_shape = (image_height, image_width)
num_workers: int = 4

# hparams for Trainer
device = 1


def test():
    data_module = ILSVRC2012_Task3(out_shape=out_shape, batch_size=batch_size, num_workers=num_workers)
    model = SNPatchGAN(image_height, image_width, image_channel,
                       mask_height, mask_width,
                       max_delta_height, max_delta_width,
                       vertical_margin, horizontal_margin,
                       guided, batch_size)
    tensorboard = TensorBoardLogger(save_dir='Experiment', name='SN_PatchGAN_logs', version='tensorboard_train_val')
    csv = CSVLogger(save_dir='Experiment', name='SN_PatchGAN_logs', version='csv_train_val')

    # initialize the Trainer
    trainer = Trainer(
        logger=[tensorboard, csv],
        default_root_dir='./SN_PatchGAN_logs',
        accelerator="auto",
        devices=device,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        log_every_n_steps=2,
    )

    # test the model
    trainer.test(model, data_module,
                 ckpt_path='Experiment/SN_PatchGAN_logs/tensorboard_train_val/checkpoints/last.ckpt')


if __name__ == '__main__':
    test()
