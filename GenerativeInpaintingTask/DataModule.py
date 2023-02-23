import os
import numpy as np
import cv2
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union

import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split


class CelebAMaskHQ(LightningDataModule):
    class _dataset(Dataset):
        def __init__(self,
                     image_paths: list[str],
                     resize_shape: Optional[tuple[int, int]], ):
            self.image_paths: list[str] = image_paths
            self.resize_shape: Optional[tuple[int, int]] = resize_shape

        def __getitem__(self, idx: int) -> torch.Tensor:
            item: np.ndarray = cv2.imread(self.image_paths[idx])

            # print(item.shape)

            if self.resize_shape is not None:
                item = cv2.resize(item, self.resize_shape)
            item: torch.Tensor = transforms.ToTensor()(item)
            item.mul_(2).add_(-1)

            # print(item.shape)

            return item

        def __len__(self) -> int:
            return len(self.image_paths)

    def __init__(
            self,
            data_dir: str = 'CelebAMask-HQ/CelebA-HQ-img',
            out_shape: Optional[tuple[int, int]] = None,
            batch_size: int = 8,
            num_workers: int = 1,
            validation_ratio: float = 0.1,
            test_ratio: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.resize_shape = out_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

    def prepare_data(self):

        def _get_filenames(data_dir: str) -> list[str]:
            image_list: list[str] = os.listdir(data_dir)
            image_paths: list[str] = [os.path.join(data_dir, _) for _ in image_list]
            return image_paths

        # list & collect all images
        image_paths: list[str] = _get_filenames(self.data_dir)
        image_paths.sort(key=lambda x: (int)(os.path.basename(x).split('.')[0]))

        # print(f'CelebA-HQ total samples: {len(image_paths)}')

        # split for training-validation & test
        self.test_count: int = (int)(len(image_paths) * self.test_ratio)
        self.val_count: int = (int)(len(image_paths) * self.validation_ratio)
        self.train_count: int = len(image_paths) - self.test_count - self.val_count
        train_val_count: int = self.train_count + self.val_count
        self.train_val_image_paths: list[str] = image_paths[0:train_val_count]
        self.test_image_paths: list[str] = image_paths[train_val_count:]

        # print(f'CelebA-HQ train_val samples: {len(self.train_val_image_paths)}')
        # print(f'CelebA-HQ test samples: {len(self.test_image_paths)}')

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_image_paths, val_image_paths = \
                random_split(self.train_val_image_paths, [self.train_count, self.val_count])
            self.celeba_hq_train: CelebAMaskHQ._dataset = CelebAMaskHQ._dataset(train_image_paths, self.resize_shape)
            self.celeba_hq_val: CelebAMaskHQ._dataset = CelebAMaskHQ._dataset(val_image_paths, self.resize_shape)

            # print(f'CelebA-HQ train dataset len: {len(self.celeba_hq_train)}')
            # print(f'CelebA-HQ val dataset len: {len(self.celeba_hq_val)}')

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.celeba_hq_test: CelebAMaskHQ._dataset = CelebAMaskHQ._dataset(self.test_image_paths, self.resize_shape)

            # print(f'CelebA-HQ test dataset len: {len(self.celeba_hq_test)}')

    def train_dataloader(self):
        return DataLoader(
            self.celeba_hq_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.celeba_hq_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.celeba_hq_test, batch_size=self.batch_size, num_workers=self.num_workers)


class ILSVRC2012_Task1_2(LightningDataModule):
    class _dataset(Dataset):
        def __init__(self,
                     image_paths: list[str],
                     crop_shape: tuple[int, int], ):
            self.image_paths: list[str] = image_paths
            self.crop_shape: tuple[int, int] = crop_shape
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomCrop(self.crop_shape),
                ]
            )

        def __getitem__(self, idx: int) -> torch.Tensor:
            item: np.ndarray = cv2.imread(self.image_paths[idx])

            # print(item.shape)
            imgw, imgh, _ = item.shape
            if imgh < self.crop_shape[0] or imgw < self.crop_shape[1]:
                item = transforms.ToPILImage()(item)
                item = transforms.Resize(min(self.crop_shape))(item)

            item: torch.Tensor = self.transform(item)
            item.mul_(2).add_(-1)

            # print(item.shape)

            return item

        def __len__(self) -> int:
            return len(self.image_paths)

    def __init__(
            self,
            train_data_dir: str = 'ILSVRC2012/ILSVRC2012_img_train',
            validation_data_dir: str = 'ILSVRC2012/ILSVRC2012_img_val',
            test_data_dir: str = 'ILSVRC2012/ILSVRC2012_img_test_v10102019',
            out_shape: Optional[tuple[int, int]] = None,
            batch_size: int = 8,
            num_workers: int = 1,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.test_data_dir = test_data_dir
        self.out_shape = out_shape
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):

        def _get_filenames(data_dir: str) -> list[str]:
            image_list: list[str] = os.listdir(data_dir)
            image_paths: list[str] = [os.path.join(data_dir, _) for _ in image_list]
            return image_paths

        # list & collect all images
        self.train_image_paths = _get_filenames(self.train_data_dir)
        self.val_image_paths = _get_filenames(self.validation_data_dir)
        self.test_image_paths = _get_filenames(self.test_data_dir)

        # counting
        # self.train_count: int = len(self.train_image_paths)
        # self.val_count: int = len(self.val_image_paths)
        # self.test_count: int = len(self.test_image_paths)
        #
        # print(f'ILSVRCt1_2 total samples: {self.train_count + self.val_count + self.test_count}')
        # print(f'ILSVRCt1_2 train samples: {self.train_count}')
        # print(f'ILSVRCt1_2 val samples: {self.val_count}')
        # print(f'ILSVRCt1_2 test samples: {self.test_count}')

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.ilsvrc_t1_2_train: ILSVRC2012_Task1_2._dataset = \
                ILSVRC2012_Task1_2._dataset(self.train_image_paths, self.out_shape)
            self.ilsvrc_val: ILSVRC2012_Task1_2._dataset = \
                ILSVRC2012_Task1_2._dataset(self.val_image_paths, self.out_shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.ilsvrc_test: ILSVRC2012_Task1_2._dataset = \
                ILSVRC2012_Task1_2._dataset(self.test_image_paths, self.out_shape)

    def train_dataloader(self):
        return DataLoader(
            self.ilsvrc_t1_2_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.ilsvrc_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ilsvrc_test, batch_size=self.batch_size, num_workers=self.num_workers)


class ILSVRC2012_Task3(LightningDataModule):
    class _dataset(Dataset):
        def __init__(self,
                     image_paths: list[str],
                     crop_shape: tuple[int, int], ):
            self.image_paths: list[str] = image_paths
            self.crop_shape: tuple[int, int] = crop_shape
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomCrop(self.crop_shape),
                ]
            )

        def __getitem__(self, idx: int) -> torch.Tensor:
            item: np.ndarray = cv2.imread(self.image_paths[idx])

            # print(item.shape)
            imgw, imgh, _ = item.shape
            if imgh < self.crop_shape[0] or imgw < self.crop_shape[1]:
                item = transforms.ToPILImage()(item)
                item = transforms.Resize(min(self.crop_shape))(item)

            item: torch.Tensor = self.transform(item)
            item.mul_(2).add_(-1)

            # print(item.shape)

            return item

        def __len__(self) -> int:
            return len(self.image_paths)

    def __init__(
            self,
            train_data_dir: str = 'ILSVRC2012/ILSVRC2012_img_train_t3',
            validation_data_dir: str = 'ILSVRC2012/ILSVRC2012_img_val',
            test_data_dir: str = 'ILSVRC2012/ILSVRC2012_img_test_v10102019',
            out_shape: Optional[tuple[int, int]] = None,
            batch_size: int = 8,
            num_workers: int = 1,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.test_data_dir = test_data_dir
        self.out_shape = out_shape
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):

        def _get_filenames(data_dir: str) -> list[str]:
            image_list: list[str] = os.listdir(data_dir)
            image_paths: list[str] = [os.path.join(data_dir, _) for _ in image_list]
            return image_paths

        # list & collect all images
        self.train_image_paths = _get_filenames(self.train_data_dir)
        self.val_image_paths = _get_filenames(self.validation_data_dir)
        self.test_image_paths = _get_filenames(self.test_data_dir)

        # counting
        # self.train_count: int = len(self.train_image_paths)
        # self.val_count: int = len(self.val_image_paths)
        # self.test_count: int = len(self.test_image_paths)
        #
        # print(f'ILSVRCt3 total samples: {self.train_count + self.val_count + self.test_count}')
        # print(f'ILSVRCt3 train samples: {self.train_count}')
        # print(f'ILSVRCt3 val samples: {self.val_count}')
        # print(f'ILSVRCt3 test samples: {self.test_count}')

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.ilsvrc_t3_train: ILSVRC2012_Task3._dataset = \
                ILSVRC2012_Task3._dataset(self.train_image_paths, self.out_shape)
            self.ilsvrc_val: ILSVRC2012_Task3._dataset = \
                ILSVRC2012_Task3._dataset(self.val_image_paths, self.out_shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.ilsvrc_test: ILSVRC2012_Task3._dataset = \
                ILSVRC2012_Task3._dataset(self.test_image_paths, self.out_shape)

    def train_dataloader(self):
        return DataLoader(
            self.ilsvrc_t3_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.ilsvrc_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ilsvrc_test, batch_size=self.batch_size, num_workers=self.num_workers)


def Test_CelebAMaskHQ():
    print('\033[34;30mTesting DataModule: CelebAMask-HQ\033[0m')
    dataset = CelebAMaskHQ(data_dir='./CelebAMask-HQ/CelebA-HQ-img',
                           batch_size=2,
                           out_shape=(256, 256),
                           validation_ratio=0.1,
                           test_ratio=0.1)
    dataset.prepare_data()
    dataset.setup()
    dataset_train = dataset.train_dataloader()
    print('dataset_train:')
    for idx, batch in enumerate(dataset_train):
        print(f'idx: {idx} batch_size: {batch.shape}')
    dataset_val = dataset.val_dataloader()
    print('dataset_val:')
    for idx, batch in enumerate(dataset_val):
        print(f'idx: {idx} batch_size: {batch.shape}')
        for i in range(batch.size(0)):
            img: torch.Tensor = batch[i].permute(1, 2, 0)
            img.add_(1).mul_(0.5)
            img = img.numpy()
            cv2.imshow(f'val_{i}', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
    dataset_test = dataset.test_dataloader()
    print('dataset_test:')
    for idx, batch in enumerate(dataset_test):
        print(f'idx: {idx} batch_size: {batch.shape}')
        for i in range(batch.size(0)):
            img = batch[i].permute(1, 2, 0)
            img.add_(1).mul_(0.5)
            img = img.numpy()
            cv2.imshow(f'test_{i}', img)
            cv2.waitKey()
            cv2.destroyAllWindows()


def Test_ILSVRC2012t1_2():
    print('\033[34;30mTesting DataModule: ILSVRC2012 Task 1&2\033[0m')
    dataset = ILSVRC2012_Task1_2(batch_size=2,
                                 out_shape=(256, 256), )
    dataset.prepare_data()
    dataset.setup()
    dataset_train = dataset.train_dataloader()
    print('dataset_train:')
    for idx, batch in enumerate(dataset_train):
        print(f'idx: {idx} batch_size: {batch.shape}')
    dataset_val = dataset.val_dataloader()
    print('dataset_val:')
    for idx, batch in enumerate(dataset_val):
        print(f'idx: {idx} batch_size: {batch.shape}')
        for i in range(batch.size(0)):
            img = batch[i].permute(1, 2, 0)
            img.add_(1).mul_(0.5)
            img = img.numpy()
            cv2.imshow(f'val_{i}', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
    dataset_test = dataset.test_dataloader()
    print('dataset_test:')
    for idx, batch in enumerate(dataset_test):
        print(f'idx: {idx} batch_size: {batch.shape}')
        for i in range(batch.size(0)):
            img = batch[i].permute(1, 2, 0)
            img.add_(1).mul_(0.5)
            img = img.numpy()
            cv2.imshow(f'test_{i}', img)
            cv2.waitKey()
            cv2.destroyAllWindows()


def Test_ILSVRC2012t3():
    print('\033[34;30mTesting DataModule: ILSVRC2012 Task 3\033[0m')
    dataset = ILSVRC2012_Task3(batch_size=2,
                               out_shape=(256, 256), )
    dataset.prepare_data()
    dataset.setup()
    dataset_train = dataset.train_dataloader()
    print('dataset_train:')
    for idx, batch in enumerate(dataset_train):
        print(f'idx: {idx} batch_size: {batch.shape}')
    dataset_val = dataset.val_dataloader()
    print('dataset_val:')
    for idx, batch in enumerate(dataset_val):
        print(f'idx: {idx} batch_size: {batch.shape}')
        for i in range(batch.size(0)):
            img = batch[i].permute(1, 2, 0)
            img.add_(1).mul_(0.5)
            img = img.numpy()
            cv2.imshow(f'val_{i}', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
    dataset_test = dataset.test_dataloader()
    print('dataset_test:')
    for idx, batch in enumerate(dataset_test):
        print(f'idx: {idx} batch_size: {batch.shape}')
        for i in range(batch.size(0)):
            img = batch[i].permute(1, 2, 0)
            img.add_(1).mul_(0.5)
            img = img.numpy()
            cv2.imshow(f'test_{i}', img)
            cv2.waitKey()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    Test_CelebAMaskHQ()
    Test_ILSVRC2012t1_2()
    Test_ILSVRC2012t3()
