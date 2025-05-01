import logging
from glob import glob
from os import listdir
from os.path import splitext

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, train, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.train = train
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.ids = list(map(int, self.ids))
        self.ids.sort()
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img, scale, p):
        img_nd = np.array(pil_img)
        # plt.imshow(img_nd)
        # plt.show()
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        idx = str(idx)
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        p = np.random.choice([0, 1])
        mask = self.preprocess(mask, self.scale, p)
        img = self.preprocess(img, self.scale, p)
        return {
            'idx': idx,
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'label': torch.from_numpy(mask).type(torch.IntTensor)
        }


class TrainBasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, train, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.train = train
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.ids = list(map(int, self.ids))
        self.ids.sort()
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img, scale, p):
        img_nd = np.array(pil_img)
        # plt.imshow(img_nd)
        # plt.show()
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        idx = str(idx)
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        p = np.random.choice([0, 1])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p),  # 指定是否翻转
        ])
        mask = transform(mask)
        img = transform(img)

        # p = np.random.choice([0, 1])
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        p = np.random.choice([0, 1])
        mask = self.preprocess(mask, self.scale, p)
        img = self.preprocess(img, self.scale, p)
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'label': torch.from_numpy(mask).type(torch.IntTensor)
        }
