import os.path
from io import BytesIO
import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import cv2
import numpy as np
import math
from torchvision import transforms


def load_img_crop(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = img/255.
    return img


def load_img_crop_gray(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = img/255.
    return img


# 数据集
class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, split='train', data_len=-1):
        self.datatype = datatype
        self.data_len = data_len
        self.split = split

        if datatype == 'img':
            if split == 'train':
                self.gt_path = Util.get_paths_from_images(os.path.join(dataroot, 'gt'))
                self.mask_path = Util.get_paths_from_images(os.path.join(dataroot, 'mask'))
                self.mask_dilated_path = Util.get_paths_from_images(os.path.join(dataroot, 'mask_dilated'))
            else:
                self.gt_path = Util.get_paths_from_images(os.path.join(dataroot, 'gt'))       # gt
                self.mask_path = Util.get_paths_from_images(os.path.join(dataroot, 'mask'))
            # self.input_path = Util.get_paths_from_images(os.path.join(dataroot, 'specular'))
            self.input_path = Util.get_paths_from_images(os.path.join(dataroot, 'ge_gt'))

            self.dataset_len = len(self.input_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        gt_path = self.gt_path[index]
        input_path = self.input_path[index]

        img_input = torch.from_numpy(load_img_crop(input_path))
        img_input = img_input.permute(2, 0, 1)  # c, h, w
        img_gt = torch.from_numpy(load_img_crop(gt_path))
        img_gt = img_gt.permute(2, 0, 1)

        if self.split == 'train':
            mask_dilated_path = self.mask_dilated_path[index]
            mask_dilated = torch.from_numpy(load_img_crop_gray(mask_dilated_path))
            mask_dilated = mask_dilated.unsqueeze(-1)
            mask_dilated = mask_dilated.permute(2, 0, 1)
            mask_dilated = mask_dilated.repeat(3, 1, 1)

            mask_path = self.mask_path[index]
            mask = torch.from_numpy(load_img_crop_gray(mask_path))
            mask = mask.unsqueeze(-1)
            mask = mask.permute(2, 0, 1)
            mask = mask.repeat(3, 1, 1)
            [img_gt, img_input, mask, mask_dilated] = Util.transform_augment(
                [img_gt, img_input, mask, mask_dilated], split=self.split, min_max=(0, 1))
            mask = mask[0:1, :, :]
            mask_dilated = mask_dilated[0:1, :, :]
            return {'GT': img_gt, 'Input': img_input, 'Mask': mask, 'Mask_Dilated': mask_dilated}
        else:
            mask_path = self.mask_path[index]
            mask = torch.from_numpy(load_img_crop_gray(mask_path))
            mask = mask.unsqueeze(-1)
            mask = mask.permute(2, 0, 1)
            mask = mask.repeat(3, 1, 1)
            [img_gt, img_input, mask] = Util.transform_augment(
                [img_gt, img_input, mask], split=self.split, min_max=(0, 1))
            mask = mask[0:1, :, :]
            return {'GT': img_gt, 'Input': img_input, 'Mask': mask}
