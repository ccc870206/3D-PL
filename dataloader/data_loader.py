import random
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
from util.kitti_util import KITTI
import numpy as np
import torch


class CreateDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt

        self.img_target_paths, self.img_target_size = make_dataset(opt.img_target_file)
        if self.opt.isTrain:
            self.lab_target_paths, self.lab_target_size = make_dataset(opt.lab_target_file)

        self.transform_augment = get_transform(opt, True)
        self.transform_no_augment = get_transform(opt, False)
        self.transform_depth_no_augment = get_depth_transform(opt, False)
        

        self.kitti = KITTI()

    def __getitem__(self, item):
        if self.opt.dataset_mode == 'paired':
            img_target_path = self.img_target_paths[item % self.img_target_size]
        elif self.opt.dataset_mode == 'unpaired':
            img_target_path = self.img_target_paths[index]
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        img_target = Image.open(img_target_path).convert('RGB')

        # KITTI depth
        w = img_target.size[0]
        h = img_target.size[1]

        velo_path = img_target_path.replace("image_02", "velodyne_points")
        velo_path = velo_path.replace("png", "bin")
        calib_path = "/".join(img_target_path.split('/')[:-4])
        gt, gt_interp = self.kitti.get_depth(calib_path,velo_path, [h, w], interp=True)

        img_target = img_target.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)
        img_target = self.transform_no_augment(img_target)

        return {'img_target': img_target,
                'lab_target': gt,
                'img_target_paths': img_target_path,
                }

    def __len__(self):
        return self.img_target_size

def dataloader(opt):
    datasets = CreateDataset()
    datasets.initialize(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=opt.shuffle, num_workers=int(opt.nThreads))

    return dataset

def get_transform(opt, augment):
    transforms_list = []

    if augment:
        if opt.isTrain:
            transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
    transforms_list += [
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    return transforms.Compose(transforms_list)

def get_depth_transform(opt, augment):
    transforms_list = []

    if augment:
        if opt.isTrain:
            transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
    transforms_list += [
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))
    ]
    
    return transforms.Compose(transforms_list)
