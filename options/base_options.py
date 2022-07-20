import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # basic define
        self.parser.add_argument('--name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                                 help='models are save here')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load')
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')
        self.parser.add_argument('--model', type=str, default='wsupervised',
                                 help='choose which model to use, [supervised] | [wsupervised]')

        # data pattern define
        self.parser.add_argument('--loadSize', type=list, default=[640, 192],
                                 help='load image into same size [256, 192]|[640, 192]')
        self.parser.add_argument('--flip', action='store_true',
                                 help='if specified, do flip the image for data augmentation')
        self.parser.add_argument('--scale_rate', type=float, default=0,
                                 help='scale images with same rate')
        self.parser.add_argument('--rotation', action='store_true',
                                 help='if specified, rotate the images for data augmentation')
        self.parser.add_argument('--crop', action='store_true',
                                 help='if specified, crop the images for data augmentation')
        self.parser.add_argument('--batchSize', type=int, default=6,
                                 help='input batch size')
        self.parser.add_argument('--nThreads', type=int, default=2,
                                 help='# threads for loading data')

        # network structure define
        self.parser.add_argument('--image_nc', type=int, default=3,
                                 help='# of input image channels')
        self.parser.add_argument('--label_nc', type=int, default=1,
                                 help='# of output label channels')
        self.parser.add_argument('--ngf', type=int, default=64,
                                 help='# of encoder filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64,
                                 help='# of discriminator filter in first conv layer')
        self.parser.add_argument('--task_layers', type=int, default=4,
                                 help='# of number of the down sample layers for task network')
        self.parser.add_argument('--task_model_type', type=str, default='UNet',
                                 help='select model for task network [UNet] |[ResNet]')
        self.parser.add_argument('--norm', type=str, default='batch',
                                 help='batch normalization or instance normalization')
        self.parser.add_argument('--activation', type=str, default='PReLU',
                                 help='ReLu, LeakyReLU, PReLU, or SELU')
        self.parser.add_argument('--init_type', type=str, default='kaiming',
                                 help='network initialization [normal|xavier|kaiming]')
        self.parser.add_argument('--drop_rate', type=float, default=0,
                                 help='# of drop rate')
        self.parser.add_argument('--U_weight', type=float, default=0.1,
                                 help='weight for Unet')
   

    def parse(self, mode):
        if not self.initialized:
            self.initialize()
        if mode == "train":
            self.opt=self.parser.parse_args()
        else:
            self.opt, unknown = self.parser.parse_known_args()

        self.opt.isTrain = self.isTrain

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >=0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids):
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('--------------Options--------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('----------------End----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if self.opt.isTrain:
            file_name = os.path.join(expr_dir, 'train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('--------------Options--------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('----------------End----------------\n')

        return self.opt