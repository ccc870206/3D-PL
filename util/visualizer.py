import numpy as np
import os
import ntpath
import time
from . import util
import matplotlib.pyplot as plt
from PIL import Image

class Visualizer():
    def __init__(self, opt):
        self.opt = opt

    # save image to the disk
    def save_images(self, save_dir, visuals, image_path):
        image_dir = os.path.join(save_dir, 'img')
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        for label, image_numpy in visuals.items():
            if label == 'lab_t_g':
                image_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(image_dir, image_name)
                image_numpy = image_numpy.reshape(192, 640)
                util.save_img(save_path, util.convert_array_vis(image_numpy))
            else:
                image_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(image_dir, image_name)
                
                util.save_img(save_path, image_numpy)
