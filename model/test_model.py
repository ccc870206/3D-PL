import torch
from torch.autograd import Variable
from .base_model import BaseModel
from . import network
from util import util
import numpy as np
from collections import OrderedDict

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert (not opt.isTrain)
        BaseModel.initialize(self, opt)

        self.loss_names = []
        self.visual_names =['img_t', 'lab_t_g']
        self.model_names = ['img2task']

        # define the task network
        self.net_img2task = network.define_G(opt.image_nc, opt.label_nc, opt.ngf, opt.task_layers, opt.norm,
                                             opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
                                             False, opt.gpu_ids, opt.U_weight)

        self.load_networks(opt.which_epoch)

    def set_input(self, input):
        self.input = input
        self.img_target = input['img_target']

        if len(self.gpu_ids) > 0:
            self.img_target = self.img_target.cuda()

    def test(self):
        self.img_t = Variable(self.img_target)

        with torch.no_grad():
            self.lab_t_g = self.net_img2task.forward(self.img_t)

        return self.lab_t_g
        
    # save_results
    def save_results(self, visualizer, wed_page):
        img_target_paths = self.input['img_target_paths']

        for i in range(self.img_t.size(0)):
            img_target = util.tensor2im(self.img_t.data[i])
            
            lab_fake_target = util.tensor2im(self.lab_t_g[-1].data[i], bytes=80., imtype=np.float32)

            visuals = OrderedDict([('img_t', img_target), ('lab_t_g', lab_fake_target)])
            print('process image ......%s' % img_target_paths[0])
            visualizer.save_images(wed_page, visuals, img_target_paths)
            img_target_paths.pop(0)