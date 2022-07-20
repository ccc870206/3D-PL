from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--results_dir', type=str, default='./results/', help = 'saves results here')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--img_target_file', type=str, default='./datasplit/test_kitti.txt',
                                 help='training and testing dataser for target domain')
        self.parser.add_argument('--shuffle', dest='shuffle', action='store_true')
        self.parser.set_defaults(shuffle=False)
        self.parser.add_argument('--dataset_mode', type=str, default='paired',
                                 help='chooses how datasets are loaded. [paired| unpaired]')
        self.isTrain = False