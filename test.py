import cv2
import numpy as np 
import os

from dataloader.data_loader import dataloader
from model.models import create_model
from options.test_options import TestOptions
from util.visualizer import Visualizer
from util import util


# load options and set batchsize
opt = TestOptions().parse("test")
opt.batchSize = 1

# load dataset
dataset = dataloader(opt)
dataset_size = len(dataset) * opt.batchSize
print ('testing images = %d ' % dataset_size)

# create model and visualizer
model = create_model(opt)
visualizer = Visualizer(opt)

# create folder to save result
save_dir = os.path.join(opt.results_dir,opt.name, '%s_%s' %(opt.phase, opt.which_epoch))
os.makedirs(os.path.join(save_dir, 'img'), exist_ok=True)

# initialize arrays that store metrics 
num_samples = len(dataset)
abs_rel = np.zeros(num_samples, np.float32)
sq_rel = np.zeros(num_samples,np.float32)
rmse = np.zeros(num_samples,np.float32)
rmse_log = np.zeros(num_samples,np.float32)
a1 = np.zeros(num_samples,np.float32)
a2 = np.zeros(num_samples,np.float32)
a3 = np.zeros(num_samples,np.float32)

# set evaluation range
MAX_DEPTH = 50. 
MIN_DEPTH = 1e-3

# testing
for i,data in enumerate(dataset):
    # get depth prediction from model
    model.set_input(data)
    predicted_depth = model.test()
    predicted_depth = predicted_depth[-1].squeeze()

    # get ground truth
    ground_depth = data['lab_target'].squeeze().data.cpu().numpy()
    height, width = ground_depth.shape

    # resize and denormalize the depth prediction
    predicted_depth = cv2.resize(predicted_depth.data.cpu().numpy(),(width, height),interpolation=cv2.INTER_LINEAR)
    predicted_depth = (predicted_depth * 0.5 + 0.5) * 80

    predicted_depth[predicted_depth < MIN_DEPTH] = MIN_DEPTH
    predicted_depth[predicted_depth > MAX_DEPTH] = MAX_DEPTH

    # evaluate the result with the ground truth depth between the range of MIN_DEPTH and MAX_DEPTH
    mask = np.logical_and(ground_depth > MIN_DEPTH, ground_depth < MAX_DEPTH)

    # crop used by Garg ECCV16
    crop = np.array([0.40810811 * height,  0.99189189 * height,
                            0.03594771 * width,   0.96405229 * width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)

    # print the result for each image
    abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i] = util.compute_errors(ground_depth[mask],predicted_depth[mask])

    print('{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f}'
            .format(i, abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i]))
    model.save_results(visualizer, save_dir)

# print final result for the whole dataset
print ('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3'))
print ('{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f}'
        .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(),a1.mean(),a2.mean(),a3.mean()))

# save the result
best_cla_txt = save_dir + '/evaluation_result.txt'
with open(best_cla_txt, 'w') as txtfile:
    txtfile.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10},  {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3')+'\n')
    txtfile.write('{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f}'
        .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(),a1.mean(),a2.mean(),a3.mean())+'\n')
    txtfile.write('{:10.3f}&{:10.3f}&{:10.3f}&{:10.3f}&{:10.3f}&{:10.3f}&{:10.3f}'
        .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(),a1.mean(),a2.mean(),a3.mean())+'\n')