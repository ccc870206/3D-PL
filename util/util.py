import numpy as np
import os
import torch
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image


def compute_errors(ground_truth, predication):
    # accuracy
    threshold = np.maximum((ground_truth / predication),(predication / ground_truth))
    a1 = (threshold < 1.25 ).mean()
    a2 = (threshold < 1.25 ** 2 ).mean()
    a3 = (threshold < 1.25 ** 3 ).mean()

    #MSE
    rmse = (ground_truth - predication) ** 2
    rmse = np.sqrt(rmse.mean())

    #MSE(log)
    rmse_log = (np.log(ground_truth) - np.log(predication)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # Abs Relative difference
    abs_rel = np.mean(np.abs(ground_truth - predication) / ground_truth)

    # Squared Relative difference
    sq_rel = np.mean(((ground_truth - predication) ** 2) / ground_truth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)


# convert a tensor into a numpy array
def numpy2im(image_numpy, bytes=255.0, imtype=np.uint8):
    image_numpy = (image_numpy * 0.5 + 0.5) * bytes

    return image_numpy.astype(imtype)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_crop_mask(size, height=192, width=640):

    crop = np.array([0.40810811 * height,  0.99189189 * height, 0.03594771 * width,   0.96405229 * width]).astype(np.int32)
    crop_mask = np.zeros(size)
    crop_mask[:,:,crop[0]:crop[1],crop[2]:crop[3]] = 1
    if torch.cuda.is_available():
        crop_mask = torch.from_numpy(crop_mask).cuda()
    else:
        crop_mask = torch.from_numpy(crop_mask)

    return crop_mask

def convert_array_vis(depth):
    mask = depth!=0
    depth = depth.clip(1, 80)
    # We actually visualize disparity as
    # the qualitative differences are easier to see
    disparity = 1/depth
    vmax = np.percentile(disparity[mask], 95)
    normalizer = mpl.colors.Normalize(vmin=disparity.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disparity)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask] = 255
    return colormapped_im


def save_img(path, result_img):
    pred_img1 = Image.fromarray((result_img).astype('uint8'))
    pred_img1.save(path)