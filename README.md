# [ECCV2022] 3D-PL: Domain Adaptive Depth Estimation with 3D-aware Pseudo-Labeling
### [Paper] [[Project Page](https://ccc870206.github.io/3D-PL/)]
<div align=center><img src="https://github.com/ccc870206/3D-PL/blob/main/figure/teaser.jpg"/></div>

## Installation
* This code was developed with Python 3.7.10 & Pytorch 1.8.1 & CUDA 11.3
* Other requirements: numpy, cv2, tensorboardX
* Clone this repo
```
git clone https://github.com/ccc870206/3D-PL.git
cd 3D-PL
```

## Dataset
Target dataset: [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)

Rename the main folder of kitti dataset as `kitti_data` and put the folder under `data/`
```
data
  |----kitti_data 
         |----2011_09_26         
         |----2011_09_28        
         |----......... 
```
Source dataset: [vKITTI](https://europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds/) (1.3.1)

## Testing
Download our [pre-trained model](https://drive.google.com/drive/folders/1ZeCHo7ktv1zAu1R-bA29DLJ0SbcLbrIy?usp=sharing) and put the folder under `checkpoints/`. 

- Test the model pre-trained with single-image setting
```
python3 test.py --model test --name best_model_single_image --which_epoch best
```

- Test the model pre-trained with stereo-pair setting
```
python3 test.py --model test --name best_model_stereo_pair --which_epoch best
```

## Acknowledgments
Code is inspired by [T^2Net](https://github.com/lyndonzheng/Synthetic2Realistic) and [GASDA](https://github.com/sshan-zhao/GASDA).
