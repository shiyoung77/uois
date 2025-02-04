import os
import sys
sys.path.append('uois')
from time import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.contrib import tenumerate

import src.data_augmentation as data_augmentation
import src.segmentation as segmentation
import src.util.utilities as util_

# Depth Seeding Network Parameters
dsn_config = {
    # Sizes
    'feature_dim' : 64, # 32 would be normal

    # Mean Shift parameters (for 3D voting)
    'max_GMS_iters' : 10, 
    'epsilon' : 0.05, # Connected Components parameter
    'sigma' : 0.02, # Gaussian bandwidth parameter
    'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
    'subsample_factor' : 5,
    
    # Misc
    'min_pixels_thresh' : 500,
    'tau' : 15.,
}

# Region Refinement Network parameters
rrn_config = {
    # Sizes
    'feature_dim' : 64, # 32 would be normal
    'img_H' : 224,
    'img_W' : 224,
    
    # architecture parameters
    'use_coordconv' : False,
}

# UOIS-Net-3D Parameters
uois3d_config = {
    # Padding for RGB Refinement Network
    'padding_percentage' : 0.25,
    
    # Open/Close Morphology for IMP (Initial Mask Processing) module
    'use_open_close_morphology' : True,
    'open_close_morphology_ksize' : 9,
    
    # Largest Connected Component for IMP module
    'use_largest_connected_component' : True,
}

checkpoint_dir = 'checkpoints'
dsn_checkpoint_path = os.path.join(checkpoint_dir, 'DepthSeedingNetwork_3D_TOD_checkpoint.pth')
rrn_checkpoint_path = os.path.join(checkpoint_dir, 'RRN_OID_checkpoint.pth')
uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_checkpoint_path
uois_net_3d = segmentation.UOISNet3D(uois3d_config, 
                                     dsn_checkpoint_path,
                                     dsn_config,
                                     rrn_checkpoint_path,
                                     rrn_config
                                    )

dataset = "../dataset"
videos = [f"{idx:04d}" for idx in [20, 21, 28, 29, 35]]
print(videos)

for video in videos:
    video_folder = os.path.join(dataset, video)
    color_folder = os.path.join(video_folder, 'color')
    prefix_list = sorted([i.split('-')[0] for i in os.listdir(color_folder)])
    depth_trunc = 2.0

    with open(os.path.join(video_folder, "config.json")) as f:
        data_cfg = json.load(f)
    cam_intr = np.asarray(data_cfg['cam_intr'])

    # output folder for saliency and segmentation prediction
    sal_pred_folder = os.path.join(dataset, video, "predictions", "uois", "sal_pred")
    seg_pred_folder = os.path.join(dataset, video, "predictions", "uois", "seg_pred")
    os.makedirs(sal_pred_folder, exist_ok=True)
    os.makedirs(seg_pred_folder, exist_ok=True)

    for i, prefix in tenumerate(prefix_list):
        color_im_path = os.path.join(video_folder, 'color', prefix_list[i] + '-color.png')
        depth_im_path = os.path.join(video_folder, 'depth', prefix_list[i] + '-depth.png')
        color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
        depth_im[depth_im > depth_trunc] = 0
        im_h, im_w = depth_im.shape

        cam_params = {
            'fx': cam_intr[0, 0],
            'fy': cam_intr[1, 1],
            'x_offset': cam_intr[0, 2],
            'y_offset': cam_intr[1, 2],
            'img_height': im_h,
            'img_width': im_w
        }
        xyz = util_.compute_xyz(depth_im, cam_params)   # XYZ is in left-handed coordinate system!
        rgb = data_augmentation.standardize_image(color_im)  # (H, W, 3)

        batch = {
            'rgb': data_augmentation.array_to_tensor(rgb).unsqueeze(0),
            'xyz': data_augmentation.array_to_tensor(xyz).unsqueeze(0)
        }

        fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)

        # saliency prediction
        fg_mask = fg_masks[0].cpu().numpy()  # label = 0: background, 1: table, 2: foreground
        fg_mask[fg_mask == 1] = 0  # merge table into background
        fg_mask[fg_mask == 2] = 1  # change foreground label to 1
        cv2.imwrite(os.path.join(sal_pred_folder, f"{prefix}.png"), fg_mask.astype(np.uint16))

        seg_mask = seg_masks[0].cpu().numpy()
        cv2.imwrite(os.path.join(seg_pred_folder, f"{prefix}.png"), seg_mask.astype(np.uint16))
            
    cv2.destroyAllWindows()
