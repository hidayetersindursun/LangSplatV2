#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import pickle
import torch
import cv2
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", gt_mask=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            
        self.gt_mask = gt_mask
        if self.gt_mask is not None:
            self.gt_mask = self.gt_mask.to(self.data_device)
            
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    def get_language_feature(self, language_feature_dir, feature_level):
        language_feature_name = os.path.join(language_feature_dir, self.image_name)
        seg_map_np = np.load(language_feature_name + '_s.npy')
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
        
        # Check if resizing is needed
        if seg_map_np.shape[1] != self.image_height or seg_map_np.shape[2] != self.image_width:
            # Resize seg_map using Nearest Neighbor to preserve integer indices
            # seg_map_np is [Levels, H, W]. We transpose to [H, W, Levels] for cv2.resize
            seg_map_resized = []
            for i in range(seg_map_np.shape[0]):
                resized_layer = cv2.resize(seg_map_np[i], (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
                seg_map_resized.append(resized_layer)
            seg_map_np = np.array(seg_map_resized)

        seg_map = torch.from_numpy(seg_map_np)
        
        y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        seg = seg_map[:, y, x].squeeze(-1).long()
        mask = seg != -1
        if feature_level == 0: # default
            point_feature1 = feature_map[seg[0:1]].squeeze(0)
            mask = mask[0:1].reshape(1, self.image_height, self.image_width)
        elif feature_level == 1: # s
            point_feature1 = feature_map[seg[1:2]].squeeze(0)
            mask = mask[1:2].reshape(1, self.image_height, self.image_width)
        elif feature_level == 2: # m
            point_feature1 = feature_map[seg[2:3]].squeeze(0)
            mask = mask[2:3].reshape(1, self.image_height, self.image_width)
        elif feature_level == 3: # l
            point_feature1 = feature_map[seg[3:4]].squeeze(0)
            mask = mask[3:4].reshape(1, self.image_height, self.image_width)
        else:
            raise ValueError("feature_level=", feature_level)
        point_feature = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
        return point_feature.cuda(), mask.cuda()

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

