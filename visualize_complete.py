import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scene import Scene, GaussianModel
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
import sys
from types import SimpleNamespace as Namespace
import random

def get_combined_args(parser : ArgumentParser, model_path):
    # Retrieve config args from model path
    cfgfilepath = os.path.join(model_path, "cfg_args")
    try:
        with open(cfgfilepath) as cfg_file:
            cfgfile_string = cfg_file.read()
    except FileNotFoundError:
        print(f"Config file not found at {cfgfilepath}")
        return None

    # cfg_args dosyası 'Namespace(...)' string'i içerir, eval ile parse ediyoruz
    try:
        # Namespace class'ının görünür olduğundan emin olalım (import types...)
        # Ancak eval stringi içinde Namespace varsa local scope'da olması gerek.
        args_cfgfile = eval(cfgfile_string)
    except Exception as e:
        print(f"Error parsing config file: {e}")
        return None
    
    # Temiz bir Namespace oluştur
    args = Namespace(**vars(args_cfgfile))
    args.model_path = model_path
    
    return args

def render_depth_map(view, gaussians, pipeline_args, full_args, background):
    # 1. Compute view-space depth
    
    # 3DGS stores W2C as Transposed [4, 4]
    w2c = view.world_view_transform
    
    pts = gaussians.get_xyz # [N, 3]
    ones = torch.ones((pts.shape[0], 1), device=pts.device)
    pts_homo = torch.cat([pts, ones], dim=1) # [N, 4]
    
    # Transform points to Camera Space
    # P_cam = P_world @ w2c
    pts_cam = pts_homo @ w2c
    depths = pts_cam[:, 2:3] # Z coordinate [N, 1]
    
    # 2. Render Depth Accumulation
    # We need 3 channels for render()
    depth_features = depths.repeat(1, 3)
    
    # Background for depth should be 0 
    bg_zero = torch.zeros(3, device="cuda", dtype=torch.float32)
    
    # Render with depth as color
    depth_pkg = render(view, gaussians, pipeline_args, bg_zero, opt=full_args, scaling_modifier=1.0, override_color=depth_features)
    depth_accum = depth_pkg["render"] # [3, H, W]
    
    # 3. Render Alpha Accumulation (Normalization weight)
    alpha_features = torch.ones_like(depth_features)
    alpha_pkg = render(view, gaussians, pipeline_args, bg_zero, opt=full_args, scaling_modifier=1.0, override_color=alpha_features)
    alpha_accum = alpha_pkg["render"]
    
    # 4. Normalize
    # Take the first channel
    depth_map = depth_accum[0] / (alpha_accum[0] + 1e-6)
    
    return depth_map

def visualize_model(model_path, dataset_args, pipeline_args, iteration):
    print(f"Processing model: {model_path} at iteration {iteration}")
    
    full_args = get_combined_args(ArgumentParser(), model_path)
    if full_args is None: return
    
    # Override iteration and flags
    full_args.iteration = iteration
    if not hasattr(full_args, "include_feature"): full_args.include_feature = False
    if not hasattr(full_args, "quick_render"): full_args.quick_render = False

    # Initialize Model and Scene
    try:
        sh_degree = getattr(full_args, "sh_degree", 3)
        gaussians = GaussianModel(sh_degree)
        scene = Scene(full_args, gaussians, load_iteration=iteration, shuffle=False)
    except Exception as e:
        print(f"Error loading scene: {e}")
        return
    
    bg_color = [1, 1, 1] if full_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Output dir
    out_dir = os.path.join(model_path, f"vis_complete_{iteration}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Pick random cameras
    cameras = scene.getTestCameras()
    if len(cameras) < 5:
        cameras = scene.getTrainCameras()
        
    random.seed(42)
    selected_indices = random.sample(range(len(cameras)), min(5, len(cameras)))
    selected_cameras = [cameras[i] for i in selected_indices]
    
    print(f"Saving to {out_dir}...")
    
    for idx, view in enumerate(selected_cameras):
        # RGB Render
        rgb_pkg = render(view, gaussians, pipeline_args, background, opt=full_args, scaling_modifier=1.0)
        rgb = rgb_pkg["render"].detach().cpu().permute(1, 2, 0).numpy()
        rgb = np.clip(rgb, 0, 1)
        
        # Depth Render
        depth_map = render_depth_map(view, gaussians, pipeline_args, full_args, background)
        depth = depth_map.detach().cpu().numpy()
        
        # Visualize Depth
        valid_mask = (depth > 1e-5)
        if valid_mask.sum() > 0:
            d_min = depth[valid_mask].min()
            d_max = depth[valid_mask].max()
            depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
            depth_norm = np.clip(depth_norm, 0, 1)
        else:
            depth_norm = depth
            
        depth_colormap = plt.get_cmap("turbo")(depth_norm)[:, :, :3] # [H, W, 3]
        depth_colormap[~valid_mask] = 0
        
        # Combine
        combined = np.hstack([rgb, depth_colormap])
        
        # Save
        plt.imsave(os.path.join(out_dir, f"view_{idx:03d}_{view.image_name}.png"), combined)
        
    print("Done for this model.")

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int) 
    # Varsayılan 30000, eğer eğitim 7000'de durduysa argümanla değiştirilmeli
    
    args = parser.parse_args(sys.argv[1:]) 
    if args.source_path is None:
        args.source_path = "dummy_path" # Dummy path to avoid os.path.abspath error
    if args.language_features_name is None:
        args.language_features_name = "dummy_feat"
    dataset_args = model.extract(args)
    pipeline_args = pipeline.extract(args)
    if not hasattr(pipeline_args, 'debug'): pipeline_args.debug = False

    base_dir = "/home/yusuf/AES/hidayet/LangSplatV2"
    models_to_process = [
        "output/waldo_kitchen_baseline_-1",
        "output/waldo_kitchen_sam2_-1",
        "output/waldo_final_ours_-1",
        "output/waldo_final_balanced_-1",
    ]
    
    for m in models_to_process:
        full_path = os.path.join(base_dir, m)
        if os.path.exists(full_path):
            vis_iteration = args.iteration
            # Klasör kontrolü yapalım, eğer 30000 yoksa 7000 vs deneyebiliriz ama user argüman vermeli.
            # Otomatik check: models path içindeki point_cloud klasörlerine bakıp en sonuncuyu alabiliriz.
            # Ama şimdilik iteration argümanına güveniyoruz.
            visualize_model(full_path, dataset_args, pipeline_args, vis_iteration)
        else:
            print(f"Skipping non-existent model: {full_path}")
