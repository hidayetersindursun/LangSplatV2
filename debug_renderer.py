import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from eval.openclip_encoder import OpenCLIPNetwork
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Helper function for language features
def render_language_feature_map_quick(gaussians, view, pipeline, background, args):
    with torch.no_grad():
        output = render(view, gaussians, pipeline, background, args)
        language_feature_weight_map = output['language_feature_weight_map']
        
        D, H, W = language_feature_weight_map.shape
        num_levels = D // 64
        
        # Reshape for memory efficiency
        language_feature_weight_map = language_feature_weight_map.view(num_levels, 64, H, W).view(num_levels, 64, H*W)
        
        # Prepare codebooks
        language_codebooks = gaussians._language_feature_codebooks.permute(0, 2, 1)
        
        # EINSUM Operation
        language_feature_map = torch.einsum('ldk,lkn->ldn', language_codebooks, language_feature_weight_map).view(num_levels, 512, H, W)
        
        # Normalization
        language_feature_map = language_feature_map / (language_feature_map.norm(dim=1, keepdim=True) + 1e-10)
        
    return language_feature_map

def debug_render(dataset, iteration, pipeline, skip_train, skip_test, args):
    with torch.no_grad():
        # Initialize Gaussian Model
        gaussians = GaussianModel(dataset.sh_degree)
        
        # Initialize Scene (loads cameras)
        # We pass load_iteration=iteration to let it find the iteration number, 
        # but we will overwrite the model with the checkpoint.
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # Manually load checkpoint to get language features
        # Scene.load_ply only loads geometry, not language features (codebooks/logits)
        checkpoint_path = os.path.join(dataset.model_path, "chkpnt{}.pth".format(scene.loaded_iter))
        if not os.path.exists(checkpoint_path):
             # Try searching for the max iteration checkpoint if exact match fails
             # But usually scene.loaded_iter comes from searchForMaxIteration
             pass
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        (model_params, first_iter) = torch.load(checkpoint_path)
        gaussians.restore(model_params, args)
        
        print("Logits Stats:")
        print(f"Min: {gaussians._language_feature_logits.min().item()}")
        print(f"Max: {gaussians._language_feature_logits.max().item()}")
        print(f"Mean: {gaussians._language_feature_logits.mean().item()}")
        print(f"Std: {gaussians._language_feature_logits.std().item()}")
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Load CLIP Model
        clip_model = OpenCLIPNetwork("cuda")
        
        # Select a view (First training camera)
        view = scene.getTrainCameras()[0]
        print(f"Rendering view: {view.image_name}")
        
        # 1. Render RGB
        rendering = render(view, gaussians, pipeline, background, args)["render"]
        rgb_img = rendering.permute(1, 2, 0).cpu().numpy()
        
        # 2. Render Language Features
        lf_map = render_language_feature_map_quick(gaussians, view, pipeline, background, args)
        # lf_map is [3, 512, H, W]
        
        # Take Level 0
        lf_map_0 = lf_map[0].permute(1, 2, 0) # [H, W, 512]
        
        # 3. Query "car"
        prompts = ["car", "tree", "road"]
        text_embeds = clip_model.encode_text(prompts, "cuda")
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds.to(lf_map_0.dtype)
        
        sims = torch.matmul(lf_map_0, text_embeds.T) # [H, W, 3]
        
        # Visualize
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.title("RGB Render")
        plt.imshow(np.clip(rgb_img, 0, 1))
        plt.axis('off')
        
        for i, prompt in enumerate(prompts):
            sim_map = sims[..., i].cpu().numpy()
            
            plt.subplot(1, 4, i + 2)
            plt.title(f"Sim: {prompt}")
            plt.imshow(sim_map, cmap='jet')
            plt.colorbar()
            plt.axis('off')
            
            print(f"{prompt} - Min: {sim_map.min():.4f}, Max: {sim_map.max():.4f}, Mean: {sim_map.mean():.4f}")

        plt.savefig("debug_render_result.png")
        print("Saved debug_render_result.png")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    # Add missing args manually
    parser.add_argument("--include_feature", action="store_true", default=True)
    parser.add_argument("--quick_render", action="store_true", default=False)
    parser.add_argument("--topk", default=1, type=int)
    parser.add_argument("--vq_layer_num", default=1, type=int)
    parser.add_argument("--codebook_size", default=64, type=int)
    parser.add_argument("--percent_dense", default=0.01, type=float)
    parser.add_argument("--language_feature_lr", default=0.0025, type=float)
    
    # Add ALL other OptimizationParams to satisfy training_setup
    parser.add_argument("--position_lr_init", default=0.00016, type=float)
    parser.add_argument("--position_lr_final", default=0.0000016, type=float)
    parser.add_argument("--position_lr_delay_mult", default=0.01, type=float)
    parser.add_argument("--position_lr_max_steps", default=30000, type=int)
    parser.add_argument("--feature_lr", default=0.0025, type=float)
    parser.add_argument("--opacity_lr", default=0.05, type=float)
    parser.add_argument("--scaling_lr", default=0.005, type=float)
    parser.add_argument("--rotation_lr", default=0.001, type=float)
    parser.add_argument("--lambda_dssim", default=0.2, type=float)
    parser.add_argument("--densification_interval", default=100, type=int)
    parser.add_argument("--opacity_reset_interval", default=3000, type=int)
    parser.add_argument("--densify_from_iter", default=500, type=int)
    parser.add_argument("--densify_until_iter", default=15000, type=int)
    parser.add_argument("--densify_grad_threshold", default=0.0002, type=float)
    # feature_level is already in ModelParams, so we don't add it here.

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    debug_render(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
