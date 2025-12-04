# eval_araba.py
import torch
from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
import os
import sys

def evaluate(dataset, pipeline, args):
    # Load Scene and Gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    
    # Load specific iteration if requested, otherwise load latest
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"Loading model from {args.model_path} at iteration {args.iteration}")

    # Get Test Cameras
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("⚠️ No test cameras found! Using train cameras for demo (first 10).")
        test_cameras = scene.getTrainCameras()[:10]

    print(f"Evaluating on {len(test_cameras)} images...")

    total_psnr = 0.0
    
    with torch.no_grad():
        for idx, view in enumerate(test_cameras):
            # Render
            rendering = render(view, gaussians, pipeline, background, args)["render"]
            
            # Ground Truth
            gt = view.original_image[0:3, :, :]
            
            # Calculate PSNR
            p = psnr(rendering, gt).mean().double()
            total_psnr += p
            
            print(f"Image {view.image_name}: PSNR = {p:.4f}")

    avg_psnr = total_psnr / len(test_cameras)
    print(f"\n✅ Evaluation Complete.")
    print(f"Average PSNR: {avg_psnr:.4f}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    
    args = get_combined_args(parser)
    safe_state(args.quiet)
    
    evaluate(model.extract(args), pipeline.extract(args), args)
