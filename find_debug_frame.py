import torch
import random
import numpy as np
import sys
from scene import Scene, GaussianModel
from arguments import ModelParams
from argparse import ArgumentParser
from utils.general_utils import safe_state

def find_first_camera(dataset):
    # Initialize system state (RNG) to match train.py
    safe_state(False)
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    first_cam = scene.getTrainCameras()[0]
    print(f"First Training Camera Name: {first_cam.image_name}")

if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = parser.parse_args(sys.argv[1:])
    find_first_camera(lp.extract(args))
