import numpy as np
import os
import cv2
import argparse

def check_shapes(dataset_path, image_name):
    img_path = os.path.join(dataset_path, "images", image_name + ".jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(dataset_path, "images", image_name + ".png")
        
    feat_path_s = os.path.join(dataset_path, "language_features", image_name + "_s.npy")
    
    img = cv2.imread(img_path)
    seg_map = np.load(feat_path_s)
    
    print(f"Image Shape: {img.shape}")
    print(f"Seg Map Shape: {seg_map.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--image_name", default="frame_00001")
    args = parser.parse_args()
    check_shapes(args.dataset_path, args.image_name)
