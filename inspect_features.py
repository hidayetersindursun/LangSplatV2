import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import argparse
from eval.openclip_encoder import OpenCLIPNetwork

def visualize_features(dataset_path, image_name):
    # Paths
    img_path = os.path.join(dataset_path, "images", image_name + ".jpg") # Assuming jpg
    if not os.path.exists(img_path):
        img_path = os.path.join(dataset_path, "images", image_name + ".png")
    
    feat_path_s = os.path.join(dataset_path, "language_features", image_name + "_s.npy")
    feat_path_f = os.path.join(dataset_path, "language_features", image_name + "_f.npy")

    if not os.path.exists(feat_path_s):
        print(f"Feature file not found: {feat_path_s}")
        return

    # Load Data
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    seg_map = np.load(feat_path_s) # [3, H, W] (levels)
    features = np.load(feat_path_f) # [N, 512]

    print(f"Image Shape: {img.shape}")
    print(f"Seg Map Shape: {seg_map.shape}")
    print(f"Features Shape: {features.shape}")

    # Query Analysis with CLIP
    # We will compute similarity of each segment with a prompt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = OpenCLIPNetwork(device)
    
    prompts = ["car", "tree", "road", "building", "sky"]
    text_embeds = clip_model.encode_text(prompts, device)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Features are [N, 512]. We need to map them back to image space using seg_map
    # seg_map[0] contains indices pointing to features.
    
    # Create a similarity map for the first prompt "car"
    h, w = seg_map.shape[1], seg_map.shape[2]
    
    # Convert features to torch
    features_t = torch.from_numpy(features).to(device).to(text_embeds.dtype)
    
    # Compute similarity for all features at once
    # [N, 512] @ [512, P] -> [N, P]
    sims = torch.matmul(features_t, text_embeds.T) # Raw Cosine Similarity

    # Compute Softmax across prompts (Contrast Enhancement)
    # Scale by 100 (temperature) before softmax to sharpen
    probs = torch.softmax(sims * 100, dim=1) 

    # Visualize for each level
    for level_idx in range(3):
        print(f"\n--- Analyzing Level {level_idx} ---")
        
        # Visualization Setup
        plt.figure(figsize=(20, 10))
        
        # 1. Original Image
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis('off')

        # 2. Segmentation
        plt.subplot(2, 3, 2)
        plt.title(f"Segmentation (Level {level_idx})")
        plt.imshow(seg_map[level_idx], cmap='tab20')
        plt.axis('off')

        # Map back to image
        indices = seg_map[level_idx].astype(int)
        valid_mask = indices != -1

        # Analyze each prompt
        for i, prompt in enumerate(prompts):
            # Raw Similarity Map
            sim_map = np.zeros((h, w))
            sim_vals = sims[:, i].detach().cpu().numpy()
            # Handle invalid indices (-1)
            mapped_sims = np.zeros_like(indices, dtype=float)
            mapped_sims[valid_mask] = sim_vals[indices[valid_mask]]
            sim_map = mapped_sims
            
            # Probability Map (Softmax)
            prob_map = np.zeros((h, w))
            prob_vals = probs[:, i].detach().cpu().numpy()
            mapped_probs = np.zeros_like(indices, dtype=float)
            mapped_probs[valid_mask] = prob_vals[indices[valid_mask]]
            prob_map = mapped_probs

            if i < 3: # Visualize first 3 prompts
                plt.subplot(2, 3, i + 4)
                plt.title(f"Prob: '{prompt}' (Softmax)")
                plt.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
                plt.colorbar()
                plt.axis('off')

        out_path = f"inspect_result_level_{level_idx}.png"
        plt.savefig(out_path)
        print(f"Visualization saved to {out_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--image_name", required=True)
    args = parser.parse_args()
    
    visualize_features(args.dataset_path, args.image_name)
