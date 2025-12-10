import torch
import torch.nn as nn
import os
import cv2
from scene import Scene, GaussianModel
from gaussian_renderer import render
from argparse import ArgumentParser
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from arguments import ModelParams, PipelineParams, OptimizationParams
import sys
from types import SimpleNamespace as Namespace

# Helper to parse args compatible with LangSplat structure
def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Loading config from:", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
            
    return Namespace(**merged_dict)

def calculate_miou(pred_mask, gt_mask, num_classes):
    iou_list = []
    # 0 (Background) dahil tüm sınıfları gez
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        target_inds = (gt_mask == cls)
        
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            continue
            
        iou_list.append(intersection / union)
        
    return sum(iou_list) / len(iou_list) if iou_list else 0.0

def evaluate(dataset_args, pipeline_args, full_args):
    # --- KRİTİK DÜZELTME: Eval Modunu Zorla ---
    dataset_args.eval = True 
    # ------------------------------------------

    # 1. Modeli Yükle
    gaussians = GaussianModel(dataset_args.sh_degree)
    scene = Scene(dataset_args, gaussians, load_iteration=full_args.iteration, shuffle=False)
    
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 2. Random Projection Matrisini Yeniden Oluştur (Train.py ile AYNI Seed olmalı!)
    num_objects = 64
    torch.manual_seed(42)
    global_projection_matrix = torch.randn((num_objects, 3), device="cuda")
    global_projection_matrix = global_projection_matrix / (global_projection_matrix.norm(dim=1, keepdim=True) + 1e-8)

    if not hasattr(gaussians, "_instance_features"):
        # Baseline modelde bu özellik olmayabilir, hata vermesin diye rastgele oluşturuyoruz
        # Not: Baseline'ın mIoU skoru bu yüzden düşük (rastgele) çıkacaktır, bu da beklenen bir şey.
        print("UYARI: Modelde '_instance_features' bulunamadı (Baseline Model). Rastgele başlatılıyor...")
        fused_point_cloud = gaussians.get_xyz
        instance_features = torch.randn((fused_point_cloud.shape[0], num_objects), dtype=torch.float, device="cuda")
        gaussians._instance_features = nn.Parameter(instance_features.requires_grad_(False))

    # Kameraları Al
    test_cameras = scene.getTestCameras()
    
    # EĞER TEST KAMERASI YOKSA (Hala 0 ise), EĞİTİM KAMERALARINI KULLAN
    if len(test_cameras) == 0:
        print("UYARI: Test kamerası bulunamadı! Eğitim (Train) kameraları kullanılıyor...")
        test_cameras = scene.getTrainCameras()
    
    total_miou = 0.0
    total_acc = 0.0
    count = 0
    
    print(f"Evaluating {len(test_cameras)} images...")
    
    # --- MASKE YOLU BELİRLEME ---
    mask_root = os.path.join(full_args.source_path, "masks_sam2")
    if not os.path.exists(mask_root):
        print(f"HATA: Maske klasörü bulunamadı: {mask_root}")
        print(f"Lütfen '{full_args.source_path}' içinde 'masks_sam2' klasörü olduğundan emin olun.")
        return

    for idx, view in enumerate(tqdm(test_cameras)):
        
        # --- Maskeyi Elle Yükle ---
        if not hasattr(view, "gt_mask") or view.gt_mask is None:
            mask_path = os.path.join(mask_root, view.image_name + ".png")
            
            if os.path.exists(mask_path):
                mask_cv = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_cv is not None:
                    # Resize gerekirse (Görüntü boyutuna uydur)
                    if mask_cv.shape[:2] != (view.image_height, view.image_width):
                        mask_cv = cv2.resize(
                            mask_cv, 
                            (view.image_width, view.image_height), 
                            interpolation=cv2.INTER_NEAREST 
                        )
                    view.gt_mask = torch.from_numpy(mask_cv.astype(np.int32)).long().cuda()
            else:
                # Maske yoksa, belki dosya uzantısı farklıdır (.JPG vs)
                # Basit bir check daha:
                if idx == 0: print(f"Aranan maske yolu bulunamadı: {mask_path}")
                continue
        # ----------------------------------------

        if view.gt_mask is None:
            continue
            
        # 3. Render
        projected_features = gaussians._instance_features @ global_projection_matrix
        
        render_pkg = render(view, gaussians, pipeline_args, background, opt=full_args, scaling_modifier=1.0, override_color=projected_features)
        
        rendered_map = render_pkg["render"] # [3, H, W]
        
        # 4. Decoding
        H, W = rendered_map.shape[1], rendered_map.shape[2]
        flat_render = rendered_map.permute(1, 2, 0).reshape(-1, 3)
        
        dists = torch.cdist(flat_render.unsqueeze(0), global_projection_matrix.unsqueeze(0)).squeeze(0) # [H*W, 64]
        pred_ids = torch.argmin(dists, dim=1).reshape(H, W) # [H, W]
        
        # 5. GT Mask
        gt_mask = view.gt_mask.long()
        
        if gt_mask.shape != pred_ids.shape:
             gt_mask = torch.nn.functional.interpolate(
                gt_mask.unsqueeze(0).unsqueeze(0).float(), 
                size=pred_ids.shape, 
                mode="nearest"
            ).long().squeeze()
            
        # 6. Metrikler
        accuracy = (pred_ids == gt_mask).float().mean().item()
        miou = calculate_miou(pred_ids, gt_mask, num_objects)
        
        total_miou += miou
        total_acc += accuracy
        count += 1
        
    if count > 0:
        print(f"\n--- SONUÇLAR ({full_args.model_path}) ---")
        print(f"İşlenen Resim Sayısı: {count}")
        print(f"Ortalama Accuracy: {total_acc / count:.4f}")
        print(f"Ortalama mIoU:   {total_miou / count:.4f}")
    else:
        print("Hiçbir test kamerasında maske bulunamadı! (Klasör yollarını veya dosya isimlerini kontrol et)")

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--include_feature", action="store_true", default=False)
    parser.add_argument("--quick_render", action="store_true", default=False)
    parser.add_argument("--topk", type=int, default=1) 
    
    args = get_combined_args(parser)
    pipeline_args = pipeline.extract(args)
    if not hasattr(pipeline_args, 'debug'):
        pipeline_args.debug = False

    dataset_args = model.extract(args)
    evaluate(dataset_args, pipeline_args, args)