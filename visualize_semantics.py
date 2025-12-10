# visualize_semantics.py
import torch
import os
import cv2
import numpy as np
from scene import Scene, GaussianModel
from gaussian_renderer import render
from argparse import ArgumentParser
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams
import sys
from types import SimpleNamespace as Namespace

def id_to_random_color(ids):
    # ids: [N] tensor of integers
    # Returns: [N, 3] tensor of colors
    N = ids.shape[0]
    num_objects = 64
    torch.manual_seed(42)
    colors_table = torch.rand((num_objects, 3), device=ids.device)
    return colors_table[ids]

def visualize(dataset_args, pipeline_args, full_args):
    # Model Yükle
    gaussians = GaussianModel(dataset_args.sh_degree)
    scene = Scene(dataset_args, gaussians, load_iteration=full_args.iteration, shuffle=False)
    
    bg_color = [0, 0, 0] # Siyah arka plan
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Projeksiyon Matrisi (Aynı seed ile)
    num_objects = 64
    torch.manual_seed(42)
    global_projection_matrix = torch.randn((num_objects, 3), device="cuda")
    global_projection_matrix = global_projection_matrix / (global_projection_matrix.norm(dim=1, keepdim=True) + 1e-8)

    # Çıktı Klasörü
    out_dir = os.path.join(full_args.model_path, "viz_semantic")
    os.makedirs(out_dir, exist_ok=True)
    
    # Test kameralarından birkaçına bak
    cameras = scene.getTestCameras()
    if len(cameras) == 0: cameras = scene.getTrainCameras()
    
    print(f"Görseller {out_dir} klasörüne kaydediliyor...")

    # İlk 5 kamera ve 71. frame (index 70)
    indices_to_process = [0, 1, 2, 3, 4]
    if len(cameras) > 70:
        indices_to_process.append(70)
    
    # Filter indices that might be out of bounds (just in case)
    indices_to_process = [i for i in indices_to_process if i < len(cameras)]

    for idx in indices_to_process:
        view = cameras[idx]
        
        # 1. Instance Feature Render
        # Eğer modelde özellik yoksa rastgele ata (Hata vermemesi için)
        if not hasattr(gaussians, "_instance_features"):
             fused_point_cloud = gaussians.get_xyz
             instance_features = torch.randn((fused_point_cloud.shape[0], num_objects), dtype=torch.float, device="cuda")
             gaussians._instance_features = torch.nn.Parameter(instance_features)

        # projected_features = gaussians._instance_features @ global_projection_matrix
        


        # 1. En yüksek olasılıklı ID'yi bul
        ids = torch.argmax(gaussians._instance_features, dim=1) # Her Gaussian için tek bir sayı (0-64)

        # 2. Bu ID'yi RASTGELE ama SABİT bir renge ata
        # (Daha önce yazdığımız id_to_random_color fonksiyonu gibi)
        colors = id_to_random_color(ids) 
        render_pkg = render(view, gaussians, pipeline_args, background, opt=full_args, scaling_modifier=1.0, override_color=colors)
        rendered_map = render_pkg["render"] # [3, H, W]
        # Görüntüye çevir (Normalize et 0-255)
        # Değerler negatif olabilir, o yüzden min-max normalization yapıyoruz
        sem_img = rendered_map.detach().cpu().permute(1, 2, 0).numpy()
        sem_img = (sem_img - sem_img.min()) / (sem_img.max() - sem_img.min() + 1e-8)
        sem_img = (sem_img * 255).astype(np.uint8)
        
        # RGB Render (Kıyaslama için)
        rgb_pkg = render(view, gaussians, pipeline_args, background, opt=full_args, scaling_modifier=1.0)
        rgb_img = rgb_pkg["render"].detach().cpu().permute(1, 2, 0).numpy()
        rgb_img = (np.clip(rgb_img, 0, 1) * 255).astype(np.uint8)
        
        # Yan yana koy
        combined = np.hstack([rgb_img, sem_img])
        
        cv2.imwrite(os.path.join(out_dir, f"view_{idx:03d}.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
    print("Tamamlandı.")

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=15000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true", default=False)
    parser.add_argument("--quick_render", action="store_true", default=False)
    parser.add_argument("--topk", type=int, default=1) 
    
    # Basit argüman parse
    args = parser.parse_args(sys.argv[1:])
    
    # Model path'i manuel verelim veya config'den okutalım (Basitlik için manuel veriyoruz)
    # Çalıştırırken: python visualize_semantics.py -m output/waldo_kitchen_sam2_-1
    
    # Helper fonksiyonu buraya kopyalayalım (metrics.py'dan)
    def get_combined_args(parser : ArgumentParser):
        cmdlne_string = sys.argv[1:]
        cfgfile_string = "Namespace()"
        args_cmdline = parser.parse_args(cmdlne_string)
        try:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
            with open(cfgfilepath) as cfg_file:
                cfgfile_string = cfg_file.read()
        except TypeError:
            pass
        args_cfgfile = eval(cfgfile_string)
        merged_dict = vars(args_cfgfile).copy()
        for k,v in vars(args_cmdline).items():
            if v != None:
                merged_dict[k] = v
        return Namespace(**merged_dict)

    args = get_combined_args(parser)
    
    # Ensure source_path exists to avoid AttributeError in model.extract
    if not hasattr(args, 'source_path') or args.source_path is None:
        args.source_path = "" # Default to empty string if missing
    
    if not hasattr(args, 'model_path') or args.model_path is None:
        print("Model path not specified. Usage: python visualize_semantics.py -m <model_path>")
        sys.exit(1)

    evaluate_args = model.extract(args)
    pipeline_args = pipeline.extract(args)
    if not hasattr(pipeline_args, 'debug'): pipeline_args.debug = False

    visualize(evaluate_args, pipeline_args, args)