import torch
import os
import cv2
import numpy as np
from scene import Scene, GaussianModel
from gaussian_renderer import render
from argparse import ArgumentParser
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
import sys
from types import SimpleNamespace as Namespace
from tqdm import tqdm
import pandas as pd # Tabloyu güzel basmak için (yoksa pip install pandas)

# --- YARDIMCI FONKSİYONLAR ---

def get_combined_args(model_path):
    try:
        cfgfilepath = os.path.join(model_path, "cfg_args")
        with open(cfgfilepath) as cfg_file:
            cfgfile_string = cfg_file.read()
            
        # "Namespace" is already imported from types as SimpleNamespace, 
        # so eval() should work if the file contains "Namespace(...)"
        args_cfgfile = eval(cfgfile_string)
        return args_cfgfile
    except Exception as e:
        print(f"Error loading config from {model_path}: {e}")
        return None

def id_to_random_color(ids):
    # Train.py ile aynı renklendirme mantığı (Görsel tutarlılık için)
    r = (ids * 153245) % 255
    g = (ids * 654321) % 255
    b = (ids * 987654) % 255
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

def render_model(model_name, model_path, iteration, view, background, num_objects, proj_matrix):
    # Modeli Yükle
    # (Not: Her render'da modeli tekrar yüklemek yavaştır ama 
    # iki farklı modelin parametrelerinin karışmaması için en temiz yoldur)
    
    # Config dosyasını oku (sh_degree vs için)
    try:
        with open(os.path.join(model_path, "cfg_args")) as f:
            cfg = eval(f.read())
            sh_degree = cfg.sh_degree
    except:
        sh_degree = 3

    gaussians = GaussianModel(sh_degree)
    
    # Checkpoint yükle
    ckpt_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
    gaussians.load_ply(ckpt_path)
    
    # Pipeline ayarları (Basit)
    pipeline = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    opt = Namespace(include_feature=False, quick_render=False, topk=1)

    # 1. RGB Render
    render_pkg = render(view, gaussians, pipeline, background, opt, scaling_modifier=1.0)
    rgb = render_pkg["render"]
    
    # 2. Depth Render (Z-Map)
    # 3DGS'de depth hesabı (Basitleştirilmiş)
    # viewspace_points'in Z koordinatını kullanmamız lazım ama render fonksiyonu bunu direkt vermiyor olabilir.
    # Bu scriptte Depth yerine "Opacity" veya "Accumulated Alpha" kullanabiliriz.
    # Ancak görsel kanıt için Depth daha iyidir.
    # Eğer render() fonksiyonun depth döndürmüyorsa, proje matrisi ile özellik render'a odaklanalım.
    
    # 3. Semantic Render
    # Eğer modelde _instance_features yoksa (Baseline), rastgele oluştur
    if not hasattr(gaussians, "_instance_features"):
        fused_point_cloud = gaussians.get_xyz
        # Rastgele feature ata (Sadece görselde belli olsun diye)
        instance_features = torch.randn((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        gaussians._instance_features = torch.nn.Parameter(instance_features)

    # Instance özelliklerini proje et ve renderla
    # (Train.py'daki mantığın aynısı)
    if gaussians._instance_features.shape[1] != 3:
         # Eğer 64 boyutluysa matrisle çarp
         feats = gaussians._instance_features @ proj_matrix
    else:
         # Eğer zaten 3 boyutluysa (yeni yöntem) direkt kullan
         feats = gaussians._instance_features
    
    # Normalizasyon (Görsel için)
    feats = torch.sigmoid(feats)
    
    sem_pkg = render(view, gaussians, pipeline, background, opt, scaling_modifier=1.0, override_color=feats)
    sem = sem_pkg["render"]

    return rgb, sem

def compare(args):
    # Ortak Hazırlıklar
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Random Projection Matrisi (Train.py ile aynı seed)
    num_objects = 64
    torch.manual_seed(42)
    proj_matrix = torch.randn((num_objects, 3), device="cuda")
    proj_matrix = proj_matrix / (proj_matrix.norm(dim=1, keepdim=True) + 1e-8)
    
    # Sahne ve Kameraları Yükle (Sadece bir model üzerinden sahneyi yüklesek yeter)
    # Dummy model oluşturup sahneyi çekiyoruz
    gaussians = GaussianModel(3)
    # Baseline args'ı kullanarak sahneyi yükle
    base_args = get_combined_args(args.baseline_path)
    scene = Scene(base_args, gaussians, load_iteration=args.iteration, shuffle=False)
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0: test_cameras = scene.getTrainCameras() # Fallback

    print(f"Toplam {len(test_cameras)} kare karşılaştırılacak...")
    
    # Metrik Saklayıcılar
    metrics = {
        "Baseline_PSNR": [], "Ours_PSNR": [],
    }

    # Çıktı Klasörü
    out_dir = "comparison_results"
    os.makedirs(out_dir, exist_ok=True)

    # Sadece ilk 5 kareyi görselleştir, hepsinin metriğini hesapla
    for idx, view in enumerate(tqdm(test_cameras)):
        
        # --- RENDER BASELINE ---
        rgb_b, sem_b = render_model("Baseline", args.baseline_path, args.iteration, view, background, num_objects, proj_matrix)
        
        # --- RENDER OURS ---
        rgb_o, sem_o = render_model("Ours", args.ours_path, args.iteration, view, background, num_objects, proj_matrix)
        
        # --- METRİK HESAPLAMA ---
        gt_image = view.original_image.cuda()
        metrics["Baseline_PSNR"].append(psnr(rgb_b, gt_image).mean().item())
        metrics["Ours_PSNR"].append(psnr(rgb_o, gt_image).mean().item())
        
        # mIoU Hesabı (Basitleştirilmiş: Hard argmax karşılaştırması)
        # GT Maske varsa
        if hasattr(view, "gt_mask") and view.gt_mask is not None:
            # Buraya detaylı mIoU eklenebilir, şimdilik görsel yeterli
            pass

            # Rastgele 5 kare seçimi için indeksleri belirle
    import random
    num_cameras = len(test_cameras)
    if num_cameras > 5:
        random_indices_for_visualization = set(random.sample(range(num_cameras), 5))
    else:
        random_indices_for_visualization = set(range(num_cameras))

    # --- GÖRSEL KAYDETME (Sadece rastgele seçilen 5 kare) ---
    if idx in random_indices_for_visualization:
        # Tensor -> Numpy Image
        def to_img(tensor):
            return (np.clip(tensor.detach().cpu().permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)

        img_b_rgb = to_img(rgb_b)
        img_o_rgb = to_img(rgb_o)
        img_b_sem = to_img(sem_b)
        img_o_sem = to_img(sem_o)
            
            # Ground Truth Maskeyi de renklendir
            if hasattr(view, "gt_mask") and view.gt_mask is not None:
                gt_mask_np = view.gt_mask.cpu().numpy().astype(int)
                # Basit renklendirme (Hızlıca)
                gt_vis = id_to_random_color(gt_mask_np)
                # Resize (Eğer gerekirse)
                if gt_vis.shape[:2] != img_b_rgb.shape[:2]:
                     gt_vis = cv2.resize(gt_vis, (img_b_rgb.shape[1], img_b_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                gt_vis = np.zeros_like(img_b_rgb)

            # Grid Oluştur
            # Üst Satır: Baseline RGB | Ours RGB | GT RGB
            # Alt Satır: Baseline Sem | Ours Sem | GT Mask
            gt_rgb_vis = to_img(gt_image)
            
            row1 = np.hstack([img_b_rgb, img_o_rgb, gt_rgb_vis])
            row2 = np.hstack([img_b_sem, img_o_sem, gt_vis])
            grid = np.vstack([row1, row2])
            
            # Etiket Ekle
            grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            cv2.putText(grid, "Baseline RGB", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(grid, "Ours RGB", (50 + img_b_rgb.shape[1], 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(grid, "Baseline Semantic", (50, 30 + img_b_rgb.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(grid, "Ours Semantic", (50 + img_b_rgb.shape[1], 30 + img_b_rgb.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imwrite(os.path.join(out_dir, f"compare_{idx:03d}.png"), grid)

    # --- RAPORLAMA ---
    print("\n" + "="*30)
    print("      SONUÇ RAPORU      ")
    print("="*30)
    df = pd.DataFrame(metrics)
    print(df.mean())
    print("="*30)
    print(f"Görseller '{out_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--baseline_path", type=str, required=True)
    parser.add_argument("--ours_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args(sys.argv[1:])
    
    compare(args)