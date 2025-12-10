# render_depth.py
import torch
import os
import cv2
import numpy as np
import sys
from argparse import ArgumentParser
from types import SimpleNamespace as Namespace
from tqdm import tqdm

# LangSplat modüllerini import et
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Config yükleniyor:", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            cfgfile_string = cfg_file.read()
    except TypeError:
        pass
    
    # Config dosyasındaki argümanları parse et
    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    
    # Komut satırından gelenleri üzerine yaz
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def render_depth(view, gaussians, pipeline, background, opt):
    """
    Gaussian'ların Z (Derinlik) değerlerini Renk gibi render ederek Depth Map oluşturur.
    """
    # 1. Gaussian'ların Kamera Uzayındaki Derinliğini (Z) Hesapla
    # World to Camera matrisi (Transposed halde gelir genelde)
    w2c = view.world_view_transform
    
    pts = gaussians.get_xyz # [N, 3]
    ones = torch.ones((pts.shape[0], 1), device=pts.device)
    pts_homo = torch.cat([pts, ones], dim=1) # [N, 4]
    
    # World -> Camera dönüşümü
    # (Not: 3DGS matrisleri bazen Transpose ister, kütüphaneye göre değişir ama genelde budur)
    pts_cam = pts_homo @ w2c
    
    # Z koordinatı derinliktir
    depths = pts_cam[:, 2:3] # [N, 1]
    
    # 2. Render için Hazırlık
    # Rasterizer 3 kanal ister, o yüzden Z'yi 3 kere kopyalıyoruz (Grayscale gibi)
    depth_features = depths.repeat(1, 3)
    
    # Derinlik renderında arka plan SİYAH (0) olmalı
    bg_zero = torch.zeros(3, device="cuda", dtype=torch.float32)
    
    # 3. Render: Derinlik (Accumulated Depth)
    # override_color ile renk yerine derinlik gönderiyoruz
    depth_pkg = render(view, gaussians, pipeline, bg_zero, opt, scaling_modifier=1.0, override_color=depth_features)
    depth_accum = depth_pkg["render"]
    
    # 4. Render: Alpha (Accumulated Opacity)
    # Doğru ortalamayı bulmak için (Weighted Average) ağırlık toplamına bölmemiz lazım
    alpha_features = torch.ones_like(depth_features)
    alpha_pkg = render(view, gaussians, pipeline, bg_zero, opt, scaling_modifier=1.0, override_color=alpha_features)
    alpha_accum = alpha_pkg["render"]
    
    # 5. Normalizasyon (Depth = Toplam / Ağırlık)
    # Sadece ilk kanalı alıyoruz
    final_depth = depth_accum[0] / (alpha_accum[0] + 1e-6)
    
    return final_depth

def save_visuals(model_path, iteration, views, gaussians, pipeline, background, opt):
    render_dir = os.path.join(model_path, "comparison_renders")
    os.makedirs(render_dir, exist_ok=True)
    
    print(f"Renderlar {render_dir} klasörüne kaydediliyor...")
    
    # İlk 5 kamerayı veya hepsini render et (vakit varsa)
    for idx, view in enumerate(tqdm(views)):
        
        # A. Normal RGB Render
        rgb_pkg = render(view, gaussians, pipeline, background, opt, scaling_modifier=1.0)
        rgb = rgb_pkg["render"].detach().cpu().permute(1, 2, 0).numpy()
        rgb = np.clip(rgb, 0, 1)
        
        # B. Depth Render
        depth = render_depth(view, gaussians, pipeline, background, opt)
        depth = depth.detach().cpu().numpy()
        
        # Depth Görselleştirme (Renkli Harita - Turbo Colormap)
        # Min-Max Normalizasyon (Görsel kontrast için)
        # Sadece dolu pikselleri (depth > 0) dikkate al
        valid_mask = depth > 0.0
        if valid_mask.sum() > 0:
            d_min = depth[valid_mask].min()
            d_max = depth[valid_mask].max()
            depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
            depth_norm = np.clip(depth_norm, 0, 1)
            # Arka planı 0 yap
            depth_norm[~valid_mask] = 0
        else:
            depth_norm = depth

        # Renklendir (Matplotlib Turbo)
        # plt.get_cmap("turbo") için matplotlib lazım, import edelim veya cv2 kullanalım
        # Matplotlib yoksa cv2.applyColorMap daha güvenli
        import matplotlib.pyplot as plt
        depth_vis = plt.get_cmap("turbo")(depth_norm)[:, :, :3]
        
        # Arka planı siyah yap (Colormap bazen boyar)
        depth_vis[~valid_mask] = 0
        
        # C. Yan Yana Koy ve Kaydet
        # RGB | DEPTH
        combined = np.hstack([rgb, depth_vis])
        
        # BGR'ye çevir (OpenCV için) ve kaydet
        save_path = os.path.join(render_dir, f"{view.image_name}.png")
        cv2.imwrite(save_path, (combined * 255).astype(np.uint8)[:, :, ::-1])

if __name__ == "__main__":
    parser = ArgumentParser(description="Depth Render Script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    # Ekstra parametreler (Hata vermemesi için)
    parser.add_argument("--include_feature", action="store_true", default=False)
    parser.add_argument("--quick_render", action="store_true", default=False)
    parser.add_argument("--topk", type=int, default=1)
    
    args = get_combined_args(parser)
    
    # Modeli Yükle
    print(f"Model Yükleniyor: {args.model_path} (Iter: {args.iteration})")
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    pipeline_args = pipeline.extract(args)
    if not hasattr(pipeline_args, "debug"): pipeline_args.debug = False
    
    opt = Namespace(include_feature=args.include_feature, quick_render=args.quick_render, topk=args.topk)

    # Test kameralarını al (Yoksa train kullan)
    cameras = scene.getTestCameras()
    if len(cameras) == 0:
        print("Test kamerası yok, Eğitim kameraları kullanılıyor...")
        cameras = scene.getTrainCameras()
    
    # Örnek olması için sadece her 10. kareyi al (Hızlı bitsin)
    # cameras = cameras[::10] 

    save_visuals(args.model_path, args.iteration, cameras, gaussians, pipeline_args, background, opt)