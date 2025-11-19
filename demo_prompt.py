# demo_prompt.py
# LangSplatV2 için Özel Prompt Sorgulama Scripti (MEMORY FIX - RESOLUTION)

import torch
import os
import cv2
import gc
import numpy as np
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from eval.openclip_encoder import OpenCLIPNetwork
from gaussian_renderer import render
from utils.vq_utils import get_weights_and_indices
from tqdm import tqdm
from pathlib import Path

# Hızlı render fonksiyonu
def render_language_feature_map_quick(gaussians, view, pipeline, background, args):
    with torch.no_grad():
        # args içinde quick_render=True olmalı
        output = render(view, gaussians, pipeline, background, args)
        language_feature_weight_map = output['language_feature_weight_map']
        
        D, H, W = language_feature_weight_map.shape
        
        # Bellek tasarrufu için reshape
        language_feature_weight_map = language_feature_weight_map.view(3, 64, H, W).view(3, 64, H*W)
        
        # Codebook'ları hazırla
        language_codebooks = gaussians._language_feature_codebooks.permute(0, 2, 1)
        
        # EINSUM İşlemi (En çok bellek yiyen yer burası)
        # Bellek yetmezse burayı float16 (half) yapabiliriz ama resolution düşürmek daha etkili.
        language_feature_map = torch.einsum('ldk,lkn->ldn', language_codebooks, language_feature_weight_map).view(3, 512, H, W)
        
        # Normalizasyon
        language_feature_map = language_feature_map / (language_feature_map.norm(dim=1, keepdim=True) + 1e-10)
        
    return language_feature_map

def run_demo(dataset, pipeline, args, text_prompt):
    device = torch.device("cuda")
    
    # Pipeline ayarları
    pipeline.include_feature = True 

    # 1. Modeli ve CLIP'i Yükle
    print(f"⏳ Model ve Sahne Yükleniyor... Prompt: '{text_prompt}'")
    clip_model = OpenCLIPNetwork(device)
    
    # Prompt'u sisteme tanıt
    clip_model.set_positives([text_prompt]) 

    # Gaussian Modelini Hazırla
    gaussians = GaussianModel(dataset.sh_degree)
    
    # Sahne Yüklemesi
    dataset.model_path = args.ckpt_paths[0]
    scene = Scene(dataset, gaussians, shuffle=False)
    
    # Kameraları al (Süreci hızlandırmak için her 30. kareyi alıyoruz)
    views = scene.getTrainCameras()
    selected_views = views[::20] 
    
    # Checkpoint yükle
    checkpoint = os.path.join(args.ckpt_paths[0], f'chkpnt{args.checkpoint}.pth')
    if not os.path.exists(checkpoint):
        print(f"❌ HATA: Checkpoint dosyası bulunamadı: {checkpoint}")
        return

    print(f"🔄 Checkpoint yükleniyor: {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, args, mode='test')

    # Dil özelliklerini (Lang Features) yükle
    language_feature_weights = []
    language_feature_indices = []
    language_feature_codebooks = []
    
    print("⏳ Dil özellikleri birleştiriliyor...")
    for level_idx in range(3):
        temp_gaussians = GaussianModel(dataset.sh_degree)
        ckpt_path = os.path.join(args.ckpt_paths[level_idx], f'chkpnt{args.checkpoint}.pth')
        (params, _) = torch.load(ckpt_path)
        temp_gaussians.restore(params, args, mode='test')
        
        language_feature_codebooks.append(temp_gaussians._language_feature_codebooks.view(-1, 512))
        weights, indices = get_weights_and_indices(temp_gaussians._language_feature_logits, 4)
        language_feature_weights.append(weights)
        language_feature_indices.append(indices + int(level_idx * temp_gaussians._language_feature_codebooks.shape[1]))

    # Listeleri Tensora çeviriyoruz
    gaussians._language_feature_codebooks = torch.stack(language_feature_codebooks, dim=0)
    gaussians._language_feature_weights = torch.cat(language_feature_weights, dim=1)
    language_feature_indices = torch.cat(language_feature_indices, dim=1) 
    gaussians._language_feature_indices = torch.from_numpy(language_feature_indices.detach().cpu().numpy()).to(device)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Çıktı klasörü ayarla
    safe_prompt = text_prompt.replace(" ", "_")
    output_dir = Path("demo_result") / f"{args.dataset_name}_{safe_prompt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Sonuçlar şuraya kaydedilecek: {output_dir}")

    # 2. Render İşlemi
    for idx, view in enumerate(tqdm(selected_views, desc="Render alınıyor")):
        
        # Bellek temizliği (Her karede yapalım)
        torch.cuda.empty_cache()
        gc.collect()

        try:
            # Görsel Özellik Haritasını Çıkar
            lf_map = render_language_feature_map_quick(gaussians, view, pipeline, background, args)
            lf_map = lf_map.permute(0, 2, 3, 1) 
            
            # CLIP ile benzerliği hesapla
            valid_map = clip_model.get_max_across_quick(lf_map)
            similarity = valid_map.mean(dim=0)[0].cpu().numpy() 
            
            # --- SMART CONTRAST ---
            raw_min = similarity.min()
            raw_max = similarity.max()
            ABS_THRESHOLD = 0.22
            
            if raw_max < ABS_THRESHOLD:
                similarity[:] = 0
            else:
                similarity = (similarity - raw_min) / (raw_max - raw_min + 1e-8)
                similarity = similarity ** 4
            
            # Heatmap oluştur
            heatmap = cv2.applyColorMap((similarity * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Orijinal RGB görüntüyü al
            rgb_out = render(view, gaussians, pipeline, background, args)["render"]
            rgb_img = rgb_out.permute(1, 2, 0).detach().cpu().numpy() * 255
            rgb_img = cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # RGB ve Heatmap boyutlarını eşitle (Resolution değiştiği için gerekebilir)
            if rgb_img.shape[:2] != heatmap.shape[:2]:
                heatmap = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))

            # Yan yana kaydet
            combined = np.hstack((rgb_img, heatmap))
            cv2.imwrite(str(output_dir / f"view_{view.image_name}.jpg"), combined)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️ HATA: {view.image_name} işlenirken bellek yetmedi, bu kare atlanıyor.")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    print(f"✅ İşlem Tamamlandı! Klasöre bakabilirsin: {output_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True, help="Aranacak nesne")
    parser.add_argument("--ckpt_root_path", default='output', type=str)
    parser.add_argument("--checkpoint", type=int, default=10000)
    
    args = get_combined_args(parser)
    
    # --- ZORUNLU PARAMETRELER ---
    args.sh_degree = 3
    args.white_background = False
    args.language_features_name = "language_features"
    args.images = "images"
    
    # --- BELLEK İÇİN ÇÖZÜM BURADA ---
    # -1 orijinal boyut demektir. 2 yaparsak boyutu yarıya düşürür (Bellek 4x azalır).
    # Eğer hala hata alırsan burayı 4 yap.
    args.resolution = 2  
    
    args.data_device = "cuda"
    args.eval = True
    args.include_feature = True
    
    # --- RENDER PARAMETRELERİ ---
    args.quick_render = True
    args.compute_cov3D_python = False
    args.convert_SHs_python = False
    args.debug = False
    
    # Yolları ayarla
    args.dataset_path = f"./data/lerf_ovs/{args.dataset_name}"
    args.source_path = args.dataset_path
    
    # Checkpoint yolları
    args.ckpt_paths = [
        os.path.join(args.ckpt_root_path, f"{args.dataset_name}_0_{level}") 
        for level in [1, 2, 3]
    ]
    
    run_demo(model.extract(args), pipeline.extract(args), args, args.prompt)