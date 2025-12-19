import torch
import os
import cv2
import numpy as np
from scene import Scene, GaussianModel
from gaussian_renderer import render
from argparse import ArgumentParser
from tqdm import tqdm
from arguments import ModelParams, PipelineParams
import sys
from types import SimpleNamespace as Namespace
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import LPIPS 

# --- YARDIMCI FONKSİYONLAR ---

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

def id_to_random_color(ids):
    # Train.py'daki Hashing fonksiyonunun AYNISI (Tutarlılık için)
    r = ((ids * 153245) % 255) / 255.0
    g = ((ids * 654321) % 255) / 255.0
    b = ((ids * 987654) % 255) / 255.0
    return torch.stack([r, g, b], dim=-1)

def calculate_miou(pred_ids, gt_mask, num_classes):
    iou_list = []
    # Sadece sahnede var olan classlar üzerinden gitmek daha hızlıdır
    present_classes = torch.unique(gt_mask)
    
    for cls in present_classes:
        cls = cls.item()
        if cls == 0: continue # Background'u atla

        pred_inds = (pred_ids == cls)
        target_inds = (gt_mask == cls)
        
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union > 0:
            iou_list.append(intersection / union)
        
    return sum(iou_list) / len(iou_list) if iou_list else 0.0

@torch.no_grad()
def evaluate(dataset_args, pipeline_args, full_args):
    dataset_args.eval = True
    gaussians = GaussianModel(dataset_args.sh_degree)
    scene = Scene(dataset_args, gaussians, load_iteration=full_args.iteration, shuffle=False)
    
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 256 ID için referans renk tablosu oluştur (Decoding için)
    num_objects = 256
    # Hashing yöntemi için referans tablosu (ID -> Renk)
    all_ids = torch.arange(num_objects, device="cuda")
    reference_colors = id_to_random_color(all_ids) # [256, 3]

    # Baseline için Matris (Eğer lazımsa)
    torch.manual_seed(42)
    global_projection_matrix = torch.randn((num_objects, 3), device="cuda")
    global_projection_matrix = global_projection_matrix / (global_projection_matrix.norm(dim=1, keepdim=True) + 1e-8)

    # Model Tipi Kontrolü
    is_3_channel = False
    if hasattr(gaussians, "_instance_features"):
        if gaussians._instance_features.shape[1] == 3:
            is_3_channel = True
            print("Tespit Edildi: 3-Kanallı Model (Ours - Hashing)")
        else:
            print("Tespit Edildi: Yüksek Boyutlu Model (Baseline/Old - Matrix)")
    else:
        print("Uyarı: Instance feature yok, rastgele init ediliyor.")
        fused_point_cloud = gaussians.get_xyz
        # Baseline ise rastgele 256 boyutludur varsayıyoruz veya renderda hata vermemesi için 3 yapıyoruz
        instance_features = torch.randn((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        gaussians._instance_features = torch.nn.Parameter(instance_features.requires_grad_(False))
        is_3_channel = True # Rastgele olduğu için fark etmez

    # LPIPS Modeli (VGG - Algısal Benzerlik için)
    lpips_model = LPIPS(net_type='vgg').to("cuda")
    lpips_model.eval()

    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0: test_cameras = scene.getTrainCameras()
    
    total_miou = 0.0
    total_acc = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0
    
    mask_root = os.path.join(full_args.source_path, "masks_sam2")

    print(f"Evaluating {len(test_cameras)} images...")

    for idx, view in enumerate(tqdm(test_cameras)):
        
        # Maske Yükleme
        if not hasattr(view, "gt_mask") or view.gt_mask is None:
            mask_path = os.path.join(mask_root, view.image_name + ".png")
            if os.path.exists(mask_path):
                mask_cv = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_cv is not None:
                    if mask_cv.shape[:2] != (view.image_height, view.image_width):
                        mask_cv = cv2.resize(mask_cv, (view.image_width, view.image_height), interpolation=cv2.INTER_NEAREST)
                    view.gt_mask = torch.from_numpy(mask_cv.astype(np.int32)).long().cuda()
            else:
                continue

        if view.gt_mask is None: continue

        # --- FEATURE HAZIRLIĞI VE RENDER ---
        if is_3_channel:
            # Ours: Sigmoid uygula
            render_input = torch.sigmoid(gaussians._instance_features)
        else:
            # Baseline: Matris çarpımı
            render_input = gaussians._instance_features @ global_projection_matrix
        
        # Render (Semantic)
        render_pkg = render(view, gaussians, pipeline_args, background, opt=full_args, scaling_modifier=1.0, override_color=render_input)
        rendered_map = render_pkg["render"] # [3, H, W]

        # --- RGB METRICS (PSNR, SSIM, LPIPS) ---
        # Standart RGB render (Override color yok)
        rgb_render_pkg = render(view, gaussians, pipeline_args, background, opt=full_args, scaling_modifier=1.0)
        image = rgb_render_pkg["render"]
        gt_image = view.original_image.cuda()

        # Metrics Calculation
        psnr_val = psnr(image, gt_image).mean().double()
        ssim_val = ssim(image, gt_image).mean().double()
        lpips_val = lpips_model(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()

        total_psnr += psnr_val
        total_ssim += ssim_val
        total_lpips += lpips_val
        
        # --- DECODING (Renk -> ID) ---
        # Render edilen (3 kanallı) haritayı tekrar ID'ye çevirmemiz lazım
        H, W = rendered_map.shape[1], rendered_map.shape[2]
        flat_render = rendered_map.permute(1, 2, 0).reshape(-1, 3) # [P, 3]
        
        # Hangi referans tablosunu kullanacağız?
        if is_3_channel:
            target_colors = reference_colors # [256, 3] (Hashing tablosu)
        else:
            # Baseline için projeksiyon matrisi renkleri temsil eder (kabaca)
            # Not: Baseline'da ID öğrenilmediği için bu zaten rastgele çıkacak.
            target_colors = global_projection_matrix 

        # En yakın ID'yi bul
        # (Pixel sayısı çoksa chunking yapmak gerekebilir ama 24GB VRAM yeter)
        dists = torch.cdist(flat_render.unsqueeze(0), target_colors.unsqueeze(0)).squeeze(0) # [P, 256]
        pred_ids = torch.argmin(dists, dim=1).reshape(H, W)
        
        # --- METRİK ---
        gt_mask = view.gt_mask.long().cuda()
        if gt_mask.shape != pred_ids.shape:
             gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0).float(), size=pred_ids.shape, mode="nearest").long().squeeze()
        
        acc = (pred_ids == gt_mask).float().mean().item()
        miou = calculate_miou(pred_ids, gt_mask, num_objects)
        
        total_acc += acc
        total_miou += miou
        count += 1

    if count > 0:
        print(f"\n--- SONUÇLAR ({full_args.model_path}) ---")
        print(f"Accuracy: {total_acc / count:.4f}")
        print(f"mIoU:     {total_miou / count:.4f}")
        print("-" * 20)
        print(f"PSNR:     {total_psnr / count:.4f}  (Higher is better)")
        print(f"SSIM:     {total_ssim / count:.4f}  (Higher is better)")
        print(f"LPIPS:    {total_lpips / count:.4f} (Lower is better)")
        print("-" * 20)

if __name__ == "__main__":
    parser = ArgumentParser()
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
    dataset_args = model.extract(args)
    
    evaluate(dataset_args, pipeline_args, args)