#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import lpips
import os
import cv2
import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, ssim, cos_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.vq_utils import load_2d_language_feature, ResidualVectorQuantizationWithClustering
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import matplotlib.pyplot as plt
import numpy as np
from eval.openclip_encoder import OpenCLIPNetwork

# Helper function for language features
def render_language_feature_map_quick(gaussians, view, pipeline, background, args):
    with torch.no_grad():
        output = render(view, gaussians, pipeline, background, args)
        language_feature_weight_map = output['language_feature_weight_map']
        
        D, H, W = language_feature_weight_map.shape
        num_levels = D // 64
        
        # Reshape for memory efficiency
        language_feature_weight_map = language_feature_weight_map.view(num_levels, 64, H, W).view(num_levels, 64, H*W)
        
        # Prepare codebooks
        language_codebooks = gaussians._language_feature_codebooks.permute(0, 2, 1)
        
        # EINSUM Operation
        language_feature_map = torch.einsum('ldk,lkn->ldn', language_codebooks, language_feature_weight_map).view(num_levels, 512, H, W)
        
        # Normalization
        language_feature_map = language_feature_map / (language_feature_map.norm(dim=1, keepdim=True) + 1e-10)
        
    return language_feature_map


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
    
    # Initialize language feature codebooks
    if opt.include_feature and first_iter == 0:
        device = torch.device("cuda")
        features = load_2d_language_feature(dataset.lf_path, device)
        rvq = ResidualVectorQuantizationWithClustering(opt.vq_layer_num, opt.codebook_size, features.shape[1], device).to(device)
        rvq.fit_quantizers(features)
        codebooks = torch.stack(rvq.quantizers, dim=0).to(device)
        with torch.no_grad():
            gaussians._language_feature_codebooks.data.copy_(codebooks)

    # --- BİZİM EKLEME: Sabit Projeksiyon Matrisi (DÖNGÜ DIŞINDA) ---

    # Initialize CLIP for debug visualization
    clip_model = None
    text_embeds = None
    debug_prompts = ["car", "tree", "road"]

    # --- LPIPS MODEL YÜKLEME ---
    if opt.lambda_lpips > 0:
        print(f"Initializing LPIPS model with lambda={opt.lambda_lpips}...")
        lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()
        for param in lpips_loss_fn.parameters():
            param.requires_grad = False
    # ---------------------------

    if opt.include_feature:
        print("Initializing CLIP for debug visualization...")
        clip_model = OpenCLIPNetwork("cuda")
        with torch.no_grad():
            text_embeds = clip_model.encode_text(debug_prompts, "cuda")
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            # Ensure dtype matches what we will render (usually float32 or float16 depending on mixed precision, but here likely float32)
            text_embeds = text_embeds.float()

        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    loss_record = []
    iter_record = []
    smooth_loss = None
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, opt, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        opt.topk = args.topk
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt)
        image, language_feature_weight_map, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["language_feature_weight_map"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        if opt.include_feature:
            # gt_language_feature [512 H W]
            gt_language_feature, language_feature_mask = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=dataset.feature_level)
            # In this paper, we select layer_num = 1
            layer_num, _, _ = gaussians.get_language_feature_codebooks.shape
            layer_idx = min(int(iteration / 10000 * layer_num), layer_num - 1)
            language_feature = gaussians.compute_layer_feature_map(language_feature_weight_map, layer_idx)
            if args.normalize:
                language_feature = language_feature / (language_feature.norm(dim=0, keepdim=True) + 1e-10)
            loss = 0
            Ll1 = torch.tensor(0.0, device="cuda")
            if args.cos_loss:
                cosloss = cos_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)
                loss += cosloss
            if args.l1_loss:
                Ll1 = l1_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)   
                loss += Ll1

            # --- RGB EĞİTİMİ + BİZİM YENİ LOSS (VRAM DOSTU VERSİYON) ---
        else:
            # -------------------------------------------------------------------
            # STANDART RGB EĞİTİMİ
            # -------------------------------------------------------------------
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            
            # 1. RGB LOSS (L1 + SSIM)
            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            # 2. [YENİ] LPIPS LOSS
            # Görüntüleri -1 ile 1 arasına normalize et
            if opt.lambda_lpips > 0:
                lpips_val = lpips_loss_fn((image - 0.5) * 2, (gt_image - 0.5) * 2).mean()
                loss = rgb_loss + (opt.lambda_lpips * lpips_val)
            else:
                loss = rgb_loss
            
            # -------------------------------------------------------------------
            # [YENİ] SEMANTIC CONSISTENCY & GEOMETRIC REGULARIZATION
            # Yöntem: Contrastive Learning (Komşuluk İlişkisi)
            # -------------------------------------------------------------------
            
            # Parametreleri al (Argüman yoksa varsayılanları kullan)
            lambda_sem = getattr(opt, "lambda_sem", 0.1) 
            lambda_scale = getattr(opt, "lambda_scale", 0.05) 
            contrastive_margin = 0.5  # Farklı nesneler feature uzayında en az bu kadar uzak olmalı

            # Sadece maske varsa ve semantik eğitim açıksa çalış
            if lambda_sem > 0 and hasattr(viewpoint_cam, "gt_mask") and viewpoint_cam.gt_mask is not None:
                
                # Modelde instance özelliği tanımlı mı?
                if hasattr(gaussians, "_instance_features"):
                    
                    # 1. ÖZELLİK RENDER (3 Kanal)
                    # Matris yok! Özellikleri direkt sigmoid ile 0-1 arasına sıkıştırıp renk gibi çiziyoruz.
                    feats = torch.sigmoid(gaussians._instance_features)
                    
                    instance_render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt, override_color=feats)
                    pred_map = instance_render_pkg["render"] # [3, H, W]

                    # Feature map'i piksel bazında L2 normalize et (Vektör boyu 1 olsun)
                    pred_map = torch.nn.functional.normalize(pred_map, dim=0, p=2)
                    
                    # 2. GT MASKE HAZIRLIĞI
                    gt_mask = viewpoint_cam.gt_mask.long()
                    
                    # Boyut uyuşmazlığı varsa düzelt (Rounding hataları için)
                    if gt_mask.shape != pred_map.shape[1:]:
                         gt_mask = torch.nn.functional.interpolate(
                            gt_mask.unsqueeze(0).unsqueeze(0).float(), 
                            size=pred_map.shape[1:], 
                            mode="nearest"
                        ).long().squeeze()

                    # 3. CONTRASTIVE SAMPLING (Akıllı Örnekleme)
                    # Tüm resme bakmak çok yavaş olur. Rastgele 4096 piksel ve komşularını seçiyoruz.
                    
                    H, W = gt_mask.shape
                    num_samples = 4096
                    
                    # Rastgele merkez koordinatları seç
                    coords_y = torch.randint(0, H, (num_samples,), device="cuda")
                    coords_x = torch.randint(0, W, (num_samples,), device="cuda")
                    
                    # Rastgele komşu ofsetleri (-5 ile +5 piksel arası)
                    offsets_y = torch.randint(-5, 6, (num_samples,), device="cuda")
                    offsets_x = torch.randint(-5, 6, (num_samples,), device="cuda")
                    
                    # Komşu koordinatları (Sınır dışına taşmasın diye clamp)
                    neigh_y = torch.clamp(coords_y + offsets_y, 0, H-1)
                    neigh_x = torch.clamp(coords_x + offsets_x, 0, W-1)
                    
                    # 4. DEĞERLERİ TOPLA
                    # Render edilen featurelar (Transposed: [N, 3])
                    f_center = pred_map[:, coords_y, coords_x].T
                    f_neigh = pred_map[:, neigh_y, neigh_x].T
                    
                    # GT Maske ID'leri
                    id_center = gt_mask[coords_y, coords_x]
                    id_neigh = gt_mask[neigh_y, neigh_x]
                    
                    # 5. SEMANTIC MANTIK KURGUSU (Logic Gates)
                    
                    # Kural 1: En az biri "Bilinen Nesne" (ID > 0) olmalı.
                    # İkisi de 0 (Siyah) ise orası "Bilinmeyen vs Bilinmeyen"dir, dokunma.
                    valid_pair_mask = (id_center > 0) | (id_neigh > 0)
                    
                    if valid_pair_mask.sum() > 0:
                        # Sadece geçerli çiftleri al
                        f1 = f_center[valid_pair_mask]
                        f2 = f_neigh[valid_pair_mask]
                        id1 = id_center[valid_pair_mask]
                        id2 = id_neigh[valid_pair_mask]
                        
                        # Mesafe (Euclidean)
                        dist = torch.norm(f1 - f2, dim=1)
                        
                        # DURUM A: PULL (Çekme)
                        # İkisi de AYNI nesne ise ve ikisi de BİLİNEN ise.
                        # (id1 == id2) AND (id1 > 0)
                        mask_pull = (id1 == id2) & (id1 > 0)
                        
                        # DURUM B: PUSH (İtme / Ayrıştırma)
                        # ID'leri FARKLI ise. (Biri 0 olsa bile farklıdır, ayrışmalıdır).
                        mask_push = (id1 != id2)
                        
                        loss_pull = 0.0
                        loss_push = 0.0
                        
                        # Aynı olanların feature'larını birbirine benzet (Mesafe 0 olsun)
                        if mask_pull.sum() > 0:
                            loss_pull = (dist[mask_pull] ** 2).mean()
                            
                        # Farklı olanların feature'larını uzaklaştır (En az 'margin' kadar olsun)
                        if mask_push.sum() > 0:
                            # Hinge Loss: Sadece margin'den yakınsa ceza ver
                            loss_push = (torch.relu(contrastive_margin - dist[mask_push]) ** 2).mean()
                        
                        contrastive_loss = loss_pull + loss_push
                        
                        # 6. SCALE-AWARE REGULARIZATION (Büyüklük Cezası)
                        # Ekranda büyük yer kaplayan Gaussian'lara bak
                        radii = instance_render_pkg["radii"]
                        vis_filter = instance_render_pkg["visibility_filter"]
                        visible_radii = radii[vis_filter]
                        
                        scale_penalty = 0.0
                        if visible_radii.numel() > 0:
                            # Ekranda 2 pikselden büyük olanların büyüklüğü
                            scale_penalty = torch.mean(torch.relu(visible_radii - 2.0))
                        
                        # 7. NİHAİ LOSS
                        # Anlamsal hata varsa ceza ver + Eğer Gaussianlar büyükse KATMERLİ ceza ver.
                        # Bu çarpım, büyük ve kararsız Gaussian'ların bölünmesini tetikler.
                        final_sem_loss = contrastive_loss + (contrastive_loss * scale_penalty * lambda_scale)
                        
                        loss += lambda_sem * final_sem_loss
                        
                        # Loglama (İsteğe bağlı)
                        if iteration % 500 == 0:
                            print(f"Iter {iteration} | RGB: {Ll1.item():.4f} | Cont: {contrastive_loss.item():.4f} | Scale: {scale_penalty:.4f}")

            loss.backward()
        iter_end.record()
        
        if iteration % 100 == 0:
            print(f"Iter {iteration} Loss: {loss.item()}")

        # Debug Visualization
        if opt.include_feature and args.debug_interval > 0 and iteration % args.debug_interval == 0:
            print(f"\n[ITER {iteration}] Generating debug visualization...")
            with torch.no_grad():
                # Pick a random camera for debug visualization
                debug_view = scene.getTrainCameras()[randint(0, len(scene.getTrainCameras())-1)]
                
                # 1. Render RGB
                rendering = render(debug_view, gaussians, pipe, background, opt)["render"]
                rgb_img = rendering.permute(1, 2, 0).cpu().numpy()
                
                # 2. Render Language Features
                # Note: We need to temporarily set quick_render=True or handle it manually
                # render_language_feature_map_quick handles the manual reconstruction
                lf_map = render_language_feature_map_quick(gaussians, debug_view, pipe, background, opt)
                
                # Take Level 0
                lf_map_0 = lf_map[0].permute(1, 2, 0) # [H, W, 512]
                
                # 3. Compute Similarity
                sims = torch.matmul(lf_map_0, text_embeds.T) # [H, W, 3]
                
                # 4. Plot and Save
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 4, 1)
                plt.title(f"RGB (Iter {iteration})")
                plt.imshow(np.clip(rgb_img, 0, 1))
                plt.axis('off')
                
                for i, prompt in enumerate(debug_prompts):
                    sim_map = sims[..., i].cpu().numpy()
                    
                    plt.subplot(1, 4, i + 2)
                    plt.title(f"Sim: {prompt}")
                    plt.imshow(sim_map, cmap='jet')
                    plt.colorbar()
                    plt.axis('off')
                
                save_path = os.path.join(dataset.model_path, f"debug_render_{iteration:05d}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"Saved debug visualization to {save_path}")

        
        iter_record.append(iteration)
        if smooth_loss is None:
            smooth_loss = loss.item()
        else:
            smooth_loss = smooth_loss * 0.99 + loss.item() * 0.01
        loss_record.append(smooth_loss)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, opt))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if not opt.include_feature:
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if (iteration < opt.iterations) and (iteration % args.accum_iter == 0):
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print(f'testing for iter {iteration}')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        # Görselleri kaydetmek için klasör oluştur
        renders_dir = os.path.join(scene.model_path, "test_renders")
        os.makedirs(renders_dir, exist_ok=True)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # Görseli Diske Kaydet (PNG)
                    # Sadece ilk 5 kareyi kaydet (çok yer kaplamasın diye) veya hepsini
                    if idx < 5: 
                         # Tensor to Numpy [H, W, 3] (0-255 uint8)
                        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        gt_np = (gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        
                        # Yan yana birleştir (Prediction | Ground Truth)
                        combined = np.hstack([img_np, gt_np])
                        
                        # RGB -> BGR (OpenCV için)
                        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                        
                        save_name = f"{config['name']}_iter{iteration:05d}_{viewpoint.image_name}.png"
                        cv2.imwrite(os.path.join(renders_dir, save_name), combined)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55557)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--cos_loss', action='store_true', default=False)
    parser.add_argument('--l1_loss', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--debug_interval', type=int, default=0, help='Interval for saving debug visualizations (0 to disable)')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    args.model_path = args.model_path + f"_{str(args.feature_level)}"
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    # All done
    print("\nTraining complete.")
