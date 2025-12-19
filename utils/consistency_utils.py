import torch
import torch.nn.functional as F

class NeighborSearch:
    """
    Eğitim kameraları arasında komşuluk ilişkilerini yönetir.
    """
    def __init__(self, cameras):
        self.cameras = cameras
        self.num_cams = len(cameras)
        # Tüm kamera merkezlerini al
        self.centers = torch.stack([cam.camera_center for cam in cameras]).cuda()
        
    def get_neighbors(self, cam_idx, k=5):
        """
        Verilen kamera indeksine fiziksel olarak en yakın k kamerayı döndürür.
        """
        current_center = self.centers[cam_idx].unsqueeze(0)
        # Öklid mesafesi hesapla
        dists = torch.norm(self.centers - current_center, dim=1)
        
        # En yakın k+1 taneyi al (kendisi dahil olacağı için +1)
        _, indices = torch.topk(dists, k + 1, largest=False)
        
        # Kendisini (mesafe=0) listeden çıkar
        neighbor_indices = indices[indices != cam_idx]
        
        return [self.cameras[i] for i in neighbor_indices[:k]]

def get_projection_matrix(cam, W, H):
    """
    Kamera parametrelerinden (FoV) K Matrisini (Intrinsics) oluşturur.
    """
    # 3DGS FoV değerlerini tutar, bunları fokal uzunluğa çevirmeliyiz
    # tan(fov/2) = (W/2) / f
    tan_fovx = torch.tan(torch.tensor(cam.FoVx) * 0.5)
    tan_fovy = torch.tan(torch.tensor(cam.FoVy) * 0.5)
    
    f_x = W / (2.0 * tan_fovx)
    f_y = H / (2.0 * tan_fovy)
    
    c_x = W / 2.0
    c_y = H / 2.0
    
    K = torch.zeros((3, 3), device="cuda")
    K[0, 0] = f_x
    K[1, 1] = f_y
    K[0, 2] = c_x
    K[1, 2] = c_y
    K[2, 2] = 1.0
    
    return K

def warp_consistency_check(image_A, depth_A, view_A, view_B, image_B, depth_B):
    """
    Frame A'yı Frame B'nin bakış açısına warp eder ve hatayı hesaplar.
    
    Args:
        image_A: [3, H, W] Kaynak resim
        depth_A: [1, H, W] Kaynak derinlik
        view_A: Kaynak kamera objesi
        view_B: Hedef kamera objesi
        image_B: [3, H, W] Hedef resim (Ground Truth veya Render)
        depth_B: [1, H, W] Hedef derinlik (Occlusion check için)
    """
    _, H, W = image_A.shape
    
    # 1. PIXEL GRID OLUŞTUR
    y, x = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing='ij')
    # Pixel koordinatları [3, H*W] -> (u, v, 1)
    pixels = torch.stack([x.flatten(), y.flatten(), torch.ones_like(x.flatten())], dim=0).float()
    
    # 2. UNPROJECT (2D -> 3D World) using Cam A
    # P_cam = K_inv * pixel * depth
    K_A = get_projection_matrix(view_A, W, H)
    K_A_inv = torch.inverse(K_A)
    
    # Derinliği düzleştir [1, H*W]
    depth_flat = depth_A.flatten().unsqueeze(0)
    
    # Kamera A koordinat sistemindeki noktalar
    P_cam_A = torch.matmul(K_A_inv, pixels) * depth_flat
    
    # World Space'e geçiş
    # 3DGS'de view.world_view_transform (R_trans) World->View matrisidir.
    # P_cam = R * P_world + t  =>  P_world = R_inv * (P_cam - t)
    # view.world_view_transform transpoze olarak saklanır (OpenGL style), dikkat!
    
    # W2C Matrisi (World to Cam)
    W2C_A = view_A.world_view_transform.transpose(0, 1) # [4, 4] normal format
    R_A = W2C_A[:3, :3]
    t_A = W2C_A[:3, 3].unsqueeze(1)
    
    # Kamera A'dan Dünya'ya
    P_world = torch.matmul(R_A.inverse(), P_cam_A - t_A)
    # Homojen koordinat (4. boyut) ekle
    P_world = torch.cat([P_world, torch.ones((1, P_world.shape[1]), device="cuda")], dim=0)
    
    # 3. PROJECT (3D World -> 2D Cam B)
    W2C_B = view_B.world_view_transform.transpose(0, 1)
    
    # Dünya'dan Kamera B'ye (P_cam_B = W2C_B * P_world)
    P_cam_B = torch.matmul(W2C_B, P_world)
    
    # Derinlik (Z) Frame B referansında
    depth_proj = P_cam_B[2:3, :] # [1, H*W]
    
    # Perspektif Bölme (Perspective Divide) -> Normalized Device Coordinates
    # Ama biz manuel K ile çarpacağız
    K_B = get_projection_matrix(view_B, W, H)
    
    # P_pixel = K * (P_cam / Z)
    # Önce sadece 3x3 rotasyon kısmını alıp [X,Y,Z]'yi çarpalım
    P_cam_B_3 = P_cam_B[:3, :]
    pixel_B_homo = torch.matmul(K_B, P_cam_B_3)
    
    # Z'ye bölerek (u, v) elde et
    # Sıfıra bölünmeyi engellemek için eps ekle
    z_safe = pixel_B_homo[2:3, :] + 1e-6
    u_proj = pixel_B_homo[0:1, :] / z_safe
    v_proj = pixel_B_homo[1:2, :] / z_safe
    

    
    # 4. SAMPLING HAZIRLIĞI (Grid Sample için [-1, 1] arasına normalize et)
    # u [0, W] -> [-1, 1]
    # v [0, H] -> [-1, 1]
    u_norm = (u_proj / (W / 2.0)) - 1.0
    v_norm = (v_proj / (H / 2.0)) - 1.0
    

    
    grid = torch.cat([u_norm, v_norm], dim=0).permute(1, 0).reshape(1, H, W, 2)
    
    # 5. EŞLEŞTİRME (Sampling)
    # image_B'den, hesapladığımız koordinatlardaki renkleri çek
    # (Burası önemli: Biz Image B'nin piksellerini çekiyoruz, Image A'yı warp etmiyoruz.
    # "A'daki bu piksel B'de şuraya düştü, orada ne renk var?" mantığı)
    sampled_rgb_B = F.grid_sample(image_B.unsqueeze(0), grid, align_corners=True).squeeze(0)
    
    # Ayrıca B'nin kendi derinliğini de çek (Occlusion check için)
    sampled_depth_B = F.grid_sample(depth_B.unsqueeze(0), grid, align_corners=True, mode='nearest').squeeze(0)
    
    # 6. MASKING (Elemeler)
    
    # A) Kadraj Dışı Maskesi (Out of Bounds)
    mask_bounds = (u_norm.abs() <= 1.0) & (v_norm.abs() <= 1.0)
    mask_bounds = mask_bounds.reshape(1, H, W)
    
    # B) Pozitif Derinlik Maskesi (Arkada kalanları ele)
    mask_z = (depth_proj.reshape(1, H, W) > 0.1)
    
    # C) OCCLUSION MASK (Z-Test)
    # Fırlattığımız derinlik (depth_proj), orada var olan derinlikten (sampled_depth_B)
    # çok büyükse, demek ki arkada kalmışız.
    # depth_proj ~ sampled_depth_B olmalı.
    tolerance = 0.1 # 10 cm tolerans (Sahne ölçeğine göre ayarlanmalı!)
    
    # projected (misafir) < current (ev sahibi) + tolerance
    # (Not: 3DGS depth değerleri bazen gürültülüdür, toleransı iyi ayarla)
    mask_occlusion = (depth_proj.reshape(1, H, W) < (sampled_depth_B + tolerance))
    
    # Hepsinin Kesişimi
    valid_mask = mask_bounds & mask_z & mask_occlusion
    
    # 7. LOSS HESABI
    # Kaynak resim (A) ile B'den örneklediğimiz renklerin farkı
    # (Eğer geometri doğruysa, Image A'daki piksel rengi ile B'de düştüğü yerdeki renk aynı olmalı)
    
    diff = torch.abs(image_A - sampled_rgb_B)
    
    # Sadece geçerli piksellerde ortalama al
    if valid_mask.sum() > 0:
        loss = (diff * valid_mask).sum() / (valid_mask.sum() * 3 + 1e-6)
    else:
        loss = torch.tensor(0.0, device="cuda")
        
    return loss, valid_mask, sampled_rgb_B