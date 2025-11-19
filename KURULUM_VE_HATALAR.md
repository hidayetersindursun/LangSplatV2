# LangSplatV2 Kurulum Notları ve Hata Çözümleri

Bu belge, **LangSplatV2** projesinin kurulumu sırasında karşılaşılan hataları ve bunların çözümlerini içermektedir. Kurulum **Ubuntu 22.04**, **RTX 3090 Ti** ve **Sistem CUDA 12.x** ortamında yapılmıştır ancak hedef ortam **Python 3.7 + CUDA 11.6**'dır.

---

## 1. Conda Ortamı ve MKL Hatası
**Hata:** `undefined symbol: iJIT_NotifyEvent`
**Sebep:** Conda, Python 3.7 ve PyTorch 1.12 için çok yeni bir Intel MKL sürümü yüklüyor.
**Çözüm:** `environment.yml` dosyasında MKL sürümü sabitlendi.

`environment.yml` dosyasındaki `dependencies` kısmına şu satır eklendi:
```yaml
dependencies:
  - ...
  - mkl=2021.4.0  # <--- BU SATIR EKLENDİ
  - ...
```

---

## 2. CUDA Versiyon Uyuşmazlığı (System vs Conda)
**Hata:** `The detected CUDA version (13.0) mismatches the version that was used to compile PyTorch (11.6).`
**Sebep:** Sistemde kurulu olan CUDA sürümü (12.x/13.x), PyTorch'un desteklediği 11.6 sürümünden çok yeni olduğu için C++ eklentileri (`diff-gaussian-rasterization`, `simple-knn`) derlenemedi.
**Çözüm:** Conda ortamının içine sanal bir CUDA derleyicisi (NVCC) kuruldu ve sistem yolu oraya yönlendirildi.

**Uygulanan Komutlar:**
```bash
# Ortam aktifken:
conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc

# Derleyici yolunu Conda içine yönlendir (Bu komut her yeni terminalde tekrar girilmeli)
export CUDA_HOME=$CONDA_PREFIX

# Kontrol (11.7 veya 11.6 görmelisiniz)
nvcc --version
```
*Bu işlemden sonra `pip install ./submodules/...` komutları hatasız çalıştı.*

---

## 3. Python Kütüphane Uyuşmazlıkları
**Hata:** `ImportError: cannot import name 'Literal' from 'typing'` ve `ModuleNotFoundError: No module named 'mediapy'`
**Sebep:** `ftfy` ve `open_clip` kütüphanelerinin son sürümleri Python 3.8+ özellikleri istiyor. Ayrıca `mediapy` eksikti.
**Çözüm:**

```bash
pip install typing_extensions
pip install ftfy==6.0.3
pip install mediapy
```

---

## 4. Veri Seti ve Klasör Yolları
**Sorun:** `eval_lerf.sh` betiği veri setini `../../data` (iki üst klasör) altında arıyordu.
**Çözüm:** Veri seti proje içine (`./data`) taşındı ve script güncellendi.

`eval_lerf.sh` içindeki yollar şu şekilde değiştirildi:
```bash
DATASET_ROOT_PATH=./data/lerf_ovs
gt_folder=./data/lerf_ovs/label
```

---

## 5. Özel Prompt Scripti (demo_prompt.py)
Projede hazır bir görselleştirici olmadığı için, metin tabanlı (text prompt) arama yapabilmek adına `demo_prompt.py` yazıldı.

**Karşılaşılan Hatalar ve Çözümleri:**
1.  **AttributeError:** `include_feature`, `sh_degree` gibi parametreler eksikti. Script içinde `args` objesine manuel olarak eklendi.
2.  **CUDA Out of Memory:** 4K render almaya çalışırken VRAM yetmedi. `args.resolution = 2` yapılarak boyut düşürüldü ve `gc.collect()` eklendi.
3.  **Hatalı Boyama (Her yerin kırmızı olması):** Normalizasyon işlemi göreceli yapıldığı için, nesnenin olmadığı karelerde bile en yüksek puanlı yer kırmızı oluyordu.
    *   **Fix:** Akıllı kontrast (`similarity ** 4`) ve mutlak eşik değeri (`ABS_THRESHOLD = 0.22`) eklendi.

**Kullanım:**
```bash
python demo_prompt.py --dataset_name waldo_kitchen --prompt "mug"
```

---

## Özet Kurulum Komutları (Sıfırdan)

```bash
# 1. Ortamı Oluştur
conda env create --file environment.yml
conda activate langsplat_v2

# 2. CUDA Fix
conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc
export CUDA_HOME=$CONDA_PREFIX

# 3. Submodule Kurulumu
pip install ./submodules/segment-anything-langsplat
pip install ./submodules/efficient-langsplat-rasterization
pip install ./submodules/simple-knn

# 4. Ek Paketler
pip install ftfy==6.0.3 mediapy typing_extensions

# 5. Çalıştırma (Veri seti data/ klasöründe olmalı)
bash eval_lerf.sh waldo_kitchen 0 10000
```

