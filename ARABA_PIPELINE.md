# LangSplat V2 - Araba Sahnesi Eğitim ve Test Pipeline'ı

Bu doküman, "araba" veri seti üzerinde LangSplat V2 modelinin **Orijinal LangSplat Metodolojisine** uygun olarak sıfırdan eğitimine, karşılaşılan sorunların çözümüne ve nihai test aşamasına kadar olan süreci özetler.

---

## 1. Veri Hazırlığı (Data Preprocessing)

Eğitime başlamadan önce ham verilerin LangSplat formatına getirilmesi gerekiyordu.

### Kullanılan Araçlar:
*   **COLMAP:** Görüntülerden kamera pozlarını (sfm) ve nokta bulutunu çıkarmak için.
*   **CLIP Feature Extractor:** Görüntülerdeki her pikselin anlamsal (semantic) özelliklerini çıkarmak için.

### İşlem Adımları:
1.  Görüntülerden COLMAP ile `sparse/0` klasörü oluşturuldu.
2.  Görüntüler CLIP modelinden geçirilerek `.npy` formatında dil özellikleri (language features) çıkarıldı.
3.  Bu özellikler Autoencoder (VQ-VAE) için hazırlandı.

### Çıktı (Elde Edilenler):
*   `/images`: Orijinal RGB görüntüler.
*   `/sparse`: Kamera pozları ve nokta bulutu.
*   `/language_features`: Her görüntüye karşılık gelen sıkıştırılmış dil özellikleri.

---

## 2. Özellik Kontrolü (Inspection)

Eğitime başlamadan önce, çıkarılan CLIP özelliklerinin (Ground Truth) doğru olup olmadığını kontrol ettik.

### Kullanılan Kod: `inspect_features.py`
*   **Amaç:** Veri setindeki `.npy` dosyalarını okuyup, belirli bir metin sorgusuna (örn: "car") göre 2D heatmap oluşturmak.
*   **Değişiklikler:** Kod, sadece Level 0 değil, Level 1 ve Level 2 özelliklerini de görselleştirecek şekilde güncellendi.

### Çıktı:
*   `inspect_result_level_0.png`, `inspect_result_level_1.png`, `inspect_result_level_2.png`: 2D özellik haritalarının görsel doğrulaması.

---

## 3. Model Eğitimi (Training) - GÜNCELLENDİ (Orijinal Yöntem)

Bu aşama, orijinal LangSplat makalesindeki yönteme sadık kalınarak yeniden yapılandırıldı.

### Yeni Eğitim Stratejisi: "Bağımsız Seviyeler" (Independent Levels)
Eski kümülatif yöntemin aksine, orijinal yöntem şu adımları izler:
1.  **RGB Eğitimi (Base Model):** Önce sahnenin geometrisi ve rengi (RGB) 30.000 iterasyon boyunca eğitilir.
2.  **Feature Eğitimi (Levels 0, 1, 2):** Her özellik seviyesi, **aynı** RGB checkpoint'inden (`chkpnt30000.pth`) başlar ve **bağımsız olarak** 10.000 iterasyon daha eğitilir.

### Kullanılan Kod: `run_all_levels.sh` ve `train.py`
*   **Amaç:** Önce RGB modelini oturtmak, sonra dil özelliklerini bu geometri üzerine "giydirmek".
*   **Parametreler:**
    *   `--cos_loss`: Cosine Similarity Loss eklendi (Yön hizalaması için).
    *   `--topk 4`: Her piksel için en iyi 4 codebook elemanı kullanıldı (Sparse Coefficient Field).
    *   `-r 2`: Feature eğitimi sırasında downsampling uygulandı.
    *   `--iterations 10000`: Her feature seviyesi için eğitim süresi.

### Çalıştırılan Komut:
```bash
bash run_all_levels.sh
```
*(Not: Script önce `output/araba_rgb_-1` olup olmadığını kontrol eder, varsa RGB eğitimini atlayıp direkt feature eğitimine geçer.)*

### Çıktı:
*   `output/araba_rgb_-1/chkpnt30000.pth`: Saf RGB modeli.
*   `output/araba_final_0_0/chkpnt10000.pth`: Level 0 Feature Modeli.
*   `output/araba_final_1_1/chkpnt10000.pth`: Level 1 Feature Modeli.
*   `output/araba_final_2_2/chkpnt10000.pth`: Level 2 Feature Modeli.

---

## 4. Backend Kurulumu (Inference Server)

Eğitilen modeli yükleyip, dışarıdan gelen sorgulara cevap veren sunucu.

### Kullanılan Kod: `backend_renderer.py`
*   **Amaç:** Eğitilmiş `.pth` dosyalarını yüklemek, sanal bir kamera oluşturmak ve metin sorgusuna göre render almak.
*   **Yapılan Geliştirmeler:**
    1.  **Checkpoint Yolları:** Yeni eğitim yapısına (`araba_final_X_X`) ve iterasyonlarına (`chkpnt10000.pth`) göre güncellendi.
    2.  **Negatif Prompt (Canonical Phrases):** LERF benzeri bir mantık eklendi. Kullanıcının sorgusu ("car"), genel kavramlarla ("object", "background", "texture") karşılaştırılarak bir **Relevancy Score** (Alaka Skoru) hesaplandı. Bu, gürültüyü ciddi oranda azalttı.
    3.  **Görselleştirme:** Raw Cosine Similarity yerine Softmax tabanlı olasılık haritası kullanıldı.

### Çalıştırılan Komut:
```bash
python backend_renderer.py --dataset_name araba --source_path /home/yusuf/AES/hidayet/LangSplatV2/custom_datasets/preprocess-araba/colmap
```

### Çıktı:
*   `localhost:5555` portunda dinleyen bir ZMQ sunucusu.

---

## 5. Frontend ve Test (Visualization)

Kullanıcının modelle etkileşime geçtiği arayüz.

### Kullanılan Kod: `frontend_viser.py`
*   **Amaç:** Web tabanlı bir 3D viewer (Viser) açmak, kullanıcıdan metin (prompt) almak ve bunu Backend'e iletmek.
*   **İşleyiş:** Kullanıcı "car" yazar -> Backend render alır -> Frontend sonucu gösterir.

### Çalıştırılan Komut:
```bash
python frontend_viser.py
```

### Nihai Sonuç:
*   Tarayıcıda (`localhost:8081`) açılan 3D sahne.
*   "car", "wheel", "door", "windshield" gibi sorgularla nesnelerin parçalarının bile ayırt edilebildiği görüldü.
*   Negatif prompt desteği sayesinde arka plan gürültüsü minimize edildi.

---

## Özet Tablo

| Aşama | Kod Dosyası | Girdi | Çıktı |
| :--- | :--- | :--- | :--- |
| **1. Hazırlık** | `colmap`, `clip_extractor` | Resimler | Sparse Cloud, Language Features |
| **2. RGB Eğitim** | `run_all_levels.sh` | Hazırlanmış Veri | `araba_rgb_-1/chkpnt30000.pth` |
| **3. Feature Eğitim** | `run_all_levels.sh` | RGB Checkpoint | `araba_final_X_X/chkpnt10000.pth` |
| **4. Backend** | `backend_renderer.py` | Tüm Checkpointler | ZMQ Server (Render Stream) |
| **5. Frontend** | `frontend_viser.py` | Kullanıcı Sorgusu | İnteraktif 3D Görselleştirme |
