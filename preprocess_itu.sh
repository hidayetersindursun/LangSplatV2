#!/bin/bash

# Kullanım (Arka planda çalıştırmak için):
# nohup ./preprocess_itu.sh > logs/preprocess_itu.log 2>&1 &
#
# Logları takip etmek için:
# tail -f logs/preprocess_itu.log

# Dataset yolu
DATASET_PATH="custom_datasets/preprocess-itu"

echo "Preprocess işlemi başlatılıyor..."
echo "Dataset: $DATASET_PATH"
echo "Resolution: 960 (Training -r 2 ile uyumlu)"

# Preprocess işlemini başlat
# --resolution 960: Görüntüleri 960px genişliğe (1080p'nin yarısı) ölçekler.
# --model_type vit_b: Daha hızlı SAM modeli.
python preprocess.py --dataset_path "$DATASET_PATH" --resolution 960 --sam_ckpt_path ckpts/sam_vit_b_01ec64.pth --model_type vit_b
