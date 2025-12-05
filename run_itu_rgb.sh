#!/bin/bash

# Dataset ve çıktı yolları
DATASET_PATH="custom_datasets/preprocess-itu"
OUTPUT_PATH="output/itu_rgb"

# Sparse klasörü için sembolik link kontrolü (train.py kök dizinde sparse arar)
if [ ! -d "$DATASET_PATH/sparse" ]; then
    if [ -d "$DATASET_PATH/colmap/sparse" ]; then
        echo "Sparse klasörü için sembolik link oluşturuluyor..."
        ln -s colmap/sparse "$DATASET_PATH/sparse"
    else
        echo "HATA: $DATASET_PATH/colmap/sparse bulunamadı! COLMAP çıktısı eksik olabilir."
        exit 1
    fi
fi

echo "RGB Eğitimi başlatılıyor..."
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_PATH"
echo "Resolution: -r 2"

# Eğitimi başlat
python train.py -s "$DATASET_PATH" -m "$OUTPUT_PATH" -r 2
