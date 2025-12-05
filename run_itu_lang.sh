#!/bin/bash

# Kullanım (Arka planda çalıştırmak için):
# nohup ./run_itu_lang.sh > logs/train_lang_itu.log 2>&1 &
#
# Logları takip etmek için:
# tail -f logs/train_lang_itu.log

# Dataset ve Yollar
DATASET_PATH="custom_datasets/preprocess-itu"
DATASET_NAME="itu"
RGB_OUTPUT_DIR="output/${DATASET_NAME}_rgb"
RGB_CKPT_PATH="${RGB_OUTPUT_DIR}_-1/chkpnt30000.pth"

# 1. RGB Checkpoint Kontrolü
if [ ! -f "$RGB_CKPT_PATH" ]; then
    echo "❌ HATA: RGB Checkpoint bulunamadı: $RGB_CKPT_PATH"
    echo "Lütfen önce RGB eğitiminin (run_itu_rgb.sh) tamamlandığından emin olun."
    exit 1
fi

echo "🎯 RGB Checkpoint bulundu: $RGB_CKPT_PATH"

# 2. Feature Training (Level 0, 1, 2)
# Her seviye aynı RGB checkpoint'inden başlar ve bağımsız eğitilir.
# -r 2 parametresi (yarı çözünürlük) kullanılıyor.

for level in 0 1 2
do
    echo "🧠 [Level ${level}] Dil Özellikleri Eğitimi Başlıyor..."
    python train.py \
        -s $DATASET_PATH \
        -m output/${DATASET_NAME}_lang_${level} \
        --start_checkpoint $RGB_CKPT_PATH \
        --include_feature \
        --feature_level ${level} \
        --iterations 30000 \
        --save_iterations 30000 \
        --checkpoint_iterations 30000 \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --cos_loss \
        --topk 4 \
        --debug_interval 1000 \
        -r 2
        
    echo "✅ Level ${level} Tamamlandı."
done

echo "🎉 Tüm Dil Eğitimleri Tamamlandı!"
