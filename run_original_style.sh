#!/bin/bash

# LangSplat V2 Original Style Training Pipeline
# 1. Train RGB Model (30k iterations)
# 2. Train Feature Model (10k iterations, starting from RGB checkpoint)

DATASET_ROOT="/home/yusuf/AES/hidayet/LangSplatV2/custom_datasets/preprocess-araba/colmap"
DATASET_NAME="araba"

# --- Step 1: RGB Training ---
echo "🎨 [1/2] Starting RGB Training (30,000 Iterations)..."
python train.py \
    -s $DATASET_ROOT \
    -m output/${DATASET_NAME}_rgb \
    --iterations 30000 \
    --save_iterations 30000 \
    --checkpoint_iterations 30000 \
    --quiet

# Check if RGB training succeeded
if [ ! -f "output/${DATASET_NAME}_rgb/chkpnt30000.pth" ]; then
    echo "❌ RGB Training failed! Checkpoint not found."
    exit 1
fi

echo "✅ RGB Training Complete."

# --- Step 2: Feature Training (Levels 1, 2, 3) ---
# Note: Original train.sh loops through levels 1, 2, 3.
# We will replicate this behavior.

for level in 1 2 3
do
    echo "🧠 [2/2] Starting Feature Training Level ${level} (10,000 Iterations)..."
    python train.py \
        -s $DATASET_ROOT \
        -m output/${DATASET_NAME}_feature_level_${level} \
        --start_checkpoint output/${DATASET_NAME}_rgb/chkpnt30000.pth \
        --include_feature \
        --feature_level ${level} \
        --iterations 10000 \
        --save_iterations 10000 \
        --checkpoint_iterations 10000 \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --quiet
done

echo "🎉 All Training Complete!"
