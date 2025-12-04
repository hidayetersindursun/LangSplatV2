#!/bin/bash

# LangSplat V2 Original Pipeline Implementation
# Usage: bash run_all_levels.sh [OPTIONAL_RGB_CHECKPOINT_PATH]

DATASET_ROOT="/home/yusuf/AES/hidayet/LangSplatV2/custom_datasets/preprocess-araba/colmap"
DATASET_NAME="araba"
RGB_OUTPUT_DIR="output/${DATASET_NAME}_rgb"
# train.py appends feature_level to model_path. For RGB training (default feature_level=-1), it becomes ..._-1
RGB_CKPT_PATH="${RGB_OUTPUT_DIR}_-1/chkpnt30000.pth"

# 1. Determine RGB Checkpoint to use
if [ ! -z "$1" ]; then
    # User provided a checkpoint
    RGB_CKPT="$1"
    echo "🎯 Using provided RGB checkpoint: $RGB_CKPT"
elif [ -f "$RGB_CKPT_PATH" ]; then
    # Default checkpoint exists
    RGB_CKPT="$RGB_CKPT_PATH"
    echo "🎯 Found existing RGB checkpoint: $RGB_CKPT"
    echo "⏩ Skipping RGB training."
else
    # No checkpoint found, train from scratch
    echo "🎨 [1/2] RGB Checkpoint not found. Starting RGB Training (30,000 Iterations)..."
    python train.py \
        -s $DATASET_ROOT \
        -m $RGB_OUTPUT_DIR \
        --iterations 30000 \
        --save_iterations 30000 \
        --checkpoint_iterations 30000 \
        --quiet
    
    RGB_CKPT="$RGB_CKPT_PATH"
    
    if [ ! -f "$RGB_CKPT" ]; then
        echo "❌ RGB Training failed! Checkpoint not created at $RGB_CKPT"
        exit 1
    fi
    echo "✅ RGB Training Complete."
fi

# --- Step 2: Feature Training (Levels 0, 1, 2) ---
# All levels start from the SAME RGB checkpoint and train independently.
# User requested downsampling (-r 2) for feature training.

for level in 0 1 2
do
    echo "🧠 [2/2] Starting Feature Training Level ${level}..."
    python train.py \
        -s $DATASET_ROOT \
        -m output/${DATASET_NAME}_final_${level} \
        --start_checkpoint $RGB_CKPT \
        --include_feature \
        --feature_level ${level} \
        --iterations 10000 \
        --save_iterations 10000 \
        --checkpoint_iterations 10000 \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --cos_loss \
        --topk 4 \
        --debug_interval 1000 \
        --quiet \
        -r 2
done

echo "🎉 All Training Complete! Ready for Backend Renderer."
