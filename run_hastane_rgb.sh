#!/bin/bash

# Hastane Dataset RGB Training
# Usage: bash run_hastane_rgb.sh

DATASET_ROOT="/home/yusuf/AES/hidayet/LangSplatV2/custom_datasets/hastane-preprocess/colmap"
DATASET_NAME="hastane"
RGB_OUTPUT_DIR="output/${DATASET_NAME}_rgb"

echo "🏥 Starting RGB Training for Hastane Dataset (30,000 Iterations)..."

# Train only RGB (no features)
python train.py \
    -s $DATASET_ROOT \
    -m $RGB_OUTPUT_DIR \
    --iterations 30000 \
    --save_iterations 3000 6000 10000 30000 \
    --checkpoint_iterations 30000 \
    --test_iterations 1000 \


echo "✅ RGB Training Complete. Checkpoint saved to ${RGB_OUTPUT_DIR}_-1/chkpnt30000.pth"
echo "👀 You can visualize it using: python simple_viser.py --ply_path ${RGB_OUTPUT_DIR}_-1/point_cloud/iteration_30000/point_cloud.ply"