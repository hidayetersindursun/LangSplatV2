#!/bin/bash

# Evaluation Script for Araba Dataset
# Usage: bash run_eval_araba.sh

DATASET_ROOT="/home/yusuf/AES/hidayet/LangSplatV2/custom_datasets/preprocess-araba/colmap"
DATASET_NAME="araba"

# Evaluate the final combined model (Level 2)
MODEL_PATH="output/araba_final_2_2" 

echo "📊 Starting Evaluation for $DATASET_NAME..."

python eval_araba.py \
    -s $DATASET_ROOT \
    -m $MODEL_PATH \
    --iteration 10000

echo "✅ Done."
