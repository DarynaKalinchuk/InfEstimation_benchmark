#!/bin/bash
#SBATCH --job-name="inf1"
#SBATCH --container-image=kalinchukd/inf_env:latest
#SBATCH --container-mount-home
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00:00 
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/InfEstimation_benchmark
#SBATCH --nodelist=dgx1

python3 --version

IFS=',' read -r MODEL_NAME TEMPLATE VAL_FLAG MAX_LENGTH BATCH_SIZE LOGGING_STEP EPOCHS LORA_R LORA_ALPHA TARGET_LAYER LOAD_8BIT < finetune_params.txt

echo "MODEL_NAME = $MODEL_NAME"
echo "TEMPLATE = $TEMPLATE"
echo "MAX_LENGTH = $MAX_LENGTH"
echo "EPOCHS = $EPOCHS"

HVP_METHOD="DataInf"

while read -r DATASET_NAME; do
    echo "DATASET_NAME = $DATASET_NAME"

    python3 influence.py \
        --model "$MODEL_NAME" \
        --dataset "$DATASET_NAME" \
        --epochs "$EPOCHS" \
        --hvp_cal "$HVP_METHOD" \
        --template "$TEMPLATE" \
        --max_length "$MAX_LENGTH"

done < dataset_settings.txt
