#!/bin/bash
#SBATCH --job-name="fntune2"
#SBATCH --container-image=kalinchukd/inf_env:latest
#SBATCH --container-mount-home
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:1
#SBATCH --time=0-5:00:00 
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/InfEstimation_benchmark

python3 --version

IFS=',' read -r MODEL_NAME TEMPLATE VAL_FLAG MAX_LENGTH BATCH_SIZE LOGGING_STEP EPOCHS LORA_R LORA_ALPHA TARGET_LAYER LOAD_8BIT < finetune_params.txt

VAL_ARG=""
if [ "$VAL_FLAG" == "True" ]; then
    VAL_ARG="--val"
fi

LOAD8BIT_ARG=""
if [ "$LOAD_8BIT" == "True" ]; then
    LOAD8BIT_ARG="--load_in_8bit"
fi

echo "MODEL_NAME = $MODEL_NAME"
echo "TEMPLATE = $TEMPLATE"
echo "MAX_LENGTH = $MAX_LENGTH"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "LOGGING_STEP = $LOGGING_STEP"
echo "EPOCHS = $EPOCHS"


while read -r DATASET_NAME; do
    echo "DATASET_NAME = $DATASET_NAME"

    python3 full_finetune.py \
        --model "$MODEL_NAME" \
        --dataset "$DATASET_NAME" \
        --template "$TEMPLATE" \
        $VAL_ARG \
        --max_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --logging_step "$LOGGING_STEP" \
        --epochs "$EPOCHS"

done < dataset_settings.txt