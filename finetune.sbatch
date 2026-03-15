#!/bin/bash
#SBATCH --job-name="fntune1"
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
echo "VAL_FLAG = $VAL_FLAG"
echo "MAX_LENGTH = $MAX_LENGTH"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "LOGGING_STEP = $LOGGING_STEP"
echo "EPOCHS = $EPOCHS"
echo "LORA_R = $LORA_R"
echo "LORA_ALPHA = $LORA_ALPHA"
echo "TARGET_LAYER = $TARGET_LAYER"
echo "LOAD_8BIT = $LOAD_8BIT"


for DATASET_NAME in backdoor emotion grammars; do

    echo "DATASET_NAME = $DATASET_NAME"

    python3 finetune.py \
        --model $MODEL_NAME \
        $LOAD8BIT_ARG \
        --dataset $DATASET_NAME \
        --template $TEMPLATE \
        $VAL_ARG \
        --max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --logging_step $LOGGING_STEP \
        --epochs $EPOCHS \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --target_layer "$TARGET_LAYER"
done