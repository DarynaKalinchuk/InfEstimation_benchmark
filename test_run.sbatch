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

MODEL_NAME="TinyLlama"       # --model
DATASET_NAME="backdoor"      # --dataset
TEMPLATE="llama2"            # --template
VAL_FLAG=False               # --val
MAX_LENGTH=128               # --max_length
BATCH_SIZE=2                 # --batch_size
LOGGING_STEP=10              # --logging_step
EPOCHS=1                     # --epochs
LORA_R=4                     # --lora_r
LORA_ALPHA=32                # --lora_alpha
TARGET_LAYER="-1"            # --target_layer
LOAD_8BIT=False              # --load_in_8bit


VAL_ARG=""
if [ "$VAL_FLAG" == "True" ]; then
    VAL_ARG="--val"
fi

LOAD8BIT_ARG=""
if [ "$LOAD_8BIT" == "True" ]; then
    LOAD8BIT_ARG="--load_in_8bit"
fi


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