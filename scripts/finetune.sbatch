#!/bin/bash
#SBATCH --job-name="fntune1"
#SBATCH --container-image=kalinchukd/inf_env3:latest
#SBATCH --container-mount-home
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24  
#SBATCH --time=0-1:00:00 
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/infl_b
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --gres=gpu:h100:1

python3 --version

IFS=',' read -r MODEL_NAME MAX_LENGTH BATCH_SIZE EPOCHS LORA_R LORA_ALPHA < settings_txt/finetune_params.txt


echo "MODEL_NAME = $MODEL_NAME"
echo "MAX_LENGTH = $MAX_LENGTH"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "EPOCHS = $EPOCHS"
echo "LORA_R = $LORA_R"
echo "LORA_ALPHA = $LORA_ALPHA"

while read -r DATASET_NAME; do
    echo "DATASET_NAME = $DATASET_NAME"

    python3 finetune.py \
        --model $MODEL_NAME \
        --dataset $DATASET_NAME \
        --max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA
        
done < settings_txt/dataset_settings.txt