#!/bin/bash
#SBATCH --job-name="inf1"
#SBATCH --container-image=kalinchukd/inf_env3:latest
#SBATCH --container-mount-home
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24
#SBATCH --time=0-23:00:00 
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/infl_b


set -e

export PYTHONHASHSEED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python3 --version

IFS=',' read -r MODEL_NAME MAX_LENGTH BATCH_SIZE EPOCHS LORA_R LORA_ALPHA < settings_txt/finetune_params.txt

echo "MODEL_NAME = $MODEL_NAME"
echo "MAX_LENGTH = $MAX_LENGTH"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "EPOCHS = $EPOCHS"


while read -r DATASET_NAME; do

    python3 -c "from postprocess_utils import generate_combined_plots; generate_combined_plots(name_begin='${DATASET_NAME}_${EPOCHS}', results_dir='results/distr_plots')"

done < settings_txt/dataset_settings.txt


python3 -c "from postprocess_utils import generate_table_metrics; generate_table_metrics()"

