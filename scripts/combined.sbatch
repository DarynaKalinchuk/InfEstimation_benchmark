#!/bin/bash
#SBATCH --job-name="ft_inf"
#SBATCH --container-image=kalinchukd/inf_env3:latest
#SBATCH --container-mount-home
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/infl_b
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --gres=gpu:h100:1

set -e

export PYTHONHASHSEED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python3 --version

IFS=',' read -r MODEL_NAME MAX_LENGTH BATCH_SIZE EPOCHS LORA_R LORA_ALPHA < settings_txt/finetune_params.txt

echo "MODEL_NAME = $MODEL_NAME"
echo "MAX_LENGTH = $MAX_LENGTH"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "EPOCHS = $EPOCHS"
echo "LORA_R = $LORA_R"
echo "LORA_ALPHA = $LORA_ALPHA"

while read -r DATASET_NAME; do
    [ -z "$DATASET_NAME" ] && continue

    echo "=============================="
    echo "FINETUNING DATASET: $DATASET_NAME"
    echo "=============================="

    python3 finetune.py \
        --model "$MODEL_NAME" \
        --dataset "$DATASET_NAME" \
        --max_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA"

    echo "=============================="
    echo "RUNNING INFLUENCE FOR: $DATASET_NAME"
    echo "=============================="

    while read -r INF_METHOD; do
        [ -z "$INF_METHOD" ] && continue

        echo "DATASET_NAME = $DATASET_NAME"
        echo "INF_METHOD = $INF_METHOD"

        if [[ "${MODEL_NAME,,}" == *"random"* && "$INF_METHOD" == "TracIn" ]]; then
            echo "TracIn disabled for randomized models."
            continue
        fi

        METRICS_PATH="results/json/${DATASET_NAME}_${MODEL_NAME}_${INF_METHOD}_metrics_results.json"
        SCORES_PATH="scores/${MODEL_NAME}/${DATASET_NAME}_${EPOCHS}${INF_METHOD}.csv"
        RUNTIME_PATH="runtime_stats/${MODEL_NAME}/${DATASET_NAME}_${EPOCHS}${INF_METHOD}.txt"

        torchrun --standalone --nproc_per_node=1 influence.py \
            --model "$MODEL_NAME" \
            --dataset "$DATASET_NAME" \
            --epochs "$EPOCHS" \
            --inf_method "$INF_METHOD" \
            --inf_args "$INF_ARGS" \
            --max_length "$MAX_LENGTH" \
            --metrics_path "$METRICS_PATH" \
            --scores_path "$SCORES_PATH" \
            --runtime_path "$RUNTIME_PATH"

        python3 -c "from postprocess_utils import run_benchmark_measures; run_benchmark_measures(metrics_path='$METRICS_PATH', method='$INF_METHOD', scores_path='$SCORES_PATH', runtime_path='$RUNTIME_PATH', dataset_path='datasets/$DATASET_NAME')"

    done < settings_txt/methods_to_run.txt

    python3 -c "from postprocess_utils import generate_combined_plots; generate_combined_plots(name_begin='${DATASET_NAME}_${EPOCHS}', results_dir='results/distr_plots')"

done < settings_txt/dataset_settings.txt

python3 -c "from postprocess_utils import generate_table_metrics; generate_table_metrics()"