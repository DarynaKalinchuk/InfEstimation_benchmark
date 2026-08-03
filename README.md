 
This repository contains the code used for the experiments in my Master's thesis "*Benchmarking Training Data Attribution Methods for Large Language Models*", submitted to the University of Vienna.

## Repository structure

```
.
├── Dockerfile                         # Docker build file
├── datasets/                          # Datasets
├── settings_txt/                      # Configuration files for fine-tuning and influence estimation
├── scripts/                           # SLURM job scripts
├── finetune.py                        # Step 1: Model fine-tuning
├── influence.py                       # Step 2: Influence estimation and post-processing
├── inf_est_methods.py                 # Implementations of influence estimation methods (one function per method)
├── postprocess_utils.py               # Utilities for post-processing and plotting
└── utils.py                           # Utilities for preprocessing, gradient collection, model loading, and Kronfluence
```

## Requirements

The experiments were run using Python 3.11.14 on a single NVIDIA H100 80 GB GPU, other dependencies are specified in the Dockerfile. Alternatively, one can use the Docker image `kalinchukd/inf_env3` (size 12.2 GB).

## Running experiments

Fine-tune a model specified in scripts/finetune_params.txt on datasets specified in scripts/dataset_settings.txt:

```bash
sbatch finetune.sbatch
```

Estimate and process influence scores on the trained LoRa weights of the model specified in scripts/finetune_params.txt, for datasets specified in scripts/dataset_settings.txt:

```bash
sbatch influence.sbatch
```
