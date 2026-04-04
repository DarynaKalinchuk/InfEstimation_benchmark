#!/bin/bash
#SBATCH --job-name="inf1"
#SBATCH --container-image=kalinchukd/inf_env:latest
#SBATCH --container-mount-home
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:2
#SBATCH --time=0-3:00:00 
#SBATCH --container-workdir=/srv/home/users/kalinchukd23cs/InfEstimation_benchmark


python3 --version

python3 ekfac.py
