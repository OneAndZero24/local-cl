#!/bin/bash
#SBATCH --job-name=local_cl_split_fmnist_resnet_ewc_dynamic_alpha
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/dynamic_alpha/fmnist/resnet18/ewc/ewc_gs"
