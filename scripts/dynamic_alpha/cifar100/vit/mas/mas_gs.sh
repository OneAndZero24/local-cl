#!/bin/bash
#SBATCH --job-name=local_cl_split_cifar100_vit_mas_dynamic_alpha
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/dynamic_alpha/cifar100/vit/mas/mas_gs"
