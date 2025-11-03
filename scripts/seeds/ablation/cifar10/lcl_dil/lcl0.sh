#!/bin/bash
#SBATCH --job-name=local_cl_cifar10_resnet18_lcl_dil
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/seeds/ablation/cifar10/lcl_dil/lcl0"
