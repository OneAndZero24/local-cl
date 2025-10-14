#!/bin/bash
#SBATCH --job-name=local_cl_split_cifar100_resnet18_lcl_cil
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/interval_activation_cil/cifar100/lcl/lcl_gs"