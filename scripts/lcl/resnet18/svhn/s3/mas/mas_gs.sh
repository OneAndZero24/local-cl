#!/bin/bash
#SBATCH --job-name=local_cl_split_cifar10_resnet18_s3_mas
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/lcl/resnet18/cifar10/s3/mas/mas_gs"
