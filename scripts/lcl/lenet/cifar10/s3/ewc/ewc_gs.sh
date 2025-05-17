#!/bin/bash
#SBATCH --job-name=local_cl_split_cifar10_lenet_s3_ewc
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/lcl/lenet/cifar10/s3/ewc/ewc_gs"
