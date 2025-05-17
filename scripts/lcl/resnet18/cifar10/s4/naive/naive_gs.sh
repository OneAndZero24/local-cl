#!/bin/bash
#SBATCH --job-name=local_cl_split_cifar10_resnet18_s4_naive
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/lcl/resnet18/cifar10/s4/naive/naive_gs"
