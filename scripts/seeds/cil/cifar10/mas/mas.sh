#!/bin/bash
#SBATCH --job-name=local_cl_cifar10_resnet18_mas_seeds
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/seeds/cil/cifar10/mas/mas"
