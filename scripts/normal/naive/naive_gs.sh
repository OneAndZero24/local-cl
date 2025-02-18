#!/bin/bash
#SBATCH --job-name=local_cl_split_mnist_naive_normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source ../../main.sh

run_sweep_and_agent "scripts/normal/naive/naive_gs"
