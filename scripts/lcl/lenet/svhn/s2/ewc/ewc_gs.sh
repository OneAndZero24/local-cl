#!/bin/bash
#SBATCH --job-name=local_cl_split_svhn_lenet_s2_ewc
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/lcl/lenet/svhn/s2/ewc/ewc_gs"
