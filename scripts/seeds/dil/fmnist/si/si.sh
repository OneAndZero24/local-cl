#!/bin/bash
#SBATCH --job-name=local_cl_split_fmnist_mlp_si_dil
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/seeds/dil/fmnist/si/si"
