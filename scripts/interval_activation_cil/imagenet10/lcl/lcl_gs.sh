#!/bin/bash
#SBATCH --job-name=local_cl_imagenet10_lcl_cil
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/interval_activation_cil/imagenet10/lcl/lcl_gs"