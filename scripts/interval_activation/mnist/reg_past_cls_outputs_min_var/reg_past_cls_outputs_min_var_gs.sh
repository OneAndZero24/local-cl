#!/bin/bash
#SBATCH --job-name=local_cl_split_mnist_mlp_reg_past_cls_outputs_min_var
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/interval_activation/mnist/mlp/reg_past_cls_outputs_min_var_gs"
