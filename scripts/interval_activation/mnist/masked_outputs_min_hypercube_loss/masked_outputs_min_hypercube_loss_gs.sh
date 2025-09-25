#!/bin/bash
#SBATCH --job-name=local_cl_split_mnist_mlp_masked_outputs_min_hypercube_loss
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/interval_activation/mnist/masked_outputs_min_hypercube_loss/masked_outputs_min_hypercube_loss_gs"