#!/bin/bash
#SBATCH --job-name=local_cl_split_mnist_neuron_reg_scenario_3
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/scenario_3/neuron_reg/neuron_reg_gs"
