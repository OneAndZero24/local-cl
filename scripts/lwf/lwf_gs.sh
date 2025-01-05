#!/bin/bash
#SBATCH --job-name=local_cl_split_mnist_lwf
#SBATCH --qos=normal
#SBATCH --mem=32G
#SBATCH --partition=dgx

if [ -f .env ]; then
  export $(cat .env | grep -v '#' | xargs) 
  echo ".env file loaded successfully"
else
  echo ".env file not found!"
  exit 1
fi

echo "Activating the conda environment: lcl"
source activate lcl

cd $HOME/LocalCL || { echo "Error: Directory $HOME/LocalCL not found!"; exit 1; }

run_sweep_and_agent () {
  SWEEP_NAME="$1"
  
  YAML_PATH="$HOME/LocalCL/$SWEEP_NAME.yaml"
  
  if [ ! -f "$YAML_PATH" ]; then
    echo "Error: YAML file '$SWEEP_NAME.yaml' not found in $HOME/LocalCL"
    exit 1
  fi
  
  echo "Running wandb sweep for: $SWEEP_NAME"
  wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$YAML_PATH" > temp_output.txt 2>&1
  
  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)
  
  rm temp_output.txt

  echo "Starting WandB agent for sweep ID: $SWEEP_ID"
  wandb agent "$SWEEP_ID"
}

run_sweep_and_agent "scripts/lwf/lwf_gs"
