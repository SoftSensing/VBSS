#!/bin/bash
#SBATCH --job-name=video_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=rome
#SBATCH --gres=gpu:1

# Load the necessary modules. Adjust as per available versions.
module purge
module load 2021
module load Python/3.9.6
module load CUDA/11.4.1

# It is recommended to run Python scripts in a virtual environment
# Replace 'myenv' with the path to your virtual environment
#source ~/myenv/bin/activate

# Execute the Python script
srun python my_script.py
