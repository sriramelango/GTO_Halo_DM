#!/bin/bash
#SBATCH --job-name=a1.0diff_bou_sam
#SBATCH --output=./output/sample_dr_v0.%A.%a.out
#SBATCH --error=./output/sample_dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mail-type=begin            # send email when job begins
#SBATCH --mail-type=end              # send email when job ends
#SBATCH --mail-user=se7159@princeton.edu
#SBATCH --array=1

# Load conda (adjust path if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-env

# Print GPU info for debugging
nvidia-smi

export WANDB_DIR="./wandb/sampling"
export WANDB_MODE=offline

# Sampling script with organized parameters
python Diffusion_Model_Scripts/GPU/sample_data_diffusion_boundary.py \
    --timesteps 500 \
    --data_num 100000 \
    --sample_num 10000 \
    --mask_val "-1.0" \
    --fixed_alpha 1.0 \
    --device auto
