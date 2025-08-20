#!/bin/bash
#SBATCH --job-name=d_bou_300
#SBATCH --output=./output/dr_v0.%A.%a.out
#SBATCH --error=./output/dr_v0.%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --constraint="amd&gpu40"
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin            # send email when job begins
#SBATCH --mail-type=end              # send email when job ends
#SBATCH --mail-user=se7159@princeton.edu
#SBATCH --array=1

# Load conda (adjust path if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-env

# Print GPU info for debugging
nvidia-smi

export WANDB_DIR="./wandb"
export WANDB_MODE=offline

# Training script with organized parameters
python Diffusion_Model_Scripts/GPU/train_classifier_free_cond_1d_improved_constrained_diffusion.py \
    --device cuda \
    --data_path ./data/training_data_boundary_100000.pkl \
    --training_data_num 100000 \
    --max_epoch 200 \
    --batch_size 2000 \
    --result_folder ./results \
    \
    --unet_dim 128 \
    --embed_class_layers_dims "128,256,512" \
    --attn_heads 8 \
    --attn_dim_head 32 \
    --resnet_block_groups 8 \
    --learned_sinusoidal_cond \
    --channel_num 3 \
    --seq_length 22 \
    --class_dim 1 \
    \
    --timesteps 500 \
    --training_random_seed 0 \
    --mask_val "-1.0" \
    \
    --constraint_loss_type NA \
    --task_type cr3bp \
    --training_data_type cr3bp_vanilla_diffusion_seed_0 \
    \
    --wandb_project_name "Diffusion boundary"
