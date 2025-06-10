#!/bin/bash
#SBATCH --job-name=TS300_boundary
#SBATCH --ntasks=1            # Total number of tasks (cores used)
#SBATCH --nodes=1               # Number of nodes (200 cores / 32 cores per node)
#SBATCH --ntasks-per-node=1    # Number of tasks per node
#SBATCH --cpus-per-task=1       # Number of CPU cores per task
#SBATCH --mem-per-cpu=4G        # Memory per CPU core
#SBATCH --partition=cpu
#SBATCH --constraint=cascade
#SBATCH --time=10:00:00         # Total run time limit (HH:MM:SS)
#SBATCH --mail-user=jg3607@princeton.edu

module purge
module load anaconda3/2021.11
conda activate direct
module load boost/1.73.0 

python /home/jg3607/Thesis/AAS_paper/python_scripts/cr3bp_earth_mission_simulator_boundary_diffusion_warmstart.py --seed_step 200 --sample_path "/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/generated_initializations/boundary/unet_128_mults_4_4_8_embed_class_256_512_timesteps_300_batch_size_2000_cond_drop_0.1_mask_val_-1.0/cr3bp_diffusion_boundary_w_5.0_training_num_100000_num_10000.pkl" --snopt_time_limit 500 --result_folder /home/jg3607/Thesis/AAS_paper/results/boundary/diffusion_warmstart/unet_128_mults_4_4_8_embed_class_256_512_timesteps_300_batch_size_2000_cond_drop_0.1_mask_val_-1.0/



