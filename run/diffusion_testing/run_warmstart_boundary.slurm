#!/bin/bash
#SBATCH --job-name=boundary
#SBATCH --ntasks=1            # Total number of tasks (cores used)
#SBATCH --nodes=1               # Number of nodes (200 cores / 32 cores per node)
#SBATCH --ntasks-per-node=1    # Number of tasks per node
#SBATCH --cpus-per-task=1       # Number of CPU cores per task
#SBATCH --mem-per-cpu=1G        # Memory per CPU core
#SBATCH --partition=cpu
#SBATCH --constraint=cascade
#SBATCH --time=10:00:00         # Total run time limit (HH:MM:SS)
#SBATCH --mail-user=jg3607@princeton.edu

module purge
module load anaconda3/2021.11
conda activate direct
module load boost/1.73.0 

python /home/jg3607/Thesis/AAS_paper/python_scripts/cr3bp_earth_mission_simulator_boundary_diffusion_warmstart.py --seed_step 200 --snopt_time_limit 500 --result_folder /home/jg3607/Thesis/AAS_paper/results/boundary/diffusion_warmstart/unet_128_mults_4_4_8_embed_class_256_512_timesteps_500_batch_size_512_cond_drop_0.1_mask_val_0.0_spher_snopt_500s/  --initial_guess_sample_mode from_diffusion



