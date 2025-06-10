#!/bin/bash
#SBATCH --job-name=boundary
#SBATCH --ntasks=32            # Total number of tasks (cores used)
#SBATCH --nodes=1               # Number of nodes (200 cores / 32 cores per node)
#SBATCH --ntasks-per-node=32    # Number of tasks per node
#SBATCH --cpus-per-task=1       # Number of CPU cores per task
#SBATCH --mem-per-cpu=1G        # Memory per CPU core
#SBATCH --partition=cpu
#SBATCH --constraint=cascade
#SBATCH --time=02:00:00         # Total run time limit (HH:MM:SS)
#SBATCH --mail-user=jg3607@princeton.edu

module purge
module load anaconda3/2021.11
conda activate direct
module load boost/1.73.0 

initial_seed=0
seed_step=32
tasks=$SLURM_NTASKS

for (( i = 0; i < tasks; i++ )); do
    ig_seed=$((initial_seed + i * seed_step))
    srun -N 1 -n 1 --exclusive python /home/jg3607/Thesis/AAS_paper/python_scripts/cr3bp_earth_mission_simulator_boundary_diffusion_warmstart.py --seed $ig_seed --seed_step $seed_step --snopt_time_limit 500 --result_folder /home/jg3607/Thesis/AAS_paper/results/boundary/diffusion_warmstart/random_alpha_1000_seeds/ --sample_path "/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/generated_initializations/boundary/unet_128_mults_4_4_8_embed_class_256_512_timesteps_500_batch_size_512_cond_drop_0.1_mask_val_-1.0/cr3bp_diffusion_boundary_w_5.0_training_num_100000_num_10000.pkl" &

done

wait



