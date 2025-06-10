#!/bin/bash
#SBATCH --job-name=a0.5_boundary
#SBATCH --ntasks=28           # Total number of tasks (cores used)
#SBATCH --nodes=1               # Number of nodes (200 cores / 32 cores per node)
#SBATCH --ntasks-per-node=28    # Number of tasks per node
#SBATCH --cpus-per-task=1       # Number of CPU cores per task
#SBATCH --mem-per-cpu=1G        # Memory per CPU core
#SBATCH --time=9:00:00         # Total run time limit (HH:MM:SS)
#SBATCH --mail-user=jg3607@princeton.edu

module purge
module load anaconda3/2021.11
conda activate direct
module load boost/1.73.0 

initial_seed=2800
seed_step=100
tasks=$SLURM_NTASKS

for (( i = 0; i < tasks; i++ )); do
    ig_seed=$((initial_seed + i * seed_step))
    echo "    Initial Guess Seed: $ig_seed"
    srun -N 1 -n 1 --exclusive python /home/jg3607/Thesis/AAS_paper/python_scripts/cr3bp_earth_mission_simulator_boundary.py --result_folder /home/jg3607/Thesis/AAS_paper/results/boundary/alpha_0.5/  --initial_guess_sample_mode uniform_sample --seed $ig_seed --seed_step $seed_step &

done

wait

