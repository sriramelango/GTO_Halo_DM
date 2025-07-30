import sys
import time

#sys.path.append('/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/python_scripts')

#from models import *  # TODO, import CVAE models and lstm models, from '/home/anjian/Desktop/project/generative_trajectory_optimization'
from Diffusion_Model_Scripts.CPU.classifier_free_guidance_cond_1d_improved_constrained_diffusion_cpu import Unet1D, GaussianDiffusion1D, Trainer1D

import numpy as np
import pickle
import torch

def main():

    # For icml
    checkpoint_path_list = [
                       f"/scratch/gpfs/jg3607/Diffusion_model/boundary/results/cr3bp_vanilla_diffusion_seed_0/unet_20_mults_4_4_8_embed_class_40_80_timesteps_100_objective_pred_noise_batch_size_512_cond_drop_0.1_mask_val_-1.0/2024-06-18_08-47-21"
    ]

    milestone_list = ["epoch-155"]

    data_num_list = [100000]

    sample_num = 2
    diffusion_w = 5.0
    thrust = 1.0
    num_seg = 20

    diffusion_type = "diffusion_boundary"
    # thrust_list = [0.15, 0.35, 0.45, 0.65, 0.85]

    save_warmstart_data = True

    for i in range(len(checkpoint_path_list)):
        data_num = data_num_list[i]
        checkpoint_path = checkpoint_path_list[i]
        milestone = milestone_list[i]

        unet_dim = 20 #128
        unet_dim_mults = "4,4,8"
        unet_dim_mults = tuple(map(int, unet_dim_mults.split(',')))
        embed_class_layers_dims = "40,80" #"256,512"
        embed_class_layers_dims = tuple(map(int, embed_class_layers_dims.split(',')))
        timesteps = 100
        objective = "pred_noise"
        mask_val = -1

        class_dim = 1
        channel = 3  # Changed to 3 channels
        seq_length = 22  # Changed to 22 sequence length
        cond_drop_prob = 0.1

        # Randomly sample halo energy
        alpha_data_normalized = torch.rand(size=(sample_num, 1), dtype=torch.float32)

        full_solution = get_sample_from_diffusion_attention(sample_num=sample_num,
                                                                                class_dim=class_dim,
                                                                                channel=channel,
                                                                                seq_length=seq_length,
                                                                                cond_drop_prob=cond_drop_prob,
                                                                                diffusion_w=diffusion_w,
                                                                                unet_dim=unet_dim,
                                                                                unet_dim_mults=unet_dim_mults,
                                                                                embed_class_layers_dims=embed_class_layers_dims,
                                                                                timesteps=timesteps,
                                                                                objective=objective,
                                                                                condition_input_data=alpha_data_normalized,
                                                                                checkpoint_path=checkpoint_path,
                                                                                milestone=milestone,
                                                                                mask_val=mask_val)

        # Data preparation #######################################################################################################
        min_shooting_time = 0
        max_shooting_time = 40
        min_coast_time = 0
        max_coast_time = 15
        min_halo_energy = 0.008
        max_halo_energy = 0.095
        min_final_fuel_mass = 408 #700-292 => cut off value at 90%
        max_final_fuel_mass = 470
        min_manifold_length = 5
        max_manifold_length = 11

        # Reconstruct 66-dimensional format from 3-channel output
        reconstructed_solutions = []
        for i in range(full_solution.shape[0]):
            # full_solution[i] has shape (3, 22) - reconstruct to (66,)
            trajectory_params = reconstruct_from_3_channels(full_solution[i])
            reconstructed_solutions.append(trajectory_params)
        
        full_solution = np.array(reconstructed_solutions)  # Shape: (sample_num, 66)

        # Unnormalize times
        full_solution[:, 0] = full_solution[:, 0] * (max_shooting_time - min_shooting_time) + min_shooting_time
        full_solution[:, 1] = full_solution[:, 1] * (max_coast_time - min_coast_time) + min_coast_time
        full_solution[:, 2] = full_solution[:, 2] * (max_coast_time - min_coast_time) + min_coast_time
        #Convert cartesian control back to correct range and to spherical coordinates
        full_solution[:, 3:-3] = full_solution[:, 3:-3] * 2 * thrust - thrust
        ux = full_solution[:,3:-3:3]
        uy = full_solution[:,4:-3:3]
        uz = full_solution[:,5:-3:3]
        alpha, beta, r = convert_to_spherical(ux, uy, uz)
        full_solution[:,3:-3:3] = alpha
        full_solution[:,4:-3:3] = beta
        full_solution[:,5:-3:3] = r
        # Unnormalize fuel mass and manifold parameters, HALO PERIOD IS NOT UNNORMALIZED, NEEDS TO BE DONE IN THE ACTUAL RUN
        full_solution[:, -3] = full_solution[:, -3] * (max_final_fuel_mass - min_final_fuel_mass) + min_final_fuel_mass
        full_solution[:, -1] = full_solution[:, -1] * (max_manifold_length - min_manifold_length) + min_manifold_length
        # Unnormalize halo energy
        halo_energies = alpha_data_normalized.detach().cpu().numpy() * (max_halo_energy - min_halo_energy) + min_halo_energy
        full_solution = np.hstack((halo_energies, full_solution))

        if save_warmstart_data:
            parent_path = "/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/generated_initializations/boundary"
            cr3bp_time_mass_alpha_control_path = f"{parent_path}/cr3bp_{diffusion_type}_w_{diffusion_w}_training_num_{data_num}_num_{sample_num}.pkl"
            with open(cr3bp_time_mass_alpha_control_path, "wb") as fp:  # write pickle
                pickle.dump(full_solution, fp)
                print(f"{cr3bp_time_mass_alpha_control_path} is saved!")


def get_sample_from_diffusion_attention(sample_num,
                                        class_dim,
                                        channel,
                                        seq_length,
                                        cond_drop_prob,
                                        diffusion_w,
                                        unet_dim,
                                        unet_dim_mults,
                                        embed_class_layers_dims,
                                        timesteps,
                                        objective,
                                        condition_input_data,
                                        checkpoint_path,
                                        milestone,
                                        mask_val):
    model = Unet1D(
        seq_length=seq_length,
        dim=unet_dim,
        channels=channel,
        dim_mults=unet_dim_mults,
        embed_class_layers_dims=embed_class_layers_dims,
        class_dim=class_dim,
        cond_drop_prob=cond_drop_prob,
        mask_val=mask_val,
    )

    diffusion = GaussianDiffusion1D(
        model=model,
        seq_length=seq_length,
        timesteps=timesteps,
        objective=objective
    )#.cuda()

    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=[0, 0, 0],
        results_folder=checkpoint_path, # Do not need to set batch size => automatically set through dimension of class variable
    )

    # milestone = "epoch-102"
    trainer.load(milestone)


    # 3. Use the loaded model for sampling
    start_time = time.time()
    sample_results = diffusion.sample(
        classes=condition_input_data,#.cuda(),
        cond_scale=diffusion_w,
    )
    end_time = time.time()
    print(f"{checkpoint_path}, {sample_num} data, takes {end_time - start_time} seconds")

    sample_results = sample_results.reshape(sample_num, -1)

    return sample_results.detach().cpu().numpy()

def convert_to_spherical(ux, uy, uz):
    u = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    theta = np.zeros_like(u)
    mask_non_zero = u != 0
    theta[mask_non_zero] = np.arcsin(uz[mask_non_zero] / u[mask_non_zero])
    alpha = np.arctan2(uy, ux)
    alpha = np.where(alpha >= 0, alpha, 2 * np.pi + alpha)

    # Make sure theta is in [0, 2*pi]
    theta = np.where(theta >= 0, theta, 2 * np.pi + theta)
    # Make sure u is not larger than 1
    u[u>1] = 1
    return alpha, theta, u

def reconstruct_from_3_channels(channels_data):
    """
    Reconstruct 66-dimensional trajectory data from 3-channel format.
    
    Args:
        channels_data: Array of shape (3, 22) with 3-channel data
    
    Returns:
        trajectory_params: Array of shape (66,) with original trajectory parameters
    """
    trajectory_params = np.zeros(66)
    
    # Index 1: Time variables (3 parameters)
    trajectory_params[0] = channels_data[0, 0]  # shooting_time
    trajectory_params[1] = channels_data[1, 0]  # initial_coast
    trajectory_params[2] = channels_data[2, 0]  # final_coast
    
    # Indices 2-21: Control vectors (60 parameters = 20 segments × 3 components)
    for i in range(20):
        # Each segment has 3 control components
        segment_start = 3 + i * 3
        trajectory_params[segment_start] = channels_data[0, i + 1]     # ux
        trajectory_params[segment_start + 1] = channels_data[1, i + 1] # uy
        trajectory_params[segment_start + 2] = channels_data[2, i + 1] # uz
    
    # Index 22: Final parameters (3 parameters)
    trajectory_params[63] = channels_data[0, 21]  # final_fuel_mass
    trajectory_params[64] = channels_data[1, 21]  # halo_period
    trajectory_params[65] = channels_data[2, 21]  # manifold_length
    
    return trajectory_params

if __name__ == "__main__":
    main()
