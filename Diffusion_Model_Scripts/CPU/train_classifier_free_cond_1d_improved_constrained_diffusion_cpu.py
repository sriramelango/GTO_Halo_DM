import os.path
import sys
sys.path.append('../')
sys.path.append('./')

from classifier_free_guidance_cond_1d_improved_constrained_diffusion_cpu import Unet1D, \
    GaussianDiffusion1D, Trainer1D, Dataset1D

#from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d_improved_constrained_diffusion import Unet1D, \
#    GaussianDiffusion1D, Trainer1D, Dataset1D


import torch
from torch.utils.data import TensorDataset
import pickle
import numpy as np
from datetime import datetime

import random
import argparse

# Version Jannik

def restructure_to_3_channels_from_67(trajectory_params):
    """
    Convert 66-dimensional trajectory parameters to 3-channel format.
    
    Args:
        trajectory_params: Array of shape (66,) with trajectory parameters
    
    Returns:
        channels_data: Array of shape (3, 22) with 3-channel data
    """
    channels_data = np.zeros((3, 22))
    
    # Index 1: Time variables (3 parameters)
    channels_data[0, 0] = trajectory_params[0]  # shooting_time -> channel 1
    channels_data[1, 0] = trajectory_params[1]  # initial_coast -> channel 2  
    channels_data[2, 0] = trajectory_params[2]  # final_coast -> channel 3
    
    # Indices 2-21: Control vectors (60 parameters = 20 segments × 3 components)
    for i in range(20):
        # Each segment has 3 control components
        segment_start = 3 + i * 3
        channels_data[0, i + 1] = trajectory_params[segment_start]     # ux -> channel 1
        channels_data[1, i + 1] = trajectory_params[segment_start + 1] # uy -> channel 2
        channels_data[2, i + 1] = trajectory_params[segment_start + 2] # uz -> channel 3
    
    # Index 22: Final parameters (3 parameters)
    channels_data[0, 21] = trajectory_params[63]  # final_fuel_mass -> channel 1
    channels_data[1, 21] = trajectory_params[64]  # halo_period -> channel 2
    channels_data[2, 21] = trajectory_params[65]  # manifold_length -> channel 3
    
    return channels_data

def main():
    #torch.set_default_device('cpu')
    ####################################################################################################################
    # Parse the arguments
    args = parse_args()
    
    # Device detection with proper MPS support
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA device")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("Using CPU device")
    elif args.device == 'cuda':
        device = torch.device('cuda')
        print("Using CUDA device")
    elif args.device == 'mps':
        device = torch.device('mps')
        print("Using MPS device (Apple Silicon)")
    elif args.device == 'cpu':
        device = torch.device('cpu')
        print("Using CPU device")
    else:
        raise ValueError(f"Invalid device selection: {args.device}")
    
    machine = args.machine
    unet_dim = args.unet_dim
    unet_dim_mults = tuple(map(int, args.unet_dim_mults.split(',')))
    embed_class_layers_dims = tuple(map(int, args.embed_class_layers_dims.split(',')))
    timesteps = args.timesteps
    objective = str(args.objective)
    batch_size = args.batch_size
    data_path = args.data_path
    cond_drop_prob = args.cond_drop_prob
    class_dim = args.class_dim
    channel_num = args.channel_num
    seq_length = args.seq_length
    training_data_type = str(args.training_data_type)
    mask_val = args.mask_val
    training_data_range = str(args.training_data_range)
    training_data_num = args.training_data_num
    wandb_project_name = str(args.wandb_project_name)
    result_folder = str(args.result_folder)
    max_epoch = args.max_epoch
    constraint_violation_weight = args.constraint_violation_weight
    constraint_condscale = args.constraint_condscale
    max_sample_step_with_constraint_loss = args.max_sample_step_with_constraint_loss
    constraint_loss_type = str(args.constraint_loss_type)
    task_type = str(args.task_type)
    constraint_gt_sample_num = args.constraint_gt_sample_num
    normalize_xt_by_mean_sigma = str(args.normalize_xt_by_mean_sigma)

    training_random_seed = args.training_random_seed
    set_seed(seed=training_random_seed)

    print(f"constraint_loss_type {constraint_loss_type}")
    print(f"normalize_xt_by_mean_sigma {normalize_xt_by_mean_sigma}")
    print(f"constraint_violation_weight {constraint_violation_weight}")


    #####################################################################################################################
    # Create WANDB folder
    # Commented out for local testing
    # if os.path.exists("/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb"):
    #     if task_type == "car":
    #         os.makedirs(f"/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/car/{training_data_type}", exist_ok=True)
    #     if task_type == "tabletop":
    #         os.makedirs(f"/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/tabletop_v2/{training_data_type}", exist_ok=True)
    # else:
    #     os.makedirs(f"/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/wandb/{task_type}/{training_data_type}", exist_ok=True)
    ####################################################################################################################
    # Build the model
    model = Unet1D(
        dim=unet_dim,
        channels=3,  # Fixed to 3 channels
        dim_mults=unet_dim_mults,
        embed_class_layers_dims=embed_class_layers_dims,
        class_dim=class_dim,
        cond_drop_prob=cond_drop_prob,
        mask_val=mask_val,
        seq_length=22,  # Fixed to 22 sequence length
    )

    diffusion = GaussianDiffusion1D(
        model=model,
        seq_length=22,  # Fixed to 22 sequence length
        timesteps=timesteps,
        objective=objective,
        # objective='pred_noise',
        constraint_violation_weight=constraint_violation_weight,
        constraint_condscale=constraint_condscale,
        max_sample_step_with_constraint_loss=max_sample_step_with_constraint_loss,
        constraint_loss_type=constraint_loss_type,
        task_type=task_type,
        constraint_gt_sample_num=constraint_gt_sample_num,
        normalize_xt_by_mean_sigma=normalize_xt_by_mean_sigma
    ).to(device)  # Use CPU device

    # # Random dataset
    # training_data_num = 64
    # training_seq = torch.rand(training_data_num, 3, 20)  # images are normalized from 0 to 1
    # training_seq_classes = torch.rand(training_data_num, 5)  # say 10 classes
    # dataset = TensorDataset(training_seq, training_seq_classes)

    # CR3BP dataset
    # data_path = "data/CR3BP/cr3bp_time_mass_alpha_control_part_4_250k_each.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Handle both old 67-dimensional format and new 3-channel format
    if isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] == 67:
        # Old format: (batch_size, 67) where first column is halo energy
        print("DEBUG - Converting old 67-dimensional format to 3-channel format")
        x_list = []
        c_list = []
        
        for i in range(len(data)):
            # Extract halo energy (index 0) and trajectory parameters (indices 1-66)
            halo_energy = data[i, 0]
            trajectory_params = data[i, 1:]
            
            # Restructure to 3-channel format
            channels_data = restructure_to_3_channels_from_67(trajectory_params)
            
            x_list.append(channels_data)
            c_list.append([halo_energy])
        
        # Convert to numpy arrays
        x = np.array(x_list).astype(np.float32)  # Shape: (batch_size, 3, 22)
        c = np.array(c_list).astype(np.float32)  # Shape: (batch_size, 1)
        
    else:
        # New format: [halo_energy, 3_channel_data] for each sample
        x_list = []
        c_list = []
        
        for sample in data:
            halo_energy = sample[0]  # Conditioning variable
            channels_data = sample[1]  # 3-channel trajectory data (3, 22)
            
            x_list.append(channels_data)
            c_list.append([halo_energy])
        
        # Convert to numpy arrays
        x = np.array(x_list).astype(np.float32)  # Shape: (batch_size, 3, 22)
        c = np.array(c_list).astype(np.float32)  # Shape: (batch_size, 1)
    
    # Debug: Print data shapes and sample values
    print(f"DEBUG - Data loading:")
    print(f"  Total samples loaded: {len(data)}")
    print(f"  X shape (trajectory data): {x.shape}")
    print(f"  C shape (conditioning): {c.shape}")
    print(f"  Sample X[0] shape: {x[0].shape}")
    print(f"  Sample C[0]: {c[0]}")
    print(f"  X[0, 0, :5] (Channel 1, first 5): {x[0, 0, :5]}")
    print(f"  X[0, 1, :5] (Channel 2, first 5): {x[0, 1, :5]}")
    print(f"  X[0, 2, :5] (Channel 3, first 5): {x[0, 2, :5]}")
    print("  ---")
    
    # Downsample if we use fewer data
    step_size = len(x) // training_data_num
    x_downsampled = x[::step_size, :]
    c_downsampled = c[::step_size, :]
    
    print(f"  After downsampling:")
    print(f"  X_downsampled shape: {x_downsampled.shape}")
    print(f"  C_downsampled shape: {c_downsampled.shape}")
    print("  ---")

    # Convert to tensor and move to device
    training_seq = torch.tensor(x_downsampled, dtype=torch.float32).to(device)
    training_seq_classes = torch.tensor(c_downsampled, dtype=torch.float32).to(device)
    if training_data_range == "0_1":
        pass
    elif training_data_range == "-1_1":
        training_seq = training_seq * 2.0 - 1.0
        training_seq_classes = training_seq_classes * 2.0 - 1.0
    else:
        print("wrong training data range!")

    # Debug: Print final tensor shapes
    print(f"DEBUG - Model input:")
    print(f"  training_seq shape: {training_seq.shape}")
    print(f"  training_seq_classes shape: {training_seq_classes.shape}")
    print(f"  training_seq dtype: {training_seq.dtype}")
    print(f"  training_seq_classes dtype: {training_seq_classes.dtype}")
    print(f"  training_seq device: {training_seq.device}")
    print(f"  training_seq_classes device: {training_seq_classes.device}")
    print(f"  Model channels: 3")
    print(f"  Model seq_length: 22")
    print("  ---")

    dataset = TensorDataset(training_seq, training_seq_classes)

    # TODO: one loss step ##################################################
    # loss = diffusion(training_seq, classes=training_seq_classes)
    # loss.backward()
    #
    # TODO: use trainer ###################################################

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Reconfigure the tuple variables to string
    unet_dim_mults_in_str = "_".join(map(str, unet_dim_mults))
    embed_class_layers_dims_in_str = "_".join(map(str, embed_class_layers_dims))

    # Set num_workers based on device to avoid multiprocessing issues
    if device.type == 'mps':
        num_workers = 0  # Avoid multiprocessing issues with MPS
    else:
        num_workers = 1

    checkpoint_folder = f"{result_folder}/{training_data_type}/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_objective_{objective}_batch_size_{batch_size}_cond_drop_{cond_drop_prob}_mask_val_{mask_val}/{current_time}"

    # if machine == "ubuntu":
    #     results_folder = f"results/diffusion/fixed_car_vary_obs/results/{training_data_type}/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_objective_{objective}_batch_size_{batch_size}_cond_drop_{cond_drop_prob}_mask_val_{mask_val}/{current_time}"
    #     num_workers = 1
    # elif machine == "autodl-car":
    #     results_folder = f"/root/autodl-tmp/project/diffusion/fixed_car_vary_obs/results/{training_data_type}/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_objective_{objective}_batch_size_{batch_size}_cond_drop_{cond_drop_prob}_mask_val_{mask_val}/{current_time}"
    #     num_workers = 1
    # elif machine == "autodl-cr3bp":
    #     results_folder = f"/root/autodl-tmp/project/diffusion/cr3bp/results/{training_data_type}/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_objective_{objective}_batch_size_{batch_size}_cond_drop_{cond_drop_prob}_mask_val_{mask_val}/{current_time}"
    #     num_workers = 1

    step_per_epoch = int(training_data_num / batch_size)
    # max_epoch = 200  # 200

    # Determine if we should use mixed precision based on device
    # Note: MPS doesn't support fp16 mixed precision in accelerate
    use_amp = device.type == 'cuda'  # Only enable for CUDA
    
    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=dataset,
        train_batch_size=batch_size,
        train_lr=8e-5,
        train_num_steps=step_per_epoch * max_epoch,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=use_amp,  # Enable mixed precision only for CUDA
        results_folder=checkpoint_folder,
        num_workers=num_workers,
        wandb_project_name=wandb_project_name,
        training_data_range=training_data_range,
        training_data_num=training_data_num,
        training_random_seed=training_random_seed,
    )
    trainer.train()

    # do above for many steps
    sampled_seq = diffusion.sample(
        classes=training_seq_classes[:10, :],#.cuda(), This was not commented
        cond_scale=6.,
        # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    )

    print(sampled_seq.shape)  # (64, 3, 20)

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for diffusion models")

    # Device selection
    parser.add_argument('--device',
                        type=str,
                        default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='Device to use for training (auto=detect, cpu=force CPU, mps=force MPS, cuda=force CUDA)')

    # Machine
    parser.add_argument('--machine',
                        type=str,
                        default="ubuntu",
                        help="Machine to run this code")

    # Unet 1D parameters
    parser.add_argument('--unet_dim',
                        type=int,
                        default=40,
                        help='Dimension of the first layer of Unet, MUST BE DIVISIBLE BY 4')
    parser.add_argument('--unet_dim_mults',
                        type=str,
                        default="4,4,8",
                        help='List of dimension multipliers for Unet, currently at most 4 layers since we can only downsample 20 dim 4 times.')
    parser.add_argument('--embed_class_layers_dims',
                        type=str,
                        default="40,80",
                        help='List of dimension for embedding class layers')
    parser.add_argument('--cond_drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of dropping the condition input')
    parser.add_argument('--channel_num',
                        type=int,
                        default=1,
                        help='Channel number of the data')
    parser.add_argument('--mask_val',
                        type=float,
                        default=0.0,
                        help='The value to mask context input')

    # GaussianDiffusion1D parameters
    parser.add_argument('--timesteps',
                        type=int,
                        default=500,
                        help='Timesteps for the diffusion process')
    parser.add_argument('--objective',
                        type=str,
                        default='pred_noise',
                        choices=['pred_v', 'pred_noise'],
                        help='Objectives for the diffusion model')
    parser.add_argument('--seq_length',
                        type=int,
                        default=6,
                        help='length of the data sequence')

    # Trainer1D parameters
    parser.add_argument('--batch_size',
                        type=int,
                        default=10,
                        help='Batch size for training')
    parser.add_argument('--data_path',
                        type=str,
                        default="/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/data/indirect/training_data_indirect_270000.pkl",
                        help="cr3bp data path")
    parser.add_argument('--wandb_project_name',
                        type=str,
                        default="diffusion_for_cr3bp_test",
                        help="project name for wandb")

    # Training data parameters
    parser.add_argument('--class_dim',
                        type=int,
                        default=1,
                        help='Dimension of the class variable')
    parser.add_argument('--training_data_type',
                        type=str,
                        default='cr3bp_cond_time_mass_alpha_data_control',
                        help='specify the condition input and the training data')
    parser.add_argument('--training_data_range',
                        type=str,
                        default="0_1",
                        help="the range of data after normalization")
    parser.add_argument('--training_data_num',
                        type=int,
                        default=100,
                        help="number of training data")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=2,
                        help="number of epochs to train")
    parser.add_argument('--result_folder',
                        type=str,
                        default="/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/checkpoint_result/",
                        help="result_folder")
    parser.add_argument('--constraint_violation_weight',
                        type=float,
                        default=0.01,
                        help="weight of the constraint violation term")
    parser.add_argument('--constraint_condscale',
                        type=float,
                        default=6.,
                        help="weight of the cond scale in constraint violation sampling")
    parser.add_argument('--training_random_seed',
                        type=int,
                        default=0,
                        help='random seed for model training')
    parser.add_argument('--max_sample_step_with_constraint_loss',
                        type=int,
                        default=500,
                        help="maximum sampling step that has constraint loss")
    parser.add_argument('--constraint_loss_type',
                        type=str,
                        default='NA',
                        help="type of constraint loss",
                        choices=["one_over_t", "gt_threshold", "gt_scaled", "gt_std", "gt_std_absolute", "gt_std_threshold", "gt_log_likelihood", "NA"])
    parser.add_argument('--task_type',
                        type=str,
                        default='cr3bp',
                        help="type of the task",
                        choices=["car", "tabletop", "cr3bp"])
    parser.add_argument('--constraint_gt_sample_num',
                        type=int,
                        default=100,
                        help="Number of samples for gt constraints")
    parser.add_argument('--normalize_xt_by_mean_sigma',
                        type=str,
                        default="False",
                        choices=["False", "True"],
                        help="whether to normalize xt by analytical mean and sigma")

    return parser.parse_args()

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

if __name__ == "__main__":
    main()
