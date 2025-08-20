import os.path
import sys
sys.path.append('../')
sys.path.append('./')

#from denoising_diffusion_pytorch.classifier_free_guidance_cond_1d_improved_constrained_diffusion import Unet1D, \
#    GaussianDiffusion1D, Trainer1D, Dataset1D

from Diffusion_Model_Scripts.GPU.classifier_free_guidance_cond_1d_improved_constrained_diffusion import Unet1D, \
    GaussianDiffusion1D, Trainer1D, Dataset1D

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
    machine = args.machine
    device_arg = args.device
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
    mask_val = float(args.mask_val)
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
    attn_heads = args.attn_heads
    attn_dim_head = args.attn_dim_head
    resnet_block_groups = args.resnet_block_groups
    self_condition = args.self_condition
    learned_sinusoidal_cond = args.learned_sinusoidal_cond
    init_dim = args.init_dim

    training_random_seed = args.training_random_seed
    set_seed(seed=training_random_seed)

    print(f"constraint_loss_type {constraint_loss_type}")
    print(f"normalize_xt_by_mean_sigma {normalize_xt_by_mean_sigma}")
    print(f"constraint_violation_weight {constraint_violation_weight}")

    #####################################################################################################################
    # Device setup
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    if device == "mps":
        print("MPS (Metal Performance Shaders) enabled for MacBook debugging")
    elif device == "cuda":
        print("CUDA enabled for GPU training")
    else:
        print("Using CPU for training")

    #####################################################################################################################
    # Create WANDB folder
    #if os.path.exists("/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb"):
    #    if task_type == "car":
    #        os.makedirs(f"/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/car/{training_data_type}", exist_ok=True)
    #    if task_type == "tabletop":
    #        os.makedirs(f"/scratch/gpfs/al5844/project/denoising-diffusion-pytorch/wandb/tabletop_v2/{training_data_type}", exist_ok=True)
    #else:
    #    os.makedirs(f"/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/wandb/{task_type}/{training_data_type}", exist_ok=True)
    ####################################################################################################################
    # Build the model
    model = Unet1D(
        dim=unet_dim,
        channels=channel_num,  # Use parameter instead of hardcoded 3
        dim_mults=unet_dim_mults,
        embed_class_layers_dims=embed_class_layers_dims,
        class_dim=class_dim,
        cond_drop_prob=cond_drop_prob,
        mask_val=mask_val,
        seq_length=22,  # Fixed to 22 sequence length
        attn_heads=attn_heads,
        attn_dim_head=attn_dim_head,
        resnet_block_groups=resnet_block_groups,
        self_condition=self_condition,
        learned_sinusoidal_cond=learned_sinusoidal_cond,
        init_dim=init_dim,
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
    ).to(device)

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

    # Convert to tensor
    training_seq = torch.tensor(x_downsampled)
    training_seq_classes = torch.tensor(c_downsampled)
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

    num_workers = 1
    checkpoint_folder = f"{result_folder}/{training_data_type}/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_batch_size_{batch_size}_cond_drop_{cond_drop_prob}_mask_val_{mask_val}_train_data_{training_data_num}/{current_time}"

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

    # Configure mixed precision based on device
    if device == "mps":
        # MPS doesn't support mixed precision, so disable it
        amp_enabled = False
        mixed_precision_type = 'no'
    else:
        amp_enabled = True
        mixed_precision_type = 'fp16'

    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=dataset,
        train_batch_size=batch_size,
        train_lr=8e-5,
        train_num_steps=step_per_epoch * max_epoch,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=amp_enabled,  # turn on mixed precision based on device
        mixed_precision_type=mixed_precision_type,
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
        classes=training_seq_classes[:10, :].to(device), #Use detected device
        cond_scale=6.,
        # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    )

    print(sampled_seq.shape)  # (64, 3, 20)

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for diffusion models")

    # Machine
    parser.add_argument('--machine',
                        type=str,
                        default="ubuntu",
                        help="Machine to run this code")
    
    # Device configuration
    parser.add_argument('--device',
                        type=str,
                        default="auto",
                        help="Device to use (auto, cuda, mps, cpu)")

    # Unet 1D parameters
    parser.add_argument('--unet_dim',
                        type=int,
                        default=20,
                        help='Dimension of the first layer of Unet')
    parser.add_argument('--unet_dim_mults',
                        type=str,
                        default="4,4,8",
                        help='List of dimension multipliers for Unet, currently at most 4 layers since we can only downsample 20 dim 4 times.')
    parser.add_argument('--embed_class_layers_dims',
                        type=str,
                        default="40,80",
                        help='List of dimensions for class embedding layers')
    parser.add_argument('--cond_drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of dropping conditioning during training')
    parser.add_argument('--channel_num',
                        type=int,
                        default=3,
                        help='Number of channels')
    parser.add_argument('--mask_val',
                        type=str,
                        default="0.0",
                        help='Value to use for masked conditioning')
    parser.add_argument('--attn_heads',
                        type=int,
                        default=4,
                        help='Number of attention heads')
    parser.add_argument('--attn_dim_head',
                        type=int,
                        default=32,
                        help='Dimension per attention head')
    parser.add_argument('--resnet_block_groups',
                        type=int,
                        default=4,
                        help='Number of groups for ResNet block GroupNorm')
    parser.add_argument('--self_condition',
                        action='store_true',
                        help='Enable self-conditioning for better quality')
    parser.add_argument('--learned_sinusoidal_cond',
                        action='store_true', 
                        help='Use learned sinusoidal positional embeddings')
    parser.add_argument('--init_dim',
                        type=int,
                        default=None,
                        help='Initial dimension after first conv (defaults to unet_dim)')
    parser.add_argument('--timesteps',
                        type=int,
                        default=500,
                        help='Number of diffusion timesteps')
    parser.add_argument('--objective',
                        type=str,
                        default="pred_noise",
                        choices=['pred_v', 'pred_noise'],
                        help='Objective function for diffusion')
    parser.add_argument('--seq_length',
                        type=int,
                        default=22,
                        help='Length of input sequences')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size for training')
    parser.add_argument('--data_path',
                        type=str,
                        default="./data/training_data_boundary_100000.pkl",
                        help='Path to training data')
    parser.add_argument('--wandb_project_name',
                        type=str,
                        default="diffusion_for_cr3bp_indirect",
                        help='Weights & Biases project name')
    parser.add_argument('--class_dim',
                        type=int,
                        default=1,
                        help='Dimension of class embeddings')
    parser.add_argument('--training_data_type',
                        type=str,
                        default="cr3bp_vanilla_diffusion_seed_0",
                        help='Type of training data')
    parser.add_argument('--training_data_range',
                        type=str,
                        default="0_1",
                        help='Range of training data')
    parser.add_argument('--training_data_num',
                        type=int,
                        default=100000,
                        help='Number of training samples')
    parser.add_argument('--max_epoch',
                        type=int,
                        default=200,
                        help='Maximum number of training epochs')
    parser.add_argument('--result_folder',
                        type=str,
                        default="./test_results",
                        help='Folder to save results')
    parser.add_argument('--constraint_violation_weight',
                        type=float,
                        default=0.001,
                        help='Weight for constraint violation loss')
    parser.add_argument('--constraint_condscale',
                        type=float,
                        default=6.0,
                        help='Conditioning scale for constraints')
    parser.add_argument('--training_random_seed',
                        type=int,
                        default=0,
                        help='Random seed for training')
    parser.add_argument('--max_sample_step_with_constraint_loss',
                        type=int,
                        default=500,
                        help='Maximum sampling step with constraint loss')
    parser.add_argument('--constraint_loss_type',
                        type=str,
                        default="NA",
                        choices=['one_over_t', 'gt_threshold', 'gt_scaled', 'gt_std', 'gt_std_absolute', 'gt_std_threshold', 'gt_log_likelihood', 'NA'],
                        help='Type of constraint loss')
    parser.add_argument('--task_type',
                        type=str,
                        default="cr3bp",
                        choices=['car', 'tabletop', 'cr3bp'],
                        help='Type of task')
    parser.add_argument('--constraint_gt_sample_num',
                        type=int,
                        default=1,
                        help='Number of ground truth samples for constraints')
    parser.add_argument('--normalize_xt_by_mean_sigma',
                        type=str,
                        default="False",
                        choices=['False', 'True'],
                        help='Whether to normalize x_t by mean and sigma')

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
