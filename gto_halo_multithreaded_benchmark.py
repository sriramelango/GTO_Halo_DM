#!/usr/bin/env python3
"""
Fast Multithreaded GTO Halo Benchmarking Module

This version reduces thread initialization delays by:
1. Pre-initializing thread pool
2. Pre-loading libraries in each thread
3. Using thread-local storage for simulators
4. Reducing file I/O overhead
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import pandas as pd
from scipy import stats
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from multiprocessing import cpu_count
from datetime import datetime

# Set matplotlib to non-interactive backend to avoid GUI issues in threads
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

warnings.filterwarnings('ignore')

# Add GTO_Halo_DM to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import from GTO_Halo_DM
try:
    # Add the correct paths for imports (matching your approach but adapted for our directory structure)
    current_dir = os.getcwd()
    sys.path.append(current_dir)
    sys.path.append(os.path.join(current_dir, 'Data_Generation'))
    sys.path.append(os.path.join(current_dir, 'Data_Generation', 'support_scripts'))
    
    from Diffusion_Model_Scripts.GPU.classifier_free_guidance_cond_1d_improved_constrained_diffusion import Unet1D, GaussianDiffusion1D, Trainer1D
    from Data_Generation.cr3bp_earth_mission_simulator_boundary_diffusion_warmstart import CR3BPEarthMissionWarmstartSimulatorBoundary
    from support_scripts.support import get_GTO_in_CR3BP_units
    GTO_HALO_DM_AVAILABLE = True
    print("✓ GTO_Halo_DM modules loaded successfully")
    print("✓ Physical validation enabled - CR3BP simulator available")
except ImportError as e:
    print(f"Warning: GTO_Halo_DM modules not available: {e}")
    CR3BPEarthMissionWarmstartSimulatorBoundary = None
    GTO_HALO_DM_AVAILABLE = False


@dataclass
class GTOHaloBenchmarkConfig:
    """Configuration for fast multithreaded GTO Halo benchmarking."""
    # Model and data config
    model_path: str
    checkpoint_path: str
    data_path: str
    
    # Sampling config
    num_samples: int = 1000
    batch_size: int = 100
    sampling_method: str = "pc"
    guidance_weight: float = 5.0  # Default from GTO Halo DM
    fixed_alpha: Optional[float] = None  # Fixed alpha value (None for random sampling)
    
    # Physical validation config
    enable_physical_validation: bool = True
    
    # Multithreading config
    max_workers: int = None  # Will default to cpu_count() if None
    chunk_size: int = 1  # Number of samples per thread
    pre_warm_threads: bool = True  # Pre-initialize threads
    
    # Output config
    output_dir: str = "benchmark_run"  # Will be placed inside "Benchmark Results"
    save_samples: bool = True
    save_plots: bool = True
    
    # Device config
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class ThreadLocalStorage:
    """Thread-local storage for pre-initialized simulators."""
    def __init__(self):
        self.local = threading.local()
    
    def get_simulator(self, thread_id: int, output_dir: str):
        """Get or create a simulator for this thread."""
        if not hasattr(self.local, f'simulator_{thread_id}'):
            # Pre-initialize simulator for this thread
            print(f"Thread {thread_id}: Pre-initializing CR3BP simulator...")
            
                        # CR3BP simulation parameters (EXACTLY from GTO Halo DM simulator)
            cr3bp_config = {
                'seed': thread_id,
                'seed_step': 1,  # Process single sample
                'quiet_snopt': True,
                'number_of_segments': 20,  # Match GTO Halo DM
                'maximum_shooting_time': 40.0,  # Match GTO Halo DM
                'minimum_shooting_time': 0.0,  # Match GTO Halo DM
                'start_bdry': 6.48423370092,  # Match GTO Halo DM
                'end_bdry': 8.0,  # Match GTO Halo DM
                'thrust': 1.0,  # Match GTO Halo DM
                'solver_mode': "optimal",  # Match GTO Halo DM (0 = optimal, "feasible" = feasible)
                'min_mass_to_sample': 408,  # Match GTO Halo DM
                'max_mass_to_sample': 470,  # Match GTO Halo DM
                'snopt_time_limit': 500.0,  # Match GTO Halo DM (original benchmarking)
                'result_folder': os.path.join(output_dir, 'cr3bp_results')
            }
            
            # Create a dummy simulator for pre-initialization
            dummy_simulator = CR3BPEarthMissionWarmstartSimulatorBoundary(
                seed=cr3bp_config['seed'],
                seed_step=cr3bp_config['seed_step'],
                quiet_snopt=cr3bp_config['quiet_snopt'],
                number_of_segments=cr3bp_config['number_of_segments'],
                maximum_shooting_time=cr3bp_config['maximum_shooting_time'],
                minimum_shooting_time=cr3bp_config['minimum_shooting_time'],
                sample_path=None,  # Will be set later
                start_bdry=cr3bp_config['start_bdry'],
                end_bdry=cr3bp_config['end_bdry'],
                thrust=cr3bp_config['thrust'],
                solver_mode=cr3bp_config['solver_mode'],
                min_mass_to_sample=cr3bp_config['min_mass_to_sample'],
                max_mass_to_sample=cr3bp_config['max_mass_to_sample'],
                snopt_time_limit=cr3bp_config['snopt_time_limit'],
                result_folder=cr3bp_config['result_folder']
            )
            
            setattr(self.local, f'simulator_{thread_id}', dummy_simulator)
            print(f"Thread {thread_id}: CR3BP simulator pre-initialized")
        
        return getattr(self.local, f'simulator_{thread_id}')


# Global thread-local storage
thread_storage = ThreadLocalStorage()


def convert_to_spherical(ux, uy, uz):
    """Convert Cartesian coordinates to spherical coordinates (from GTO Halo DM)."""
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


def get_latest_file(folder_path):
    """Get the latest folder from the checkpoint directory (from GTO Halo DM)."""
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Date format in the filenames
    date_format = "%Y-%m-%d_%H-%M-%S"
    
    latest_time = None
    latest_file = None
    
    for file in files:
        try:
            # Extract the date and time from the filename
            file_time = datetime.strptime(file, date_format)
            # Check if this file is the latest one
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
                latest_file = file
        except ValueError:
            # Skip files that don't match the date format
            continue
    
    return latest_file

def reconstruct_from_3_channels(channels_data):
    """
    Reconstruct 66-dimensional trajectory data from 3-channel format (from GTO Halo DM).
    
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


def process_single_sample_fast(args):
    """Process a single sample with pre-initialized thread resources."""
    sample_idx, sample_data, halo_energy, output_dir, thread_id = args
    
    try:
        # Add the correct paths for imports (matching your approach)
        import sys
        import os
        
        # Add the current directory and Data_Generation to the path
        current_dir = os.getcwd()
        sys.path.append(current_dir)
        sys.path.append(os.path.join(current_dir, 'Data_Generation'))
        sys.path.append(os.path.join(current_dir, 'Data_Generation', 'support_scripts'))
        
        # Import the simulator
        from Data_Generation.cr3bp_earth_mission_simulator_boundary_diffusion_warmstart import CR3BPEarthMissionWarmstartSimulatorBoundary
        
        # Force immediate output with flush
        print(f"Thread {thread_id}: Starting sample {sample_idx} with halo energy {halo_energy:.6f}", flush=True)
        
        # Create a fresh simulator for each sample to avoid output conflicts
        # Create directories for outputs
        temp_dir = os.path.join(output_dir, 'temp_samples')
        os.makedirs(temp_dir, exist_ok=True)
        cr3bp_results_dir = os.path.join(output_dir, 'cr3bp_results')
        os.makedirs(cr3bp_results_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f'sample_{sample_idx}.pkl')
        
        # CR3BP simulation parameters (hardcoded to match GTO Halo DM exactly)
        cr3bp_config = {
            'seed': sample_idx,
            'seed_step': 1,  # Process single sample
            'quiet_snopt': True,  # Suppress SNOPT output for cleaner telemetry
            'number_of_segments': 20,  # Match GTO Halo DM
            'maximum_shooting_time': 40.0,  # Match GTO Halo DM
            'minimum_shooting_time': 0.0,  # Match GTO Halo DM
            'start_bdry': 6.48423370092,  # Match GTO Halo DM
            'end_bdry': 8.0,  # Match GTO Halo DM
            'thrust': 1.0,  # Match GTO Halo DM
            'solver_mode': "optimal",  # Match GTO Halo DM
            'min_mass_to_sample': 408,  # Match GTO Halo DM
            'max_mass_to_sample': 470,  # Match GTO Halo DM
            'snopt_time_limit': 500.0,  # Match GTO Halo DM
            'result_folder': cr3bp_results_dir
        }
        
        # Save single sample in the format expected by CR3BP simulator
        # Format: [halo_energy, trajectory_params] where trajectory_params is 66-dimensional
        sample_for_cr3bp = np.array([halo_energy] + list(sample_data))
        with open(temp_file, 'wb') as f:
            pickle.dump(sample_for_cr3bp.reshape(1, -1), f)
        
        # Create fresh simulator for this sample
        simulator = CR3BPEarthMissionWarmstartSimulatorBoundary(
            seed=cr3bp_config['seed'],
            seed_step=cr3bp_config['seed_step'],
            quiet_snopt=cr3bp_config['quiet_snopt'],
            number_of_segments=cr3bp_config['number_of_segments'],
            maximum_shooting_time=cr3bp_config['maximum_shooting_time'],
            minimum_shooting_time=cr3bp_config['minimum_shooting_time'],
            sample_path=temp_file,
            start_bdry=cr3bp_config['start_bdry'],
            end_bdry=cr3bp_config['end_bdry'],
            thrust=cr3bp_config['thrust'],
            solver_mode=cr3bp_config['solver_mode'],
            min_mass_to_sample=cr3bp_config['min_mass_to_sample'],
            max_mass_to_sample=cr3bp_config['max_mass_to_sample'],
            snopt_time_limit=cr3bp_config['snopt_time_limit'],
            result_folder=cr3bp_config['result_folder']
        )
        
        # Set the halo energy before simulation (required by the simulator)
        simulator.halo_energy = halo_energy
        
        # Run simulation
        print(f"📊 TELEMETRY: Thread {thread_id} - Sample {sample_idx} - SNOPT STARTING", flush=True)
        result_data = simulator.simulate(earth_initial_guess=sample_data)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Analyze result
        convergence_info = "UNKNOWN"
        if result_data:
            if isinstance(result_data, dict):
                print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Result data keys: {list(result_data.keys())}", flush=True)
                if "feasibility" in result_data:
                    convergence_info = "FEASIBLE" if result_data["feasibility"] else "INFEASIBLE"
                    print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Feasibility: {result_data['feasibility']}", flush=True)
                elif 'error' in result_data:
                    convergence_info = f"ERROR: {result_data['error']}"
            else:
                convergence_info = "PROCESSED"
        
        print(f"📊 TELEMETRY: Thread {thread_id} - Sample {sample_idx} - SNOPT COMPLETED - Result: {convergence_info}", flush=True)
        
        return {
            'sample_idx': sample_idx,
            'halo_energy': halo_energy,
            'result_data': result_data,
            'initial_guess': sample_data
        }
        
    except Exception as e:
        print(f"Thread {thread_id}: Error processing sample {sample_idx}: {e}", flush=True)
        return {
            'sample_idx': sample_idx,
            'halo_energy': halo_energy,
            'result_data': {'feasibility': False, 'error': str(e)},
            'initial_guess': sample_data
        }


class GTOHaloBenchmarker:
    """Fast multithreaded GTO Halo specific benchmarking for diffusion models."""
    
    def __init__(self, config: GTOHaloBenchmarkConfig):
        """Initialize the benchmarker."""
        self.config = config
        
        # Set up multithreading
        if self.config.max_workers is None:
            self.config.max_workers = cpu_count()
        
        print(f"✓ Using {self.config.max_workers} CPU cores for parallel processing")
        
        # Initialize model and data
        self.model = None
        self.diffusion = None
        self.trainer = None
        
        # Load model and data
        self.load_model()
        self.load_reference_data()
        
        # Pre-warm threads if enabled
        if self.config.pre_warm_threads:
            self.pre_warm_threads()
    
    def pre_warm_threads(self):
        """Pre-initialize threads to reduce startup delays."""
        print("Pre-warming threads to reduce initialization delays...")
        
        # Create a dummy task to warm up each thread
        def warm_up_thread(thread_id):
            print(f"Warming up thread {thread_id}...")
            try:
                # Just import the library to warm up the thread
                import sys
                import os
                
                # Add the correct paths for imports (matching your approach)
                current_dir = os.getcwd()
                sys.path.append(current_dir)
                sys.path.append(os.path.join(current_dir, 'Data_Generation'))
                sys.path.append(os.path.join(current_dir, 'Data_Generation', 'support_scripts'))
                
                from Data_Generation.cr3bp_earth_mission_simulator_boundary_diffusion_warmstart import CR3BPEarthMissionWarmstartSimulatorBoundary
                return f"Thread {thread_id} warmed up"
            except Exception as e:
                return f"Thread {thread_id} warm-up failed: {e}"
        
        # Warm up all threads
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(warm_up_thread, i) for i in range(self.config.max_workers)]
            for future in as_completed(futures):
                result = future.result()
                print(f"✓ {result}")
        
        print("✓ All threads pre-warmed")
    
    def load_model(self):
        """Load the trained diffusion model from GTO Halo DM."""
        print(f"Loading model from {self.config.model_path}")
        
        # Model parameters (EXACTLY from GTO Halo DM sampling script)
        unet_dim = 128
        unet_dim_mults = (4, 4, 8)
        embed_class_layers_dims = (256, 512)
        timesteps = 300
        objective = "pred_noise"
        class_dim = 1
        channel = 3  # Fixed to 3 channels
        seq_length = 22  # Fixed to 22 sequence length
        cond_drop_prob = 0.1
        mask_val = -1.0
        
        # Build checkpoint path (EXACTLY as in sampling script)
        unet_dim_mults_in_str = "_".join(map(str, unet_dim_mults))
        embed_class_layers_dims_in_str = "_".join(map(str, embed_class_layers_dims))
        checkpoint_path = f"results/cr3bp_vanilla_diffusion_seed_0/unet_{unet_dim}_mults_{unet_dim_mults_in_str}_embed_class_{embed_class_layers_dims_in_str}_timesteps_{timesteps}_batch_size_2000_cond_drop_0.1_mask_val_{mask_val}_train_data_100000/"
        
        # Get latest folder (EXACTLY as in sampling script)
        folder_name = get_latest_file(checkpoint_path)
        checkpoint_path = checkpoint_path + folder_name
        
        # Create model (EXACTLY as in sampling script)
        self.model = Unet1D(
            seq_length=seq_length,
            dim=unet_dim,
            channels=channel,
            dim_mults=unet_dim_mults,
            embed_class_layers_dims=embed_class_layers_dims,
            class_dim=class_dim,
            cond_drop_prob=cond_drop_prob,
            mask_val=mask_val,
        )
        
        # Create diffusion (EXACTLY as in sampling script)
        self.diffusion = GaussianDiffusion1D(
            model=self.model,
            seq_length=seq_length,
            timesteps=timesteps,
            objective=objective
        ).to(self.config.device)
        
        # Create trainer and load checkpoint
        self.trainer = Trainer1D(
            diffusion_model=self.diffusion,
            dataset=[0, 0, 0],  # Dummy dataset
            results_folder=checkpoint_path,
        )
        
        # Load checkpoint (EXACTLY as in sampling script)
        milestone = "epoch-199"  # Use latest checkpoint
        self.trainer.load(milestone)
        
        print(f"✓ Successfully loaded checkpoint from {checkpoint_path}")
    
    def load_reference_data(self):
        """Load reference data for comparison."""
        if os.path.exists(self.config.data_path):
            with open(self.config.data_path, 'rb') as f:
                self.reference_data = pickle.load(f)
            print(f"✓ Reference data loaded: {self.reference_data.shape}")
        else:
            print(f"⚠️  Reference data not found: {self.config.data_path}")
            self.reference_data = None
    
    def generate_samples(self) -> Tuple[np.ndarray, List[float]]:
        """Generate samples using GTO Halo DM diffusion model."""
        print(f"Generating {self.config.num_samples} samples...")
        
        samples = []
        sampling_times = []
        
        num_batches = (self.config.num_samples + self.config.batch_size - 1) // self.config.batch_size
        
        for i in range(num_batches):
            batch_size = min(self.config.batch_size, self.config.num_samples - i * self.config.batch_size)
            
            # For GTO Halo data, use uniform sampling of class labels in [0, 1] (EXACTLY as in sampling script)
            if self.config.fixed_alpha is not None:
                class_labels = torch.full(size=(batch_size, 1), fill_value=self.config.fixed_alpha, dtype=torch.float32, device=self.config.device)
            else:
                torch.manual_seed(1000000)  # Same seed as in sampling script
                class_labels = torch.rand(batch_size, 1, device=self.config.device)
            
            # Generate samples
            start_time = time.time()
            
            sample_results = self.diffusion.sample(
                classes=class_labels,
                cond_scale=self.config.guidance_weight,
            )
            
            end_time = time.time()
            sampling_time = end_time - start_time
            
            # Convert to numpy
            sample_np = sample_results.detach().cpu().numpy()
            samples.append(sample_np)
            sampling_times.append(sampling_time)
            
            print(f"Batch {i+1}/{num_batches}: Generated {batch_size} samples in {sampling_time:.2f}s")
        
        # Concatenate all samples
        all_samples = np.concatenate(samples, axis=0)
        all_samples = all_samples[:self.config.num_samples]  # Ensure exact number
        
        # Apply GTO Halo DM data transformation (EXACTLY as in sampling script)
        transformed_samples = self.transform_samples_gto_halo_dm(all_samples)
        
        return transformed_samples, sampling_times
    
    def transform_samples_gto_halo_dm(self, samples: np.ndarray) -> np.ndarray:
        """Transform samples using GTO Halo DM data processing logic (EXACTLY as in sampling script)."""
        print("Applying GTO Halo DM data transformation...")
        
        # Data preparation parameters (EXACTLY from GTO Halo DM sampling script)
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
        thrust = 1.0
        
        # Reconstruct 66-dimensional format from 3-channel output (EXACTLY as in sampling script)
        reconstructed_solutions = []
        for i in range(samples.shape[0]):
            # samples[i] has shape (3, 22) - reconstruct to (66,)
            trajectory_params = reconstruct_from_3_channels(samples[i])
            reconstructed_solutions.append(trajectory_params)
        
        full_solution = np.array(reconstructed_solutions)  # Shape: (sample_num, 66)
        
        # Unnormalize times (EXACTLY as in sampling script)
        full_solution[:, 0] = full_solution[:, 0] * (max_shooting_time - min_shooting_time) + min_shooting_time
        full_solution[:, 1] = full_solution[:, 1] * (max_coast_time - min_coast_time) + min_coast_time
        full_solution[:, 2] = full_solution[:, 2] * (max_coast_time - min_coast_time) + min_coast_time
        
        # Convert cartesian control back to correct range, then to spherical (EXACTLY as in sampling script)
        full_solution[:, 3:-3] = full_solution[:, 3:-3] * 2 * thrust - thrust
        ux = full_solution[:,3:-3:3]
        uy = full_solution[:,4:-3:3]
        uz = full_solution[:,5:-3:3]
        alpha, beta, r = convert_to_spherical(ux, uy, uz)
        full_solution[:,3:-3:3] = alpha
        full_solution[:,4:-3:3] = beta
        full_solution[:,5:-3:3] = r
        
        # Unnormalize fuel mass and manifold parameters (EXACTLY as in sampling script)
        full_solution[:, -3] = full_solution[:, -3] * (max_final_fuel_mass - min_final_fuel_mass) + min_final_fuel_mass
        full_solution[:, -1] = full_solution[:, -1] * (max_manifold_length - min_manifold_length) + min_manifold_length
        
        # Generate halo energies (EXACTLY as in sampling script)
        if self.config.fixed_alpha is not None:
            # Use fixed alpha value
            alpha_data_normalized = np.full((samples.shape[0], 1), self.config.fixed_alpha)
        else:
            # Use same seed as in sampling script
            torch.manual_seed(1000000)
            alpha_data_normalized = torch.rand(samples.shape[0], 1).numpy()
        
        halo_energies = alpha_data_normalized * (max_halo_energy - min_halo_energy) + min_halo_energy
        
        # Combine halo energies with trajectory parameters (EXACTLY as in sampling script)
        final_samples = np.hstack((halo_energies, full_solution))
        
        print(f"✓ Transformed samples shape: {final_samples.shape}")
        return final_samples
    
    def compute_physical_validation_metrics(self, samples: np.ndarray) -> Dict[str, Any]:
        """Compute physical validation metrics using CR3BP simulator with fast multithreading."""
        print(f"DEBUG: enable_physical_validation = {self.config.enable_physical_validation}")
        print(f"DEBUG: GTO_HALO_DM_AVAILABLE = {GTO_HALO_DM_AVAILABLE}")
        
        if not self.config.enable_physical_validation or not GTO_HALO_DM_AVAILABLE:
            print("⚠️  CRITICAL: Physical validation disabled - GTO_Halo_DM modules not available")
            print("⚠️  This means NO feasibility checking, NO optimality analysis, NO trajectory validation")
            print("⚠️  The GTO Halo benchmarking will only provide component statistics, not physical validation")
            return {
                'physical_validation_disabled': True, 
                'reason': 'GTO_Halo_DM modules not available',
                'missing_metrics': [
                    'feasible_solution_ratio',
                    'local_optimal_solution_ratio', 
                    'average_final_mass_feasible',
                    'average_final_mass_optimal',
                    'snopt_inform_distribution',
                    'solving_time_analysis'
                ]
            }
        
        print("Computing physical validation metrics using CR3BP simulator with fast multithreading...")
        
        # Prepare samples for parallel processing
        num_samples = len(samples)
        sample_args = []
        
        # Create arguments for each sample with thread ID
        for i in range(num_samples):
            halo_energy = samples[i, 0]  # First column is the physical halo energy
            initial_guess = samples[i, 1:]  # Rest is the initial guess data
            thread_id = i % self.config.max_workers  # Distribute across threads
            sample_args.append((i, initial_guess, halo_energy, self.config.output_dir, thread_id))
        
        print(f"Processing {num_samples} samples using {self.config.max_workers} pre-warmed threads...")
        
        # Process samples in parallel with pre-warmed threads
        all_results = []
        sample_status = {}  # Track status of each sample
        active_samples = set()  # Track currently running samples
        
        print(f"Starting parallel processing of {num_samples} samples with {self.config.max_workers} threads...", flush=True)
        print(f"📊 TELEMETRY: Initializing {num_samples} samples...", flush=True)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all samples
            future_to_sample = {executor.submit(process_single_sample_fast, args): args[0] for args in sample_args}
            
            # Initialize status for all samples
            for sample_idx in range(num_samples):
                sample_status[sample_idx] = "QUEUED"
            
            # Track active samples
            active_samples = set(range(num_samples))
            
            # Print initial status
            print(f"📊 TELEMETRY: All {num_samples} samples submitted to thread pool", flush=True)
            print(f"📊 TELEMETRY: Active samples: {sorted(active_samples)}", flush=True)
            
            # Collect results as they complete
            start_time = time.time()
            last_status_update = start_time
            
            for future in as_completed(future_to_sample):
                sample_idx = future_to_sample[future]
                result = future.result()
                all_results.append(result)
                
                # Periodic status update every 10 seconds
                current_time = time.time()
                if current_time - last_status_update > 10:
                    print(f"📊 TELEMETRY: STATUS UPDATE at {current_time - start_time:.1f}s", flush=True)
                    print(f"📊 TELEMETRY: Active samples: {sorted(active_samples)}", flush=True)
                    print(f"📊 TELEMETRY: Completed: {len(all_results)}/{num_samples}", flush=True)
                    last_status_update = current_time
                
                # Update status
                sample_status[sample_idx] = "COMPLETED"
                active_samples.discard(sample_idx)
                
                # Extract convergence info
                convergence_status = "UNKNOWN"
                if 'result_data' in result and result['result_data']:
                    if isinstance(result['result_data'], dict):
                        if 'feasibility' in result['result_data']:
                            convergence_status = "FEASIBLE" if result['result_data']['feasibility'] else "INFEASIBLE"
                        elif 'error' in result['result_data']:
                            convergence_status = f"ERROR: {result['result_data']['error']}"
                    else:
                        convergence_status = "PROCESSED"
                
                # Print detailed progress
                completed = len(all_results)
                print(f"📊 TELEMETRY: Sample {sample_idx} COMPLETED - Status: {convergence_status}", flush=True)
                print(f"📊 TELEMETRY: Progress: {completed}/{num_samples} samples ({completed/num_samples*100:.1f}%)", flush=True)
                print(f"📊 TELEMETRY: Remaining active samples: {sorted(active_samples)}", flush=True)
                
                # Print summary of all samples
                print(f"📊 TELEMETRY: Sample Status Summary:", flush=True)
                for idx in range(num_samples):
                    status = sample_status.get(idx, "UNKNOWN")
                    if idx in active_samples:
                        status = "RUNNING"
                    print(f"   Sample {idx}: {status}", flush=True)
                print("-" * 50, flush=True)
        
        # Sort results by sample index to maintain order
        all_results.sort(key=lambda x: x['sample_idx'])
        
        # Extract result data and initial guesses
        result_data_list = [result['result_data'] for result in all_results]
        initial_guesses_list = [result['initial_guess'] for result in all_results]
        
        # Compute statistics
        physical_metrics = self.compute_cr3bp_statistics(result_data_list, initial_guesses_list)
        
        return physical_metrics
    
    def compute_gto_halo_metrics(self, samples: np.ndarray) -> Dict[str, Any]:
        """Compute GTO Halo specific metrics."""
        print("Computing GTO Halo specific metrics...")
        
        # Samples should be in (N, 67) format: [halo_energy, trajectory_params]
        print(f"Input samples shape: {samples.shape}")
        
        # Extract components from 67-vector
        halo_energies = samples[:, 0]  # First value is halo energy
        trajectory_params = samples[:, 1:]  # Rest is trajectory parameters (66 values)
        
        # Ensure we have valid data
        if samples.size == 0:
            print("Warning: No samples generated")
            return {}
        
        # Helper to safely compute stats
        def safe_stat(arr, func, default=None):
            return func(arr) if arr.size > 0 else default
        
        # Compute metrics
        metrics = {}
        
        # Halo energy statistics
        metrics['halo_energy_mean'] = float(safe_stat(halo_energies, np.mean, None))
        metrics['halo_energy_std'] = float(safe_stat(halo_energies, np.std, None))
        metrics['halo_energy_min'] = float(safe_stat(halo_energies, np.min, None))
        metrics['halo_energy_max'] = float(safe_stat(halo_energies, np.max, None))
        
        # Trajectory parameters statistics
        metrics['trajectory_params_mean'] = float(safe_stat(trajectory_params, np.mean, None))
        metrics['trajectory_params_std'] = float(safe_stat(trajectory_params, np.std, None))
        metrics['trajectory_params_min'] = float(safe_stat(trajectory_params, np.min, None))
        metrics['trajectory_params_max'] = float(safe_stat(trajectory_params, np.max, None))
        
        # Data quality checks
        metrics['has_nan'] = bool(np.any(np.isnan(samples)))
        metrics['has_inf'] = bool(np.any(np.isinf(samples)))
        
        return metrics
    
    def compute_cr3bp_statistics(self, result_data_list: List[Dict], initial_guesses_list: List[np.ndarray]) -> Dict[str, Any]:
        """Compute CR3BP simulation statistics."""
        if not result_data_list:
            return {'error': 'No CR3BP results available'}
        
        # Extract metrics
        feasible_count = sum(1 for result in result_data_list if result.get("feasibility", False))
        total_count = len(result_data_list)
        feasible_ratio = feasible_count / total_count if total_count > 0 else 0
        
        # Local optimal solutions (SNOPT inform = 1)
        local_optimal_count = sum(1 for result in result_data_list 
                                if result.get("feasibility", False) and result.get("snopt_inform", 0) == 1)
        local_optimal_ratio = local_optimal_count / total_count if total_count > 0 else 0
        
        # Final mass analysis (extract from results.control[-3])
        final_masses_feasible = []
        final_masses_optimal = []
        for result in result_data_list:
            if result.get("feasibility", False):
                if "results.control" in result and result["results.control"] is not None:
                    final_masses_feasible.append(result["results.control"][-3])
                if result.get("snopt_inform", 0) == 1:
                    if "results.control" in result and result["results.control"] is not None:
                        final_masses_optimal.append(result["results.control"][-3])
        
        avg_final_mass_feasible = np.mean(final_masses_feasible) if final_masses_feasible else 0
        avg_final_mass_optimal = np.mean(final_masses_optimal) if final_masses_optimal else 0
        
        # Solving time analysis
        solving_times = [result.get("solving_time", 0) for result in result_data_list 
                        if result.get("feasibility", False)]
        avg_solving_time = np.mean(solving_times) if solving_times else 0
        
        # SNOPT inform distribution
        snopt_informs = [result.get("snopt_inform", 0) for result in result_data_list]
        snopt_inform_distribution = {}
        for inform in snopt_informs:
            snopt_inform_distribution[inform] = snopt_inform_distribution.get(inform, 0) + 1
        
        return {
            'feasible_ratio': feasible_ratio,
            'avg_final_mass_feasible': avg_final_mass_feasible,
            'local_optimal_ratio': local_optimal_ratio,
            'avg_final_mass_optimal': avg_final_mass_optimal,
            'avg_solving_time': avg_solving_time,
            'snopt_inform_distribution': snopt_inform_distribution,
            'total_tested': total_count,
            'feasible_count': feasible_count,
            'local_optimal_count': local_optimal_count
        }
    
    def compute_sampling_efficiency_metrics(self, sampling_times: List[float]) -> Dict[str, float]:
        """Compute sampling efficiency metrics."""
        if not sampling_times:
            return {}
        
        total_sampling_time = sum(sampling_times)
        avg_sampling_time = np.mean(sampling_times)
        sampling_time_std = np.std(sampling_times)
        samples_per_second = self.config.num_samples / total_sampling_time if total_sampling_time > 0 else 0
        
        return {
            'total_sampling_time': total_sampling_time,
            'average_sampling_time_per_sample': avg_sampling_time,
            'sampling_time_std': sampling_time_std,
            'samples_per_second': samples_per_second,
            'min_sampling_time': min(sampling_times),
            'max_sampling_time': max(sampling_times)
        }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark."""
        print("=" * 60)
        print("RUNNING GTO HALO BENCHMARK (FAST MULTITHREADED)")
        print("=" * 60)
        
        # Generate samples
        samples, sampling_times = self.generate_samples()
        
        # Compute metrics
        gto_halo_metrics = self.compute_gto_halo_metrics(samples)
        physical_validation_metrics = self.compute_physical_validation_metrics(samples)
        sampling_efficiency_metrics = self.compute_sampling_efficiency_metrics(sampling_times)
        
        # Combine results
        results = {
            'gto_halo_metrics': gto_halo_metrics,
            'physical_validation_metrics': physical_validation_metrics,
            'sampling_efficiency_metrics': sampling_efficiency_metrics,
            'num_samples': len(samples),
            'multithreading_config': {
                'max_workers': self.config.max_workers,
                'chunk_size': self.config.chunk_size,
                'pre_warm_threads': self.config.pre_warm_threads
            }
        }
        
        # Save results
        self.save_results(results, samples)
        
        return results
    
    def save_results(self, results: Dict[str, Any], samples: np.ndarray):
        """Save benchmark results."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save samples
        if self.config.save_samples:
            samples_path = os.path.join(self.config.output_dir, 'samples.npy')
            np.save(samples_path, samples)
            print(f"✓ Samples saved to {samples_path}")
        
        # Save results as JSON
        results_path = os.path.join(self.config.output_dir, 'gto_halo_benchmark_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results saved to {results_path}")
        
        # Save summary
        self.save_summary(results)
        
        # Generate plots
        if self.config.save_plots:
            self.generate_plots(results, samples)
    
    def save_summary(self, results: Dict[str, Any]):
        """Save human-readable summary."""
        summary_path = os.path.join(self.config.output_dir, 'summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("GTO HALO BENCHMARK RESULTS (FAST MULTITHREADED)\n")
            f.write("=" * 60 + "\n\n")
            
            # GTO Halo metrics
            gto_metrics = results.get('gto_halo_metrics', {})
            if gto_metrics:
                f.write("GTO HALO METRICS:\n")
                for key, value in gto_metrics.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Physical validation metrics
            physical_metrics = results.get('physical_validation_metrics', {})
            if physical_metrics:
                f.write("PHYSICAL VALIDATION METRICS:\n")
                for key, value in physical_metrics.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Sampling efficiency metrics
            sampling_metrics = results.get('sampling_efficiency_metrics', {})
            if sampling_metrics:
                f.write("SAMPLING EFFICIENCY:\n")
                for key, value in sampling_metrics.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Multithreading info
            multithreading_config = results.get('multithreading_config', {})
            if multithreading_config:
                f.write("MULTITHREADING CONFIG:\n")
                for key, value in multithreading_config.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"✓ Summary saved to {summary_path}")
    
    def generate_plots(self, results: Dict[str, Any], samples: np.ndarray):
        """Generate visualization plots."""
        # This can be implemented similar to the original benchmarking script
        # For now, we'll skip plotting to focus on the multithreading functionality
        print("✓ Plot generation skipped (focusing on multithreading)")


def main():
    """Main function to run the fast multithreaded benchmark."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Fast Multithreaded GTO Halo Benchmarking')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--data_path', type=str, required=True, help='Path to reference data')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for sampling')
    parser.add_argument('--max_workers', type=int, default=None, help='Number of worker threads (default: CPU count)')
    parser.add_argument('--chunk_size', type=int, default=1, help='Number of samples per thread')
    parser.add_argument('--pre_warm_threads', action='store_true', default=True, help='Pre-warm threads to reduce initialization delays')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name (auto-generated as "benchmark_TIMESTAMP_samples_N" if not specified)')
    parser.add_argument('--enable_physical_validation', action='store_true', default=True, help='Enable physical validation')
    parser.add_argument('--fixed_alpha', type=float, default=None, help='Fixed alpha value (None for random sampling)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create organized output directory structure
    if args.output_dir is None:
        # Create automatically named directory based on timestamp and samples
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"benchmark_{timestamp}_samples_{args.num_samples}"
        args.output_dir = os.path.join("Benchmark Results", folder_name)
    else:
        # If user specifies output_dir, make sure it's inside "Benchmark Results"
        if not args.output_dir.startswith("Benchmark Results"):
            args.output_dir = os.path.join("Benchmark Results", args.output_dir)
    
    # Ensure the "Benchmark Results" directory exists
    os.makedirs("Benchmark Results", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"✓ Results will be saved to: {args.output_dir}")
    
    # Create config
    config = GTOHaloBenchmarkConfig(
        model_path=args.model_path,
        checkpoint_path=args.model_path,  # Use model_path as checkpoint_path
        data_path=args.data_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        pre_warm_threads=args.pre_warm_threads,
        output_dir=args.output_dir,
        enable_physical_validation=args.enable_physical_validation,
        fixed_alpha=args.fixed_alpha
    )
    
    # Run benchmark
    benchmarker = GTOHaloBenchmarker(config)
    results = benchmarker.run_benchmark()
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total sampling time: {results['sampling_efficiency_metrics'].get('total_sampling_time', 0):.2f}s")
    print(f"Samples per second: {results['sampling_efficiency_metrics'].get('samples_per_second', 0):.3f}")
    print(f"Halo energy mean: {results['gto_halo_metrics'].get('halo_energy_mean', 0):.6f}")
    print(f"Feasible ratio: {results['physical_validation_metrics'].get('feasible_ratio', 0):.3f}")
    print(f"Local optimal ratio: {results['physical_validation_metrics'].get('local_optimal_ratio', 0):.3f}")
    print(f"Results saved to: {args.output_dir}")
    print("Check the summary.txt files in each subdirectory for detailed results.")


if __name__ == "__main__":
    main() 