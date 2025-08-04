#!/usr/bin/env python3
"""
Fast Multithreaded GTO Halo Benchmarking Module for Original 1-Channel Model

This version benchmarks the original 1-channel approach that uses 67-dimensional input vectors directly,
compared to the newer 3-channel approach. Saves data in identical format for direct comparison.

Key differences from 3-channel version:
1. Uses 1 channel, 67 sequence length model
2. No 3-channel reconstruction needed
3. Direct 67D vector processing
4. Identical dataset structure for comparison
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
    # Add the correct paths for imports
    current_dir = os.getcwd()
    sys.path.append(current_dir)
    sys.path.append(os.path.join(current_dir, 'Data_Generation'))
    sys.path.append(os.path.join(current_dir, 'Data_Generation', 'support_scripts'))
    
    from Diffusion_Model_Scripts.GPU.classifier_free_guidance_cond_1d_improved_constrained_diffusion import Unet1D, GaussianDiffusion1D, Trainer1D
    from Data_Generation.cr3bp_earth_mission_simulator_boundary_diffusion_warmstart import CR3BPEarthMissionWarmstartSimulatorBoundary
    from support_scripts.support import get_GTO_in_CR3BP_units
    GTO_HALO_DM_AVAILABLE = True
    print("✓ GTO_Halo_DM modules loaded successfully for ORIGINAL 1-CHANNEL model")
    print("✓ Physical validation enabled - CR3BP simulator available")
except ImportError as e:
    print(f"Warning: GTO_Halo_DM modules not available: {e}")
    CR3BPEarthMissionWarmstartSimulatorBoundary = None
    GTO_HALO_DM_AVAILABLE = False


@dataclass
class GTOHaloBenchmarkConfig:
    """Configuration for fast multithreaded GTO Halo benchmarking (Original 1-Channel Model)."""
    # Model and data config
    model_path: str
    checkpoint_path: str
    data_path: str
    checkpoint_file: str = "model-epoch-175.pt"  # Original model checkpoint
    
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
    output_dir: str = "benchmark_run_original"  # Will be placed inside "Benchmark Results"
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
                'solver_mode': "optimal",  # Match GTO Halo DM
                'min_mass_to_sample': 408,  # Match GTO Halo DM
                'max_mass_to_sample': 470,  # Match GTO Halo DM
                'snopt_time_limit': 500.0,  # Match GTO Halo DM
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


def process_single_sample_fast(args):
    """Process a single sample with pre-initialized thread resources and collect comprehensive data."""
    sample_idx, sample_data, halo_energy, output_dir, thread_id = args
    
    try:
        # Add the correct paths for imports
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
        
        # Create comprehensive dataset entry (IDENTICAL structure to 3-channel version)
        dataset_entry = {
            # Sample identification
            'sample_idx': sample_idx,
            'thread_id': thread_id,
            
            # Input data
            'generated_sample': {
                'halo_energy': halo_energy,
                'trajectory_params': sample_data.tolist(),  # Convert to list for JSON serialization
                'full_sample_vector': sample_for_cr3bp.tolist(),
                'model_type': 'ORIGINAL_1_CHANNEL'  # Add model identifier
            },
            
            # SNOPT simulation parameters
            'simulation_config': cr3bp_config,
            
            # SNOPT results and statistics
            'snopt_results': {},
            
            # Converged trajectory data (if feasible)
            'converged_trajectory': {},
            
            # Physical trajectory states (if feasible)
            'physical_trajectories': {},
            
            # Analysis metadata
            'processing_metadata': {
                'timestamp': time.time(),
                'convergence_status': "UNKNOWN",
                'model_type': 'ORIGINAL_1_CHANNEL'
            }
        }
        
        # Extract and organize SNOPT results (IDENTICAL to 3-channel version)
        if result_data and isinstance(result_data, dict):
            print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Result data keys: {list(result_data.keys())}", flush=True)
            
            # Core SNOPT statistics
            dataset_entry['snopt_results'] = {
                'feasibility': result_data.get('feasibility', False),
                'snopt_inform': result_data.get('snopt_inform', None),
                'solving_time': result_data.get('solving_time', None),
                'thrust': result_data.get('thrust', None),
                'cost_alpha': result_data.get('cost_alpha', None),
                'has_snopt_control_evaluations': result_data.get('snopt_control_evaluations', None) is not None
            }
            
            # If feasible, extract complete trajectory data
            if result_data.get('feasibility', False):
                if 'results.control' in result_data and result_data['results.control'] is not None:
                    control_vector = result_data['results.control']
                    
                    # Store the complete control vector
                    dataset_entry['converged_trajectory']['control_vector'] = control_vector.tolist()
                    
                    # Extract key trajectory parameters
                    dataset_entry['converged_trajectory']['trajectory_parameters'] = {
                        'shooting_time': float(control_vector[0]),
                        'initial_coast_time': float(control_vector[1]),
                        'final_coast_time': float(control_vector[2]),
                        'final_fuel_mass': float(control_vector[-3]),
                        'halo_period': float(control_vector[-2]),
                        'manifold_length': float(control_vector[-1])
                    }
                    
                    # Extract control segments (thrust directions for each segment)
                    control_segments = []
                    for i in range(20):  # 20 segments
                        segment_start = 3 + i * 3
                        control_segments.append({
                            'segment_id': i,
                            'alpha': float(control_vector[segment_start]),      # ux component
                            'beta': float(control_vector[segment_start + 1]),   # uy component  
                            'thrust_magnitude': float(control_vector[segment_start + 2])  # uz component
                        })
                    
                    dataset_entry['converged_trajectory']['control_segments'] = control_segments
                
                # Extract physical trajectory states if available
                if 'results.states' in result_data and result_data['results.states'] is not None:
                    converged_states = result_data['results.states']
                    dataset_entry['physical_trajectories']['converged_states'] = converged_states.tolist()
                    dataset_entry['physical_trajectories']['converged_states_shape'] = list(converged_states.shape)
                    print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Captured converged trajectory states: {converged_states.shape}", flush=True)
                
                if 'results_DM.states' in result_data and result_data['results_DM.states'] is not None:
                    predicted_states = result_data['results_DM.states']
                    dataset_entry['physical_trajectories']['predicted_states'] = predicted_states.tolist()
                    dataset_entry['physical_trajectories']['predicted_states_shape'] = list(predicted_states.shape)
                    print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Captured predicted trajectory states: {predicted_states.shape}", flush=True)
                
                # Extract manifold arcs if available
                if 'manifold_arc' in result_data and result_data['manifold_arc'] is not None:
                    converged_manifold = result_data['manifold_arc']
                    dataset_entry['physical_trajectories']['converged_manifold'] = converged_manifold.tolist()
                    dataset_entry['physical_trajectories']['converged_manifold_shape'] = list(converged_manifold.shape)
                
                if 'manifold_arc_DM' in result_data and result_data['manifold_arc_DM'] is not None:
                    predicted_manifold = result_data['manifold_arc_DM']
                    dataset_entry['physical_trajectories']['predicted_manifold'] = predicted_manifold.tolist()
                    dataset_entry['physical_trajectories']['predicted_manifold_shape'] = list(predicted_manifold.shape)
                
                # Store SNOPT control evaluations if available
                if result_data.get('snopt_control_evaluations', None) is not None:
                    snopt_evals = result_data['snopt_control_evaluations']
                    dataset_entry['snopt_results']['snopt_control_evaluations'] = {
                        'shape': list(snopt_evals.shape),
                        'data': snopt_evals.tolist(),  # Full evaluation history
                        'num_evaluations': snopt_evals.shape[0] if len(snopt_evals.shape) > 0 else 0
                    }
                
                dataset_entry['processing_metadata']['convergence_status'] = "FEASIBLE"
                
                # Check if solution is locally optimal
                if result_data.get('snopt_inform', 0) == 1:
                    dataset_entry['processing_metadata']['convergence_status'] = "LOCALLY_OPTIMAL"
            else:
                dataset_entry['processing_metadata']['convergence_status'] = "INFEASIBLE"
                
                # Even for infeasible solutions, store what we can
                if 'error' in result_data:
                    dataset_entry['snopt_results']['error_message'] = str(result_data['error'])
                
                # Try to capture predicted trajectory states even if SNOPT failed
                if 'results_DM.states' in result_data and result_data['results_DM.states'] is not None:
                    predicted_states = result_data['results_DM.states']
                    dataset_entry['physical_trajectories']['predicted_states'] = predicted_states.tolist()
                    dataset_entry['physical_trajectories']['predicted_states_shape'] = list(predicted_states.shape)
                    print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Captured predicted trajectory states (infeasible case): {predicted_states.shape}", flush=True)
                
                if 'manifold_arc_DM' in result_data and result_data['manifold_arc_DM'] is not None:
                    predicted_manifold = result_data['manifold_arc_DM']
                    dataset_entry['physical_trajectories']['predicted_manifold'] = predicted_manifold.tolist()
                    dataset_entry['physical_trajectories']['predicted_manifold_shape'] = list(predicted_manifold.shape)
        else:
            dataset_entry['processing_metadata']['convergence_status'] = "ERROR"
            if result_data:
                dataset_entry['snopt_results']['raw_result'] = str(result_data)
        
        print(f"📊 TELEMETRY: Thread {thread_id} - Sample {sample_idx} - SNOPT COMPLETED - Result: {dataset_entry['processing_metadata']['convergence_status']}", flush=True)
        
        return dataset_entry
        
    except Exception as e:
        print(f"Thread {thread_id}: Error processing sample {sample_idx}: {e}", flush=True)
        
        # Return error entry for dataset
        return {
            'sample_idx': sample_idx,
            'thread_id': thread_id,
            'generated_sample': {
                'halo_energy': halo_energy,
                'trajectory_params': sample_data.tolist() if hasattr(sample_data, 'tolist') else list(sample_data),
                'full_sample_vector': None,
                'model_type': 'ORIGINAL_1_CHANNEL'
            },
            'simulation_config': {},
            'snopt_results': {
                'feasibility': False,
                'error_message': str(e)
            },
            'converged_trajectory': {},
            'processing_metadata': {
                'timestamp': time.time(),
                'convergence_status': "PROCESSING_ERROR",
                'model_type': 'ORIGINAL_1_CHANNEL'
            }
        }


class GTOHaloBenchmarkerOriginal:
    """Fast multithreaded GTO Halo benchmarking for original 1-channel diffusion model."""
    
    def __init__(self, config: GTOHaloBenchmarkConfig):
        """Initialize the benchmarker."""
        self.config = config
        
        # Set up multithreading
        print(f"✓ Using {self.config.max_workers} worker threads for parallel processing")
        print(f"  (Note: System has {cpu_count()} CPU cores, but you can use more threads if desired)")
        
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
                
                # Add the correct paths for imports
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
        """Load the trained original 1-channel diffusion model."""
        print(f"Loading ORIGINAL 1-CHANNEL model from {self.config.model_path}")
        
        # Model parameters for ORIGINAL 1-channel model (match the checkpoint)
        unet_dim = 128
        unet_dim_mults = (4, 4, 8)
        embed_class_layers_dims = (256, 512)
        timesteps = 500  # ORIGINAL: 500 timesteps (from checkpoint)
        objective = "pred_noise"
        class_dim = 1
        channel = 1  # ORIGINAL: 1 channel instead of 3
        seq_length = 67  # ORIGINAL: 67 sequence length instead of 22
        cond_drop_prob = 0.1
        mask_val = -1.0
        
        # Create model for ORIGINAL architecture
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
        
        # Create diffusion for ORIGINAL architecture
        self.diffusion = GaussianDiffusion1D(
            model=self.model,
            seq_length=seq_length,
            timesteps=timesteps,
            objective=objective
        ).to(self.config.device)
        
        # Load checkpoint directly from the specified file
        checkpoint_path = os.path.join(self.config.model_path, self.config.checkpoint_file)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # Load model state - try the standard approaches
        if 'ema' in checkpoint:
            self.diffusion.load_state_dict(checkpoint['ema'], strict=False)
            print("✓ Loaded EMA weights")
        elif 'model' in checkpoint:
            self.diffusion.load_state_dict(checkpoint['model'], strict=False)
            print("✓ Loaded model weights")
        else:
            self.diffusion.load_state_dict(checkpoint, strict=False)
            print("✓ Loaded state dict directly")
        
        print(f"✓ Successfully loaded ORIGINAL 1-CHANNEL model from {checkpoint_path}")
    
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
        """Generate samples using ORIGINAL 1-channel diffusion model."""
        print(f"Generating {self.config.num_samples} samples with ORIGINAL 1-CHANNEL model...")
        
        samples = []
        sampling_times = []
        
        num_batches = (self.config.num_samples + self.config.batch_size - 1) // self.config.batch_size
        
        for i in range(num_batches):
            batch_size = min(self.config.batch_size, self.config.num_samples - i * self.config.batch_size)
            
            # For GTO Halo data, use uniform sampling of class labels in [0, 1]
            if self.config.fixed_alpha is not None:
                class_labels = torch.full(size=(batch_size, 1), fill_value=self.config.fixed_alpha, dtype=torch.float32, device=self.config.device)
            else:
                torch.manual_seed(1000000)  # Same seed as other version for comparison
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
        
        # Apply data transformation for ORIGINAL format
        transformed_samples = self.transform_samples_original(all_samples)
        
        return transformed_samples, sampling_times
    
    def transform_samples_original(self, samples: np.ndarray) -> np.ndarray:
        """Transform samples from ORIGINAL 1-channel model (no 3-channel reconstruction needed)."""
        print("Applying ORIGINAL 1-channel data transformation...")
        
        # Data preparation parameters (same as 3-channel version)
        min_shooting_time = 0
        max_shooting_time = 40
        min_coast_time = 0
        max_coast_time = 15
        min_halo_energy = 0.008
        max_halo_energy = 0.095
        min_final_fuel_mass = 408
        max_final_fuel_mass = 470
        min_manifold_length = 5
        max_manifold_length = 11
        thrust = 1.0
        
        # ORIGINAL model outputs shape (batch_size, 1, 67) directly
        # Extract the 67-dimensional vectors
        if len(samples.shape) == 3 and samples.shape[1] == 1:
            # Remove the channel dimension: (batch_size, 1, 67) -> (batch_size, 67)
            full_solution = samples[:, 0, :]
        else:
            # Already in correct format
            full_solution = samples
        
        print(f"Original model output shape after reshaping: {full_solution.shape}")
        
        # The ORIGINAL model directly outputs the 67D vector in the same format as training data:
        # [classifier_norm, shooting_time_norm, initial_coast_norm, final_coast_norm, 
        #  control_cart_norm (60 values), final_fuel_mass_norm, halo_period_norm, manifold_length_norm]
        
        # Extract and unnormalize components
        classifier_norm = full_solution[:, 0]  # Halo energy (already normalized)
        shooting_time_norm = full_solution[:, 1]
        initial_coast_norm = full_solution[:, 2]
        final_coast_norm = full_solution[:, 3]
        control_cart_norm = full_solution[:, 4:64]  # 60 values (20 segments × 3 components)
        final_fuel_mass_norm = full_solution[:, 64]
        halo_period_norm = full_solution[:, 65]
        manifold_length_norm = full_solution[:, 66]
        
        # Unnormalize times
        shooting_time = shooting_time_norm * (max_shooting_time - min_shooting_time) + min_shooting_time
        initial_coast = initial_coast_norm * (max_coast_time - min_coast_time) + min_coast_time
        final_coast = final_coast_norm * (max_coast_time - min_coast_time) + min_coast_time
        
        # Unnormalize control (convert from normalized cartesian back to spherical)
        control_cart = control_cart_norm * 2 * thrust - thrust  # Unnormalize cartesian
        control_cart_reshaped = control_cart.reshape(-1, 20, 3)  # (batch, 20_segments, 3_components)
        
        # Convert cartesian back to spherical coordinates
        trajectory_params = []
        for sample_idx in range(full_solution.shape[0]):
            sample_params = [shooting_time[sample_idx], initial_coast[sample_idx], final_coast[sample_idx]]
            
            # Convert each segment from cartesian to spherical
            for seg in range(20):
                ux = control_cart_reshaped[sample_idx, seg, 0]
                uy = control_cart_reshaped[sample_idx, seg, 1]
                uz = control_cart_reshaped[sample_idx, seg, 2]
                
                # Convert to spherical (handle scalar values)
                u = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
                if u != 0:
                    theta = np.arcsin(uz / u)
                else:
                    theta = 0
                alpha = np.arctan2(uy, ux)
                alpha = alpha if alpha >= 0 else 2 * np.pi + alpha
                theta = theta if theta >= 0 else 2 * np.pi + theta
                r = min(u, 1.0)  # Clamp to 1
                
                sample_params.extend([alpha, theta, r])
            
            # Unnormalize final parameters
            final_fuel_mass = final_fuel_mass_norm[sample_idx] * (max_final_fuel_mass - min_final_fuel_mass) + min_final_fuel_mass
            manifold_length = manifold_length_norm[sample_idx] * (max_manifold_length - min_manifold_length) + min_manifold_length
            
            # Note: halo_period needs special handling - for now use normalized value
            halo_period = halo_period_norm[sample_idx]
            
            sample_params.extend([final_fuel_mass, halo_period, manifold_length])
            trajectory_params.append(sample_params)
        
        trajectory_params = np.array(trajectory_params)
        
        # Generate halo energies (unnormalize)
        halo_energies = classifier_norm * (max_halo_energy - min_halo_energy) + min_halo_energy
        
        # Combine halo energies with trajectory parameters
        final_samples = np.hstack((halo_energies.reshape(-1, 1), trajectory_params))
        
        print(f"✓ Transformed ORIGINAL samples shape: {final_samples.shape}")
        return final_samples
    
    def compute_physical_validation_metrics(self, samples: np.ndarray) -> Dict[str, Any]:
        """Compute physical validation metrics (identical to 3-channel version)."""
        print(f"DEBUG: enable_physical_validation = {self.config.enable_physical_validation}")
        print(f"DEBUG: GTO_HALO_DM_AVAILABLE = {GTO_HALO_DM_AVAILABLE}")
        
        if not self.config.enable_physical_validation or not GTO_HALO_DM_AVAILABLE:
            print("⚠️  CRITICAL: Physical validation disabled - GTO_Halo_DM modules not available")
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
                if 'snopt_results' in result and result['snopt_results']:
                    if 'feasibility' in result['snopt_results']:
                        convergence_status = "FEASIBLE" if result['snopt_results']['feasibility'] else "INFEASIBLE"
                    elif 'error_message' in result['snopt_results']:
                        convergence_status = f"ERROR: {result['snopt_results']['error_message']}"
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
        
        # Save the comprehensive dataset
        self.save_comprehensive_dataset(all_results)
        
        # Extract result data and initial guesses for legacy statistics computation
        result_data_list = []
        initial_guesses_list = []
        
        for result in all_results:
            # Extract legacy format for compatibility with existing statistics
            if 'snopt_results' in result:
                legacy_result = {
                    'feasibility': result['snopt_results'].get('feasibility', False),
                    'snopt_inform': result['snopt_results'].get('snopt_inform', None),
                    'solving_time': result['snopt_results'].get('solving_time', None),
                    'results.control': None
                }
                
                # Extract control vector if available
                if 'converged_trajectory' in result and 'control_vector' in result['converged_trajectory']:
                    legacy_result['results.control'] = np.array(result['converged_trajectory']['control_vector'])
                
                result_data_list.append(legacy_result)
            else:
                # Handle old format (backward compatibility)
                result_data_list.append(result.get('result_data', {}))
            
            # Extract initial guess
            if 'generated_sample' in result:
                initial_guesses_list.append(np.array(result['generated_sample']['trajectory_params']))
            else:
                # Handle old format (backward compatibility)
                initial_guesses_list.append(result.get('initial_guess', []))
        
        # Compute statistics
        physical_metrics = self.compute_cr3bp_statistics(result_data_list, initial_guesses_list)
        
        # Add dataset information to metrics
        physical_metrics['comprehensive_dataset_saved'] = True
        physical_metrics['dataset_entries'] = len(all_results)
        physical_metrics['model_type'] = 'ORIGINAL_1_CHANNEL'
        
        return physical_metrics
    
    def save_comprehensive_dataset(self, all_results: List[Dict]):
        """Save the comprehensive dataset (identical structure to 3-channel version)."""
        print("Saving comprehensive dataset for ORIGINAL 1-CHANNEL model...")
        
        # Create dataset directory
        dataset_dir = os.path.join(self.config.output_dir, 'comprehensive_dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Organize data by convergence status for easier analysis
        dataset_by_status = {
            'LOCALLY_OPTIMAL': [],
            'FEASIBLE': [],
            'INFEASIBLE': [],
            'PROCESSING_ERROR': [],
            'UNKNOWN': []
        }
        
        for result in all_results:
            status = result.get('processing_metadata', {}).get('convergence_status', 'UNKNOWN')
            dataset_by_status[status].append(result)
        
        # Save complete dataset as pickle for easy loading in Python
        complete_dataset_path = os.path.join(dataset_dir, 'complete_dataset.pkl')
        with open(complete_dataset_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"✓ Complete dataset saved to {complete_dataset_path}")
        
        # Save complete dataset as JSON for readability
        complete_dataset_json_path = os.path.join(dataset_dir, 'complete_dataset.json')
        with open(complete_dataset_json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"✓ Complete dataset (JSON) saved to {complete_dataset_json_path}")
        
        # Save datasets by convergence status
        for status, data in dataset_by_status.items():
            if data:  # Only save if there's data for this status
                status_path = os.path.join(dataset_dir, f'dataset_{status.lower()}.pkl')
                with open(status_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"✓ {status} dataset saved to {status_path} ({len(data)} samples)")
        
        # Save feasible trajectories separately for quick access
        feasible_trajectories = []
        for result in all_results:
            if (result.get('processing_metadata', {}).get('convergence_status') in ['FEASIBLE', 'LOCALLY_OPTIMAL'] 
                and 'converged_trajectory' in result 
                and result['converged_trajectory']):
                feasible_trajectories.append({
                    'sample_idx': result['sample_idx'],
                    'halo_energy': result['generated_sample']['halo_energy'],
                    'convergence_status': result['processing_metadata']['convergence_status'],
                    'snopt_inform': result['snopt_results']['snopt_inform'],
                    'solving_time': result['snopt_results']['solving_time'],
                    'trajectory_parameters': result['converged_trajectory']['trajectory_parameters'],
                    'control_vector': result['converged_trajectory']['control_vector'],
                    'control_segments': result['converged_trajectory']['control_segments'],
                    'model_type': 'ORIGINAL_1_CHANNEL'
                })
        
        if feasible_trajectories:
            feasible_traj_path = os.path.join(dataset_dir, 'feasible_trajectories.pkl')
            with open(feasible_traj_path, 'wb') as f:
                pickle.dump(feasible_trajectories, f)
            print(f"✓ Feasible trajectories saved to {feasible_traj_path} ({len(feasible_trajectories)} trajectories)")
        
        # Create dataset summary
        summary = {
            'total_samples': len(all_results),
            'by_status': {status: len(data) for status, data in dataset_by_status.items()},
            'feasible_trajectories': len(feasible_trajectories),
            'model_type': 'ORIGINAL_1_CHANNEL',
            'dataset_structure': {
                'sample_identification': ['sample_idx', 'thread_id'],
                'input_data': ['generated_sample'],
                'simulation_config': ['simulation_config'],
                'snopt_results': ['snopt_results'],
                'converged_trajectory': ['converged_trajectory'],
                'physical_trajectories': ['physical_trajectories'],
                'processing_metadata': ['processing_metadata']
            },
            'files_saved': [
                'complete_dataset.pkl',
                'complete_dataset.json',
                'feasible_trajectories.pkl'
            ] + [f'dataset_{status.lower()}.pkl' for status, data in dataset_by_status.items() if data]
        }
        
        # Save dataset summary
        summary_path = os.path.join(dataset_dir, 'dataset_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Dataset summary saved to {summary_path}")
        
        # Create usage guide
        usage_guide = """
# GTO Halo Comprehensive Dataset Usage Guide (ORIGINAL 1-CHANNEL MODEL)

This directory contains a comprehensive dataset with all generated samples from the ORIGINAL 1-channel model,
SNOPT simulation results, converged trajectory data, and physical orbital trajectories.

## Model Type: ORIGINAL_1_CHANNEL
- Input: 67-dimensional vectors directly 
- Architecture: 1 channel, 67 sequence length
- No 3-channel conversion needed

## Files:

1. **complete_dataset.pkl**: Full dataset in Python pickle format
   - Load with: `data = pickle.load(open('complete_dataset.pkl', 'rb'))`
   
2. **complete_dataset.json**: Full dataset in JSON format (human-readable)

3. **feasible_trajectories.pkl**: Only the feasible/optimal trajectory data
   - Quick access to successful convergence results
   
4. **dataset_[status].pkl**: Data organized by convergence status
   - dataset_locally_optimal.pkl: SNOPT inform = 1 solutions
   - dataset_feasible.pkl: Feasible but not necessarily optimal
   - dataset_infeasible.pkl: Failed convergence
   - dataset_processing_error.pkl: Processing errors

5. **dataset_summary.json**: Overview of dataset contents

## Data Structure:

Each entry contains:
- **sample_idx**: Sample identifier
- **generated_sample**: Original diffusion model output
  - halo_energy: Physical halo energy parameter
  - trajectory_params: 66-dimensional trajectory parameters  
  - full_sample_vector: Complete input vector for SNOPT
  - model_type: 'ORIGINAL_1_CHANNEL'
- **simulation_config**: SNOPT simulation parameters used
- **snopt_results**: SNOPT convergence statistics
- **converged_trajectory**: Complete trajectory data (if feasible)
- **physical_trajectories**: Physical orbital state trajectories
- **processing_metadata**: Timestamps, convergence status, and model type

## Comparing with 3-Channel Model:

```python
import pickle

# Load original model results
with open('complete_dataset.pkl', 'rb') as f:
    original_data = pickle.load(f)

# Load 3-channel model results  
with open('../3channel_results/complete_dataset.pkl', 'rb') as f:
    channel3_data = pickle.load(f)

# Compare model performance
original_feasible = [entry for entry in original_data 
                    if entry['snopt_results']['feasibility']]
                    
channel3_feasible = [entry for entry in channel3_data 
                    if entry['snopt_results']['feasibility']]

print(f"Original 1-channel: {len(original_feasible)}/{len(original_data)} feasible")
print(f"3-channel: {len(channel3_feasible)}/{len(channel3_data)} feasible")
```
"""
        
        usage_guide_path = os.path.join(dataset_dir, 'README.md')
        with open(usage_guide_path, 'w') as f:
            f.write(usage_guide)
        print(f"✓ Usage guide saved to {usage_guide_path}")
        
        print(f"✓ Comprehensive dataset completed for ORIGINAL 1-CHANNEL model:")
        print(f"   - Total samples: {len(all_results)}")
        print(f"   - Locally optimal: {len(dataset_by_status['LOCALLY_OPTIMAL'])}")
        print(f"   - Feasible: {len(dataset_by_status['FEASIBLE'])}")
        print(f"   - Infeasible: {len(dataset_by_status['INFEASIBLE'])}")
        print(f"   - Processing errors: {len(dataset_by_status['PROCESSING_ERROR'])}")
        print(f"   - Feasible trajectories: {len(feasible_trajectories)}")
    
    def compute_gto_halo_metrics(self, samples: np.ndarray) -> Dict[str, Any]:
        """Compute GTO Halo specific metrics (identical to 3-channel version)."""
        print("Computing GTO Halo specific metrics for ORIGINAL 1-CHANNEL model...")
        
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
        
        # Add model type
        metrics['model_type'] = 'ORIGINAL_1_CHANNEL'
        
        return metrics
    
    def compute_cr3bp_statistics(self, result_data_list: List[Dict], initial_guesses_list: List[np.ndarray]) -> Dict[str, Any]:
        """Compute CR3BP simulation statistics (identical to 3-channel version)."""
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
            'local_optimal_count': local_optimal_count,
            'model_type': 'ORIGINAL_1_CHANNEL'
        }
    
    def compute_sampling_efficiency_metrics(self, sampling_times: List[float]) -> Dict[str, float]:
        """Compute sampling efficiency metrics (identical to 3-channel version)."""
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
            'max_sampling_time': max(sampling_times),
            'model_type': 'ORIGINAL_1_CHANNEL'
        }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark."""
        print("=" * 60)
        print("RUNNING GTO HALO BENCHMARK (ORIGINAL 1-CHANNEL MODEL)")
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
            'model_info': {
                'model_type': 'ORIGINAL_1_CHANNEL',
                'channels': 1,
                'sequence_length': 67,
                'checkpoint_file': self.config.checkpoint_file
            },
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
        """Save benchmark results (identical to 3-channel version)."""
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
            f.write("GTO HALO BENCHMARK RESULTS (ORIGINAL 1-CHANNEL MODEL)\n")
            f.write("=" * 60 + "\n\n")
            
            # Model info
            model_info = results.get('model_info', {})
            if model_info:
                f.write("MODEL INFORMATION:\n")
                for key, value in model_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
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
                
                # Add dataset information if available
                if physical_metrics.get('comprehensive_dataset_saved', False):
                    f.write("COMPREHENSIVE DATASET:\n")
                    f.write(f"  dataset_saved: {physical_metrics.get('comprehensive_dataset_saved', False)}\n")
                    f.write(f"  total_entries: {physical_metrics.get('dataset_entries', 0)}\n")
                    f.write(f"  location: {self.config.output_dir}/comprehensive_dataset/\n")
                    f.write("  files: complete_dataset.pkl, feasible_trajectories.pkl, dataset_summary.json\n")
                    f.write("  usage_guide: comprehensive_dataset/README.md\n")
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
            
            # Add important file locations
            f.write("OUTPUT FILES:\n")
            f.write(f"  benchmark_results: gto_halo_benchmark_results.json\n")
            f.write(f"  generated_samples: samples.npy\n")
            f.write(f"  comprehensive_dataset: comprehensive_dataset/\n")
            f.write(f"  cr3bp_simulation_results: cr3bp_results/\n")
            f.write(f"  summary: summary.txt\n")
        
        print(f"✓ Summary saved to {summary_path}")
    
    def generate_plots(self, results: Dict[str, Any], samples: np.ndarray):
        """Generate visualization plots."""
        # This can be implemented similar to the original benchmarking script
        # For now, we'll skip plotting to focus on the multithreading functionality
        print("✓ Plot generation skipped (focusing on multithreading)")


def main():
    """Main function to run the original 1-channel benchmark."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Fast Multithreaded GTO Halo Benchmarking (Original 1-Channel Model)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--checkpoint_file', type=str, default='model-epoch-175.pt', help='Checkpoint filename')
    parser.add_argument('--data_path', type=str, required=True, help='Path to reference data')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for sampling')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of worker threads (default: 4)')
    parser.add_argument('--chunk_size', type=int, default=1, help='Number of samples per thread')
    parser.add_argument('--pre_warm_threads', action='store_true', default=True, help='Pre-warm threads to reduce initialization delays')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name (auto-generated if not specified)')
    parser.add_argument('--enable_physical_validation', action='store_true', default=True, help='Enable physical validation')
    parser.add_argument('--fixed_alpha', type=float, default=None, help='Fixed alpha value (None for random sampling)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create organized output directory structure
    if args.output_dir is None:
        # Create automatically named directory based on timestamp and samples
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"benchmark_original_{timestamp}_samples_{args.num_samples}"
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
        checkpoint_file=args.checkpoint_file,
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
    benchmarker = GTOHaloBenchmarkerOriginal(config)
    results = benchmarker.run_benchmark()
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY (ORIGINAL 1-CHANNEL MODEL)")
    print("=" * 60)
    print(f"Model type: {results['model_info'].get('model_type', 'ORIGINAL_1_CHANNEL')}")
    print(f"Total sampling time: {results['sampling_efficiency_metrics'].get('total_sampling_time', 0):.2f}s")
    print(f"Samples per second: {results['sampling_efficiency_metrics'].get('samples_per_second', 0):.3f}")
    print(f"Halo energy mean: {results['gto_halo_metrics'].get('halo_energy_mean', 0):.6f}")
    print(f"Feasible ratio: {results['physical_validation_metrics'].get('feasible_ratio', 0):.3f}")
    print(f"Local optimal ratio: {results['physical_validation_metrics'].get('local_optimal_ratio', 0):.3f}")
    print(f"Results saved to: {args.output_dir}")
    
    # Comprehensive dataset information
    if results['physical_validation_metrics'].get('comprehensive_dataset_saved', False):
        print(f"Comprehensive dataset: {args.output_dir}/comprehensive_dataset/")
        print(f"Dataset entries: {results['physical_validation_metrics'].get('dataset_entries', 0)}")
        print("Dataset includes: complete_dataset.pkl, feasible_trajectories.pkl, and usage guide")
    
    print("Check the summary.txt and comprehensive_dataset/README.md for detailed results.")


if __name__ == "__main__":
    main()