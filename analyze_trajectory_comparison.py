#!/usr/bin/env python3
"""
Trajectory Comparison Analysis Script

This script loads comprehensive datasets from the GTO Halo benchmarking system
and creates detailed plots comparing predicted trajectories (from diffusion model)
with converged trajectories (from SNOPT simulation).

Features:
1. Parameter-by-parameter comparison plots
2. Control segment analysis
3. Key metrics comparison (fuel mass, shooting time, etc.)
4. Statistical analysis of prediction accuracy
5. Trajectory visualization in parameter space
"""

import os
import sys
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import json

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrajectoryComparisonAnalyzer:
    """Analyze and visualize trajectory predictions vs converged solutions."""
    
    def __init__(self, dataset_path: str, output_dir: str = None):
        """Initialize the analyzer with dataset path."""
        self.dataset_path = dataset_path
        self.output_dir = output_dir or os.path.join(os.path.dirname(dataset_path), 'trajectory_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the comprehensive dataset
        self.load_dataset()
        
        # Trajectory parameter names for plotting
        self.param_names = self._get_parameter_names()
        
    def load_dataset(self):
        """Load the comprehensive dataset."""
        print(f"Loading dataset from {self.dataset_path}")
        
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Dataset loaded: {len(self.data)} samples")
        
        # Filter for feasible trajectories only
        self.feasible_data = [
            entry for entry in self.data 
            if entry['snopt_results']['feasibility'] and 
               'converged_trajectory' in entry and 
               entry['converged_trajectory']
        ]
        
        # Filter for data with physical trajectories
        self.physical_data = [
            entry for entry in self.data
            if 'physical_trajectories' in entry and 
               entry['physical_trajectories'] and
               'predicted_states' in entry['physical_trajectories'] and
               entry['physical_trajectories']['predicted_states'] is not None
        ]
        
        print(f"Feasible trajectories: {len(self.feasible_data)}")
        
        if len(self.feasible_data) == 0:
            raise ValueError("No feasible trajectories found in dataset!")
            
    def _get_parameter_names(self):
        """Get parameter names for the 66-dimensional trajectory vector."""
        names = []
        
        # Time parameters (3)
        names.extend(['Shooting Time', 'Initial Coast', 'Final Coast'])
        
        # Control segments (60 parameters = 20 segments × 3 components)
        for i in range(20):
            names.extend([f'Seg{i+1}_Alpha', f'Seg{i+1}_Beta', f'Seg{i+1}_Thrust'])
        
        # Final parameters (3)
        names.extend(['Final Fuel Mass', 'Halo Period', 'Manifold Length'])
        
        return names
    
    def extract_trajectory_data(self):
        """Extract predicted and converged trajectory data."""
        print("Extracting trajectory data...")
        
        predicted_trajectories = []
        converged_trajectories = []
        halo_energies = []
        sample_indices = []
        
        for entry in self.feasible_data:
            # Predicted trajectory (from diffusion model)
            predicted = np.array(entry['generated_sample']['trajectory_params'])
            predicted_trajectories.append(predicted)
            
            # Converged trajectory (from SNOPT)
            converged = np.array(entry['converged_trajectory']['control_vector'])
            converged_trajectories.append(converged)
            
            # Metadata
            halo_energies.append(entry['generated_sample']['halo_energy'])
            sample_indices.append(entry['sample_idx'])
        
        self.predicted_trajectories = np.array(predicted_trajectories)
        self.converged_trajectories = np.array(converged_trajectories)
        self.halo_energies = np.array(halo_energies)
        self.sample_indices = np.array(sample_indices)
        
        print(f"Extracted {len(predicted_trajectories)} trajectory pairs")
        
    def compute_comparison_metrics(self):
        """Compute comparison metrics between predicted and converged trajectories."""
        print("Computing comparison metrics...")
        
        # Absolute differences
        self.abs_differences = np.abs(self.predicted_trajectories - self.converged_trajectories)
        
        # Relative differences (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.rel_differences = np.abs(
                (self.predicted_trajectories - self.converged_trajectories) / 
                (self.converged_trajectories + 1e-10)
            )
            self.rel_differences[~np.isfinite(self.rel_differences)] = 0
        
        # Summary statistics
        self.metrics = {
            'mean_abs_error': np.mean(self.abs_differences, axis=0),
            'std_abs_error': np.std(self.abs_differences, axis=0),
            'mean_rel_error': np.mean(self.rel_differences, axis=0),
            'std_rel_error': np.std(self.rel_differences, axis=0),
            'max_abs_error': np.max(self.abs_differences, axis=0),
            'max_rel_error': np.max(self.rel_differences, axis=0)
        }
        
        # Overall accuracy metrics
        # Safe correlation calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                correlation = np.corrcoef(
                    self.predicted_trajectories.flatten(), 
                    self.converged_trajectories.flatten()
                )[0, 1]
                if not np.isfinite(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
        
        self.overall_metrics = {
            'overall_mae': np.mean(self.abs_differences),
            'overall_mape': np.mean(self.rel_differences) * 100,
            'correlation': correlation
        }
        
        print(f"Overall MAE: {self.overall_metrics['overall_mae']:.6f}")
        print(f"Overall MAPE: {self.overall_metrics['overall_mape']:.2f}%")
        print(f"Correlation: {self.overall_metrics['correlation']:.4f}")
    
    def plot_error_distribution(self):
        """Plot error distribution analysis."""
        print("Creating error distribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Absolute error distribution
        axes[0, 0].hist(self.abs_differences.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Absolute Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Absolute Errors')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Relative error distribution (cap at 200% for visualization)
        rel_errors_capped = np.clip(self.rel_differences.flatten() * 100, 0, 200)
        axes[0, 1].hist(rel_errors_capped, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Relative Error (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Relative Errors (capped at 200%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error vs halo energy
        mean_abs_error_per_sample = np.mean(self.abs_differences, axis=1)
        axes[1, 0].scatter(self.halo_energies, mean_abs_error_per_sample, alpha=0.7, s=60)
        axes[1, 0].set_xlabel('Halo Energy')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].set_title('Prediction Accuracy vs Halo Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Parameter-wise error ranking
        param_indices = np.arange(len(self.param_names))
        axes[1, 1].bar(param_indices, self.metrics['mean_abs_error'])
        axes[1, 1].set_xlabel('Parameter Index')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Prediction Error by Parameter')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Rotate x-axis labels for readability
        if len(self.param_names) <= 20:
            axes[1, 1].set_xticks(param_indices[::3])  # Show every 3rd label
            axes[1, 1].set_xticklabels([self.param_names[i] for i in param_indices[::3]], 
                                      rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_control_segments_comparison(self):
        """Plot control segments comparison."""
        print("Creating control segments comparison...")
        
        # Extract control segments
        predicted_controls = []
        converged_controls = []
        
        for i, entry in enumerate(self.feasible_data):
            pred_params = np.array(entry['generated_sample']['trajectory_params'])
            conv_params = np.array(entry['converged_trajectory']['control_vector'])
            
            # Extract control segments (parameters 3-62, 20 segments × 3 components)
            pred_controls = pred_params[3:63].reshape(20, 3)  # [alpha, beta, thrust]
            conv_controls = conv_params[3:63].reshape(20, 3)
            
            predicted_controls.append(pred_controls)
            converged_controls.append(conv_controls)
        
        predicted_controls = np.array(predicted_controls)
        converged_controls = np.array(converged_controls)
        
        # Plot average control profile comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        component_names = ['Alpha (Thrust Direction)', 'Beta (Thrust Direction)', 'Thrust Magnitude']
        
        for comp in range(3):
            # Average profiles
            pred_avg = np.mean(predicted_controls[:, :, comp], axis=0)
            conv_avg = np.mean(converged_controls[:, :, comp], axis=0)
            pred_std = np.std(predicted_controls[:, :, comp], axis=0)
            conv_std = np.std(converged_controls[:, :, comp], axis=0)
            
            segments = np.arange(1, 21)
            
            axes[comp].plot(segments, pred_avg, 'b-o', label='Predicted (Diffusion)', linewidth=2, markersize=6)
            axes[comp].fill_between(segments, pred_avg - pred_std, pred_avg + pred_std, 
                                   alpha=0.3, color='blue')
            
            axes[comp].plot(segments, conv_avg, 'r-s', label='Converged (SNOPT)', linewidth=2, markersize=6)
            axes[comp].fill_between(segments, conv_avg - conv_std, conv_avg + conv_std, 
                                   alpha=0.3, color='red')
            
            axes[comp].set_xlabel('Control Segment')
            axes[comp].set_ylabel(f'{component_names[comp]}')
            axes[comp].set_title(f'Average {component_names[comp]} Profile')
            axes[comp].legend()
            axes[comp].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'control_segments_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def extract_physical_trajectory_data(self):
        """Extract physical trajectory state data for comparison."""
        print("Extracting physical trajectory data...")
        
        self.predicted_physical_trajectories = []
        self.converged_physical_trajectories = []
        self.physical_sample_indices = []
        self.physical_halo_energies = []
        
        for entry in self.physical_data:
            # Get predicted trajectory states
            if 'predicted_states' in entry['physical_trajectories']:
                pred_states = np.array(entry['physical_trajectories']['predicted_states'])
                self.predicted_physical_trajectories.append(pred_states)
                
                # Get converged trajectory states if available
                if ('converged_states' in entry['physical_trajectories'] and 
                    entry['physical_trajectories']['converged_states'] is not None):
                    conv_states = np.array(entry['physical_trajectories']['converged_states'])
                    self.converged_physical_trajectories.append(conv_states)
                else:
                    # If no converged trajectory, use None placeholder
                    self.converged_physical_trajectories.append(None)
                
                # Store metadata
                self.physical_sample_indices.append(entry['sample_idx'])
                self.physical_halo_energies.append(entry['generated_sample']['halo_energy'])
        
        print(f"Extracted {len(self.predicted_physical_trajectories)} physical trajectory pairs")
        
        # Filter out entries where we don't have both predicted and converged
        valid_pairs = []
        for i, (pred, conv) in enumerate(zip(self.predicted_physical_trajectories, self.converged_physical_trajectories)):
            if conv is not None:
                valid_pairs.append(i)
        
        print(f"Found {len(valid_pairs)} complete physical trajectory pairs")
        self.valid_physical_indices = valid_pairs
    
    def compute_trajectory_deviations(self):
        """Compute deviations between predicted and converged physical trajectories."""
        print("Computing trajectory deviations...")
        
        self.trajectory_deviations = []
        self.position_deviations = []
        self.velocity_deviations = []
        self.max_position_deviations = []
        self.max_velocity_deviations = []
        self.final_position_deviations = []
        
        for idx in self.valid_physical_indices:
            pred_traj = self.predicted_physical_trajectories[idx]
            conv_traj = self.converged_physical_trajectories[idx]
            
            # Ensure trajectories have same length (interpolate if needed)
            min_length = min(len(pred_traj), len(conv_traj))
            pred_traj = pred_traj[:min_length]
            conv_traj = conv_traj[:min_length]
            
            # Extract positions (x, y, z) and velocities (vx, vy, vz)
            # Assuming state format: [x, y, z, vx, vy, vz, ...]
            pred_pos = pred_traj[:, :3]  # [x, y, z]
            conv_pos = conv_traj[:, :3]
            pred_vel = pred_traj[:, 3:6]  # [vx, vy, vz]
            conv_vel = conv_traj[:, 3:6]
            
            # Compute position and velocity deviations
            pos_dev = np.linalg.norm(pred_pos - conv_pos, axis=1)
            vel_dev = np.linalg.norm(pred_vel - conv_vel, axis=1)
            
            self.position_deviations.append(pos_dev)
            self.velocity_deviations.append(vel_dev)
            self.max_position_deviations.append(np.max(pos_dev))
            self.max_velocity_deviations.append(np.max(vel_dev))
            self.final_position_deviations.append(pos_dev[-1])
            
            # Store full trajectory deviation data
            self.trajectory_deviations.append({
                'position_deviation': pos_dev,
                'velocity_deviation': vel_dev,
                'predicted_positions': pred_pos,
                'converged_positions': conv_pos,
                'predicted_velocities': pred_vel,
                'converged_velocities': conv_vel
            })
        
        print(f"Computed deviations for {len(self.trajectory_deviations)} trajectory pairs")
    
    def compute_aggregate_trajectory_statistics(self):
        """Compute comprehensive aggregate statistics for physical trajectory deviations."""
        print("Computing aggregate trajectory statistics...")
        
        if not hasattr(self, 'trajectory_deviations') or not self.trajectory_deviations:
            print("No trajectory deviations available for aggregate analysis")
            return
        
        # Initialize aggregate statistics
        self.aggregate_stats = {
            'position_deviation': {
                'mean_over_time': [],
                'std_over_time': [],
                'max_over_time': [],
                'final_values': [],
                'cumulative_stats': {}
            },
            'velocity_deviation': {
                'mean_over_time': [],
                'std_over_time': [],
                'max_over_time': [],
                'final_values': [],
                'cumulative_stats': {}
            },
            'trajectory_length': [],
            'halo_energy_correlation': {},
            'time_normalized_stats': {}
        }
        
        # Collect all trajectory data for aggregate analysis
        all_position_deviations = []
        all_velocity_deviations = []
        all_trajectory_lengths = []
        all_halo_energies = []
        
        for idx in self.valid_physical_indices:
            pred_traj = self.predicted_physical_trajectories[idx]
            conv_traj = self.converged_physical_trajectories[idx]
            
            # Ensure trajectories have same length
            min_length = min(len(pred_traj), len(conv_traj))
            pred_traj = pred_traj[:min_length]
            conv_traj = conv_traj[:min_length]
            
            # Extract positions and velocities
            pred_pos = pred_traj[:, :3]
            conv_pos = conv_traj[:, :3]
            pred_vel = pred_traj[:, 3:6]
            conv_vel = conv_traj[:, 3:6]
            
            # Compute deviations
            pos_dev = np.linalg.norm(pred_pos - conv_pos, axis=1)
            vel_dev = np.linalg.norm(pred_vel - conv_vel, axis=1)
            
            all_position_deviations.append(pos_dev)
            all_velocity_deviations.append(vel_dev)
            all_trajectory_lengths.append(min_length)
            all_halo_energies.append(self.physical_halo_energies[idx])
        
        # Convert to numpy arrays (handle variable lengths)
        all_trajectory_lengths = np.array(all_trajectory_lengths)
        all_halo_energies = np.array(all_halo_energies)
        
        # Find maximum trajectory length for time normalization
        max_length = np.max(all_trajectory_lengths)
        
        # Pad shorter trajectories with NaN for proper statistics
        padded_pos_deviations = np.full((len(all_position_deviations), max_length), np.nan)
        padded_vel_deviations = np.full((len(all_velocity_deviations), max_length), np.nan)
        
        for i, (pos_dev, vel_dev, length) in enumerate(zip(all_position_deviations, all_velocity_deviations, all_trajectory_lengths)):
            padded_pos_deviations[i, :length] = pos_dev
            padded_vel_deviations[i, :length] = vel_dev
        
        # Compute time-wise statistics
        self.aggregate_stats['position_deviation']['mean_over_time'] = np.nanmean(padded_pos_deviations, axis=0)
        self.aggregate_stats['position_deviation']['std_over_time'] = np.nanstd(padded_pos_deviations, axis=0)
        self.aggregate_stats['position_deviation']['max_over_time'] = np.nanmax(padded_pos_deviations, axis=0)
        
        self.aggregate_stats['velocity_deviation']['mean_over_time'] = np.nanmean(padded_vel_deviations, axis=0)
        self.aggregate_stats['velocity_deviation']['std_over_time'] = np.nanstd(padded_vel_deviations, axis=0)
        self.aggregate_stats['velocity_deviation']['max_over_time'] = np.nanmax(padded_vel_deviations, axis=0)
        
        # Compute final values (last valid point for each trajectory)
        final_pos_deviations = []
        final_vel_deviations = []
        for i, length in enumerate(all_trajectory_lengths):
            final_pos_deviations.append(padded_pos_deviations[i, length-1])
            final_vel_deviations.append(padded_vel_deviations[i, length-1])
        
        self.aggregate_stats['position_deviation']['final_values'] = np.array(final_pos_deviations)
        self.aggregate_stats['velocity_deviation']['final_values'] = np.array(final_vel_deviations)
        
        # Compute cumulative statistics
        self.aggregate_stats['position_deviation']['cumulative_stats'] = {
            'mean': np.mean(final_pos_deviations),
            'std': np.std(final_pos_deviations),
            'median': np.median(final_pos_deviations),
            'min': np.min(final_pos_deviations),
            'max': np.max(final_pos_deviations),
            'q25': np.percentile(final_pos_deviations, 25),
            'q75': np.percentile(final_pos_deviations, 75)
        }
        
        self.aggregate_stats['velocity_deviation']['cumulative_stats'] = {
            'mean': np.mean(final_vel_deviations),
            'std': np.std(final_vel_deviations),
            'median': np.median(final_vel_deviations),
            'min': np.min(final_vel_deviations),
            'max': np.max(final_vel_deviations),
            'q25': np.percentile(final_vel_deviations, 25),
            'q75': np.percentile(final_vel_deviations, 75)
        }
        
        # Compute halo energy correlations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                pos_corr = np.corrcoef(all_halo_energies, final_pos_deviations)[0, 1]
                vel_corr = np.corrcoef(all_halo_energies, final_vel_deviations)[0, 1]
                if not np.isfinite(pos_corr):
                    pos_corr = 0.0
                if not np.isfinite(vel_corr):
                    vel_corr = 0.0
            except:
                pos_corr = 0.0
                vel_corr = 0.0
        
        self.aggregate_stats['halo_energy_correlation'] = {
            'position_deviation_correlation': pos_corr,
            'velocity_deviation_correlation': vel_corr
        }
        
        # Store trajectory lengths
        self.aggregate_stats['trajectory_length'] = {
            'mean': np.mean(all_trajectory_lengths),
            'std': np.std(all_trajectory_lengths),
            'min': np.min(all_trajectory_lengths),
            'max': np.max(all_trajectory_lengths),
            'values': all_trajectory_lengths
        }
        
        # Time-normalized statistics (for trajectories of different lengths)
        time_normalized_pos = []
        time_normalized_vel = []
        
        for i, length in enumerate(all_trajectory_lengths):
            # Normalize time to [0, 1] for each trajectory
            time_points = np.linspace(0, 1, length)
            pos_dev = padded_pos_deviations[i, :length]
            vel_dev = padded_vel_deviations[i, :length]
            
            time_normalized_pos.append(pos_dev)
            time_normalized_vel.append(vel_dev)
        
        # Compute statistics over normalized time
        max_normalized_length = max(len(traj) for traj in time_normalized_pos)
        normalized_pos_array = np.full((len(time_normalized_pos), max_normalized_length), np.nan)
        normalized_vel_array = np.full((len(time_normalized_vel), max_normalized_length), np.nan)
        
        for i, (pos_traj, vel_traj) in enumerate(zip(time_normalized_pos, time_normalized_vel)):
            normalized_pos_array[i, :len(pos_traj)] = pos_traj
            normalized_vel_array[i, :len(vel_traj)] = vel_traj
        
        self.aggregate_stats['time_normalized_stats'] = {
            'position_mean_over_normalized_time': np.nanmean(normalized_pos_array, axis=0),
            'position_std_over_normalized_time': np.nanstd(normalized_pos_array, axis=0),
            'velocity_mean_over_normalized_time': np.nanmean(normalized_vel_array, axis=0),
            'velocity_std_over_normalized_time': np.nanstd(normalized_vel_array, axis=0)
        }
        
        print(f"Aggregate statistics computed for {len(self.valid_physical_indices)} trajectory pairs")
        print(f"Position deviation - Mean: {self.aggregate_stats['position_deviation']['cumulative_stats']['mean']:.6f} DU")
        print(f"Velocity deviation - Mean: {self.aggregate_stats['velocity_deviation']['cumulative_stats']['mean']:.6f} DU/TU")

    def plot_aggregate_trajectory_analysis(self):
        """Create comprehensive aggregate analysis plots for physical trajectories."""
        print("Creating aggregate trajectory analysis plots...")
        
        if not hasattr(self, 'aggregate_stats'):
            print("No aggregate statistics available. Run compute_aggregate_trajectory_statistics() first.")
            return
        
        # Create a figure with only the top row (4 plots)
        fig = plt.figure(figsize=(20, 5))
        
        # 1. Time-wise mean and std plots
        ax1 = fig.add_subplot(1, 4, 1)
        time_points = np.arange(len(self.aggregate_stats['position_deviation']['mean_over_time']))
        pos_mean = self.aggregate_stats['position_deviation']['mean_over_time']
        pos_std = self.aggregate_stats['position_deviation']['std_over_time']
        
        ax1.plot(time_points, pos_mean, 'b-', linewidth=2, label='Mean Position Deviation')
        ax1.fill_between(time_points, pos_mean - pos_std, pos_mean + pos_std, 
                        alpha=0.3, color='blue', label='±1σ')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Position Deviation (DU)')
        ax1.set_title('Position Deviation Over Time\n(Mean ± Std)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Velocity deviation over time
        ax2 = fig.add_subplot(1, 4, 2)
        vel_mean = self.aggregate_stats['velocity_deviation']['mean_over_time']
        vel_std = self.aggregate_stats['velocity_deviation']['std_over_time']
        
        ax2.plot(time_points, vel_mean, 'r-', linewidth=2, label='Mean Velocity Deviation')
        ax2.fill_between(time_points, vel_mean - vel_std, vel_mean + vel_std, 
                        alpha=0.3, color='red', label='±1σ')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Velocity Deviation (DU/TU)')
        ax2.set_title('Velocity Deviation Over Time\n(Mean ± Std)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Final deviation distributions
        ax3 = fig.add_subplot(1, 4, 3)
        final_pos = self.aggregate_stats['position_deviation']['final_values']
        final_vel = self.aggregate_stats['velocity_deviation']['final_values']
        
        ax3.hist(final_pos, bins=20, alpha=0.7, color='blue', label='Position', density=True)
        ax3.hist(final_vel, bins=20, alpha=0.7, color='red', label='Velocity', density=True)
        ax3.set_xlabel('Final Deviation')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution of Final Deviations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Halo energy correlation
        ax4 = fig.add_subplot(1, 4, 4)
        valid_halo_energies = [self.physical_halo_energies[i] for i in self.valid_physical_indices]
        ax4.scatter(valid_halo_energies, final_pos, alpha=0.7, s=60, c='blue', label='Position')
        ax4.scatter(valid_halo_energies, final_vel, alpha=0.7, s=60, c='red', label='Velocity')
        ax4.set_xlabel('Halo Energy')
        ax4.set_ylabel('Final Deviation')
        ax4.set_title('Final Deviation vs Halo Energy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Remove all remaining subplots (5-12) since we only want the top row
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'aggregate_trajectory_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed statistics to JSON
        stats_path = os.path.join(self.output_dir, 'aggregate_trajectory_statistics.json')
        with open(stats_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_stats = {}
            for key, value in self.aggregate_stats.items():
                if isinstance(value, dict):
                    json_stats[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            json_stats[key][subkey] = subvalue.tolist()
                        elif isinstance(subvalue, (np.integer, np.floating)):
                            json_stats[key][subkey] = float(subvalue)
                        else:
                            json_stats[key][subkey] = subvalue
                else:
                    json_stats[key] = value
            json.dump(json_stats, f, indent=2)
        
        print(f"Aggregate trajectory analysis saved to: {self.output_dir}")
        print(f"  - aggregate_trajectory_analysis.png: Comprehensive analysis plots")
        print(f"  - aggregate_trajectory_statistics.json: Detailed statistics")
    
    def create_summary_report(self):
        """Create a summary report of the analysis."""
        print("Creating summary report...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_path': self.dataset_path,
            'total_samples': len(self.data),
            'feasible_samples': len(self.feasible_data),
            'physical_trajectory_samples': len(self.physical_data) if hasattr(self, 'physical_data') else 0,
            'overall_metrics': self.overall_metrics,
            'parameter_metrics': {
                'mean_absolute_errors': self.metrics['mean_abs_error'].tolist(),
                'mean_relative_errors': self.metrics['mean_rel_error'].tolist(),
                'parameter_names': self.param_names
            },
            'worst_predicted_parameters': [],
            'best_predicted_parameters': []
        }
        
        # Add physical trajectory metrics if available
        if hasattr(self, 'trajectory_deviations') and self.trajectory_deviations:
            report['physical_trajectory_metrics'] = {
                'valid_trajectory_pairs': len(self.valid_physical_indices),
                'mean_max_position_deviation': float(np.mean(self.max_position_deviations)),
                'std_max_position_deviation': float(np.std(self.max_position_deviations)),
                'mean_max_velocity_deviation': float(np.mean(self.max_velocity_deviations)),
                'std_max_velocity_deviation': float(np.std(self.max_velocity_deviations)),
                'mean_final_position_deviation': float(np.mean(self.final_position_deviations)),
                'std_final_position_deviation': float(np.std(self.final_position_deviations))
            }
        
        # Find worst and best predicted parameters
        sorted_indices = np.argsort(self.metrics['mean_abs_error'])
        report['worst_predicted_parameters'] = [
            {'parameter': self.param_names[i], 'mae': float(self.metrics['mean_abs_error'][i])}
            for i in sorted_indices[-5:]  # Top 5 worst
        ]
        report['best_predicted_parameters'] = [
            {'parameter': self.param_names[i], 'mae': float(self.metrics['mean_abs_error'][i])}
            for i in sorted_indices[:5]  # Top 5 best
        ]
        
        # Save report
        report_path = os.path.join(self.output_dir, 'trajectory_comparison_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable summary
        summary_path = os.path.join(self.output_dir, 'analysis_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("TRAJECTORY COMPARISON ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Total samples: {len(self.data)}\n")
            f.write(f"Feasible samples analyzed: {len(self.feasible_data)}\n")
            if hasattr(self, 'physical_data'):
                f.write(f"Physical trajectory samples: {len(self.physical_data)}\n")
            f.write("\n")
            
            f.write("OVERALL PREDICTION ACCURACY:\n")
            f.write(f"  Mean Absolute Error: {self.overall_metrics['overall_mae']:.6f}\n")
            f.write(f"  Mean Absolute Percentage Error: {self.overall_metrics['overall_mape']:.2f}%\n")
            f.write(f"  Correlation coefficient: {self.overall_metrics['correlation']:.4f}\n\n")
            
            f.write("WORST PREDICTED PARAMETERS:\n")
            for param in report['worst_predicted_parameters'][::-1]:  # Reverse for worst first
                f.write(f"  {param['parameter']}: MAE = {param['mae']:.6f}\n")
            
            f.write("\nBEST PREDICTED PARAMETERS:\n")
            for param in report['best_predicted_parameters']:
                f.write(f"  {param['parameter']}: MAE = {param['mae']:.6f}\n")
            
            # Add physical trajectory metrics if available
            if 'physical_trajectory_metrics' in report:
                phys_metrics = report['physical_trajectory_metrics']
                f.write("\nPHYSICAL TRAJECTORY DEVIATIONS:\n")
                f.write(f"  Valid trajectory pairs: {phys_metrics['valid_trajectory_pairs']}\n")
                f.write(f"  Mean max position deviation: {phys_metrics['mean_max_position_deviation']:.6f} DU\n")
                f.write(f"  Mean max velocity deviation: {phys_metrics['mean_max_velocity_deviation']:.6f} DU/TU\n")
                f.write(f"  Mean final position deviation: {phys_metrics['mean_final_position_deviation']:.6f} DU\n")
            
            f.write(f"\nAnalysis plots saved to: {self.output_dir}\n")
        
        print(f"Summary report saved to {summary_path}")
    
    def run_complete_analysis(self):
        """Run the complete trajectory comparison analysis."""
        print("Starting trajectory comparison analysis...")
        
        # Control parameter analysis
        self.extract_trajectory_data()
        self.compute_comparison_metrics()
        self.analyze_parameter_deviations()
        self.plot_parameter_deviation_analysis()
        self.analyze_manifold_and_time_deviations()
        self.plot_manifold_and_time_analysis()
        self.analyze_feasibility_vs_halo_energy()
        self.plot_feasibility_analysis()
        self.plot_error_distribution()
        self.plot_control_segments_comparison()
        
        # Physical trajectory analysis (if data available)
        if hasattr(self, 'physical_data') and self.physical_data:
            print("\nStarting physical trajectory analysis...")
            self.extract_physical_trajectory_data()
            if hasattr(self, 'valid_physical_indices') and self.valid_physical_indices:
                self.compute_trajectory_deviations()
                self.compute_aggregate_trajectory_statistics()
                self.plot_aggregate_trajectory_analysis()
                print("Physical trajectory analysis complete!")
            else:
                print("No valid physical trajectory pairs found")
        else:
            print("No physical trajectory data available")
        
        self.create_summary_report()
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        print(f"Key plots created:")
        print(f"  - error_distribution.png: Error distribution analysis")
        print(f"  - control_segments_comparison.png: Control profile comparison")
        if hasattr(self, 'valid_physical_indices') and self.valid_physical_indices:
            print(f"  - aggregate_trajectory_analysis.png: Aggregate trajectory analysis")
        print(f"  - analysis_summary.txt: Human-readable summary")

    def analyze_parameter_deviations(self):
        """Analyze which parameters have the greatest deviation from converged trajectories."""
        print("Analyzing parameter-wise deviations...")
        
        # Compute deviation statistics for each parameter
        param_deviations = {
            'mean_abs_error': self.metrics['mean_abs_error'],
            'std_abs_error': self.metrics['std_abs_error'],
            'mean_rel_error': self.metrics['mean_rel_error'],
            'max_abs_error': self.metrics['max_abs_error'],
            'parameter_names': self.param_names
        }
        
        # Rank parameters by different metrics
        param_ranking = {}
        
        # Rank by mean absolute error
        mean_abs_ranking = np.argsort(param_deviations['mean_abs_error'])[::-1]  # Worst first
        param_ranking['by_mean_abs_error'] = [
            {'parameter': self.param_names[i], 'value': param_deviations['mean_abs_error'][i]}
            for i in mean_abs_ranking
        ]
        
        # Rank by mean relative error
        mean_rel_ranking = np.argsort(param_deviations['mean_rel_error'])[::-1]  # Worst first
        param_ranking['by_mean_rel_error'] = [
            {'parameter': self.param_names[i], 'value': param_deviations['mean_rel_error'][i]}
            for i in mean_rel_ranking
        ]
        
        # Rank by max absolute error
        max_abs_ranking = np.argsort(param_deviations['max_abs_error'])[::-1]  # Worst first
        param_ranking['by_max_abs_error'] = [
            {'parameter': self.param_names[i], 'value': param_deviations['max_abs_error'][i]}
            for i in max_abs_ranking
        ]
        
        # Group parameters by category
        time_params = [0, 1, 2]  # shooting_time, initial_coast, final_coast
        control_params = list(range(3, 63))  # 20 segments × 3 components
        final_params = [63, 64, 65]  # final_fuel_mass, halo_period, manifold_length
        
        # Analyze deviations by parameter category
        category_deviations = {
            'time_parameters': {
                'mean_abs_error': np.mean(param_deviations['mean_abs_error'][time_params]),
                'mean_rel_error': np.mean(param_deviations['mean_rel_error'][time_params]),
                'parameters': [self.param_names[i] for i in time_params]
            },
            'control_parameters': {
                'mean_abs_error': np.mean(param_deviations['mean_abs_error'][control_params]),
                'mean_rel_error': np.mean(param_deviations['mean_rel_error'][control_params]),
                'parameters': [self.param_names[i] for i in control_params]
            },
            'final_parameters': {
                'mean_abs_error': np.mean(param_deviations['mean_abs_error'][final_params]),
                'mean_rel_error': np.mean(param_deviations['mean_rel_error'][final_params]),
                'parameters': [self.param_names[i] for i in final_params]
            }
        }
        
        self.param_deviations = param_deviations
        self.param_ranking = param_ranking
        self.category_deviations = category_deviations
        
        print(f"Parameter deviation analysis complete!")
        print(f"Top 5 worst parameters by mean absolute error:")
        for i, param in enumerate(param_ranking['by_mean_abs_error'][:5]):
            print(f"  {i+1}. {param['parameter']}: {param['value']:.6f}")
        
        print(f"\nCategory-wise mean absolute errors:")
        print(f"  Time parameters: {category_deviations['time_parameters']['mean_abs_error']:.6f}")
        print(f"  Control parameters: {category_deviations['control_parameters']['mean_abs_error']:.6f}")
        print(f"  Final parameters: {category_deviations['final_parameters']['mean_abs_error']:.6f}")

    def plot_parameter_deviation_analysis(self):
        """Create comprehensive plots for parameter deviation analysis."""
        print("Creating parameter deviation analysis plots...")
        
        if not hasattr(self, 'param_deviations'):
            print("No parameter deviation data available. Run analyze_parameter_deviations() first.")
            return
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Top 20 worst parameters by mean absolute error
        ax1 = fig.add_subplot(3, 3, 1)
        top_20_worst = self.param_ranking['by_mean_abs_error'][:20]
        param_names = [p['parameter'] for p in top_20_worst]
        param_values = [p['value'] for p in top_20_worst]
        
        bars = ax1.barh(range(len(param_names)), param_values, color='red', alpha=0.7)
        ax1.set_yticks(range(len(param_names)))
        ax1.set_yticklabels(param_names, fontsize=8)
        ax1.set_xlabel('Mean Absolute Error')
        ax1.set_title('Top 20 Worst Parameters\n(by Mean Absolute Error)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Top 20 worst parameters by mean relative error
        ax2 = fig.add_subplot(3, 3, 2)
        top_20_rel_worst = self.param_ranking['by_mean_rel_error'][:20]
        param_names_rel = [p['parameter'] for p in top_20_rel_worst]
        param_values_rel = [p['value'] for p in top_20_rel_worst]
        
        bars = ax2.barh(range(len(param_names_rel)), param_values_rel, color='orange', alpha=0.7)
        ax2.set_yticks(range(len(param_names_rel)))
        ax2.set_yticklabels(param_names_rel, fontsize=8)
        ax2.set_xlabel('Mean Relative Error')
        ax2.set_title('Top 20 Worst Parameters\n(by Mean Relative Error)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter deviation heatmap
        ax3 = fig.add_subplot(3, 3, 3)
        # Create a heatmap showing deviation for each parameter across all samples
        deviation_matrix = self.abs_differences  # [n_samples, n_parameters]
        
        # Normalize for better visualization
        normalized_deviations = (deviation_matrix - np.mean(deviation_matrix, axis=0)) / (np.std(deviation_matrix, axis=0) + 1e-10)
        
        im = ax3.imshow(normalized_deviations.T, cmap='viridis', aspect='auto')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Parameter Index')
        ax3.set_title('Normalized Parameter Deviations\n(Heatmap)')
        plt.colorbar(im, ax=ax3, label='Normalized Deviation')
        
        # 4. Category-wise comparison
        ax4 = fig.add_subplot(3, 3, 4)
        categories = ['Time Parameters', 'Control Parameters', 'Final Parameters']
        category_means = [
            self.category_deviations['time_parameters']['mean_abs_error'],
            self.category_deviations['control_parameters']['mean_abs_error'],
            self.category_deviations['final_parameters']['mean_abs_error']
        ]
        category_stds = [
            np.std(self.abs_differences[:, [0, 1, 2]]),
            np.std(self.abs_differences[:, 3:63]),
            np.std(self.abs_differences[:, [63, 64, 65]])
        ]
        
        bars = ax4.bar(categories, category_means, yerr=category_stds, capsize=5, alpha=0.7)
        ax4.set_ylabel('Mean Absolute Error')
        ax4.set_title('Deviation by Parameter Category')
        ax4.grid(True, alpha=0.3)
        
        # 5. Parameter correlation with halo energy
        ax5 = fig.add_subplot(3, 3, 5)
        correlations = []
        for i in range(len(self.param_names)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    corr = np.corrcoef(self.halo_energies, self.abs_differences[:, i])[0, 1]
                    if not np.isfinite(corr):
                        corr = 0.0
                except:
                    corr = 0.0
            correlations.append(abs(corr))
        
        # Show top 15 parameters with highest correlation
        top_corr_indices = np.argsort(correlations)[-15:]
        top_corr_params = [self.param_names[i] for i in top_corr_indices]
        top_corr_values = [correlations[i] for i in top_corr_indices]
        
        bars = ax5.barh(range(len(top_corr_params)), top_corr_values, color='green', alpha=0.7)
        ax5.set_yticks(range(len(top_corr_params)))
        ax5.set_yticklabels(top_corr_params, fontsize=8)
        ax5.set_xlabel('|Correlation with Halo Energy|')
        ax5.set_title('Parameters Most Correlated\nwith Halo Energy')
        ax5.grid(True, alpha=0.3)
        
        # 6. Control segment analysis
        ax6 = fig.add_subplot(3, 3, 6)
        # Analyze control segments (20 segments, 3 components each)
        control_deviations = self.abs_differences[:, 3:63].reshape(-1, 20, 3)
        mean_control_dev = np.mean(control_deviations, axis=0)  # [20, 3]
        
        segments = np.arange(1, 21)
        ax6.plot(segments, mean_control_dev[:, 0], 'b-o', label='Alpha', linewidth=2, markersize=6)
        ax6.plot(segments, mean_control_dev[:, 1], 'r-s', label='Beta', linewidth=2, markersize=6)
        ax6.plot(segments, mean_control_dev[:, 2], 'g-^', label='Thrust', linewidth=2, markersize=6)
        ax6.set_xlabel('Control Segment')
        ax6.set_ylabel('Mean Absolute Error')
        ax6.set_title('Control Parameter Deviations\nby Segment')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Statistical summary table
        ax7 = fig.add_subplot(3, 3, 7)
        ax7.axis('off')
        
        # Get top 10 worst parameters
        top_10_worst = self.param_ranking['by_mean_abs_error'][:10]
        
        summary_text = f"""PARAMETER DEVIATION SUMMARY

Top 10 Worst Parameters:
"""
        for i, param in enumerate(top_10_worst):
            summary_text += f"  {i+1}. {param['parameter']}: {param['value']:.6f}\n"
        
        summary_text += f"""
Category Statistics:
  Time Parameters: {self.category_deviations['time_parameters']['mean_abs_error']:.6f}
  Control Parameters: {self.category_deviations['control_parameters']['mean_abs_error']:.6f}
  Final Parameters: {self.category_deviations['final_parameters']['mean_abs_error']:.6f}

Overall Statistics:
  Mean MAE: {np.mean(self.abs_differences):.6f}
  Std MAE: {np.std(self.abs_differences):.6f}
  Max MAE: {np.max(self.abs_differences):.6f}"""
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 8. Deviation distribution by parameter type
        ax8 = fig.add_subplot(3, 3, 8)
        time_deviations = self.abs_differences[:, [0, 1, 2]].flatten()
        control_deviations_flat = self.abs_differences[:, 3:63].flatten()
        final_deviations = self.abs_differences[:, [63, 64, 65]].flatten()
        
        ax8.hist(time_deviations, bins=30, alpha=0.7, label='Time', density=True, color='blue')
        ax8.hist(control_deviations_flat, bins=30, alpha=0.7, label='Control', density=True, color='red')
        ax8.hist(final_deviations, bins=30, alpha=0.7, label='Final', density=True, color='green')
        ax8.set_xlabel('Absolute Error')
        ax8.set_ylabel('Density')
        ax8.set_title('Deviation Distribution\nby Parameter Type')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Error growth analysis by parameter
        ax9 = fig.add_subplot(3, 3, 9)
        # Analyze how errors grow with halo energy for different parameter categories
        ax9.scatter(self.halo_energies, np.mean(self.abs_differences[:, [0, 1, 2]], axis=1), 
                   alpha=0.7, s=60, c='blue', label='Time Parameters')
        ax9.scatter(self.halo_energies, np.mean(self.abs_differences[:, 3:63], axis=1), 
                   alpha=0.7, s=60, c='red', label='Control Parameters')
        ax9.scatter(self.halo_energies, np.mean(self.abs_differences[:, [63, 64, 65]], axis=1), 
                   alpha=0.7, s=60, c='green', label='Final Parameters')
        ax9.set_xlabel('Halo Energy')
        ax9.set_ylabel('Mean Absolute Error')
        ax9.set_title('Error Growth by Parameter Category')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_deviation_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed statistics to JSON
        stats_path = os.path.join(self.output_dir, 'parameter_deviation_statistics.json')
        with open(stats_path, 'w') as f:
            json_stats = {
                'parameter_deviations': {
                    'mean_abs_error': self.param_deviations['mean_abs_error'].tolist(),
                    'std_abs_error': self.param_deviations['std_abs_error'].tolist(),
                    'mean_rel_error': self.param_deviations['mean_rel_error'].tolist(),
                    'max_abs_error': self.param_deviations['max_abs_error'].tolist(),
                    'parameter_names': self.param_names
                },
                'parameter_ranking': self.param_ranking,
                'category_deviations': self.category_deviations
            }
            json.dump(json_stats, f, indent=2)
        
        print(f"Parameter deviation analysis saved to: {self.output_dir}")
        print(f"  - parameter_deviation_analysis.png: Comprehensive parameter analysis plots")
        print(f"  - parameter_deviation_statistics.json: Detailed parameter statistics")

    def analyze_manifold_and_time_deviations(self):
        """Analyze deviations specifically for manifold and time-related parameters."""
        print("Analyzing manifold and time-specific deviations...")
        
        # Extract specific parameters
        manifold_length_idx = 65  # Manifold Length parameter
        shooting_time_idx = 0     # Shooting Time parameter
        initial_coast_idx = 1     # Initial Coast parameter
        final_coast_idx = 2       # Final Coast parameter
        
        # Analyze manifold length deviations
        manifold_deviations = self.abs_differences[:, manifold_length_idx]
        manifold_rel_deviations = self.rel_differences[:, manifold_length_idx]
        
        # Analyze time parameter deviations
        time_deviations = self.abs_differences[:, [shooting_time_idx, initial_coast_idx, final_coast_idx]]
        time_rel_deviations = self.rel_differences[:, [shooting_time_idx, initial_coast_idx, final_coast_idx]]
        
        # Compute statistics
        manifold_stats = {
            'mean_abs_error': np.mean(manifold_deviations),
            'std_abs_error': np.std(manifold_deviations),
            'mean_rel_error': np.mean(manifold_rel_deviations),
            'max_abs_error': np.max(manifold_deviations),
            'min_abs_error': np.min(manifold_deviations),
            'correlation_with_halo_energy': np.corrcoef(self.halo_energies, manifold_deviations)[0, 1] if len(self.halo_energies) > 1 else 0.0
        }
        
        time_stats = {
            'shooting_time': {
                'mean_abs_error': np.mean(time_deviations[:, 0]),
                'std_abs_error': np.std(time_deviations[:, 0]),
                'mean_rel_error': np.mean(time_rel_deviations[:, 0]),
                'correlation_with_halo_energy': np.corrcoef(self.halo_energies, time_deviations[:, 0])[0, 1] if len(self.halo_energies) > 1 else 0.0
            },
            'initial_coast': {
                'mean_abs_error': np.mean(time_deviations[:, 1]),
                'std_abs_error': np.std(time_deviations[:, 1]),
                'mean_rel_error': np.mean(time_rel_deviations[:, 1]),
                'correlation_with_halo_energy': np.corrcoef(self.halo_energies, time_deviations[:, 1])[0, 1] if len(self.halo_energies) > 1 else 0.0
            },
            'final_coast': {
                'mean_abs_error': np.mean(time_deviations[:, 2]),
                'std_abs_error': np.std(time_deviations[:, 2]),
                'mean_rel_error': np.mean(time_rel_deviations[:, 2]),
                'correlation_with_halo_energy': np.corrcoef(self.halo_energies, time_deviations[:, 2])[0, 1] if len(self.halo_energies) > 1 else 0.0
            }
        }
        
        # Analyze control segment patterns
        control_deviations = self.abs_differences[:, 3:63].reshape(-1, 20, 3)
        
        # Find which control segments have highest deviations
        mean_segment_deviations = np.mean(control_deviations, axis=(0, 2))  # Average across samples and components
        worst_segments = np.argsort(mean_segment_deviations)[::-1]  # Worst first
        
        segment_analysis = {
            'worst_segments': [
                {'segment': i+1, 'mean_deviation': mean_segment_deviations[i]}
                for i in worst_segments
            ],
            'best_segments': [
                {'segment': i+1, 'mean_deviation': mean_segment_deviations[i]}
                for i in worst_segments[::-1]  # Best first
            ],
            'segment_statistics': {
                'mean_deviation': np.mean(mean_segment_deviations),
                'std_deviation': np.std(mean_segment_deviations),
                'max_deviation': np.max(mean_segment_deviations),
                'min_deviation': np.min(mean_segment_deviations)
            }
        }
        
        self.manifold_stats = manifold_stats
        self.time_stats = time_stats
        self.segment_analysis = segment_analysis
        
        print(f"Manifold and time deviation analysis complete!")
        print(f"Manifold Length - Mean MAE: {manifold_stats['mean_abs_error']:.6f}")
        print(f"Shooting Time - Mean MAE: {time_stats['shooting_time']['mean_abs_error']:.6f}")
        print(f"Worst control segment: {segment_analysis['worst_segments'][0]['segment']} "
              f"(MAE: {segment_analysis['worst_segments'][0]['mean_deviation']:.6f})")

    def plot_manifold_and_time_analysis(self):
        """Create plots for manifold and time-specific deviation analysis."""
        print("Creating manifold and time analysis plots...")
        
        if not hasattr(self, 'manifold_stats'):
            print("No manifold/time statistics available. Run analyze_manifold_and_time_deviations() first.")
            return
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Manifold length deviation analysis
        ax1 = fig.add_subplot(2, 4, 1)
        manifold_length_idx = 65
        manifold_deviations = self.abs_differences[:, manifold_length_idx]
        
        ax1.hist(manifold_deviations, bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(self.manifold_stats['mean_abs_error'], color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {self.manifold_stats["mean_abs_error"]:.6f}')
        ax1.set_xlabel('Manifold Length Deviation')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Manifold Length Deviation\nDistribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Manifold deviation vs halo energy
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.scatter(self.halo_energies, manifold_deviations, alpha=0.7, s=60)
        ax2.set_xlabel('Halo Energy')
        ax2.set_ylabel('Manifold Length Deviation')
        ax2.set_title('Manifold Deviation vs\nHalo Energy')
        ax2.grid(True, alpha=0.3)
        
        # 3. Time parameter deviations
        ax3 = fig.add_subplot(2, 4, 3)
        time_params = ['Shooting Time', 'Initial Coast', 'Final Coast']
        time_means = [
            self.time_stats['shooting_time']['mean_abs_error'],
            self.time_stats['initial_coast']['mean_abs_error'],
            self.time_stats['final_coast']['mean_abs_error']
        ]
        time_stds = [
            self.time_stats['shooting_time']['std_abs_error'],
            self.time_stats['initial_coast']['std_abs_error'],
            self.time_stats['final_coast']['std_abs_error']
        ]
        
        bars = ax3.bar(time_params, time_means, yerr=time_stds, capsize=5, alpha=0.7)
        ax3.set_ylabel('Mean Absolute Error')
        ax3.set_title('Time Parameter Deviations')
        ax3.grid(True, alpha=0.3)
        
        # 4. Control segment deviation ranking
        ax4 = fig.add_subplot(2, 4, 4)
        top_10_worst_segments = self.segment_analysis['worst_segments'][:10]
        segment_numbers = [seg['segment'] for seg in top_10_worst_segments]
        segment_deviations = [seg['mean_deviation'] for seg in top_10_worst_segments]
        
        bars = ax4.barh(range(len(segment_numbers)), segment_deviations, color='red', alpha=0.7)
        ax4.set_yticks(range(len(segment_numbers)))
        ax4.set_yticklabels([f'Seg {num}' for num in segment_numbers])
        ax4.set_xlabel('Mean Absolute Error')
        ax4.set_title('Top 10 Worst Control\nSegments')
        ax4.grid(True, alpha=0.3)
        
        # 5. Time parameter correlation with halo energy
        ax5 = fig.add_subplot(2, 4, 5)
        time_correlations = [
            self.time_stats['shooting_time']['correlation_with_halo_energy'],
            self.time_stats['initial_coast']['correlation_with_halo_energy'],
            self.time_stats['final_coast']['correlation_with_halo_energy']
        ]
        
        bars = ax5.bar(time_params, time_correlations, alpha=0.7)
        ax5.set_ylabel('Correlation with Halo Energy')
        ax5.set_title('Time Parameter Correlations\nwith Halo Energy')
        ax5.grid(True, alpha=0.3)
        
        # 6. Control segment deviation pattern
        ax6 = fig.add_subplot(2, 4, 6)
        control_deviations = self.abs_differences[:, 3:63].reshape(-1, 20, 3)
        mean_control_dev = np.mean(control_deviations, axis=0)  # [20, 3]
        
        segments = np.arange(1, 21)
        ax6.plot(segments, mean_control_dev[:, 0], 'b-o', label='Alpha', linewidth=2, markersize=6)
        ax6.plot(segments, mean_control_dev[:, 1], 'r-s', label='Beta', linewidth=2, markersize=6)
        ax6.plot(segments, mean_control_dev[:, 2], 'g-^', label='Thrust', linewidth=2, markersize=6)
        ax6.set_xlabel('Control Segment')
        ax6.set_ylabel('Mean Absolute Error')
        ax6.set_title('Control Parameter Deviations\nby Segment')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Statistical summary
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.axis('off')
        
        summary_text = f"""MANIFOLD & TIME ANALYSIS

Manifold Length:
  Mean MAE: {self.manifold_stats['mean_abs_error']:.6f}
  Std MAE:  {self.manifold_stats['std_abs_error']:.6f}
  Max MAE:  {self.manifold_stats['max_abs_error']:.6f}
  Corr with Halo Energy: {self.manifold_stats['correlation_with_halo_energy']:.4f}

Time Parameters:
  Shooting Time MAE: {self.time_stats['shooting_time']['mean_abs_error']:.6f}
  Initial Coast MAE: {self.time_stats['initial_coast']['mean_abs_error']:.6f}
  Final Coast MAE: {self.time_stats['final_coast']['mean_abs_error']:.6f}

Control Segments:
  Mean Segment MAE: {self.segment_analysis['segment_statistics']['mean_deviation']:.6f}
  Worst Segment: {self.segment_analysis['worst_segments'][0]['segment']}
  Best Segment: {self.segment_analysis['best_segments'][0]['segment']}"""
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 8. Deviation heatmap for time and manifold parameters
        ax8 = fig.add_subplot(2, 4, 8)
        # Focus on time and manifold parameters
        time_manifold_deviations = self.abs_differences[:, [0, 1, 2, 65]]  # shooting, initial_coast, final_coast, manifold_length
        param_labels = ['Shooting Time', 'Initial Coast', 'Final Coast', 'Manifold Length']
        
        im = ax8.imshow(time_manifold_deviations.T, cmap='viridis', aspect='auto')
        ax8.set_xlabel('Sample Index')
        ax8.set_yticks(range(len(param_labels)))
        ax8.set_yticklabels(param_labels)
        ax8.set_title('Time & Manifold Parameter\nDeviations (Heatmap)')
        plt.colorbar(im, ax=ax8, label='Absolute Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'manifold_time_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed statistics to JSON
        stats_path = os.path.join(self.output_dir, 'manifold_time_statistics.json')
        with open(stats_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_stats = {
                'manifold_stats': {},
                'time_stats': {},
                'segment_analysis': {}
            }
            
            # Convert manifold_stats
            for key, value in self.manifold_stats.items():
                if isinstance(value, (np.integer, np.floating)):
                    json_stats['manifold_stats'][key] = float(value)
                else:
                    json_stats['manifold_stats'][key] = value
            
            # Convert time_stats
            for time_param, stats in self.time_stats.items():
                json_stats['time_stats'][time_param] = {}
                for key, value in stats.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_stats['time_stats'][time_param][key] = float(value)
                    else:
                        json_stats['time_stats'][time_param][key] = value
            
            # Convert segment_analysis
            json_stats['segment_analysis']['worst_segments'] = []
            for seg in self.segment_analysis['worst_segments']:
                json_stats['segment_analysis']['worst_segments'].append({
                    'segment': int(seg['segment']),
                    'mean_deviation': float(seg['mean_deviation'])
                })
            
            json_stats['segment_analysis']['best_segments'] = []
            for seg in self.segment_analysis['best_segments']:
                json_stats['segment_analysis']['best_segments'].append({
                    'segment': int(seg['segment']),
                    'mean_deviation': float(seg['mean_deviation'])
                })
            
            json_stats['segment_analysis']['segment_statistics'] = {}
            for key, value in self.segment_analysis['segment_statistics'].items():
                if isinstance(value, (np.integer, np.floating)):
                    json_stats['segment_analysis']['segment_statistics'][key] = float(value)
                else:
                    json_stats['segment_analysis']['segment_statistics'][key] = value
            
            json.dump(json_stats, f, indent=2)
        
        print(f"Manifold and time analysis saved to: {self.output_dir}")
        print(f"  - manifold_time_analysis.png: Manifold and time analysis plots")
        print(f"  - manifold_time_statistics.json: Detailed manifold/time statistics")

    def analyze_feasibility_vs_halo_energy(self):
        """Analyze the correlation between halo energy and trajectory feasibility."""
        print("Analyzing feasibility vs halo energy correlation...")
        
        # Extract halo energies and feasibility for all samples
        all_halo_energies = []
        all_feasibility = []
        
        for entry in self.data:
            halo_energy = entry['generated_sample']['halo_energy']
            feasibility = entry['snopt_results']['feasibility']
            
            all_halo_energies.append(halo_energy)
            all_feasibility.append(feasibility)
        
        all_halo_energies = np.array(all_halo_energies)
        all_feasibility = np.array(all_feasibility)
        
        # Compute statistics
        feasible_energies = all_halo_energies[all_feasibility]
        infeasible_energies = all_halo_energies[~all_feasibility]
        
        feasibility_stats = {
            'total_samples': len(all_halo_energies),
            'feasible_count': np.sum(all_feasibility),
            'infeasible_count': np.sum(~all_feasibility),
            'feasibility_rate': np.mean(all_feasibility),
            'feasible_energies': {
                'mean': np.mean(feasible_energies) if len(feasible_energies) > 0 else 0,
                'std': np.std(feasible_energies) if len(feasible_energies) > 0 else 0,
                'min': np.min(feasible_energies) if len(feasible_energies) > 0 else 0,
                'max': np.max(feasible_energies) if len(feasible_energies) > 0 else 0,
                'values': feasible_energies.tolist()
            },
            'infeasible_energies': {
                'mean': np.mean(infeasible_energies) if len(infeasible_energies) > 0 else 0,
                'std': np.std(infeasible_energies) if len(infeasible_energies) > 0 else 0,
                'min': np.min(infeasible_energies) if len(infeasible_energies) > 0 else 0,
                'max': np.max(infeasible_energies) if len(infeasible_energies) > 0 else 0,
                'values': infeasible_energies.tolist()
            }
        }
        
        # Compute correlation between halo energy and feasibility
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                correlation = np.corrcoef(all_halo_energies, all_feasibility.astype(float))[0, 1]
                if not np.isfinite(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
        
        feasibility_stats['correlation'] = correlation
        
        self.feasibility_stats = feasibility_stats
        
        print(f"Feasibility analysis complete!")
        print(f"Total samples: {feasibility_stats['total_samples']}")
        print(f"Feasible: {feasibility_stats['feasible_count']} ({feasibility_stats['feasibility_rate']*100:.1f}%)")
        print(f"Infeasible: {feasibility_stats['infeasible_count']} ({(1-feasibility_stats['feasibility_rate'])*100:.1f}%)")
        print(f"Correlation with halo energy: {correlation:.4f}")
        
        if len(feasible_energies) > 0:
            print(f"Feasible halo energy - Mean: {feasibility_stats['feasible_energies']['mean']:.6f}")
        if len(infeasible_energies) > 0:
            print(f"Infeasible halo energy - Mean: {feasibility_stats['infeasible_energies']['mean']:.6f}")

    def plot_feasibility_analysis(self):
        """Create plots for feasibility vs halo energy analysis."""
        print("Creating feasibility analysis plots...")
        
        if not hasattr(self, 'feasibility_stats'):
            print("No feasibility statistics available. Run analyze_feasibility_vs_halo_energy() first.")
            return
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Feasibility rate vs halo energy (scatter plot)
        ax1 = fig.add_subplot(2, 4, 1)
        all_halo_energies = []
        all_feasibility = []
        
        for entry in self.data:
            halo_energy = entry['generated_sample']['halo_energy']
            feasibility = entry['snopt_results']['feasibility']
            all_halo_energies.append(halo_energy)
            all_feasibility.append(feasibility)
        
        all_halo_energies = np.array(all_halo_energies)
        all_feasibility = np.array(all_feasibility)
        
        # Color code by feasibility
        colors = ['red' if not feasible else 'green' for feasible in all_feasibility]
        alpha_values = [0.7 if feasible else 0.5 for feasible in all_feasibility]
        
        ax1.scatter(all_halo_energies, all_feasibility, c=colors, alpha=alpha_values, s=60)
        ax1.set_xlabel('Halo Energy')
        ax1.set_ylabel('Feasibility (0=Infeasible, 1=Feasible)')
        ax1.set_title('Feasibility vs Halo Energy\n(Green=Feasible, Red=Infeasible)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution of halo energies by feasibility
        ax2 = fig.add_subplot(2, 4, 2)
        feasible_energies = all_halo_energies[all_feasibility]
        infeasible_energies = all_halo_energies[~all_feasibility]
        
        if len(feasible_energies) > 0:
            ax2.hist(feasible_energies, bins=20, alpha=0.7, color='green', label='Feasible', density=True)
        if len(infeasible_energies) > 0:
            ax2.hist(infeasible_energies, bins=20, alpha=0.7, color='red', label='Infeasible', density=True)
        
        ax2.set_xlabel('Halo Energy')
        ax2.set_ylabel('Density')
        ax2.set_title('Halo Energy Distribution\nby Feasibility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feasibility rate by energy bins
        ax3 = fig.add_subplot(2, 4, 3)
        # Create energy bins and compute feasibility rate in each bin
        energy_bins = np.linspace(np.min(all_halo_energies), np.max(all_halo_energies), 10)
        bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
        feasibility_rates = []
        bin_counts = []
        
        for i in range(len(energy_bins) - 1):
            mask = (all_halo_energies >= energy_bins[i]) & (all_halo_energies < energy_bins[i+1])
            if np.sum(mask) > 0:
                rate = np.mean(all_feasibility[mask])
                count = np.sum(mask)
                feasibility_rates.append(rate)
                bin_counts.append(count)
            else:
                feasibility_rates.append(0)
                bin_counts.append(0)
        
        ax3.bar(bin_centers, feasibility_rates, width=energy_bins[1]-energy_bins[0], alpha=0.7, color='blue')
        ax3.set_xlabel('Halo Energy')
        ax3.set_ylabel('Feasibility Rate')
        ax3.set_title('Feasibility Rate by\nEnergy Bins')
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot of energy distributions
        ax4 = fig.add_subplot(2, 4, 4)
        data_to_plot = []
        labels = []
        
        if len(feasible_energies) > 0:
            data_to_plot.append(feasible_energies)
            labels.append('Feasible')
        if len(infeasible_energies) > 0:
            data_to_plot.append(infeasible_energies)
            labels.append('Infeasible')
        
        if data_to_plot:
            bp = ax4.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            colors = ['lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            ax4.set_ylabel('Halo Energy')
            ax4.set_title('Halo Energy Distribution\n(Box Plot)')
            ax4.grid(True, alpha=0.3)
        
        # 5. Statistical summary
        ax5 = fig.add_subplot(2, 4, 5)
        ax5.axis('off')
        
        stats = self.feasibility_stats
        # Calculate feasibility ratio
        feasibility_ratio = stats['feasible_count'] / stats['infeasible_count'] if stats['infeasible_count'] > 0 else float('inf')
        feasibility_ratio_str = f"{feasibility_ratio:.2f}" if np.isfinite(feasibility_ratio) else "∞"
        
        summary_text = f"""FEASIBILITY ANALYSIS SUMMARY

Overall Statistics:
  Total Samples: {stats['total_samples']}
  Feasible: {stats['feasible_count']} ({stats['feasibility_rate']*100:.1f}%)
  Infeasible: {stats['infeasible_count']} ({(1-stats['feasibility_rate'])*100:.1f}%)
  Feasibility Ratio: {feasibility_ratio_str} (Feasible:Infeasible)
  Correlation with Halo Energy: {stats['correlation']:.4f}

Feasible Trajectories:
  Count: {stats['feasible_count']}
  Mean Energy: {stats['feasible_energies']['mean']:.6f}
  Std Energy: {stats['feasible_energies']['std']:.6f}
  Energy Range: {stats['feasible_energies']['min']:.6f} - {stats['feasible_energies']['max']:.6f}

Infeasible Trajectories:
  Count: {stats['infeasible_count']}
  Mean Energy: {stats['infeasible_energies']['mean']:.6f}
  Std Energy: {stats['infeasible_energies']['std']:.6f}
  Energy Range: {stats['infeasible_energies']['min']:.6f} - {stats['infeasible_energies']['max']:.6f}"""
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 6. Cumulative distribution functions
        ax6 = fig.add_subplot(2, 4, 6)
        if len(feasible_energies) > 0 and len(infeasible_energies) > 0:
            feasible_sorted = np.sort(feasible_energies)
            infeasible_sorted = np.sort(infeasible_energies)
            
            cdf_feasible = np.arange(1, len(feasible_sorted) + 1) / len(feasible_sorted)
            cdf_infeasible = np.arange(1, len(infeasible_sorted) + 1) / len(infeasible_sorted)
            
            ax6.plot(feasible_sorted, cdf_feasible, 'g-', linewidth=2, label='Feasible')
            ax6.plot(infeasible_sorted, cdf_infeasible, 'r-', linewidth=2, label='Infeasible')
            ax6.set_xlabel('Halo Energy')
            ax6.set_ylabel('Cumulative Probability')
            ax6.set_title('Cumulative Distribution\nFunctions')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Energy range analysis
        ax7 = fig.add_subplot(2, 4, 7)
        # Show the energy ranges where feasibility changes
        energy_range = np.linspace(np.min(all_halo_energies), np.max(all_halo_energies), 50)
        feasibility_by_energy = []
        
        for energy in energy_range:
            # Find trajectories within a small energy window
            window = 0.01  # Small energy window
            mask = (all_halo_energies >= energy - window) & (all_halo_energies <= energy + window)
            if np.sum(mask) > 0:
                rate = np.mean(all_feasibility[mask])
                feasibility_by_energy.append(rate)
            else:
                feasibility_by_energy.append(0)
        
        ax7.plot(energy_range, feasibility_by_energy, 'b-', linewidth=2)
        ax7.set_xlabel('Halo Energy')
        ax7.set_ylabel('Feasibility Rate')
        ax7.set_title('Feasibility Rate vs\nHalo Energy (Smoothed)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Energy threshold analysis
        ax8 = fig.add_subplot(2, 4, 8)
        # Find the energy threshold that best separates feasible from infeasible
        if len(feasible_energies) > 0 and len(infeasible_energies) > 0:
            # Use ROC analysis to find optimal threshold
            from sklearn.metrics import roc_curve, auc
            
            # Create binary classification data
            y_true = all_feasibility
            y_score = all_halo_energies  # Use halo energy as score
            
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Find optimal threshold (closest to top-left corner)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            ax8.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
            ax8.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax8.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
                       label=f'Optimal Threshold: {optimal_threshold:.4f}')
            ax8.set_xlabel('False Positive Rate')
            ax8.set_ylabel('True Positive Rate')
            ax8.set_title('ROC Curve for\nFeasibility Prediction')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feasibility_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed statistics to JSON
        stats_path = os.path.join(self.output_dir, 'feasibility_statistics.json')
        with open(stats_path, 'w') as f:
            json_stats = {
                'feasibility_stats': {
                    'total_samples': int(self.feasibility_stats['total_samples']),
                    'feasible_count': int(self.feasibility_stats['feasible_count']),
                    'infeasible_count': int(self.feasibility_stats['infeasible_count']),
                    'feasibility_rate': float(self.feasibility_stats['feasibility_rate']),
                    'correlation': float(self.feasibility_stats['correlation'])
                },
                'feasible_energies': {
                    'mean': float(self.feasibility_stats['feasible_energies']['mean']),
                    'std': float(self.feasibility_stats['feasible_energies']['std']),
                    'min': float(self.feasibility_stats['feasible_energies']['min']),
                    'max': float(self.feasibility_stats['feasible_energies']['max']),
                    'values': [float(x) for x in self.feasibility_stats['feasible_energies']['values']]
                },
                'infeasible_energies': {
                    'mean': float(self.feasibility_stats['infeasible_energies']['mean']),
                    'std': float(self.feasibility_stats['infeasible_energies']['std']),
                    'min': float(self.feasibility_stats['infeasible_energies']['min']),
                    'max': float(self.feasibility_stats['infeasible_energies']['max']),
                    'values': [float(x) for x in self.feasibility_stats['infeasible_energies']['values']]
                }
            }
            json.dump(json_stats, f, indent=2)
        
        print(f"Feasibility analysis saved to: {self.output_dir}")
        print(f"  - feasibility_analysis.png: Feasibility vs halo energy analysis plots")
        print(f"  - feasibility_statistics.json: Detailed feasibility statistics")


def main():
    """Main function to run trajectory comparison analysis."""
    parser = argparse.ArgumentParser(description='Analyze trajectory predictions vs converged solutions')
    parser.add_argument('dataset_path', type=str, help='Path to comprehensive dataset pickle file')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for plots and analysis (default: auto-generated)')
    parser.add_argument('--feasible_only', action='store_true', default=True,
                       help='Analyze only feasible trajectories (default: True)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found: {args.dataset_path}")
        return
    
    # Run analysis
    analyzer = TrajectoryComparisonAnalyzer(args.dataset_path, args.output_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()