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
    
    def plot_parameter_comparison(self):
        """Create parameter-by-parameter comparison plots."""
        print("Creating parameter comparison plots...")
        
        n_params = len(self.param_names)
        n_cols = 4
        n_rows = int(np.ceil(n_params / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i in range(n_params):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(self.predicted_trajectories[:, i], 
                      self.converged_trajectories[:, i],
                      alpha=0.7, s=60, c=self.halo_energies, cmap='viridis')
            
            # Perfect prediction line
            min_val = min(np.min(self.predicted_trajectories[:, i]), 
                         np.min(self.converged_trajectories[:, i]))
            max_val = max(np.max(self.predicted_trajectories[:, i]), 
                         np.max(self.converged_trajectories[:, i]))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Predicted (Diffusion Model)')
            ax.set_ylabel('Converged (SNOPT)')
            ax.set_title(f'{self.param_names[i]}\nMAE: {self.metrics["mean_abs_error"][i]:.4f}')
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient (with safe calculation)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    corr = np.corrcoef(self.predicted_trajectories[:, i], 
                                      self.converged_trajectories[:, i])[0, 1]
                    if not np.isfinite(corr):
                        corr = 0.0
                except:
                    corr = 0.0
            ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create colorbar separately
        fig, ax = plt.subplots(figsize=(8, 1))
        im = ax.scatter([], [], c=[], cmap='viridis')
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, orientation='horizontal')
        cbar.set_label('Halo Energy')
        ax.remove()
        plt.savefig(os.path.join(self.output_dir, 'halo_energy_colorbar.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
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
    
    def plot_key_metrics_comparison(self):
        """Plot comparison of key trajectory metrics."""
        print("Creating key metrics comparison...")
        
        # Extract key metrics
        predicted_metrics = []
        converged_metrics = []
        
        for entry in self.feasible_data:
            # Predicted metrics (from trajectory parameters)
            pred_params = np.array(entry['generated_sample']['trajectory_params'])
            predicted_metrics.append({
                'shooting_time': pred_params[0],
                'initial_coast': pred_params[1],
                'final_coast': pred_params[2],
                'final_fuel_mass': pred_params[-3],
                'halo_period': pred_params[-2],
                'manifold_length': pred_params[-1]
            })
            
            # Converged metrics (from SNOPT solution)
            conv_params = entry['converged_trajectory']['trajectory_parameters']
            converged_metrics.append(conv_params)
        
        # Create comparison plots
        metrics_to_plot = ['shooting_time', 'final_fuel_mass', 'manifold_length']
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
        
        for i, metric in enumerate(metrics_to_plot):
            pred_values = [m[metric] for m in predicted_metrics]
            conv_values = [m[metric] for m in converged_metrics]
            
            axes[i].scatter(pred_values, conv_values, alpha=0.7, s=80, 
                           c=self.halo_energies, cmap='viridis')
            
            # Perfect prediction line
            min_val = min(min(pred_values), min(conv_values))
            max_val = max(max(pred_values), max(conv_values))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            axes[i].set_xlabel(f'Predicted {metric.replace("_", " ").title()}')
            axes[i].set_ylabel(f'Converged {metric.replace("_", " ").title()}')
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].grid(True, alpha=0.3)
            
            # Add correlation (with safe calculation)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    corr = np.corrcoef(pred_values, conv_values)[0, 1]
                    if not np.isfinite(corr):
                        corr = 0.0
                except:
                    corr = 0.0
            axes[i].text(0.05, 0.95, f'r={corr:.3f}', transform=axes[i].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'key_metrics_comparison.png'), 
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
    
    def plot_physical_trajectory_comparison(self):
        """Plot comparison of physical orbital trajectories."""
        print("Creating physical trajectory comparison plots...")
        
        # Create a comprehensive trajectory comparison plot
        n_trajectories = len(self.valid_physical_indices)
        if n_trajectories == 0:
            print("No valid trajectory pairs for physical comparison")
            return
        
        # Plot individual trajectories in 3D space
        fig = plt.figure(figsize=(20, 15))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
        
        for i, idx in enumerate(self.valid_physical_indices[:5]):  # Limit to first 5 for clarity
            pred_traj = self.predicted_physical_trajectories[idx]
            conv_traj = self.converged_physical_trajectories[idx]
            
            # Extract positions
            pred_pos = pred_traj[:, :3]
            conv_pos = conv_traj[:, :3]
            
            ax1.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
                    '--', color=colors[i], alpha=0.7, linewidth=2, 
                    label=f'Predicted {idx}' if i < 3 else "")
            ax1.plot(conv_pos[:, 0], conv_pos[:, 1], conv_pos[:, 2], 
                    '-', color=colors[i], alpha=0.9, linewidth=2,
                    label=f'Converged {idx}' if i < 3 else "")
        
        ax1.set_xlabel('X (DU)')
        ax1.set_ylabel('Y (DU)')
        ax1.set_zlabel('Z (DU)')
        ax1.set_title('3D Trajectory Comparison\n(Dashed=Predicted, Solid=Converged)')
        ax1.legend()
        
        # 2D X-Y projection
        ax2 = fig.add_subplot(2, 3, 2)
        for i, idx in enumerate(self.valid_physical_indices[:5]):
            pred_traj = self.predicted_physical_trajectories[idx]
            conv_traj = self.converged_physical_trajectories[idx]
            
            pred_pos = pred_traj[:, :3]
            conv_pos = conv_traj[:, :3]
            
            ax2.plot(pred_pos[:, 0], pred_pos[:, 1], '--', color=colors[i], alpha=0.7, linewidth=2)
            ax2.plot(conv_pos[:, 0], conv_pos[:, 1], '-', color=colors[i], alpha=0.9, linewidth=2)
        
        ax2.set_xlabel('X (DU)')
        ax2.set_ylabel('Y (DU)')
        ax2.set_title('X-Y Plane Projection')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Position deviation over time
        ax3 = fig.add_subplot(2, 3, 3)
        for i, idx in enumerate(self.valid_physical_indices[:5]):
            deviation = self.trajectory_deviations[i]['position_deviation']
            time_points = np.linspace(0, 1, len(deviation))
            ax3.plot(time_points, deviation, color=colors[i], linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Normalized Time')
        ax3.set_ylabel('Position Deviation (DU)')
        ax3.set_title('Position Deviation Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Velocity deviation over time
        ax4 = fig.add_subplot(2, 3, 4)
        for i, idx in enumerate(self.valid_physical_indices[:5]):
            deviation = self.trajectory_deviations[i]['velocity_deviation']
            time_points = np.linspace(0, 1, len(deviation))
            ax4.plot(time_points, deviation, color=colors[i], linewidth=2, alpha=0.8)
        
        ax4.set_xlabel('Normalized Time')
        ax4.set_ylabel('Velocity Deviation (DU/TU)')
        ax4.set_title('Velocity Deviation Over Time')
        ax4.grid(True, alpha=0.3)
        
        # Maximum deviation vs halo energy
        ax5 = fig.add_subplot(2, 3, 5)
        valid_halo_energies = [self.physical_halo_energies[i] for i in self.valid_physical_indices]
        ax5.scatter(valid_halo_energies, self.max_position_deviations, 
                   alpha=0.7, s=80, c='blue', label='Position')
        ax5.scatter(valid_halo_energies, self.max_velocity_deviations, 
                   alpha=0.7, s=80, c='red', label='Velocity')
        ax5.set_xlabel('Halo Energy')
        ax5.set_ylabel('Maximum Deviation')
        ax5.set_title('Max Deviation vs Halo Energy')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Final position deviation
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.hist(self.final_position_deviations, bins=min(10, len(self.final_position_deviations)), 
                alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Final Position Deviation (DU)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Distribution of Final Position Deviations')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'physical_trajectory_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_manifold_comparison(self):
        """Plot comparison of manifold arcs."""
        print("Creating manifold comparison plots...")
        
        manifold_data = []
        for entry in self.physical_data:
            if ('predicted_manifold' in entry['physical_trajectories'] and 
                'converged_manifold' in entry['physical_trajectories'] and
                entry['physical_trajectories']['predicted_manifold'] is not None and
                entry['physical_trajectories']['converged_manifold'] is not None):
                
                pred_manifold = np.array(entry['physical_trajectories']['predicted_manifold'])
                conv_manifold = np.array(entry['physical_trajectories']['converged_manifold'])
                manifold_data.append({
                    'predicted': pred_manifold,
                    'converged': conv_manifold,
                    'sample_idx': entry['sample_idx'],
                    'halo_energy': entry['generated_sample']['halo_energy']
                })
        
        if not manifold_data:
            print("No manifold data available for comparison")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(manifold_data)))
        
        for i, data in enumerate(manifold_data[:5]):  # Limit to first 5
            pred_manifold = data['predicted']
            conv_manifold = data['converged']
            
            # X-Y projection
            axes[0].plot(pred_manifold[:, 0], pred_manifold[:, 1], '--', 
                        color=colors[i], alpha=0.7, linewidth=2, 
                        label=f'Predicted {data["sample_idx"]}' if i < 3 else "")
            axes[0].plot(conv_manifold[:, 0], conv_manifold[:, 1], '-', 
                        color=colors[i], alpha=0.9, linewidth=2,
                        label=f'Converged {data["sample_idx"]}' if i < 3 else "")
        
        axes[0].set_xlabel('X (DU)')
        axes[0].set_ylabel('Y (DU)')
        axes[0].set_title('Manifold Arc Comparison (X-Y)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # Manifold deviation analysis
        manifold_deviations = []
        for data in manifold_data:
            pred_manifold = data['predicted']
            conv_manifold = data['converged']
            
            # Ensure same length
            min_length = min(len(pred_manifold), len(conv_manifold))
            pred_manifold = pred_manifold[:min_length]
            conv_manifold = conv_manifold[:min_length]
            
            # Compute position deviation
            deviation = np.linalg.norm(pred_manifold[:, :3] - conv_manifold[:, :3], axis=1)
            manifold_deviations.append(np.mean(deviation))
        
        axes[1].hist(manifold_deviations, bins=min(10, len(manifold_deviations)), 
                    alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Mean Manifold Deviation (DU)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Manifold Deviations')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'manifold_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
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
        self.plot_parameter_comparison()
        self.plot_error_distribution()
        self.plot_key_metrics_comparison()
        self.plot_control_segments_comparison()
        
        # Physical trajectory analysis (if data available)
        if hasattr(self, 'physical_data') and self.physical_data:
            print("\nStarting physical trajectory analysis...")
            self.extract_physical_trajectory_data()
            if hasattr(self, 'valid_physical_indices') and self.valid_physical_indices:
                self.compute_trajectory_deviations()
                self.plot_physical_trajectory_comparison()
                self.plot_manifold_comparison()
                print("Physical trajectory analysis complete!")
            else:
                print("No valid physical trajectory pairs found")
        else:
            print("No physical trajectory data available")
        
        self.create_summary_report()
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        print(f"Key plots created:")
        print(f"  - parameter_comparison.png: Parameter-by-parameter comparisons")
        print(f"  - error_distribution.png: Error distribution analysis")
        print(f"  - key_metrics_comparison.png: Key trajectory metrics comparison")
        print(f"  - control_segments_comparison.png: Control profile comparison")
        if hasattr(self, 'valid_physical_indices') and self.valid_physical_indices:
            print(f"  - physical_trajectory_comparison.png: Physical orbital path comparisons")
            print(f"  - manifold_comparison.png: Manifold arc comparisons")
        print(f"  - analysis_summary.txt: Human-readable summary")


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