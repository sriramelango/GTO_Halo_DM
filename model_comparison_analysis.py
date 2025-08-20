import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import pickle

class ModelComparisonAnalyzer:
    def __init__(self, original_dataset_path, three_channel_dataset_path, output_dir="model_comparison_results"):
        self.original_dataset_path = original_dataset_path
        self.three_channel_dataset_path = three_channel_dataset_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load datasets
        self.load_datasets()
        
    def load_datasets(self):
        """Load both datasets and extract key metrics."""
        print("Loading original model dataset...")
        with open(self.original_dataset_path, 'rb') as f:
            self.original_data = pickle.load(f)
            
        print("Loading 3-channel model dataset...")
        with open(self.three_channel_dataset_path, 'rb') as f:
            self.three_channel_data = pickle.load(f)
            
        # Extract feasibility rates
        self.extract_feasibility_rates()
        
        # Extract position/velocity deviations
        self.extract_deviation_metrics()
        
    def extract_feasibility_rates(self):
        """Extract feasibility rates for both models."""
        # Original model
        original_feasible = sum(1 for sample in self.original_data if sample.get('snopt_results', {}).get('feasibility', False))
        self.original_feasibility_rate = original_feasible / len(self.original_data)
        self.original_total_samples = len(self.original_data)
        self.original_feasible_samples = original_feasible
        
        # 3-channel model
        three_channel_feasible = sum(1 for sample in self.three_channel_data if sample.get('snopt_results', {}).get('feasibility', False))
        self.three_channel_feasibility_rate = three_channel_feasible / len(self.three_channel_data)
        self.three_channel_total_samples = len(self.three_channel_data)
        self.three_channel_feasible_samples = three_channel_feasible
        
        print(f"Original Model: {self.original_feasible_samples}/{self.original_total_samples} feasible ({self.original_feasibility_rate:.1%})")
        print(f"3-Channel Model: {self.three_channel_feasible_samples}/{self.three_channel_total_samples} feasible ({self.three_channel_feasibility_rate:.1%})")
        
    def extract_deviation_metrics(self):
        """Extract position and velocity deviation metrics for both models."""
        # Original model deviations
        self.original_position_deviations = []
        self.original_velocity_deviations = []
        self.original_aggregate_position_deviations = []
        self.original_aggregate_velocity_deviations = []
        
        for sample in self.original_data:
            if sample.get('snopt_results', {}).get('feasibility', False) and 'physical_trajectories' in sample:
                trajectory_data = sample['physical_trajectories']
                if 'converged_states' in trajectory_data and 'predicted_states' in trajectory_data:
                    # Calculate deviations between converged and predicted states
                    converged_states = np.array(trajectory_data['converged_states'])
                    predicted_states = np.array(trajectory_data['predicted_states'])
                    
                    if len(converged_states) > 0 and len(predicted_states) > 0:
                        # Use minimum length to avoid shape mismatch
                        min_length = min(len(converged_states), len(predicted_states))
                        if min_length > 0:
                            # Calculate final position and velocity deviations
                            final_converged = converged_states[-1]
                            final_predicted = predicted_states[-1]
                            
                            # Calculate final position deviation (first 3 elements)
                            pos_dev = np.linalg.norm(final_converged[:3] - final_predicted[:3])
                            # Calculate final velocity deviation (elements 3-6)
                            vel_dev = np.linalg.norm(final_converged[3:6] - final_predicted[3:6])
                            
                            self.original_position_deviations.append(pos_dev)
                            self.original_velocity_deviations.append(vel_dev)
                            
                            # Calculate aggregate deviations across entire trajectory
                            # Truncate both arrays to minimum length
                            converged_truncated = converged_states[:min_length]
                            predicted_truncated = predicted_states[:min_length]
                            
                            # Calculate position deviations across all time steps
                            pos_deviations = np.linalg.norm(converged_truncated[:, :3] - predicted_truncated[:, :3], axis=1)
                            vel_deviations = np.linalg.norm(converged_truncated[:, 3:6] - predicted_truncated[:, 3:6], axis=1)
                            
                            # Calculate aggregate metrics (mean across entire trajectory)
                            aggregate_pos_dev = np.mean(pos_deviations)
                            aggregate_vel_dev = np.mean(vel_deviations)
                            
                            self.original_aggregate_position_deviations.append(aggregate_pos_dev)
                            self.original_aggregate_velocity_deviations.append(aggregate_vel_dev)
        
        # 3-channel model deviations
        self.three_channel_position_deviations = []
        self.three_channel_velocity_deviations = []
        self.three_channel_aggregate_position_deviations = []
        self.three_channel_aggregate_velocity_deviations = []
        
        for sample in self.three_channel_data:
            if sample.get('snopt_results', {}).get('feasibility', False) and 'physical_trajectories' in sample:
                trajectory_data = sample['physical_trajectories']
                if 'converged_states' in trajectory_data and 'predicted_states' in trajectory_data:
                    # Calculate deviations between converged and predicted states
                    converged_states = np.array(trajectory_data['converged_states'])
                    predicted_states = np.array(trajectory_data['predicted_states'])
                    
                    if len(converged_states) > 0 and len(predicted_states) > 0:
                        # Use minimum length to avoid shape mismatch
                        min_length = min(len(converged_states), len(predicted_states))
                        if min_length > 0:
                            # Calculate final position and velocity deviations
                            final_converged = converged_states[-1]
                            final_predicted = predicted_states[-1]
                            
                            # Calculate final position deviation (first 3 elements)
                            pos_dev = np.linalg.norm(final_converged[:3] - final_predicted[:3])
                            # Calculate final velocity deviation (elements 3-6)
                            vel_dev = np.linalg.norm(final_converged[3:6] - final_predicted[3:6])
                            
                            self.three_channel_position_deviations.append(pos_dev)
                            self.three_channel_velocity_deviations.append(vel_dev)
                            
                            # Calculate aggregate deviations across entire trajectory
                            # Truncate both arrays to minimum length
                            converged_truncated = converged_states[:min_length]
                            predicted_truncated = predicted_states[:min_length]
                            
                            # Calculate position deviations across all time steps
                            pos_deviations = np.linalg.norm(converged_truncated[:, :3] - predicted_truncated[:, :3], axis=1)
                            vel_deviations = np.linalg.norm(converged_truncated[:, 3:6] - predicted_truncated[:, 3:6], axis=1)
                            
                            # Calculate aggregate metrics (mean across entire trajectory)
                            aggregate_pos_dev = np.mean(pos_deviations)
                            aggregate_vel_dev = np.mean(vel_deviations)
                            
                            self.three_channel_aggregate_position_deviations.append(aggregate_pos_dev)
                            self.three_channel_aggregate_velocity_deviations.append(aggregate_vel_dev)
        
        # Calculate statistics
        self.calculate_deviation_statistics()
        
        # Extract solving times for feasible samples
        self.original_solving_times = []
        self.three_channel_solving_times = []
        
        for sample in self.original_data:
            if sample.get('snopt_results', {}).get('feasibility', False):
                solving_time = sample.get('snopt_results', {}).get('solving_time', 0)
                if solving_time > 0:
                    self.original_solving_times.append(solving_time)
        
        for sample in self.three_channel_data:
            if sample.get('snopt_results', {}).get('feasibility', False):
                solving_time = sample.get('snopt_results', {}).get('solving_time', 0)
                if solving_time > 0:
                    self.three_channel_solving_times.append(solving_time)
        
        # Calculate solving time statistics
        if self.original_solving_times:
            self.original_solve_mean = np.mean(self.original_solving_times)
            self.original_solve_std = np.std(self.original_solving_times)
        else:
            self.original_solve_mean = self.original_solve_std = 0
            
        if self.three_channel_solving_times:
            self.three_channel_solve_mean = np.mean(self.three_channel_solving_times)
            self.three_channel_solve_std = np.std(self.three_channel_solving_times)
        else:
            self.three_channel_solve_mean = self.three_channel_solve_std = 0
            
        print(f"\nSNOPT Solving Time Statistics (Feasible Samples):")
        print(f"Original Model: {self.original_solve_mean:.2f}±{self.original_solve_std:.2f}s ({len(self.original_solving_times)} samples)")
        print(f"3-Channel Model: {self.three_channel_solve_mean:.2f}±{self.three_channel_solve_std:.2f}s ({len(self.three_channel_solving_times)} samples)")
        
    def calculate_deviation_statistics(self):
        """Calculate statistics for position and velocity deviations."""
        # Original model statistics
        if self.original_position_deviations:
            self.original_pos_mean = np.mean(self.original_position_deviations)
            self.original_pos_std = np.std(self.original_position_deviations)
            self.original_pos_median = np.median(self.original_position_deviations)
        else:
            self.original_pos_mean = self.original_pos_std = self.original_pos_median = 0
            
        if self.original_velocity_deviations:
            self.original_vel_mean = np.mean(self.original_velocity_deviations)
            self.original_vel_std = np.std(self.original_velocity_deviations)
            self.original_vel_median = np.median(self.original_velocity_deviations)
        else:
            self.original_vel_mean = self.original_vel_std = self.original_vel_median = 0
            
        # Original model aggregate statistics
        if self.original_aggregate_position_deviations:
            self.original_agg_pos_mean = np.mean(self.original_aggregate_position_deviations)
            self.original_agg_pos_std = np.std(self.original_aggregate_position_deviations)
            self.original_agg_pos_median = np.median(self.original_aggregate_position_deviations)
        else:
            self.original_agg_pos_mean = self.original_agg_pos_std = self.original_agg_pos_median = 0
            
        if self.original_aggregate_velocity_deviations:
            self.original_agg_vel_mean = np.mean(self.original_aggregate_velocity_deviations)
            self.original_agg_vel_std = np.std(self.original_aggregate_velocity_deviations)
            self.original_agg_vel_median = np.median(self.original_aggregate_velocity_deviations)
        else:
            self.original_agg_vel_mean = self.original_agg_vel_std = self.original_agg_vel_median = 0
            
        # 3-channel model statistics
        if self.three_channel_position_deviations:
            self.three_channel_pos_mean = np.mean(self.three_channel_position_deviations)
            self.three_channel_pos_std = np.std(self.three_channel_position_deviations)
            self.three_channel_pos_median = np.median(self.three_channel_position_deviations)
        else:
            self.three_channel_pos_mean = self.three_channel_pos_std = self.three_channel_pos_median = 0
            
        if self.three_channel_velocity_deviations:
            self.three_channel_vel_mean = np.mean(self.three_channel_velocity_deviations)
            self.three_channel_vel_std = np.std(self.three_channel_velocity_deviations)
            self.three_channel_vel_median = np.median(self.three_channel_velocity_deviations)
        else:
            self.three_channel_vel_mean = self.three_channel_vel_std = self.three_channel_vel_median = 0
            
        # 3-channel model aggregate statistics
        if self.three_channel_aggregate_position_deviations:
            self.three_channel_agg_pos_mean = np.mean(self.three_channel_aggregate_position_deviations)
            self.three_channel_agg_pos_std = np.std(self.three_channel_aggregate_position_deviations)
            self.three_channel_agg_pos_median = np.median(self.three_channel_aggregate_position_deviations)
        else:
            self.three_channel_agg_pos_mean = self.three_channel_agg_pos_std = self.three_channel_agg_pos_median = 0
            
        if self.three_channel_aggregate_velocity_deviations:
            self.three_channel_agg_vel_mean = np.mean(self.three_channel_aggregate_velocity_deviations)
            self.three_channel_agg_vel_std = np.std(self.three_channel_aggregate_velocity_deviations)
            self.three_channel_agg_vel_median = np.median(self.three_channel_aggregate_velocity_deviations)
        else:
            self.three_channel_agg_vel_mean = self.three_channel_agg_vel_std = self.three_channel_agg_vel_median = 0
            
        print(f"\nFinal Deviation Statistics:")
        print(f"Original Model - Position: mean={self.original_pos_mean:.4f}, std={self.original_pos_std:.4f}, median={self.original_pos_median:.4f}")
        print(f"Original Model - Velocity: mean={self.original_vel_mean:.4f}, std={self.original_vel_std:.4f}, median={self.original_vel_median:.4f}")
        print(f"3-Channel Model - Position: mean={self.three_channel_pos_mean:.4f}, std={self.three_channel_pos_std:.4f}, median={self.three_channel_pos_median:.4f}")
        print(f"3-Channel Model - Velocity: mean={self.three_channel_vel_mean:.4f}, std={self.three_channel_vel_std:.4f}, median={self.three_channel_vel_median:.4f}")
        
        print(f"\nAggregate Deviation Statistics (Across Entire Trajectory):")
        print(f"Original Model - Position: mean={self.original_agg_pos_mean:.4f}, std={self.original_agg_pos_std:.4f}, median={self.original_agg_pos_median:.4f}")
        print(f"Original Model - Velocity: mean={self.original_agg_vel_mean:.4f}, std={self.original_agg_vel_std:.4f}, median={self.original_agg_vel_median:.4f}")
        print(f"3-Channel Model - Position: mean={self.three_channel_agg_pos_mean:.4f}, std={self.three_channel_agg_pos_std:.4f}, median={self.three_channel_agg_pos_median:.4f}")
        print(f"3-Channel Model - Velocity: mean={self.three_channel_agg_vel_mean:.4f}, std={self.three_channel_agg_vel_std:.4f}, median={self.three_channel_agg_vel_median:.4f}")
        
    def create_comparison_plot(self):
        """Create a comprehensive comparison plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Feasibility Rate Comparison
        models = ['Original Model', '3-Channel Model']
        feasibility_rates = [self.original_feasibility_rate, self.three_channel_feasibility_rate]
        feasible_counts = [self.original_feasible_samples, self.three_channel_feasible_samples]
        total_counts = [self.original_total_samples, self.three_channel_total_samples]
        
        bars = ax1.bar(models, feasibility_rates, color=['#ff7f0e', '#2ca02c'], alpha=0.7)
        ax1.set_ylabel('Feasibility Rate')
        ax1.set_title('Model Feasibility Rate Comparison')
        ax1.set_ylim(0, 1)
        
        # Add percentage labels on bars
        for i, (bar, rate, feasible, total) in enumerate(zip(bars, feasibility_rates, feasible_counts, total_counts)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}\n({feasible}/{total})', ha='center', va='bottom', fontweight='bold')
        
        # 2. Aggregate Position Deviation Comparison
        if self.original_aggregate_position_deviations and self.three_channel_aggregate_position_deviations:
            ax2.hist(self.original_aggregate_position_deviations, bins=20, alpha=0.7, label='Original Model', 
                    color='#ff7f0e', density=True)
            ax2.hist(self.three_channel_aggregate_position_deviations, bins=20, alpha=0.7, label='3-Channel Model', 
                    color='#2ca02c', density=True)
            ax2.set_xlabel('Aggregate Position Deviation (DU)')
            ax2.set_ylabel('Density')
            ax2.set_title('Aggregate Position Deviation Distribution\n(Across Entire Trajectory)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No aggregate position deviation data available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Aggregate Position Deviation Distribution')
        
        # 3. Aggregate Velocity Deviation Comparison
        if self.original_aggregate_velocity_deviations and self.three_channel_aggregate_velocity_deviations:
            ax3.hist(self.original_aggregate_velocity_deviations, bins=20, alpha=0.7, label='Original Model', 
                    color='#ff7f0e', density=True)
            ax3.hist(self.three_channel_aggregate_velocity_deviations, bins=20, alpha=0.7, label='3-Channel Model', 
                    color='#2ca02c', density=True)
            ax3.set_xlabel('Aggregate Velocity Deviation (DU/TU)')
            ax3.set_ylabel('Density')
            ax3.set_title('Aggregate Velocity Deviation Distribution\n(Across Entire Trajectory)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No aggregate velocity deviation data available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Aggregate Velocity Deviation Distribution')
        
        # 4. SNOPT Solving Time Comparison
        # Create bar plot with error bars
        models = ['Original Model', '3-Channel Model']
        solve_means = [self.original_solve_mean, self.three_channel_solve_mean]
        solve_stds = [self.original_solve_std, self.three_channel_solve_std]
        
        bars = ax4.bar(models, solve_means, yerr=solve_stds, capsize=5, 
                       color=['#ff7f0e', '#2ca02c'], alpha=0.7)
        ax4.set_ylabel('SNOPT Solving Time (seconds)')
        ax4.set_title('Average SNOPT Solving Time Comparison\n(Feasible Samples Only)')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, solve_means, solve_stds)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                    f'{mean:.2f}±{std:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3)
        
        # Calculate improvements for JSON
        feasibility_improvement = 0
        if self.original_feasibility_rate > 0:
            feasibility_improvement = ((self.three_channel_feasibility_rate - self.original_feasibility_rate) / self.original_feasibility_rate * 100)
        
        position_improvement = 0
        if self.original_pos_mean > 0:
            position_improvement = ((self.original_pos_mean - self.three_channel_pos_mean) / self.original_pos_mean * 100)
            
        velocity_improvement = 0
        if self.original_vel_mean > 0:
            velocity_improvement = ((self.original_vel_mean - self.three_channel_vel_mean) / self.original_vel_mean * 100)
            
        agg_position_improvement = 0
        if self.original_agg_pos_mean > 0:
            agg_position_improvement = ((self.original_agg_pos_mean - self.three_channel_agg_pos_mean) / self.original_agg_pos_mean * 100)
            
        agg_velocity_improvement = 0
        if self.original_agg_vel_mean > 0:
            agg_velocity_improvement = ((self.original_agg_vel_mean - self.three_channel_agg_vel_mean) / self.original_agg_vel_mean * 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed statistics to JSON
        comparison_stats = {
            'timestamp': datetime.now().isoformat(),
            'original_model': {
                'dataset_path': self.original_dataset_path,
                'total_samples': self.original_total_samples,
                'feasible_samples': self.original_feasible_samples,
                'feasibility_rate': float(self.original_feasibility_rate),
                'position_deviations': {
                    'mean': float(self.original_pos_mean),
                    'std': float(self.original_pos_std),
                    'median': float(self.original_pos_median),
                    'values': [float(x) for x in self.original_position_deviations]
                },
                'velocity_deviations': {
                    'mean': float(self.original_vel_mean),
                    'std': float(self.original_vel_std),
                    'median': float(self.original_vel_median),
                    'values': [float(x) for x in self.original_velocity_deviations]
                },
                'aggregate_position_deviations': {
                    'mean': float(self.original_agg_pos_mean),
                    'std': float(self.original_agg_pos_std),
                    'median': float(self.original_agg_pos_median),
                    'values': [float(x) for x in self.original_aggregate_position_deviations]
                },
                'aggregate_velocity_deviations': {
                    'mean': float(self.original_agg_vel_mean),
                    'std': float(self.original_agg_vel_std),
                    'median': float(self.original_agg_vel_median),
                    'values': [float(x) for x in self.original_aggregate_velocity_deviations]
                },
                'solving_times': {
                    'mean': float(self.original_solve_mean),
                    'std': float(self.original_solve_std),
                    'values': [float(x) for x in self.original_solving_times]
                }
            },
            'three_channel_model': {
                'dataset_path': self.three_channel_dataset_path,
                'total_samples': self.three_channel_total_samples,
                'feasible_samples': self.three_channel_feasible_samples,
                'feasibility_rate': float(self.three_channel_feasibility_rate),
                'position_deviations': {
                    'mean': float(self.three_channel_pos_mean),
                    'std': float(self.three_channel_pos_std),
                    'median': float(self.three_channel_pos_median),
                    'values': [float(x) for x in self.three_channel_position_deviations]
                },
                'velocity_deviations': {
                    'mean': float(self.three_channel_vel_mean),
                    'std': float(self.three_channel_vel_std),
                    'median': float(self.three_channel_vel_median),
                    'values': [float(x) for x in self.three_channel_velocity_deviations]
                },
                'aggregate_position_deviations': {
                    'mean': float(self.three_channel_agg_pos_mean),
                    'std': float(self.three_channel_agg_pos_std),
                    'median': float(self.three_channel_agg_pos_median),
                    'values': [float(x) for x in self.three_channel_aggregate_position_deviations]
                },
                'aggregate_velocity_deviations': {
                    'mean': float(self.three_channel_agg_vel_mean),
                    'std': float(self.three_channel_agg_vel_std),
                    'median': float(self.three_channel_agg_vel_median),
                    'values': [float(x) for x in self.three_channel_aggregate_velocity_deviations]
                },
                'solving_times': {
                    'mean': float(self.three_channel_solve_mean),
                    'std': float(self.three_channel_solve_std),
                    'values': [float(x) for x in self.three_channel_solving_times]
                }
            },
            'improvements': {
                'feasibility_rate_improvement': float(feasibility_improvement),
                'position_accuracy_improvement': float(position_improvement),
                'velocity_accuracy_improvement': float(velocity_improvement),
                'aggregate_position_accuracy_improvement': float(agg_position_improvement),
                'aggregate_velocity_accuracy_improvement': float(agg_velocity_improvement),
                'solving_time_improvement': float(((self.original_solve_mean - self.three_channel_solve_mean) / self.original_solve_mean * 100) if self.original_solve_mean > 0 else 0)
            }
        }
        
        stats_path = os.path.join(self.output_dir, 'model_comparison_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(comparison_stats, f, indent=2)
        
        print(f"\nModel comparison analysis completed!")
        print(f"  - Comparison plot: {os.path.join(self.output_dir, 'model_comparison_analysis.png')}")
        print(f"  - Statistics: {os.path.join(self.output_dir, 'model_comparison_statistics.json')}")

def main():
    # Define dataset paths
    original_dataset = "Benchmark Results/benchmark_original_2025-08-04_15-59-21_samples_250_merged/comprehensive_dataset/complete_dataset.pkl"
    three_channel_dataset = "Benchmark Results/benchmark_2025-08-04_15-59-28_samples_250_merged/comprehensive_dataset/complete_dataset.pkl"
    
    # Create analyzer and run comparison
    analyzer = ModelComparisonAnalyzer(original_dataset, three_channel_dataset)
    analyzer.create_comparison_plot()

if __name__ == "__main__":
    main() 